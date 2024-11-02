import torch
import time
import math
from torch.nn import functional as F
from torch import nn

device = "mps" if torch.backends.mps.is_available() else "cpu"


class LoRALayer(nn.Module):
    def __init__(self, weight, bias, rank=8, alpha=8):
        super(LoRALayer, self).__init__()

        row, column = weight.shape

        # Restore Linear layer
        if bias is None:
            self.linear = nn.Linear(column, row, bias=False)
            self.linear.load_state_dict({"weight": weight})
        else:
            self.linear = nn.Linear(column, row)
            self.linear.load_state_dict({"weight": weight, "bias": bias})

        # Create LoRA weights with initialization
        self.lora_A = nn.Parameter(torch.zeros(column, rank))  # Weight for LoRA
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_B = nn.Parameter(torch.zeros(rank, row))  # Bias for LoRA
        nn.init.zeros_(self.lora_B)

        # Scaling factor for the low-rank adaptation
        self.scaling = alpha / rank

    def forward(self, input):
        # Standard linear transformation
        x = self.linear(input)

        # Low-rank adaptation
        # Using the alpha to scale the contribution of the LoRA weights
        y = self.scaling * (input @ self.lora_A @ self.lora_B)

        return x + y


class TLoRALayer(nn.Module):
    def __init__(self, weight, bias, rank=8, alpha=8):
        super(TLoRALayer, self).__init__()
        rank = rank - 1

        row, column = weight.shape

        # Restore Linear layer
        if bias is None:
            self.linear = nn.Linear(column, row, bias=False)
            self.linear.load_state_dict({"weight": weight})
        else:
            self.linear = nn.Linear(column, row)
            self.linear.load_state_dict({"weight": weight, "bias": bias})

        # Create TLoRA weights with initialization
        self.lora_A = nn.Parameter(torch.zeros(column, rank))  # First matrix
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_B = nn.Parameter(torch.zeros(rank, rank))  # Second matrix
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

        self.lora_C = nn.Parameter(torch.zeros(rank, row))  # Third matrix
        nn.init.zeros_(self.lora_C)


    def forward(self, input):
        # Standard linear transformation
        x = self.linear(input)

        # Low-rank adaptation with tri-matrix TLoRA
        # Using the scaling to control the LoRA output
        # scaling = self.lora_B.abs().sum()
        scaling = torch.maximum(
            self.lora_B.abs().sum(), torch.tensor(1.0, device=input.device)
        )
        y = scaling * (input @ self.lora_A @ self.lora_B @ self.lora_C)

        return x + y


def create_lora_model(model, rank=8, lora_type="lora"):
    # Get target module name
    target_names = []

    # Check the model type and find the appropriate layers for LoRA
    model_type = model.config.model_type

    if model_type == "roberta":
        # RoBERTa: Look for Linear layers within the RoBERTa model's attention and output layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "encoder.layer." in name:
                if "attention.self.query" in name or "attention.self.key" in name:
                    #'attention.self.value' in name or \
                    #'intermediate.dense' in name or \
                    #'output.dense' in name
                    target_names.append(name)

    elif model_type == "opt":
        # OPT: Look for Linear layers within the decoder
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "decoder.layers." in name:
                if "self_attn.q_proj" in name or "self_attn.v_proj" in name:
                    target_names.append(name)

    else:
        raise ValueError(f"Model type '{model_type}' is not supported for LoRA.")

    # Replace each target module with LoRA
    for name in target_names:
        name_struct = name.split(".")
        # Get target module
        module_list = [model]
        for struct in name_struct:
            module_list.append(getattr(module_list[-1], struct))
        # Build LoRA
        if lora_type == "lora":
            lora = LoRALayer(
                weight = module_list[-1].weight,
                bias = module_list[-1].bias,
                rank = rank,
            ).to(device)
        elif lora_type == "tlora":
            lora = TLoRALayer(
                weight = module_list[-1].weight,
                bias = module_list[-1].bias,
                rank = rank,
            ).to(device)
        # Replace with the LoRA layer
        module_list[-2].__setattr__(name_struct[-1], lora)

    # Set requires_grad based on the parameter names
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def get_accuracy(logits, labels):
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == labels).sum().item() / len(labels)
    return accuracy


def train_one_epoch(model, train_loader, optimizer):
    loss_function = torch.nn.CrossEntropyLoss()
    model.train()
    total_loss, total_acc = 0, 0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute loss and accuracy
        loss = loss_function(outputs.logits, labels)
        acc = get_accuracy(outputs.logits, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss and accuracy
        total_loss += loss.item()
        total_acc += acc

        # Log batch progress every 20% of total batches
        if (batch_idx + 1) % (len(train_loader) // 5) == 0:
            print(
                f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f} - Accuracy: {acc:.4f}"
            )

    # Calculate average loss and accuracy for the epoch
    train_loss = total_loss / len(train_loader)
    train_acc = total_acc / len(train_loader)
    return model, train_loss, train_acc


def validate(model, val_loader):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Compute loss and accuracy
            loss = loss_function(outputs.logits, labels)
            acc = get_accuracy(outputs.logits, labels)

            # Accumulate loss and accuracy
            total_loss += loss.item()
            total_acc += acc

    # Calculate average loss and accuracy for validation
    avg_loss = total_loss / len(val_loader)
    avg_acc = total_acc / len(val_loader)
    return avg_loss, avg_acc


def train(model, train_loader, val_loader, epochs, optimizer):
    total_time = 0

    # Dictionary to store history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": [],
    }

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train for one epoch
        start_time = time.time()
        model, train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        end_time = time.time()
        epoch_time = end_time - start_time
        total_time += epoch_time

        # Validate after each epoch
        val_loss, val_acc = validate(model, val_loader)

        # Store metrics for this epoch
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(val_acc)

        # Log epoch results
        print(
            f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Time: {epoch_time:.2f}s"
        )

    # Best val acc
    history["best_val_acc"] = max(history["val_acc"])

    # Print average time per epoch
    avg_time_per_epoch = total_time / epochs
    print(f"Average Time per Epoch: {avg_time_per_epoch:.2f}s")

    return model, history
