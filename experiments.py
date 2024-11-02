from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSequenceClassification
from torchinfo import summary
import sys
import os
import torch
from itertools import product
import json

from data import (
    prepare_tokenized_dataloaders_mrpc,
    prepare_tokenized_dataloaders_cola,
    prepare_tokenized_dataloaders_rte,
    prepare_tokenized_dataloaders_sst2,
)
from tlora import create_lora_model, train, validate

device = "mps" if torch.backends.mps.is_available() else "cpu"


class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout  # Original stdout
        self.log = open(log_file, "w")  # Log file for writing

    def write(self, message):
        self.terminal.write(message)  # Write to console
        self.log.write(message)  # Write to log file

    def flush(self):
        # Needed for compatibility with Python's flush behavior
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def setup_logger(data_name, model_name, lora_type, log_dir="logs"):
    # Ensure the logs directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Define log file path
    short_model_name = model_name.split("/")[-1]
    log_file = os.path.join(
        log_dir, f"log_{data_name}_{short_model_name}_{lora_type}.txt"
    )

    # Redirect stdout to Logger
    sys.stdout = Logger(log_file)


def reset_logger():
    # Reset stdout to default and close log file explicitly
    sys.stdout.close()
    sys.stdout = sys.__stdout__


def run_experiment(data_name, model_name, lora_type, batch_size=32, rank=8, epochs=20):
    print(
        f"Running experiment with data: {data_name}, model: {model_name}, LoRA type: {lora_type}"
    )

    # Load dataset based on the provided data_name
    if data_name == "glue_mrpc":
        raw_datasets = load_dataset("glue", "mrpc")
    elif data_name == "glue_cola":
        raw_datasets = load_dataset("glue", "cola")
    elif data_name == "glue_rte":
        raw_datasets = load_dataset("glue", "rte")
    elif data_name == "glue_sst2":
        raw_datasets = load_dataset("glue", "sst2")
    else:
        raise ValueError(
            f"Dataset {
                data_name} not recognized. Please provide a valid dataset."
        )

    # Prepare tokenized dataloaders
    if data_name == "glue_mrpc":
        train_dataloader, val_dataloader = prepare_tokenized_dataloaders_mrpc(
            raw_datasets, model_name, batch_size=batch_size
        )
    elif data_name == "glue_cola":
        train_dataloader, val_dataloader = prepare_tokenized_dataloaders_cola(
            raw_datasets, model_name, batch_size=batch_size
        )
    elif data_name == "glue_rte":
        train_dataloader, val_dataloader = prepare_tokenized_dataloaders_rte(
            raw_datasets, model_name, batch_size=batch_size
        )
    elif data_name == "glue_sst2":
        train_dataloader, val_dataloader = prepare_tokenized_dataloaders_sst2(
            raw_datasets, model_name, batch_size=batch_size
        )

    # Base model
    config = AutoConfig.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    ).to(device)
    print(base_model)
    print(summary(base_model))

    # Create LoRA model/adapter
    lora_model = create_lora_model(base_model, rank=rank, lora_type=lora_type)
    print(lora_model)
    print(summary(lora_model))

    # Training setup
    optimizer = torch.optim.Adam(params=lora_model.parameters(), lr=1e-5)
    epochs = 20
    trained_model, history = train(
        lora_model, train_dataloader, val_dataloader, epochs, optimizer
    )

    # Validate and report
    avg_loss, avg_acc = validate(trained_model, val_dataloader)
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_acc:.4f}")

    # Save history
    short_model_name = model_name.split("/")[-1]
    loss_curve_file = os.path.join(
        "logs", f"loss_curve_{data_name}_{short_model_name}_{lora_type}.json"
    )
    with open(loss_curve_file, "w") as f:
        json.dump(history, f, indent=4)

    return None


# Define the experiment parameters
data_names = ["glue_mrpc", "glue_rte", "glue_cola", "glue_sst2"]
model_names = ["facebook/opt-125m", "FacebookAI/roberta-base"]
lora_types = ["tlora", "lora"]

# Loop through all combinations of parameters and run experiments
for data_name, model_name, lora_type in product(data_names, model_names, lora_types):
    # Set up the logger for each combination
    setup_logger(data_name, model_name, lora_type)

    # Run the experiment
    run_experiment(data_name, model_name, lora_type)

    # Reset logger
    reset_logger()
