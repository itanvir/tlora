import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.spatial.distance import cosine

plt.style.use("ggplot")


def plot_trainable_parameters():

    # Data for the bar chart
    methods_1 = ["Full FT", "LoRA"]
    trainable_parameters_1 = [355362819, 3145728]

    methods_2 = ["LoRA", "TLoRA"]
    trainable_parameters_2 = [3145728, 49200]

    # Create subplots
    fig, axes = plt.subplots(1, 2)

    # First subplot: Full FT vs LoRA
    axes[0].bar(methods_1, trainable_parameters_1, color=["#636EFA", "#EF553B"])
    axes[0].set_ylabel("Trainable parameters")
    axes[0].set_title("Full FT vs LoRA")

    # Add values on top of bars for the first subplot
    for i, value in enumerate(trainable_parameters_1):
        axes[0].text(
            i, value + 1000, f"{value:,}", ha="center", va="bottom", fontsize=10
        )

    # Second subplot: LoRA vs TLoRA
    axes[1].bar(methods_2, trainable_parameters_2, color=["#EF553B", "#00CC96"])
    axes[1].set_ylabel("Trainable parameters")
    axes[1].set_title("LoRA vs TLoRA")

    # Add values on top of bars for the second subplot
    for i, value in enumerate(trainable_parameters_2):
        axes[1].text(
            i, value + 1000, f"{value:,}", ha="center", va="bottom", fontsize=10
        )

    # Show the plots
    plt.tight_layout()
    plt.show()

    return None


def plot_train_val_loss(history_file):
    # Load LoRA history
    with open(history_file, "r") as f:
        history = json.load(f)

    train_loss_lora = history["train_loss"]
    val_loss_lora = history["val_loss"]

    # Epochs range
    epochs = range(1, len(train_loss_lora) + 1)

    # Create subplots: one for training loss and one for validation loss
    plt.figure()
    plt.plot(
        epochs,
        train_loss_lora,
        label="Training Loss",
        color="#636EFA",
        linestyle="-",
        marker="o",
    )
    plt.plot(
        epochs,
        val_loss_lora,
        label="Validation Loss",
        color="#EF553B",
        linestyle="-",
        marker="s",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_l2_norms(history_file):

    with open(history_file, "r") as f:
        history = json.load(f)

    scalings = history["scalings"]

    scalings_q = np.array([[layer[0] for layer in epoch] for epoch in scalings])
    scalings_v = np.array([[layer[1] for layer in epoch] for epoch in scalings])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for ax, scaling_matrix, title in zip(axes, [scalings_q, scalings_v], ["q", "v"]):
        im = ax.imshow(scaling_matrix.T, cmap="inferno", interpolation="nearest")
        ax.set_title(f"L2 Norms for {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Layer")

        cbar = fig.colorbar(im)

    plt.tight_layout()
    plt.show()


def plot_scalings(history_file):

    with open(history_file, "r") as f:
        history = json.load(f)

    scalings = history["scalings"]

    scalings_q = np.array([[layer[2] for layer in epoch] for epoch in scalings])
    scalings_v = np.array([[layer[3] for layer in epoch] for epoch in scalings])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for ax, scaling_matrix, title in zip(axes, [scalings_q, scalings_v], ["q", "v"]):
        im = ax.imshow(scaling_matrix.T, cmap="inferno", interpolation="nearest")
        ax.set_title(f"Scaling Factors for {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Layer")

        cbar = fig.colorbar(im)

    plt.tight_layout()
    plt.show()


def compute_transformation_matrix_tlora(W0, lora_A, lora_B, lora_C, scaling):
    return W0 + scaling * (lora_A @ lora_B @ lora_C)


def compute_transformation_matrix_lora(W0, lora_A, lora_B):
    return W0 + lora_A @ lora_B


def plot_eigenvalue_distribution(lora_model_file, tlora_model_file):

    model = torch.load(tlora_model_file)
    lora_A = (
        model.roberta.encoder.layer[0]
        .attention.self.query.random_A.detach()
        .cpu()
        .numpy()
    )
    lora_B = (
        model.roberta.encoder.layer[0]
        .attention.self.query.lora_B.detach()
        .cpu()
        .numpy()
    )
    lora_C = (
        model.roberta.encoder.layer[0]
        .attention.self.query.random_C.detach()
        .cpu()
        .numpy()
    )
    scaling = (
        model.roberta.encoder.layer[0]
        .attention.self.query.lora_scaling.detach()
        .cpu()
        .numpy()
    )
    W_0 = (
        model.roberta.encoder.layer[0]
        .attention.self.query.linear.weight.detach()
        .cpu()
        .numpy()
    )
    W_tlora = compute_transformation_matrix_tlora(W_0, lora_A, lora_B, lora_C, scaling)

    model = torch.load(lora_model_file)
    lora_A = (
        model.roberta.encoder.layer[0]
        .attention.self.query.lora_A.detach()
        .cpu()
        .numpy()
    )
    lora_B = (
        model.roberta.encoder.layer[0]
        .attention.self.query.lora_B.detach()
        .cpu()
        .numpy()
    )
    W_0 = (
        model.roberta.encoder.layer[0]
        .attention.self.query.linear.weight.detach()
        .cpu()
        .numpy()
    )
    W_lora = compute_transformation_matrix_lora(W_0, lora_A, lora_B)

    # Calculate eigenvalues for TLoRA and LoRA transformation matrices
    eigenvalues_tlora = np.linalg.eigvals(W_tlora)
    eigenvalues_lora = np.linalg.eigvals(W_lora)

    # Separate real positive and negative eigenvalues
    eigenvalues_tlora_pos = np.real(eigenvalues_tlora[eigenvalues_tlora > 0])
    eigenvalues_tlora_neg = np.abs(np.real(eigenvalues_tlora[eigenvalues_tlora < 0]))
    eigenvalues_lora_pos = np.real(eigenvalues_lora[eigenvalues_lora > 0])
    eigenvalues_lora_neg = np.abs(np.real(eigenvalues_lora[eigenvalues_lora < 0]))

    # Plot the absolute values on a log scale for better visibility
    plt.figure(figsize=(14, 6))

    # Positive eigenvalues (Log scale)
    plt.subplot(1, 2, 1)
    sns.kdeplot(eigenvalues_lora_pos, color="orange", label="LoRA", fill=True)
    sns.kdeplot(eigenvalues_tlora_pos, color="blue", label="TLoRA", fill=True)
    plt.xlabel("Eigenvalue Magnitude (Positive)")
    plt.ylabel("Density")
    plt.title("Positive Eigenvalue Distribution (TLoRA vs. LoRA)")
    plt.legend()

    # Negative eigenvalues (Log scale on absolute value)
    plt.subplot(1, 2, 2)
    sns.kdeplot(eigenvalues_lora_neg, color="orange", label="LoRA", fill=True)
    sns.kdeplot(eigenvalues_tlora_neg, color="blue", label="TLoRA", fill=True)
    plt.xlabel("Eigenvalue Magnitude (Absolute Negative)")
    plt.ylabel("Density")
    plt.title("Negative Eigenvalue Distribution (TLoRA vs. LoRA)")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return None


def plot_layer_matrix_norms(model):
    # Initialize lists to store norms for each layer (for TLoRA)
    norms_A = []
    norms_B = []
    norms_C = []

    # Number of layers in the model (assuming model is a roberta-based model)
    num_layers = len(
        model.roberta.encoder.layer
    )  # Adjust this depending on the actual model architecture

    # Loop through each layer and extract norms
    for layer_idx in range(num_layers):
        # Extracting lora matrices for TLoRA (or LoRA if only A and B exist)
        if hasattr(
            model.roberta.encoder.layer[layer_idx].attention.self.query, "random_C"
        ):
            lora_A = (
                model.roberta.encoder.layer[layer_idx]
                .attention.self.query.random_A.detach()
                .cpu()
                .numpy()
            )
            lora_B = (
                model.roberta.encoder.layer[layer_idx]
                .attention.self.query.lora_B.detach()
                .cpu()
                .numpy()
            )
            lora_C = (
                model.roberta.encoder.layer[layer_idx]
                .attention.self.query.random_C.detach()
                .cpu()
                .numpy()
            )
        else:
            lora_A = (
                model.roberta.encoder.layer[layer_idx]
                .attention.self.query.lora_A.detach()
                .cpu()
                .numpy()
            )
            lora_B = (
                model.roberta.encoder.layer[layer_idx]
                .attention.self.query.lora_B.detach()
                .cpu()
                .numpy()
            )
            lora_C = None

        # Compute L2 norms for each matrix
        norms_A.append(np.linalg.norm(lora_A, ord=2))
        norms_B.append(np.linalg.norm(lora_B, ord=2))

        # Only append norms for C if it exists (i.e., TLoRA)
        if lora_C is not None:
            norms_C.append(np.linalg.norm(lora_C, ord=2))

    # Now we build the heatmap matrix
    # If we're using LoRA (only A and B), we can just append two columns (A, B).
    if lora_C is None:
        norms_matrix = np.column_stack([norms_A, norms_B])
        norm_labels = ["Norm(A)", "Norm(B)"]
    else:
        norms_matrix = np.column_stack([norms_A, norms_B, norms_C])
        norm_labels = ["Norm(A)", "Norm(B)", "Norm(C)"]

    # Plot the heatmap
    plt.figure(figsize=(8, 8))
    sns.heatmap(
        norms_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=norm_labels,
        yticklabels=[f"Layer {i+1}" for i in range(num_layers)],
    )
    plt.xlabel("Matrix")
    plt.ylabel("Layer")
    plt.title("Heatmap of Matrix Norms Across Layers")
    plt.show()

    return None


def plot_tlora_weight_distribution(model_file):
    model = torch.load(model_file)

    lora_A = (
        model.roberta.encoder.layer[0]
        .attention.self.query.random_A.detach()
        .cpu()
        .numpy()
    )
    lora_B = (
        model.roberta.encoder.layer[0]
        .attention.self.query.lora_B.detach()
        .cpu()
        .numpy()
    )
    lora_C = (
        model.roberta.encoder.layer[0]
        .attention.self.query.random_C.detach()
        .cpu()
        .numpy()
    )
    scaling = (
        model.roberta.encoder.layer[0]
        .attention.self.query.lora_scaling.detach()
        .cpu()
        .numpy()
    )
    W_0 = (
        model.roberta.encoder.layer[0]
        .attention.self.query.linear.weight.detach()
        .cpu()
        .numpy()
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.hist(W_0.flatten(), bins=50, color="blue", alpha=0.7)
    plt.title("Original Weight Hist")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")

    plt.subplot(1, 4, 2)
    plt.hist(lora_A.flatten(), bins=50, color="green", alpha=0.7)
    plt.title("TLoRA A Weight Hist")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")

    plt.subplot(1, 4, 3)
    plt.hist(lora_B.flatten(), bins=50, color="orange", alpha=0.7)
    plt.title("TLoRA B Weight Hist")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")

    plt.subplot(1, 4, 4)
    plt.hist(lora_C.flatten(), bins=50, color="red", alpha=0.7)
    plt.title("TLoRA C Weight Hist")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    return None


def calculate_final_weights(
    model_file_lora, model_file_tlora, layer_idx, component="query"
):
    """
    Calculates W_lora and W_tlora for a given layer and attention component (query, key, or value).
    Args:
    - model_file_lora (str): Path to LoRA model file.
    - model_file_tlora (str): Path to TLoRA model file.
    - layer_idx (int): Index of the layer to analyze.
    - component (str): Attention component ('query', 'key', or 'value').

    Returns:
    - W_lora (np.ndarray): Transformed weight matrix for LoRA.
    - W_tlora (np.ndarray): Transformed weight matrix for TLoRA.
    """
    # Load TLoRA model and extract layer weights
    tlora_model = torch.load(model_file_tlora)
    lora_A = (
        getattr(tlora_model.roberta.encoder.layer[layer_idx].attention.self, component)
        .random_A.detach()
        .cpu()
        .numpy()
    )
    lora_B = (
        getattr(tlora_model.roberta.encoder.layer[layer_idx].attention.self, component)
        .lora_B.detach()
        .cpu()
        .numpy()
    )
    lora_C = (
        getattr(tlora_model.roberta.encoder.layer[layer_idx].attention.self, component)
        .random_C.detach()
        .cpu()
        .numpy()
    )
    scaling = (
        getattr(tlora_model.roberta.encoder.layer[layer_idx].attention.self, component)
        .lora_scaling.detach()
        .cpu()
        .numpy()
    )
    W_0 = (
        getattr(tlora_model.roberta.encoder.layer[layer_idx].attention.self, component)
        .linear.weight.detach()
        .cpu()
        .numpy()
    )
    W_tlora = compute_transformation_matrix_tlora(W_0, lora_A, lora_B, lora_C, scaling)

    # Load LoRA model and extract layer weights
    lora_model = torch.load(model_file_lora)
    lora_A = (
        getattr(lora_model.roberta.encoder.layer[layer_idx].attention.self, component)
        .lora_A.detach()
        .cpu()
        .numpy()
    )
    lora_B = (
        getattr(lora_model.roberta.encoder.layer[layer_idx].attention.self, component)
        .lora_B.detach()
        .cpu()
        .numpy()
    )
    W_0 = (
        getattr(lora_model.roberta.encoder.layer[layer_idx].attention.self, component)
        .linear.weight.detach()
        .cpu()
        .numpy()
    )
    W_lora = compute_transformation_matrix_lora(W_0, lora_A, lora_B)

    return W_lora, W_tlora


def plot_layer_cosine_similarity(model_file_lora, model_file_tlora, num_layers=12):
    """
    Plots cosine similarity between W_lora and W_tlora for each attention component across layers.
    Args:
    - model_file_lora (str): Path to LoRA model file.
    - model_file_tlora (str): Path to TLoRA model file.
    - num_layers (int): Number of layers to analyze.
    """
    components = ["query", "value"]
    similarity_matrix = np.zeros((num_layers, len(components)))

    for layer_idx in range(num_layers):
        for i, component in enumerate(components):
            # Compute LoRA and TLoRA weights for the current layer and component
            W_lora, W_tlora = calculate_final_weights(
                model_file_lora, model_file_tlora, layer_idx, component
            )
            # Flatten the weights to compute cosine similarity
            W_lora_flat = W_lora.flatten()
            W_tlora_flat = W_tlora.flatten()
            # Calculate cosine similarity
            cosine_sim = 1 - cosine(W_lora_flat, W_tlora_flat)
            similarity_matrix[layer_idx, i] = cosine_sim

    # Plot heatmap
    plt.figure(figsize=(8, 8))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        cmap="coolwarm",
        xticklabels=components,
        yticklabels=[f"Layer {i}" for i in range(num_layers)],
        cbar=True,
        vmin=-1,
        vmax=1,
    )
    plt.title("Cosine Similarity between LoRA and TLoRA Weights Across Layers")
    plt.xlabel("Attention Component")
    plt.ylabel("Layer Index")
    plt.show()

    return None


def plot_diff_heatmap(model_file_lora, model_file_tlora):
    W_lora, W_tlora = calculate_final_weights(model_file_lora, model_file_tlora, 0)
    diff = W_lora - W_tlora
    sns.heatmap(diff, cmap="bwr", cbar=True)
    plt.title("LoRA - TLoRA Weight Matrix")

    return None
