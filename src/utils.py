"""
Helper functions shared by train.py and evaluate.py:
reproducibility, device selection, saving/loading models, and plotting.
"""

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw plots to files without needing a display window
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

# Plot colours: one colour per series (train vs validation), and a single-hue
# light-to-dark blue scale for the confusion matrix.
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


# --------------------------------------------------------------------------- #
# Reproducibility and hardware
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed every random number generator so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use an NVIDIA GPU (e.g. Colab) or Apple GPU if available, otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Saving and loading
# --------------------------------------------------------------------------- #
def save_checkpoint(model: torch.nn.Module, path: Path, **info) -> None:
    """
    Save the model's learned weights (its state_dict) plus some metadata.

    Saving the state_dict, rather than the whole model object, is the
    recommended PyTorch practice: the file only holds tensors, and it stays
    loadable even if you later move or rename the Python code.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **info}, path)


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> dict:
    """Load weights saved by save_checkpoint into `model`; return the checkpoint dict."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=MUTED_COLOR, labelsize=9)
    ax.xaxis.label.set_color(MUTED_COLOR)
    ax.yaxis.label.set_color(MUTED_COLOR)


def plot_history(history: dict, path: Path) -> None:
    """Plot training/validation loss and accuracy per epoch, side by side."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)

    panels = [
        (ax_loss, "train_loss", "val_loss", "Loss", "Cross-entropy loss"),
        (ax_acc, "train_acc", "val_acc", "Accuracy", "Accuracy (%)"),
    ]
    for ax, train_key, val_key, title, ylabel in panels:
        _style_axes(ax)
        ax.plot(epochs, history[train_key], color=TRAIN_COLOR, linewidth=2,
                marker="o", markersize=4, label="Train")
        ax.plot(epochs, history[val_key], color=VAL_COLOR, linewidth=2,
                marker="o", markersize=4, label="Validation")
        ax.set_title(title, color=TEXT_COLOR, fontsize=12, loc="left")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, labelcolor=TEXT_COLOR)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list, path: Path) -> None:
    """
    Plot the confusion matrix as a heatmap.

    Row = the true class, column = the predicted class. Each row is normalised
    to percentages, so the diagonal shows the accuracy for that class and the
    off-diagonal cells show which classes get mixed up.
    """
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor=SURFACE)
    image = ax.imshow(cm_pct, cmap=BLUES, vmin=0, vmax=100)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.tick_params(colors=MUTED_COLOR, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Write the percentage in every cell that is at least 1%.
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            value = cm_pct[i, j]
            if value >= 1:
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if value > 50 else TEXT_COLOR)

    ax.set_xlabel("Predicted class", color=MUTED_COLOR)
    ax.set_ylabel("True class", color=MUTED_COLOR)
    ax.set_title("Confusion matrix on the test set (% of each true class)",
                 color=TEXT_COLOR, fontsize=12, loc="left")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=MUTED_COLOR, labelsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_sample_predictions(images, labels, preds, class_names, mean, std, path: Path) -> None:
    """Show a grid of test images with the predicted and true label (wrong ones flagged)."""
    n = len(images)
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.2), facecolor=SURFACE)

    for ax, img, label, pred in zip(axes.flat, images, labels, preds):
        img = img.squeeze().numpy() * std[0] + mean[0]   # undo normalisation for display
        ax.imshow(img, cmap="gray_r")
        correct = label == pred
        mark = "" if correct else "✗ "
        ax.set_title(f"{mark}{class_names[pred]}\n(true: {class_names[label]})",
                     fontsize=8, color=TEXT_COLOR if correct else "#d03b3b")
        ax.axis("off")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
