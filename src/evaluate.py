"""
Evaluate the trained model on the Fashion-MNIST test set.

Run from the repository root (after python -m src.train):
    python -m src.evaluate

The test set (10,000 images) is never used during training or model
selection, so this is an honest estimate of how the model performs on new data.

Outputs:
  results/results.json           test accuracy, loss and per-class accuracy
  results/confusion_matrix.png   which classes get confused with which
  results/sample_predictions.png a grid of test images with predictions
"""

import numpy as np
import torch

from src import config
from src.data import get_dataloaders
from src.engine import build_criterion, evaluate
from src.model import FashionCNN, count_parameters
from src.utils import (
    get_device, load_checkpoint, plot_confusion_matrix,
    plot_sample_predictions, save_json,
)


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    """cm[i, j] = number of images of true class i that were predicted as class j."""
    return np.bincount(
        labels * num_classes + preds, minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)


def main() -> None:
    device = get_device()
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {config.MODEL_PATH}. Run `python -m src.train` first."
        )

    model = FashionCNN(num_classes=config.NUM_CLASSES).to(device)
    checkpoint = load_checkpoint(model, config.MODEL_PATH, device)
    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"(validation accuracy {checkpoint['val_acc']:.2f}%)")

    _, _, test_loader = get_dataloaders()
    test_loss, test_acc, labels, preds = evaluate(
        model, test_loader, build_criterion(), device, return_predictions=True
    )
    labels, preds = labels.numpy(), preds.numpy()
    print(f"\nTest accuracy: {test_acc:.2f}%  (loss {test_loss:.4f}, "
          f"{len(labels):,} images)\n")

    cm = confusion_matrix(labels, preds, config.NUM_CLASSES)
    per_class = cm.diagonal() / cm.sum(axis=1) * 100

    print(f"{'Class':<13} Accuracy")
    for name, acc in zip(config.CLASSES, per_class):
        print(f"{name:<13} {acc:6.2f}%")

    save_json({
        "test_accuracy": round(test_acc, 2),
        "test_loss": round(test_loss, 4),
        "best_epoch": checkpoint["epoch"],
        "val_accuracy": round(checkpoint["val_acc"], 2),
        "trainable_parameters": count_parameters(model),
        "per_class_accuracy": {
            name: round(float(acc), 2) for name, acc in zip(config.CLASSES, per_class)
        },
    }, config.METRICS_PATH)

    plot_confusion_matrix(cm, config.CLASSES, config.RESULTS_DIR / "confusion_matrix.png")

    # A grid of the first 18 test images with the model's predictions.
    images, sample_labels = next(iter(test_loader))
    images, sample_labels = images[:18], sample_labels[:18]
    model.eval()
    with torch.no_grad():
        sample_preds = model(images.to(device)).argmax(dim=1).cpu()
    plot_sample_predictions(
        images, sample_labels.tolist(), sample_preds.tolist(), config.CLASSES,
        config.MEAN, config.STD, config.RESULTS_DIR / "sample_predictions.png",
    )
    print("\nSaved results/results.json, results/confusion_matrix.png "
          "and results/sample_predictions.png")


if __name__ == "__main__":
    main()
