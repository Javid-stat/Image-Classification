"""
Train the Fashion-MNIST CNN.

Run from the repository root:
    python -m src.train

What it does:
  1. Downloads Fashion-MNIST (first run only) and splits off a validation set
  2. Trains FashionCNN with the Adam optimiser for config.EPOCHS epochs
  3. Checks validation accuracy after every epoch and saves the best model
     to models/fashion_cnn.pth
  4. Saves the per-epoch history to results/history.json and plots the
     learning curves to results/training_curves.png

Evaluate the saved model on the test set afterwards with:
    python -m src.evaluate
"""

import time

import torch

from src import config
from src.data import get_dataloaders
from src.engine import build_criterion, evaluate, train_one_epoch
from src.model import FashionCNN, count_parameters
from src.utils import get_device, plot_history, save_checkpoint, save_json, set_seed


def main(epochs: int = config.EPOCHS) -> None:
    set_seed(config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders()
    print(f"Train images: {len(train_loader.dataset):,} | "
          f"Validation images: {len(val_loader.dataset):,}")

    model = FashionCNN(num_classes=config.NUM_CLASSES).to(device)
    print(f"Model has {count_parameters(model):,} trainable parameters")

    criterion = build_criterion()
    # Adam adapts the step size for each weight individually, so it works well
    # out of the box with a learning rate of 1e-3. weight_decay adds a small
    # penalty on large weights to reduce overfitting.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch = 0.0, 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(round(value, 4))

        # Keep the weights from the epoch with the best validation accuracy.
        # Later epochs can start to overfit, so the last epoch is not always the best.
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc, best_epoch = val_acc, epoch
            save_checkpoint(model, config.MODEL_PATH, epoch=epoch, val_acc=val_acc)

        print(f"Epoch {epoch:2d}/{epochs} | "
              f"train loss {train_loss:.4f}, acc {train_acc:5.2f}% | "
              f"val loss {val_loss:.4f}, acc {val_acc:5.2f}%"
              f"{'  <- best, saved' if improved else ''}")

    minutes = (time.time() - start) / 60
    print(f"\nTraining finished in {minutes:.1f} min. "
          f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Model saved to {config.MODEL_PATH.relative_to(config.ROOT_DIR)}")

    history.update({
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 2),
        "training_minutes": round(minutes, 1),
        "device": str(device),
    })
    save_json(history, config.HISTORY_PATH)
    plot_history(history, config.RESULTS_DIR / "training_curves.png")
    print("Learning curves saved to results/training_curves.png")


if __name__ == "__main__":
    # This guard is required on Windows when DataLoader uses worker processes,
    # and is good practice everywhere.
    main()
