"""
The core training and evaluation steps, shared by train.py and evaluate.py.
"""

import torch
from torch import nn


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    One full pass over the training data. Returns (average loss, accuracy %).

    For every batch:
      1. forward pass   - compute the model's predictions
      2. compute loss   - how wrong are the predictions?
      3. backward pass  - compute the gradient of the loss w.r.t. every weight
      4. optimiser step - nudge every weight a little against its gradient
    """
    model.train()  # turns ON dropout and makes BatchNorm use batch statistics
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()              # gradients accumulate by default, so reset them
        outputs = model(images)            # 1. forward pass -> logits [batch, 10]
        loss = criterion(outputs, labels)  # 2. loss
        loss.backward()                    # 3. backpropagation
        optimizer.step()                   # 4. update weights

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()  # no gradients needed for evaluation: faster and uses less memory
def evaluate(model, loader, criterion, device, return_predictions: bool = False):
    """
    Measure loss and accuracy on data the model does not learn from.

    Returns (average loss, accuracy %), plus (all_labels, all_predictions)
    as tensors when return_predictions=True.
    """
    model.eval()  # turns OFF dropout and makes BatchNorm use its running statistics
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)

        preds = outputs.argmax(dim=1)      # the class with the highest score
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if return_predictions:
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    loss, acc = total_loss / total, 100.0 * correct / total
    if return_predictions:
        return loss, acc, torch.cat(all_labels), torch.cat(all_preds)
    return loss, acc


def build_criterion() -> nn.Module:
    """
    Cross-entropy loss: the standard loss for multi-class classification.
    It applies softmax to the logits and penalises a low probability on the
    correct class (loss = -log p(correct class)).
    """
    return nn.CrossEntropyLoss()
