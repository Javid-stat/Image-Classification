"""
Data loading and preprocessing for Fashion-MNIST.

torchvision downloads the dataset automatically the first time you run the
code (about 30 MB) and stores it in data/.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src import config


def get_transform() -> transforms.Compose:
    """Return the preprocessing pipeline applied to every image."""
    return transforms.Compose([
        # PIL image (28x28, values 0-255) -> float tensor of shape [1, 28, 28]
        # with values scaled to [0, 1].
        transforms.ToTensor(),
        # Shift and scale each pixel: (x - mean) / std. Centred inputs keep the
        # activations in a healthy range, so the network trains faster.
        transforms.Normalize(config.MEAN, config.STD),
    ])


def get_dataloaders(batch_size: int = config.BATCH_SIZE):
    """
    Build the train, validation and test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    transform = get_transform()

    full_train = datasets.FashionMNIST(
        root=config.DATA_DIR, train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=config.DATA_DIR, train=False, download=True, transform=transform
    )

    # Split the official training set into train / validation.
    # A seeded generator makes the split identical on every run.
    val_size = int(len(full_train) * config.VAL_FRACTION)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(config.SEED)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    # pin_memory speeds up copying batches to a GPU; it does nothing on CPU.
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,   # shuffle so every epoch sees a new order
        num_workers=config.NUM_WORKERS, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=pin,
    )
    return train_loader, val_loader, test_loader
