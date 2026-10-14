"""
The convolutional neural network (CNN) used to classify Fashion-MNIST images.

Input:  a batch of grayscale images, shape [batch, 1, 28, 28]
Output: 10 raw scores (logits) per image, one per clothing class

Architecture (shapes for one image):

    [1, 28, 28]
      -> ConvBlock(1 -> 32)    -> [32, 14, 14]
      -> ConvBlock(32 -> 64)   -> [64, 7, 7]
      -> Flatten               -> [3136]
      -> Linear(3136 -> 128)   -> [128]
      -> Linear(128 -> 10)     -> [10]
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    Two 3x3 convolutions followed by 2x2 max pooling.

    Stacking two small 3x3 convolutions "sees" a 5x5 area of the image, like a
    single 5x5 convolution, but with fewer weights and an extra non-linearity.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            # Conv2d slides learnable 3x3 filters over the image. Each filter
            # learns to detect a local pattern (an edge, a corner, a texture).
            # padding=1 keeps the height and width unchanged.
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # BatchNorm re-centres and re-scales each channel's activations per
            # batch. This stabilises training and allows a higher learning rate.
            nn.BatchNorm2d(out_channels),
            # ReLU (max(0, x)) adds non-linearity. Without it, stacked layers
            # would collapse into one big linear function.
            nn.ReLU(inplace=True),

            # The second convolution combines the first layer's simple patterns
            # into more complex ones (e.g. edges -> a collar or a sole).
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # MaxPool keeps the strongest response in each 2x2 window, halving
            # height and width. This cuts computation and makes the features
            # slightly robust to small shifts of the object in the image.
            nn.MaxPool2d(kernel_size=2),
            # Dropout2d randomly switches off whole feature maps during training,
            # so the network cannot rely on any single feature (less overfitting).
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FashionCNN(nn.Module):
    """A small, classic CNN: two convolutional blocks and a two-layer classifier."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Feature extractor: learns *what* is in the image.
        # Channels grow (1 -> 32 -> 64) while the spatial size shrinks
        # (28 -> 14 -> 7): fewer positions, but richer features at each.
        self.features = nn.Sequential(
            ConvBlock(1, 32),    # [1, 28, 28]  -> [32, 14, 14]
            ConvBlock(32, 64),   # [32, 14, 14] -> [64, 7, 7]
        )

        # Classifier: turns the features into a decision.
        self.classifier = nn.Sequential(
            # Flatten the 64 feature maps of 7x7 into one vector of 3136 values,
            # because Linear layers expect a 1D input per image.
            nn.Flatten(),
            # A fully connected (Linear) layer combines features from the
            # whole image, unlike convolutions, which only look locally.
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # Final layer: one output (logit) per class. No softmax here,
            # because nn.CrossEntropyLoss applies log-softmax internally.
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters (weights and biases) in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check: python -m src.model
    model = FashionCNN()
    dummy = torch.randn(4, 1, 28, 28)           # a fake batch of 4 images
    print(model)
    print(f"Output shape: {tuple(model(dummy).shape)}")   # expect (4, 10)
    print(f"Trainable parameters: {count_parameters(model):,}")
