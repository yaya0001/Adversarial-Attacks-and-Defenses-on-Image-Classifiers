"""
SimpleCNN Model Definition
===========================
A lightweight Convolutional Neural Network designed for MNIST digit
classification (28×28 grayscale images → 10 classes).

Architecture overview:
    Input(1×28×28)
      → Conv(32 filters, 3×3) → ReLU → MaxPool(2×2)   → (32×14×14)
      → Conv(64 filters, 3×3) → ReLU → MaxPool(2×2)   → (64×7×7)
      → Flatten                                         → (3136)
      → FC(128) → ReLU                                  → (128)
      → FC(10)                                          → (10) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A two-layer convolutional network for MNIST classification.

    The model uses two convolutional blocks (conv → relu → max-pool),
    followed by two fully-connected layers.  The output produces raw
    logits (no softmax), suitable for use with ``CrossEntropyLoss``.
    """

    def __init__(self) -> None:
        super(SimpleCNN, self).__init__()

        # --- Convolutional Feature Extractor ---
        # Conv block 1: 1 input channel (grayscale) → 32 feature maps
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, padding=1)
        # Conv block 2: 32 → 64 feature maps
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)
        # Shared 2×2 max-pooling layer (halves spatial dimensions)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Fully-Connected Classifier ---
        # After two pooling stages: 28→14→7, so feature map is 64×7×7 = 3136
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # 10 digit classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape ``(batch_size, 1, 28, 28)``.

        Returns:
            Raw logits of shape ``(batch_size, 10)`` — one score per class.
        """
        # Block 1: Conv → ReLU → Pool  →  (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv1(x)))
        # Block 2: Conv → ReLU → Pool  →  (B, 64, 7, 7)
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten feature maps into a 1-D vector per sample
        x = x.view(x.size(0), -1)               # → (B, 3136)
        # Classifier head
        x = F.relu(self.fc1(x))                  # → (B, 128)
        x = self.fc2(x)                          # → (B, 10) logits
        return x