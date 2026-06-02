"""
CifarCNN Model Definition
===========================
A lightweight Convolutional Neural Network designed for CIFAR-10
image classification (32×32 RGB images → 10 classes).

Architecture overview:
    Input(3×32×32)
      → Conv(32 filters, 3×3, pad=1) → BN → ReLU               → (32×32×32)
      → Conv(64 filters, 3×3, pad=1) → BN → ReLU → MaxPool(2×2) → (64×16×16)
      → Conv(128 filters, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)→ (128×8×8)
      → Flatten                                                   → (8192)
      → FC(256) → ReLU → Dropout(0.25)                           → (256)
      → FC(10)                                                    → (10) logits

Design rationale:
    - Batch Normalization accelerates convergence on CIFAR-10 and
      stabilises adversarial training.
    - Dropout(0.25) provides light regularisation without hurting
      robust accuracy significantly.
    - Three conv layers (vs two for MNIST) handle the 3-channel RGB
      input and richer spatial structure of CIFAR-10.
    - The model is deliberately kept lightweight (~200K params) to
      remain trainable on CPU in a reasonable time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarCNN(nn.Module):
    """
    A three-layer convolutional network for CIFAR-10 classification.

    Uses batch normalisation after each convolutional layer and light
    dropout before the final classifier.  Outputs raw logits (no
    softmax), suitable for use with ``CrossEntropyLoss``.
    """

    def __init__(self) -> None:
        super(CifarCNN, self).__init__()

        # --- Convolutional Feature Extractor ---
        # Block 1: 3 input channels (RGB) → 32 feature maps
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: 32 → 64 feature maps
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3: 64 → 128 feature maps
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Shared 2×2 max-pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Light regularisation
        self.dropout = nn.Dropout(0.25)

        # --- Fully-Connected Classifier ---
        # After two pooling stages: 32→16→8, feature map is 128×8×8 = 8192
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape ``(batch_size, 3, 32, 32)``.

        Returns:
            Raw logits of shape ``(batch_size, 10)`` — one score per class.
        """
        # Block 1: Conv → BN → ReLU  (no pooling yet)  → (B, 32, 32, 32)
        x = F.relu(self.bn1(self.conv1(x)))
        # Block 2: Conv → BN → ReLU → Pool              → (B, 64, 16, 16)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Block 3: Conv → BN → ReLU → Pool              → (B, 128, 8, 8)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Flatten feature maps
        x = x.view(x.size(0), -1)                        # → (B, 8192)
        # Classifier head
        x = self.dropout(F.relu(self.fc1(x)))             # → (B, 256)
        x = self.fc2(x)                                   # → (B, 10) logits
        return x
