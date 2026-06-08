"""
this file work on CIFAR-10 dataset, which is a more complex dataset than MNIST.
The datashape is (batch_size, 3, 32, 32) because CIFAR-10 images are RGB (3 channels) and 32×32 pixels in size.

We used a lightweight CNN for CIFAR-10 classification.
The input is a 32×32 RGB image.
The network has three convolutional layers that gradually increase the number of feature maps from 32 to 128.
Batch normalization is used after each convolution to stabilize training,and max pooling reduces the spatial size from 32×32 to 8×8.
After flattening, the features are passed through two fully connected layers, producing 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarCNN(nn.Module):
    def __init__(self) -> None:
        super(CifarCNN, self).__init__()

        # --- Convolutional Feature Extractor ---
        # Block 1: 3 input channels (RGB)
        # image size is 3*32*32 
        # padding=1 to save image size (32×32 → 32×32) after convolution.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3, padding=1)
        # this work on 32 feature maps that extracted from conv1,
        # Batch Normalization stabilizes training and helps the model converge faster
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: 32 input feature maps → 64 output feature maps
        # Model learn more complex features. 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3: 64 → 128 feature maps
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Shared 2×2 max-pooling layer
        # max pooling reduces spatial dimensions by half,
        # Max pooling take the maximum value in each 2*2 window 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Light regularisation
        # this mean the model will randomly set 25% of the activations to zero during training,
        # which helps prevent overfitting by encouraging the model to learn more robust features that are not reliant on any single activation.
        self.dropout = nn.Dropout(0.25)


        # --- Fully-Connected Classifier ---
        # After two pooling stages: 32→16→8, feature map is 128×8×8 = 8192
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1: Conv → BN → ReLU  (no pooling yet)  → from (B, 3, 32, 32) to(B, 32, 32, 32)
        x = F.relu(self.bn1(self.conv1(x)))
        # Block 2: Conv → BN → ReLU → Pool              → from (B, 32, 32, 32) to(B, 64, 16, 16)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Block 3: Conv → BN → ReLU → Pool              → from (B, 64, 16, 16) to (B, 128, 8, 8)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Flatten feature maps this convert feature maps into 1-D vector 
        x = x.view(x.size(0), -1)                        # from (B, 128, 8, 8) to(B, 8192)
        # Classifier head
        x = self.dropout(F.relu(self.fc1(x)))             # from (B, 8192) to (B, 256)
        x = self.fc2(x)                                   # from (B, 256) to (B, 10) logits
        return x
