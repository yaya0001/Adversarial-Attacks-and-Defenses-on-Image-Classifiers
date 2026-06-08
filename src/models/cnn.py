'''We used a lightweight CNN for MNIST classification.
The model has two convolutional blocks for feature extraction, 
followed by two fully connected layers for classification.
It outputs raw logits for the 10 digit classes,
which are used with cross-entropy loss during training.'''

import torch
import torch.nn as nn    #contain all the layers (e.g., Conv2d, Linear, MaxPool2d)
import torch.nn.functional as F   #contain all the activation functions (e.g., relu, softmax)

class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super(SimpleCNN, self).__init__()

        # --- Convolutional Feature Extractor ---
        # Conv block 1: 1 input channel (grayscale)
        # 32 output channels (feature maps) this mean it learn in 32 different ways to extract features from the input image(e.g., edges, curves, corners ,textures)
        # 3×3 kernel means each filter 3×3 pixels in size, and it will slide across the input image to compute feature maps
        # padding=1 to save image size (28×28 → 28×28) after convolution.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,kernel_size=3, padding=1)
        # Conv block 2: 32 → 64 feature maps (learn more complex features by combining the 32 from previous layer)
        # first conv layer extract low-level features (e.g., edges), and the second conv layer can combine those to learn higher-level features (e.g., shapes, patterns)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3, padding=1)
        # max pooling  later reduces spatial dimensions by half (28×28 → 14×14 after first pool, then 14×14 → 7×7 after second pool)
        # this helps to reduce computational load and also makes the model more robust to small translations in the input image.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        # --- Fully-Connected Classifier ---
        # After two pooling stages: 28→14→7, so feature map is 64×7×7 = 3136
        #fc1 takes the flattened feature maps and outputs 128 hidden units.
        
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        #fc2 takes those 128 hidden units and outputs 10 logits, one for each digit class (0-9).
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # 10 digit classes
        
        
        #why there is no softmax layer at the end ? Because bytorch make internal softmax when we use CrossEntropyLoss.
        #and if we add softmax here, it will apply softmax twice and this make a problem.

    #an important part of the model is the forward method. 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            x: Input tensor of shape ``(batch_size, 1, 28, 28)``.
        Returns:
            Raw logits of shape ``(batch_size, 10)`` — one score per class.
        """
        # Block 1: Conv → ReLU → Pool  →  (B, 32, 14, 14) ,B is batch size, 32 is number of feature maps, and 14×14 is the spatial dimension after pooling.
        x = self.pool(F.relu(self.conv1(x)))
        # Block 2: Conv → ReLU → Pool  →  (B, 64, 7, 7)
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten feature maps into a 1-D vector per sample
        x = x.view(x.size(0), -1)               # → (B, 3136) 64*7*7
        # Classifier head
        x = F.relu(self.fc1(x))                  # → (B, 128)
        x = self.fc2(x)                          # → (B, 10) logits
        return x