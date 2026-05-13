"""
Model Training Script
=====================
Trains a SimpleCNN on the MNIST dataset using the Adam optimizer
and Cross-Entropy loss. After training, the model checkpoint is saved
to ``results/checkpoints/mnist_cnn.pth``.

Usage:
    python src/train.py
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cnn import SimpleCNN

# ──────────────────────────────────────────────────────────────
#  Hyperparameters & Configuration
# ──────────────────────────────────────────────────────────────
BATCH_SIZE = 64          # Mini-batch size for training and evaluation
LEARNING_RATE = 1e-3     # Adam learning rate
EPOCHS = 5               # Number of full passes over the training set
DATA_DIR = "./data"      # Root directory for MNIST data
CHECKPOINT_DIR = "results/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "mnist_cnn.pth")


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate model accuracy on clean (unperturbed) test data.

    Args:
        model:       Trained model to evaluate.
        test_loader: DataLoader providing the test set.
        device:      Device on which to run inference.

    Returns:
        Test accuracy as a percentage (0–100).
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  # No gradients needed during evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    print(f"  ↳ Test Accuracy: {acc:.2f}%")
    return acc


def train() -> None:
    """
    Full training pipeline: data loading → training loop → checkpoint saving.
    """
    # --- Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Preparation ---
    # MNIST images are 28×28 grayscale; ToTensor() scales pixels to [0, 1]
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root=DATA_DIR, train=True,
                                   download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False,
                                  download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model, Loss, Optimizer ---
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print(f"\nStarting training for {EPOCHS} epoch(s)…\n")

    for epoch in range(EPOCHS):
        model.train()                    # Switch to training mode (enables dropout, BN, etc.)
        running_loss = 0.0

        for images, labels in tqdm(train_loader,
                                   desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()        # Reset gradients from previous step
            outputs = model(images)      # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()              # Backward pass — compute gradients
            optimizer.step()             # Update model weights

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"  Epoch [{epoch + 1}/{EPOCHS}]  ─  Avg Loss: {avg_loss:.4f}")

        # Evaluate on test set after every epoch
        evaluate(model, test_loader, device)

    # --- Save Trained Model ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\n✓ Model checkpoint saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()