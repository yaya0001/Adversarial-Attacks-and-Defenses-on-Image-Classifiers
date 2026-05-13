"""
Model Training Script
=====================
Trains a SimpleCNN on the MNIST dataset using the Adam optimizer
and Cross-Entropy loss. After training, the model checkpoint is saved
to ``results/checkpoints/mnist_cnn.pth``.

Reproducibility:
    A fixed random seed (42) is set across Python, NumPy, and PyTorch
    to ensure deterministic, reproducible training runs.

Usage:
    python src/train.py
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.cnn import SimpleCNN

# ──────────────────────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ──────────────────────────────────────────────────────────────
#  Hyperparameters & Configuration
# ──────────────────────────────────────────────────────────────
BATCH_SIZE = 64          # Mini-batch size for training and evaluation
LEARNING_RATE = 1e-3     # Adam learning rate
EPOCHS = 5               # Number of full passes over the training set
DATA_DIR = "./data"      # Root directory for MNIST data
CHECKPOINT_DIR = "results/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "mnist_cnn.pth")
METRICS_DIR = "results"
METRICS_PATH = os.path.join(METRICS_DIR, "training_metrics.json")

# Train/val split sizes (MNIST training set has 60,000 samples)
TRAIN_SIZE = 50000
VAL_SIZE = 10000


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device,
             label: str = "Test") -> float:
    """
    Evaluate model accuracy on clean (unperturbed) data.

    Args:
        model:       Trained model to evaluate.
        data_loader: DataLoader providing the evaluation set.
        device:      Device on which to run inference.
        label:       Descriptive label printed alongside the result.

    Returns:
        Accuracy as a percentage (0–100).
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  # No gradients needed during evaluation
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    print(f"  ↳ {label} Accuracy: {acc:.2f}%")
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

    full_train_dataset = datasets.MNIST(root=DATA_DIR, train=True,
                                        download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False,
                                  download=True, transform=transform)

    # Split the 60k training set into 50k train / 10k validation
    train_dataset, val_dataset = random_split(
        full_train_dataset, [TRAIN_SIZE, VAL_SIZE],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model, Loss, Optimizer ---
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print(f"\nStarting training for {EPOCHS} epoch(s)…\n")
    metrics = {"epochs": [], "seed": SEED,
               "hyperparameters": {"batch_size": BATCH_SIZE,
                                   "learning_rate": LEARNING_RATE,
                                   "optimizer": "Adam",
                                   "epochs": EPOCHS}}

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

        # Evaluate on validation set after every epoch
        val_acc = evaluate(model, val_loader, device, label="Validation")

        # Also evaluate on test set for reporting
        test_acc = evaluate(model, test_loader, device, label="Test")

        metrics["epochs"].append({
            "epoch": epoch + 1,
            "avg_loss": round(avg_loss, 4),
            "val_accuracy": round(val_acc, 2),
            "test_accuracy": round(test_acc, 2),
        })

    # --- Save Trained Model ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\n✓ Model checkpoint saved to: {CHECKPOINT_PATH}")

    # --- Save Training Metrics ---
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Training metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    train()