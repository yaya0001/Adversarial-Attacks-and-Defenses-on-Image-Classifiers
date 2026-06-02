"""
CIFAR-10 Baseline Training Script
====================================
Trains a CifarCNN on the CIFAR-10 dataset using the Adam optimizer
and Cross-Entropy loss.  After training, the model checkpoint is saved
to ``results/checkpoints/cifar10_cnn.pth``.

Data augmentation:
    - RandomHorizontalFlip
    - RandomCrop(32, padding=4)
    (standard CIFAR-10 augmentation — applied only during training)

Note: pixel values are kept in [0, 1] (no channel-wise normalisation)
so that adversarial perturbations can be applied directly.

Reproducibility:
    A fixed random seed (42) is set across Python, NumPy, and PyTorch.

Usage:
    python train_cifar.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

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

from models.cifar_cnn import CifarCNN

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
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 20
DATA_DIR = "../../data"
CHECKPOINT_DIR = "../../results/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "cifar10_cnn.pth")
METRICS_DIR = "../../results"
METRICS_PATH = os.path.join(METRICS_DIR, "cifar10_training_metrics.json")

# Train/val split sizes (CIFAR-10 training set has 50,000 samples)
TRAIN_SIZE = 45000
VAL_SIZE = 5000

# CIFAR-10 class names for reference
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    label: str = "Test",
) -> float:
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

    with torch.no_grad():
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
    """Full training pipeline: data loading → training loop → checkpoint saving."""

    # --- Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Preparation ---
    # Training: augmentation + ToTensor (keep [0, 1] range)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    # Evaluation: ToTensor only
    eval_transform = transforms.ToTensor()

    full_train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True,
                                          download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False,
                                    download=True, transform=eval_transform)

    # Split 50k training into 45k train / 5k val
    train_dataset, val_dataset = random_split(
        full_train_dataset, [TRAIN_SIZE, VAL_SIZE],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model, Loss, Optimizer ---
    model = CifarCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # --- Training Loop ---
    print(f"\nStarting CIFAR-10 training for {EPOCHS} epoch(s)…\n")
    metrics = {
        "dataset": "CIFAR-10",
        "epochs": [],
        "seed": SEED,
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "epochs": EPOCHS,
        },
    }

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader,
                                   desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"  Epoch [{epoch + 1}/{EPOCHS}]  ─  Avg Loss: {avg_loss:.4f}")

        # Evaluate on validation and test sets
        val_acc = evaluate(model, val_loader, device, label="Validation")
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
