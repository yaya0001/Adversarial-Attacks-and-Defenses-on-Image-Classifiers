"""
CIFAR-10 Adversarial Training Script
=======================================
Trains a CifarCNN on CIFAR-10 using adversarial training (AT).
Two variants are supported:

    1. FGSM-AT  — adversarial examples generated via single-step FGSM
    2. PGD-AT   — adversarial examples generated via multi-step PGD (7 steps)

Uses standard CIFAR-10 ℓ∞ perturbation budget:
    ε = 8/255 ≈ 0.031  (Madry et al., 2018)
    α = 2/255          (PGD step size)

Usage:
    python train_adversarial_cifar.py --mode fgsm-at
    python train_adversarial_cifar.py --mode pgd-at
    python train_adversarial_cifar.py --mode both
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import argparse
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
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack

# ──────────────────────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────────────────────
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────
#  Hyperparameters & Configuration
# ──────────────────────────────────────────────────────────────
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 15
DATA_DIR = "../../data"
CHECKPOINT_DIR = "../../results/checkpoints"
METRICS_DIR = "../../results"

# CIFAR-10 standard adversarial training parameters (Madry et al.)
AT_EPSILON = 8 / 255       # ≈ 0.031 — standard CIFAR-10 ℓ∞ budget
PGD_ALPHA = 2 / 255        # ≈ 0.0078 — standard PGD step size
PGD_STEPS = 7              # Number of PGD inner iterations

# Train/val split
TRAIN_SIZE = 45000
VAL_SIZE = 5000


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    label: str = "Test",
) -> float:
    """Evaluate model accuracy on clean data."""
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


def train_adversarial(mode: str) -> None:
    """
    Full adversarial training pipeline for CIFAR-10.

    Args:
        mode: One of 'fgsm-at' or 'pgd-at'.
    """
    set_seed(SEED)

    mode_label = mode.upper().replace("-", "_")
    checkpoint_name = f"cifar10_cnn_{mode.replace('-', '_')}.pth"
    metrics_name = f"cifar10_training_metrics_{mode.replace('-', '_')}.json"

    print(f"\n{'='*60}")
    print(f"  CIFAR-10 Adversarial Training — {mode_label}")
    print(f"  ε = {AT_EPSILON:.4f} ({8}/255), Epochs = {EPOCHS}")
    if mode == "pgd-at":
        print(f"  PGD: α = {PGD_ALPHA:.4f} ({2}/255), steps = {PGD_STEPS}")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Preparation ---
    # No data augmentation during AT — we want the model to learn
    # from adversarial perturbations, not random crops
    transform = transforms.ToTensor()

    full_train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True,
                                          download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False,
                                    download=True, transform=transform)

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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # --- Training Loop ---
    print(f"\nStarting adversarial training for {EPOCHS} epoch(s)…\n")
    metrics = {
        "dataset": "CIFAR-10",
        "mode": mode,
        "epochs": [],
        "seed": SEED,
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "epochs": EPOCHS,
            "at_epsilon": round(AT_EPSILON, 6),
            "at_epsilon_fraction": "8/255",
        },
    }

    if mode == "pgd-at":
        metrics["hyperparameters"]["pgd_alpha"] = round(PGD_ALPHA, 6)
        metrics["hyperparameters"]["pgd_alpha_fraction"] = "2/255"
        metrics["hyperparameters"]["pgd_steps"] = PGD_STEPS

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader,
                                   desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            # Generate adversarial examples
            model.eval()
            if mode == "fgsm-at":
                adv_images = fgsm_attack(model, images, labels, AT_EPSILON)
            else:  # pgd-at
                adv_images = pgd_attack(model, images, labels,
                                        epsilon=AT_EPSILON,
                                        alpha=PGD_ALPHA,
                                        iters=PGD_STEPS)
            model.train()

            # Train on adversarial examples
            optimizer.zero_grad()
            outputs = model(adv_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"  Epoch [{epoch + 1}/{EPOCHS}]  ─  Avg Loss: {avg_loss:.4f}")

        val_acc = evaluate(model, val_loader, device, label="Validation")
        test_acc = evaluate(model, test_loader, device, label="Test")

        metrics["epochs"].append({
            "epoch": epoch + 1,
            "avg_loss": round(avg_loss, 4),
            "val_accuracy": round(val_acc, 2),
            "test_accuracy": round(test_acc, 2),
        })

    # --- Save ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n✓ Model checkpoint saved to: {checkpoint_path}")

    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = os.path.join(METRICS_DIR, metrics_name)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Training metrics saved to: {metrics_path}")


def main() -> None:
    """Parse CLI arguments and launch adversarial training."""
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Adversarial Training for CifarCNN"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fgsm-at", "pgd-at", "both"],
        default="both",
        help="Training mode: 'fgsm-at', 'pgd-at', or 'both' (default: both)",
    )
    args = parser.parse_args()

    if args.mode == "both":
        train_adversarial("fgsm-at")
        train_adversarial("pgd-at")
    else:
        train_adversarial(args.mode)

    print("\n✓ All CIFAR-10 adversarial training complete.")


if __name__ == "__main__":
    main()
