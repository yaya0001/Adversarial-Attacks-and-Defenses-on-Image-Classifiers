"""
CIFAR-10 Ablation Study Script
=================================
Runs two ablation experiments on PGD-based adversarial training
for CIFAR-10 (reduced scope for CPU feasibility):

    1. Effect of PGD inner steps during training (3, 7 steps)
    2. Effect of training epochs (10, 15 epochs)

Usage:
    python ablation_study_cifar.py
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
from attacks.pgd import pgd_attack

# ──────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DATA_DIR = "../../data"
CHECKPOINT_DIR = "results/checkpoints/ablation"
RESULTS_DIR = "../../results"
RESULTS_PATH = os.path.join(RESULTS_DIR, "cifar10_ablation_results.json")

# CIFAR-10 AT parameters
AT_EPSILON = 8 / 255
PGD_ALPHA = 2 / 255
DEFAULT_PGD_STEPS = 7
DEFAULT_EPOCHS = 15

# Evaluation attack (fixed)
EVAL_PGD_EPSILON = 8 / 255
EVAL_PGD_ALPHA = 2 / 255
EVAL_PGD_ITERS = 20

# Reduced ablation scope for CPU
ABLATION_PGD_STEPS = [3, 7]
ABLATION_EPOCHS = [10, 15]

# Train/val split
TRAIN_SIZE = 45000
VAL_SIZE = 5000


def set_seed(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(device: torch.device):
    """Create train, validation, and test data loaders."""
    transform = transforms.ToTensor()
    full_train = datasets.CIFAR10(root=DATA_DIR, train=True,
                                  download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False,
                                    download=True, transform=transform)

    train_ds, val_ds = random_split(
        full_train, [TRAIN_SIZE, VAL_SIZE],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


def evaluate_clean(model: nn.Module, loader: DataLoader,
                   device: torch.device) -> float:
    """Measure clean accuracy."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100.0 * correct / total


def evaluate_pgd(model: nn.Module, loader: DataLoader,
                 device: torch.device) -> float:
    """Measure robust accuracy under PGD evaluation attack."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd_attack(model, images, labels,
                                epsilon=EVAL_PGD_EPSILON,
                                alpha=EVAL_PGD_ALPHA,
                                iters=EVAL_PGD_ITERS)
        preds = model(adv_images).argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return 100.0 * correct / total


def train_pgd_at(
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    pgd_steps: int,
    tag: str,
) -> dict:
    """Train a PGD-AT CifarCNN with specified epochs and PGD steps."""
    set_seed(SEED)

    model = CifarCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n  Training: {tag}  (epochs={epochs}, pgd_steps={pgd_steps})")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader,
                                   desc=f"    Epoch {epoch+1}/{epochs}",
                                   leave=False):
            images, labels = images.to(device), labels.to(device)

            model.eval()
            adv_images = pgd_attack(model, images, labels,
                                    epsilon=AT_EPSILON,
                                    alpha=PGD_ALPHA,
                                    iters=pgd_steps)
            model.train()

            optimizer.zero_grad()
            outputs = model(adv_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"    Epoch {epoch+1}/{epochs}  ─  Loss: {avg_loss:.4f}")

    # Save checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"cifar10_{tag}.pth")
    torch.save(model.state_dict(), ckpt_path)

    # Evaluate
    clean_acc = evaluate_clean(model, test_loader, device)
    robust_acc = evaluate_pgd(model, test_loader, device)

    print(f"    ✓ Clean: {clean_acc:.2f}%  |  Robust (PGD ε=8/255): {robust_acc:.2f}%")

    return {
        "tag": tag,
        "epochs": epochs,
        "pgd_steps": pgd_steps,
        "clean_accuracy": round(clean_acc, 2),
        "robust_accuracy": round(robust_acc, 2),
    }


def main() -> None:
    """Run all CIFAR-10 ablation experiments."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_data_loaders(device)

    results = {"ablation_pgd_steps": [], "ablation_epochs": []}

    # Ablation 1: PGD inner steps
    print(f"\n{'='*60}")
    print("  CIFAR-10 Ablation 1: PGD Inner Steps")
    print(f"  Fixed: epochs={DEFAULT_EPOCHS}, ε=8/255")
    print(f"  Varying: pgd_steps ∈ {ABLATION_PGD_STEPS}")
    print(f"{'='*60}")

    for steps in ABLATION_PGD_STEPS:
        result = train_pgd_at(
            device, train_loader, val_loader, test_loader,
            epochs=DEFAULT_EPOCHS, pgd_steps=steps,
            tag=f"pgd_at_steps_{steps}",
        )
        results["ablation_pgd_steps"].append(result)

    # Ablation 2: Training epochs
    print(f"\n{'='*60}")
    print("  CIFAR-10 Ablation 2: Training Epochs")
    print(f"  Fixed: pgd_steps={DEFAULT_PGD_STEPS}, ε=8/255")
    print(f"  Varying: epochs ∈ {ABLATION_EPOCHS}")
    print(f"{'='*60}")

    for ep in ABLATION_EPOCHS:
        result = train_pgd_at(
            device, train_loader, val_loader, test_loader,
            epochs=ep, pgd_steps=DEFAULT_PGD_STEPS,
            tag=f"pgd_at_epochs_{ep}",
        )
        results["ablation_epochs"].append(result)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ CIFAR-10 ablation results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
