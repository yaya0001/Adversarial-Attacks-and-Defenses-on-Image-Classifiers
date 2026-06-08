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

from models.cnn import SimpleCNN
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from Evaluationfunction import evaluate 

SEED = 42



#H.P
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
DATA_DIR = "../../data"
CHECKPOINT_DIR = "../../results/checkpoints"
METRICS_DIR = "../../results"

# CIFAR-10 standard adversarial training parameters (Madry et al.)
AT_EPSILON = 8 / 255       # ≈ 0.031 — standard CIFAR-10 ℓ∞ budget
PGD_ALPHA = 2 / 255        # ≈ 0.0078 — standard PGD step size
PGD_STEPS = 7              # Number of PGD inner iterations

# Train/val split
TRAIN_SIZE = 50000
VAL_SIZE = 10000

def train_adversarial(mode: str):
    """
    Full adversarial training pipeline for CIFAR-10.

    Args:
        mode: One of 'fgsm-at' or 'pgd-at'.
    """


    mode_label = mode.upper().replace("-", "_")
    checkpoint_name = f"MNIST_cnn_{mode.replace('-', '_')}.pth"
    metrics_name = f"MNIST_training_metrics_{mode.replace('-', '_')}.json"

    print(f"\n{'='*60}")
    print(f"  MNIST Adversarial Training — {mode_label}")
    print(f"  ε = {AT_EPSILON:.4f} ({8}/255), Epochs = {EPOCHS}")
    if mode == "pgd-at":
        print(f"  PGD: α = {PGD_ALPHA:.4f} ({2}/255), steps = {PGD_STEPS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Data Preparation
    transform = transforms.ToTensor()

    full_train_dataset = datasets.MNIST(root=DATA_DIR, train=True,
                                          download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False,
                                    download=True, transform=transform)

    train_dataset, val_dataset = random_split(
        full_train_dataset, [TRAIN_SIZE, VAL_SIZE],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #Model, Loss, Optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    #Training Loop
    metrics = {
        "dataset": "MNIST",
        "mode": mode,
        "epochs": [],
        "seed": SEED,
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "epochs": EPOCHS,
            "at_epsilon": AT_EPSILON,
            "at_epsilon_fraction": "8/255",
        },
    }

    if mode == "pgd-at":
        metrics["hyperparameters"]["pgd_alpha"] = PGD_ALPHA
        metrics["hyperparameters"]["pgd_alpha_fraction"] = "2/255"
        metrics["hyperparameters"]["pgd_steps"] = PGD_STEPS

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
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
        print(f"Epoch [{epoch + 1}/{EPOCHS}]  ─  Avg Loss: {avg_loss:.4f}")
        # Evaluate on clean validation and test sets
        val_acc = evaluate(model, val_loader, device, label="Validation")
        test_acc = evaluate(model, test_loader, device, label="Test")
        #update matrics
        metrics["epochs"].append({
            "epoch": epoch + 1,
            "avg_loss": round(avg_loss, 4),
            "val_accuracy": round(val_acc, 2),
            "test_accuracy": round(test_acc, 2),
        })

    #Save
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\ncheckpoint saved to: {checkpoint_path}")

    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = os.path.join(METRICS_DIR, metrics_name)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"metrics saved to: {metrics_path}")


#Training the model on the two different adversarial training modes
train_adversarial(mode="fgsm-at")
train_adversarial(mode="pgd-at")    
