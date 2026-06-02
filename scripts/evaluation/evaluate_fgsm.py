"""
FGSM Robustness Evaluation
===========================
Evaluates a pre-trained SimpleCNN on the MNIST test set under
both clean and FGSM-adversarial conditions at multiple epsilon
levels to quantify the model's adversarial robustness.

Results are saved to ``results/fgsm_evaluation.json`` for
reproducibility and report generation.

Usage:
    python src/evaluate_fgsm.py

Expected output:
    Clean Accuracy:               ~99%
    FGSM Accuracy (ε = 0.05):     ~90–95%
    FGSM Accuracy (ε = 0.30):     ~20–40%
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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn import SimpleCNN
from attacks.fgsm import fgsm_attack

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
#  Configuration
# ──────────────────────────────────────────────────────────────
BATCH_SIZE = 64
DATA_DIR = "../../data"
CHECKPOINT_PATH = "../../results/checkpoints/mnist_cnn.pth"
RESULTS_DIR = "../../results"
RESULTS_PATH = os.path.join(RESULTS_DIR, "fgsm_evaluation.json")

# Epsilon values to sweep — larger ε ⟹ stronger attack, lower accuracy
EPSILONS = [0.05, 0.1, 0.2, 0.3]


def evaluate_clean(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Measure model accuracy on the original (unperturbed) test set.

    Args:
        model:       Trained classifier in eval mode.
        test_loader: DataLoader providing MNIST test images.
        device:      Computation device (CPU / CUDA).

    Returns:
        Classification accuracy as a percentage (0–100).
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient tracking for speed
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


def evaluate_fgsm(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    epsilon: float,
) -> float:
    """
    Measure model accuracy on FGSM-adversarial test images.

    For each batch the attack generates adversarial examples using the
    specified epsilon, then feeds them through the model.

    Args:
        model:       Trained classifier in eval mode.
        test_loader: DataLoader providing MNIST test images.
        device:      Computation device (CPU / CUDA).
        epsilon:     FGSM perturbation magnitude (ℓ∞ budget).

    Returns:
        Adversarial accuracy as a percentage (0–100).
    """
    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Generate adversarial images for this batch
        adv_images = fgsm_attack(model, images, labels, epsilon)

        # Evaluate the model on the adversarial images
        outputs = model(adv_images)
        _, predicted = torch.max(outputs, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


def main() -> None:
    """
    Entry point: loads the model, evaluates clean & adversarial accuracy,
    prints a summary table, and saves results to JSON.
    """
    # --- Device Selection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- Load Test Data ---
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False,
                                  download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Load Pre-trained Model ---
    model = SimpleCNN().to(device)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"✓ Model loaded from: {CHECKPOINT_PATH}\n")

    # --- Clean Accuracy (baseline) ---
    clean_acc = evaluate_clean(model, test_loader, device)
    print(f"{'Clean Accuracy':<35} {clean_acc:>6.2f}%")

    # --- Adversarial Accuracy at each epsilon ---
    print("-" * 45)
    results = {"clean_accuracy": round(clean_acc, 2), "fgsm_results": []}

    for epsilon in EPSILONS:
        adv_acc = evaluate_fgsm(model, test_loader, device, epsilon)
        print(f"FGSM Accuracy (ε = {epsilon:<5})       {adv_acc:>6.2f}%")
        results["fgsm_results"].append({
            "epsilon": epsilon,
            "accuracy": round(adv_acc, 2),
        })

    print("-" * 45)
    print("Evaluation complete.")

    # --- Save Results ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()