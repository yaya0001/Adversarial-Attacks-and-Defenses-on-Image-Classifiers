"""
Full Model Evaluation Script
==============================
Evaluates all three model variants (Baseline, FGSM-AT, PGD-AT) under
both FGSM and PGD adversarial attacks at multiple epsilon levels.

Produces a unified JSON file containing the complete evaluation matrix:
    3 models × (1 clean + 4 FGSM epsilons + 3 PGD settings) = 24 data points

Usage:
    python evaluate_all.py
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
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cnn import SimpleCNN
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack

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
RESULTS_DIR = "../../results"
RESULTS_PATH = os.path.join(RESULTS_DIR, "full_evaluation.json")

# Model checkpoints to evaluate
MODELS = {
    "Baseline": "results/checkpoints/mnist_cnn.pth",
    "FGSM-AT":  "results/checkpoints/mnist_cnn_fgsm_at.pth",
    "PGD-AT":   "results/checkpoints/mnist_cnn_pgd_at.pth",
}

# FGSM epsilon values to sweep
FGSM_EPSILONS = [0.05, 0.1, 0.2, 0.3]

# PGD attack configurations
PGD_SETTINGS = [
    {"epsilon": 0.1, "alpha": 0.01, "iters": 10},
    {"epsilon": 0.2, "alpha": 0.01, "iters": 20},
    {"epsilon": 0.3, "alpha": 0.01, "iters": 40},
]


def evaluate_clean(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Measure model accuracy on unperturbed test data."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


def evaluate_under_attack(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    attack_fn,
    attack_kwargs: dict,
) -> float:
    """
    Measure model accuracy under an adversarial attack.

    Args:
        model:         Classifier in eval mode.
        test_loader:   DataLoader for the test set.
        device:        Computation device.
        attack_fn:     Attack function (fgsm_attack or pgd_attack).
        attack_kwargs: Keyword arguments passed to the attack function
                       (e.g. epsilon, alpha, iters).
    Returns:
        Adversarial accuracy as a percentage (0–100).
    """
    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack_fn(model, images, labels, **attack_kwargs)

        outputs = model(adv_images)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


def main() -> None:
    """Evaluate all models under all attacks and save results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- Load Test Data ---
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False,
                                  download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Evaluate Each Model ---
    all_results = {}

    for model_name, checkpoint_path in MODELS.items():
        print(f"\n{'='*55}")
        print(f"  Evaluating: {model_name}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"{'='*55}")

        # Check checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"  ⚠ Checkpoint not found — skipping {model_name}")
            continue

        # Load model
        model = SimpleCNN().to(device)
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        model.eval()
        print(f"  ✓ Model loaded.\n")

        model_results = {}

        # --- Clean Accuracy ---
        clean_acc = evaluate_clean(model, test_loader, device)
        model_results["clean_accuracy"] = round(clean_acc, 2)
        print(f"  {'Clean Accuracy':<40} {clean_acc:>6.2f}%")
        print(f"  {'-'*50}")

        # --- FGSM Attack ---
        model_results["fgsm_results"] = []
        for eps in FGSM_EPSILONS:
            acc = evaluate_under_attack(
                model, test_loader, device,
                fgsm_attack, {"epsilon": eps}
            )
            model_results["fgsm_results"].append({
                "epsilon": eps,
                "accuracy": round(acc, 2),
            })
            print(f"  FGSM  ε={eps:<6}                          {acc:>6.2f}%")

        print(f"  {'-'*50}")

        # --- PGD Attack ---
        model_results["pgd_results"] = []
        for setting in PGD_SETTINGS:
            acc = evaluate_under_attack(
                model, test_loader, device,
                pgd_attack, {
                    "epsilon": setting["epsilon"],
                    "alpha": setting["alpha"],
                    "iters": setting["iters"],
                }
            )
            model_results["pgd_results"].append({
                "epsilon": setting["epsilon"],
                "alpha": setting["alpha"],
                "iters": setting["iters"],
                "accuracy": round(acc, 2),
            })
            print(f"  PGD   ε={setting['epsilon']:<5} α={setting['alpha']:<5} "
                  f"iters={setting['iters']:<4}   {acc:>6.2f}%")

        all_results[model_name] = model_results

    # --- Save Results ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Full evaluation results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
