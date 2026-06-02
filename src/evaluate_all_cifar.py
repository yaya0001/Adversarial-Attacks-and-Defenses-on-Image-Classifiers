"""
CIFAR-10 Full Model Evaluation Script
========================================
Evaluates all three CIFAR-10 model variants (Baseline, FGSM-AT, PGD-AT)
under both FGSM and PGD adversarial attacks at multiple epsilon levels.

Uses standard CIFAR-10 ℓ∞ epsilon values: {2/255, 4/255, 8/255, 16/255}

Usage:
    python evaluate_all_cifar.py
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cifar_cnn import CifarCNN
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
DATA_DIR = "./data"
RESULTS_DIR = "results"
RESULTS_PATH = os.path.join(RESULTS_DIR, "cifar10_full_evaluation.json")

# Model checkpoints
MODELS = {
    "Baseline": "results/checkpoints/cifar10_cnn.pth",
    "FGSM-AT":  "results/checkpoints/cifar10_cnn_fgsm_at.pth",
    "PGD-AT":   "results/checkpoints/cifar10_cnn_pgd_at.pth",
}

# Standard CIFAR-10 ℓ∞ epsilon values
FGSM_EPSILONS = [2/255, 4/255, 8/255, 16/255]

# PGD attack configurations
PGD_ALPHA = 2 / 255
PGD_SETTINGS = [
    {"epsilon": 4/255,  "alpha": PGD_ALPHA, "iters": 10},
    {"epsilon": 8/255,  "alpha": PGD_ALPHA, "iters": 20},
    {"epsilon": 16/255, "alpha": PGD_ALPHA, "iters": 40},
]


def evaluate_clean(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Measure model accuracy on unperturbed test data."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100.0 * correct / total


def evaluate_under_attack(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    attack_fn,
    attack_kwargs: dict,
) -> float:
    """Measure model accuracy under an adversarial attack."""
    model.eval()
    correct = total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack_fn(model, images, labels, **attack_kwargs)
        preds = model(adv_images).argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return 100.0 * correct / total


def main() -> None:
    """Evaluate all CIFAR-10 models under all attacks."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load test data (no augmentation)
    transform = transforms.ToTensor()
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False,
                                    download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_results = {}

    for model_name, checkpoint_path in MODELS.items():
        print(f"\n{'='*55}")
        print(f"  Evaluating: {model_name} (CIFAR-10)")
        print(f"{'='*55}")

        if not os.path.exists(checkpoint_path):
            print(f"  ⚠ Checkpoint not found — skipping {model_name}")
            continue

        model = CifarCNN().to(device)
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        model.eval()
        print(f"  ✓ Model loaded.\n")

        model_results = {}

        # Clean accuracy
        clean_acc = evaluate_clean(model, test_loader, device)
        model_results["clean_accuracy"] = round(clean_acc, 2)
        print(f"  {'Clean Accuracy':<45} {clean_acc:>6.2f}%")
        print(f"  {'-'*55}")

        # FGSM attack
        model_results["fgsm_results"] = []
        for eps in FGSM_EPSILONS:
            acc = evaluate_under_attack(
                model, test_loader, device,
                fgsm_attack, {"epsilon": eps}
            )
            model_results["fgsm_results"].append({
                "epsilon": round(eps, 6),
                "epsilon_fraction": f"{int(eps*255)}/255",
                "accuracy": round(acc, 2),
            })
            print(f"  FGSM  ε={int(eps*255):>2}/255 ({eps:.4f})              {acc:>6.2f}%")

        print(f"  {'-'*55}")

        # PGD attack
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
                "epsilon": round(setting["epsilon"], 6),
                "epsilon_fraction": f"{int(setting['epsilon']*255)}/255",
                "alpha": round(setting["alpha"], 6),
                "iters": setting["iters"],
                "accuracy": round(acc, 2),
            })
            print(f"  PGD   ε={int(setting['epsilon']*255):>2}/255  "
                  f"α={int(setting['alpha']*255)}/255  "
                  f"iters={setting['iters']:<4}   {acc:>6.2f}%")

        all_results[model_name] = model_results

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ CIFAR-10 full evaluation saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
