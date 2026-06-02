"""
CIFAR-10 Adversarial Attack Visualization Suite
==================================================
Generates a comprehensive set of six figures illustrating
how FGSM and PGD adversarial attacks affect a CifarCNN trained
on CIFAR-10.  All figures are saved to ``reports/figures/``.

Generated figures:
    C1. Original CIFAR-10 test samples
    C2. Clean predictions (true vs predicted labels)
    C3. FGSM: original → adversarial → pixel-difference map
    C4. PGD:  original → adversarial → pixel-difference map
    C5. FGSM epsilon sweep (ε = 2/255, 4/255, 8/255, 16/255)
    C6. PGD settings comparison (varying ε and iterations)

Usage:
    python visualize_cifar.py
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cifar_cnn import CifarCNN
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack

# ──────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 16
DATA_DIR = "./data"
CHECKPOINT_PATH = "results/checkpoints/cifar10_cnn.pth"
FIGURE_DIR = "reports/figures"
DPI = 300

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# FGSM epsilon values (standard CIFAR-10)
FGSM_EPSILONS = [2/255, 4/255, 8/255, 16/255]

# PGD attack configurations
PGD_ALPHA = 2 / 255
PGD_SETTINGS = [
    {"epsilon": 4/255,  "alpha": PGD_ALPHA, "iters": 10},
    {"epsilon": 8/255,  "alpha": PGD_ALPHA, "iters": 20},
    {"epsilon": 16/255, "alpha": PGD_ALPHA, "iters": 40},
]


def _save_figure(filename: str) -> None:
    """Save the current matplotlib figure and close it."""
    path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def _show_cifar_image(ax, img_tensor):
    """Display a CIFAR-10 image tensor (C, H, W) on a matplotlib axes."""
    # Transpose from (C, H, W) to (H, W, C) for display
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img)


def main() -> None:
    """Generate all six CIFAR-10 adversarial visualization figures."""

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(FIGURE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load test data
    transform = transforms.ToTensor()
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False,
                                    download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load pre-trained model
    model = CifarCNN().to(device)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"✓ Model loaded from: {CHECKPOINT_PATH}\n")

    # Grab one batch
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # =========================================================
    #  Figure C1 — Original CIFAR-10 Test Samples
    # =========================================================
    print("Generating Figure C1: Original CIFAR-10 samples…")
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(BATCH_SIZE):
        ax = axes[i // 4][i % 4]
        _show_cifar_image(ax, images[i])
        ax.set_title(CIFAR10_CLASSES[labels[i].item()], fontsize=9)
        ax.axis("off")
    fig.suptitle("CIFAR-10 Original Test Images", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure("c1_cifar10_original_samples.png")

    # =========================================================
    #  Figure C2 — Clean Predictions
    # =========================================================
    print("Generating Figure C2: Clean predictions…")
    with torch.no_grad():
        clean_preds = model(images).argmax(dim=1)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for i in range(12):
        ax = axes[i // 4][i % 4]
        _show_cifar_image(ax, images[i])
        true_cls = CIFAR10_CLASSES[labels[i].item()]
        pred_cls = CIFAR10_CLASSES[clean_preds[i].item()]
        is_correct = labels[i].item() == clean_preds[i].item()
        color = "green" if is_correct else "red"
        ax.set_title(f"True: {true_cls}\nPred: {pred_cls}",
                     fontsize=9, color=color)
        ax.axis("off")
    fig.suptitle("CIFAR-10 Clean Predictions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure("c2_cifar10_clean_predictions.png")

    # =========================================================
    #  Figure C3 — FGSM: Original vs Adversarial vs Difference
    # =========================================================
    print("Generating Figure C3: FGSM attack breakdown…")
    fgsm_eps = 8 / 255
    single_image = images[0:1]
    single_label = labels[0:1]

    fgsm_image = fgsm_attack(model, single_image, single_label, fgsm_eps)
    with torch.no_grad():
        orig_pred = model(single_image).argmax(dim=1).item()
        fgsm_pred = model(fgsm_image).argmax(dim=1).item()

    diff = torch.abs(fgsm_image - single_image)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    _show_cifar_image(axes[0], single_image[0])
    axes[0].set_title(f"Original\nPred: {CIFAR10_CLASSES[orig_pred]}", fontsize=10)
    axes[0].axis("off")

    _show_cifar_image(axes[1], fgsm_image[0])
    axes[1].set_title(f"FGSM Adversarial\nPred: {CIFAR10_CLASSES[fgsm_pred]}", fontsize=10)
    axes[1].axis("off")

    # Show perturbation amplified for visibility (×10)
    diff_amplified = (diff[0] * 10).clamp(0, 1)
    _show_cifar_image(axes[2], diff_amplified)
    axes[2].set_title("Perturbation (×10)\n(|adv − orig|)", fontsize=10)
    axes[2].axis("off")

    fig.suptitle(f"FGSM Attack on CIFAR-10  |  ε = 8/255",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure("c3_cifar10_fgsm_breakdown.png")

    # =========================================================
    #  Figure C4 — PGD: Original vs Adversarial vs Difference
    # =========================================================
    print("Generating Figure C4: PGD attack breakdown…")
    pgd_eps = 8 / 255
    single_image = images[1:2]
    single_label = labels[1:2]

    pgd_image = pgd_attack(model, single_image, single_label,
                            epsilon=pgd_eps, alpha=PGD_ALPHA, iters=20)
    with torch.no_grad():
        orig_pred = model(single_image).argmax(dim=1).item()
        pgd_pred_val = model(pgd_image).argmax(dim=1).item()

    diff = torch.abs(pgd_image - single_image)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    _show_cifar_image(axes[0], single_image[0])
    axes[0].set_title(f"Original\nPred: {CIFAR10_CLASSES[orig_pred]}", fontsize=10)
    axes[0].axis("off")

    _show_cifar_image(axes[1], pgd_image[0])
    axes[1].set_title(f"PGD Adversarial\nPred: {CIFAR10_CLASSES[pgd_pred_val]}", fontsize=10)
    axes[1].axis("off")

    diff_amplified = (diff[0] * 10).clamp(0, 1)
    _show_cifar_image(axes[2], diff_amplified)
    axes[2].set_title("Perturbation (×10)\n(|adv − orig|)", fontsize=10)
    axes[2].axis("off")

    fig.suptitle(f"PGD Attack on CIFAR-10  |  ε = 8/255, 20 iters",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure("c4_cifar10_pgd_breakdown.png")

    # =========================================================
    #  Figure C5 — FGSM Epsilon Comparison
    # =========================================================
    print("Generating Figure C5: FGSM epsilon sweep…")
    single_image = images[2:3]
    single_label = labels[2:3]

    fig, axes = plt.subplots(1, len(FGSM_EPSILONS) + 1, figsize=(14, 3))

    _show_cifar_image(axes[0], single_image[0])
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    for i, eps in enumerate(FGSM_EPSILONS):
        adv = fgsm_attack(model, single_image, single_label, eps)
        with torch.no_grad():
            pred = model(adv).argmax(dim=1).item()
        _show_cifar_image(axes[i + 1], adv[0])
        axes[i + 1].set_title(f"ε = {int(eps*255)}/255\nPred: {CIFAR10_CLASSES[pred]}",
                              fontsize=10)
        axes[i + 1].axis("off")

    fig.suptitle("FGSM on CIFAR-10 — Increasing Epsilon",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure("c5_cifar10_fgsm_epsilon_sweep.png")

    # =========================================================
    #  Figure C6 — PGD Settings Comparison
    # =========================================================
    print("Generating Figure C6: PGD settings comparison…")
    single_image = images[3:4]
    single_label = labels[3:4]

    fig, axes = plt.subplots(1, len(PGD_SETTINGS) + 1, figsize=(14, 3))

    _show_cifar_image(axes[0], single_image[0])
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    for i, setting in enumerate(PGD_SETTINGS):
        adv = pgd_attack(model, single_image, single_label,
                          epsilon=setting["epsilon"],
                          alpha=setting["alpha"],
                          iters=setting["iters"])
        with torch.no_grad():
            pred = model(adv).argmax(dim=1).item()
        _show_cifar_image(axes[i + 1], adv[0])
        axes[i + 1].set_title(
            f"ε={int(setting['epsilon']*255)}/255\n"
            f"iters={setting['iters']}\n"
            f"Pred: {CIFAR10_CLASSES[pred]}",
            fontsize=9)
        axes[i + 1].axis("off")

    fig.suptitle("PGD on CIFAR-10 — Different Attack Settings",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure("c6_cifar10_pgd_settings.png")

    print(f"\n✓ All 6 CIFAR-10 visualizations saved to: {FIGURE_DIR}/")


if __name__ == "__main__":
    main()
