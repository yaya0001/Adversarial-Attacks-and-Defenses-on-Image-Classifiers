"""
Adversarial Attack Visualization Suite
=======================================
Generates a comprehensive set of six figures that illustrate
how FGSM and PGD adversarial attacks affect a SimpleCNN trained
on MNIST. All figures are saved to ``reports/figures/``.

Generated figures:
    1.  Original MNIST test samples
    2.  Clean predictions (true vs predicted labels)
    3.  FGSM: original → adversarial → pixel-difference map
    4.  PGD:  original → adversarial → pixel-difference map
    5.  FGSM epsilon sweep (ε = 0.05, 0.1, 0.2, 0.3)
    6.  PGD settings comparison (varying ε, iterations)

Usage:
    python src/visualize_steps.py
"""

import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn import SimpleCNN
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack

# ──────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────
BATCH_SIZE = 16
DATA_DIR = "./data"
CHECKPOINT_PATH = "results/checkpoints/mnist_cnn.pth"
FIGURE_DIR = "reports/figures"
DPI = 300  # Resolution for saved figures

# FGSM epsilon values to compare in Figure 5
FGSM_EPSILONS = [0.05, 0.1, 0.2, 0.3]

# PGD attack configurations to compare in Figure 6
PGD_SETTINGS = [
    {"epsilon": 0.1, "alpha": 0.01, "iters": 10},
    {"epsilon": 0.2, "alpha": 0.01, "iters": 20},
    {"epsilon": 0.3, "alpha": 0.01, "iters": 40},
]


def _save_figure(filename: str) -> None:
    """Save the current matplotlib figure and close it."""
    path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ──────────────────────────────────────────────────────────────
#  Setup: device, data, and model
# ──────────────────────────────────────────────────────────────
os.makedirs(FIGURE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# MNIST test set — ToTensor() scales pixels to [0, 1]
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root=DATA_DIR, train=False,
                              download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load pre-trained model
model = SimpleCNN().to(device)
model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=device)
)
model.eval()
print(f"✓ Model loaded from: {CHECKPOINT_PATH}\n")

# Grab one batch of images for all visualizations
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)


# =========================================================
#  Figure 1 — Original Dataset Samples
# =========================================================
print("Generating Figure 1: Original dataset samples…")

plt.figure(figsize=(8, 8))
for i in range(BATCH_SIZE):
    plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].cpu().squeeze(), cmap="gray")
    plt.title(f"Label: {labels[i].item()}", fontsize=9)
    plt.axis("off")

plt.suptitle("MNIST Original Test Images", fontsize=13, fontweight="bold")
plt.tight_layout()
_save_figure("01_original_dataset_samples.png")


# =========================================================
#  Figure 2 — Clean Predictions
# =========================================================
print("Generating Figure 2: Clean predictions…")

with torch.no_grad():
    clean_outputs = model(images)
    clean_preds = clean_outputs.argmax(dim=1)

plt.figure(figsize=(10, 7))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images[i].cpu().squeeze(), cmap="gray")

    # Colour the title green if correct, red if wrong
    is_correct = labels[i].item() == clean_preds[i].item()
    color = "green" if is_correct else "red"
    plt.title(f"True: {labels[i].item()} | Pred: {clean_preds[i].item()}",
              fontsize=9, color=color)
    plt.axis("off")

plt.suptitle("Clean Image Predictions", fontsize=13, fontweight="bold")
plt.tight_layout()
_save_figure("02_clean_predictions.png")


# =========================================================
#  Figure 3 — FGSM: Original vs Adversarial vs Difference
# =========================================================
print("Generating Figure 3: FGSM attack breakdown…")

fgsm_epsilon = 0.3
single_image = images[0:1]   # Take the first image (keep batch dim)
single_label = labels[0:1]

fgsm_image = fgsm_attack(model, single_image, single_label, fgsm_epsilon)

with torch.no_grad():
    original_pred = model(single_image).argmax(dim=1).item()
    fgsm_pred = model(fgsm_image).argmax(dim=1).item()

# Pixel-wise absolute difference highlights the perturbation
fgsm_difference = torch.abs(fgsm_image - single_image)

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(single_image.detach().cpu().squeeze(), cmap="gray")
plt.title(f"Original\nPred: {original_pred}", fontsize=10)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(fgsm_image.detach().cpu().squeeze(), cmap="gray")
plt.title(f"FGSM Adversarial\nPred: {fgsm_pred}", fontsize=10)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(fgsm_difference.detach().cpu().squeeze(), cmap="hot")
plt.title("Perturbation\n(|adv − orig|)", fontsize=10)
plt.axis("off")

plt.suptitle(f"FGSM Attack Visualization  |  ε = {fgsm_epsilon}",
             fontsize=13, fontweight="bold")
plt.tight_layout()
_save_figure("03_fgsm_original_vs_attack.png")


# =========================================================
#  Figure 4 — PGD: Original vs Adversarial vs Difference
# =========================================================
print("Generating Figure 4: PGD attack breakdown…")

pgd_epsilon = 0.3
pgd_alpha = 0.01
pgd_iters = 40

single_image = images[1:2]
single_label = labels[1:2]

pgd_image = pgd_attack(model, single_image, single_label,
                        epsilon=pgd_epsilon, alpha=pgd_alpha, iters=pgd_iters)

with torch.no_grad():
    original_pred = model(single_image).argmax(dim=1).item()
    pgd_pred = model(pgd_image).argmax(dim=1).item()

pgd_difference = torch.abs(pgd_image - single_image)

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(single_image.detach().cpu().squeeze(), cmap="gray")
plt.title(f"Original\nPred: {original_pred}", fontsize=10)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(pgd_image.detach().cpu().squeeze(), cmap="gray")
plt.title(f"PGD Adversarial\nPred: {pgd_pred}", fontsize=10)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pgd_difference.detach().cpu().squeeze(), cmap="hot")
plt.title("Perturbation\n(|adv − orig|)", fontsize=10)
plt.axis("off")

plt.suptitle(f"PGD Attack Visualization  |  ε = {pgd_epsilon}, iters = {pgd_iters}",
             fontsize=13, fontweight="bold")
plt.tight_layout()
_save_figure("04_pgd_original_vs_attack.png")


# =========================================================
#  Figure 5 — FGSM Epsilon Comparison
# =========================================================
print("Generating Figure 5: FGSM epsilon sweep…")

single_image = images[2:3]
single_label = labels[2:3]

plt.figure(figsize=(12, 3))

# First subplot: the clean original
plt.subplot(1, len(FGSM_EPSILONS) + 1, 1)
plt.imshow(single_image.detach().cpu().squeeze(), cmap="gray")
plt.title("Original", fontsize=10)
plt.axis("off")

# Remaining subplots: one per epsilon value
for i, eps in enumerate(FGSM_EPSILONS):
    adv_image = fgsm_attack(model, single_image, single_label, eps)

    with torch.no_grad():
        pred = model(adv_image).argmax(dim=1).item()

    plt.subplot(1, len(FGSM_EPSILONS) + 1, i + 2)
    plt.imshow(adv_image.detach().cpu().squeeze(), cmap="gray")
    plt.title(f"ε = {eps}\nPred: {pred}", fontsize=10)
    plt.axis("off")

plt.suptitle("FGSM Effect with Increasing Epsilon",
             fontsize=13, fontweight="bold")
plt.tight_layout()
_save_figure("05_fgsm_epsilon_comparison.png")


# =========================================================
#  Figure 6 — PGD Settings Comparison
# =========================================================
print("Generating Figure 6: PGD settings comparison…")

single_image = images[3:4]
single_label = labels[3:4]

plt.figure(figsize=(12, 3))

# First subplot: the clean original
plt.subplot(1, len(PGD_SETTINGS) + 1, 1)
plt.imshow(single_image.detach().cpu().squeeze(), cmap="gray")
plt.title("Original", fontsize=10)
plt.axis("off")

# Remaining subplots: one per PGD configuration
for i, setting in enumerate(PGD_SETTINGS):
    adv_image = pgd_attack(
        model, single_image, single_label,
        epsilon=setting["epsilon"],
        alpha=setting["alpha"],
        iters=setting["iters"],
    )

    with torch.no_grad():
        pred = model(adv_image).argmax(dim=1).item()

    plt.subplot(1, len(PGD_SETTINGS) + 1, i + 2)
    plt.imshow(adv_image.detach().cpu().squeeze(), cmap="gray")
    plt.title(
        f"ε={setting['epsilon']}\n"
        f"iters={setting['iters']}\n"
        f"Pred: {pred}",
        fontsize=9,
    )
    plt.axis("off")

plt.suptitle("PGD Effect with Different Attack Settings",
             fontsize=13, fontweight="bold")
plt.tight_layout()
_save_figure("06_pgd_settings_comparison.png")


# ──────────────────────────────────────────────────────────────
print(f"\n✓ All 6 visualizations saved to: {FIGURE_DIR}/")