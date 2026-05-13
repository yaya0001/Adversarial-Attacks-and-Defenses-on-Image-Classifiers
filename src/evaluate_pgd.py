"""
PGD Robustness Evaluation
==========================
Evaluates a pre-trained SimpleCNN on the MNIST test set under
PGD-adversarial conditions at multiple (ε, α, iterations) settings.

PGD is a stronger, iterative variant of FGSM and is considered the
"standard" adversarial attack for robustness benchmarking.

Usage:
    python src/evaluate_pgd.py
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn import SimpleCNN
from attacks.pgd import pgd_attack

# ──────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────
BATCH_SIZE = 64
DATA_DIR = "./data"
CHECKPOINT_PATH = "results/checkpoints/mnist_cnn.pth"

# Each dict specifies one PGD evaluation setting.
# Larger epsilon + more iterations ⟹ stronger attack.
PGD_SETTINGS = [
    {"epsilon": 0.1, "alpha": 0.01, "iters": 10},
    {"epsilon": 0.2, "alpha": 0.01, "iters": 20},
    {"epsilon": 0.3, "alpha": 0.01, "iters": 40},
]


def evaluate_pgd(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    epsilon: float,
    alpha: float,
    iters: int,
) -> float:
    """
    Measure model accuracy on PGD-adversarial test images.

    Args:
        model:       Trained classifier in eval mode.
        test_loader: DataLoader providing MNIST test images.
        device:      Computation device (CPU / CUDA).
        epsilon:     Maximum ℓ∞ perturbation budget.
        alpha:       Step size per PGD iteration.
        iters:       Number of PGD iterations.

    Returns:
        Adversarial accuracy as a percentage (0–100).
    """
    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Generate adversarial images using iterative PGD
        adv_images = pgd_attack(model, images, labels,
                                epsilon=epsilon, alpha=alpha, iters=iters)

        # Evaluate the model on the adversarial batch
        outputs = model(adv_images)
        _, predicted = torch.max(outputs, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


def main() -> None:
    """
    Entry point: loads the model, evaluates accuracy under each PGD
    configuration, and prints a formatted summary.
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
        torch.load(CHECKPOINT_PATH, map_location=device)
    )
    model.eval()
    print(f"✓ Model loaded from: {CHECKPOINT_PATH}\n")

    # --- Evaluate under each PGD setting ---
    print(f"{'ε':<8} {'α':<8} {'Iters':<8} {'Accuracy':>10}")
    print("-" * 38)

    for setting in PGD_SETTINGS:
        acc = evaluate_pgd(
            model, test_loader, device,
            epsilon=setting["epsilon"],
            alpha=setting["alpha"],
            iters=setting["iters"],
        )
        print(f"{setting['epsilon']:<8} {setting['alpha']:<8} "
              f"{setting['iters']:<8} {acc:>9.2f}%")

    print("-" * 38)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()