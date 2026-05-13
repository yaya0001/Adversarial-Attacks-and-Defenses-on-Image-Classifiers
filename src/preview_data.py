"""
MNIST Data Preview
==================
Loads a random batch of 16 training images from the MNIST dataset and
displays them in a 4×4 grid. The figure is saved to
``reports/figures/mnist_samples.png`` for inclusion in the project report.

Usage:
    python src/preview_data.py
"""

import os

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────
BATCH_SIZE = 16
DATA_DIR = "./data"
OUTPUT_PATH = "reports/figures/mnist_samples.png"

# ──────────────────────────────────────────────────────────────
#  Data Loading
# ──────────────────────────────────────────────────────────────
# ToTensor() converts PIL images to float tensors in [0, 1]
transform = transforms.Compose([transforms.ToTensor()])

dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform,
)
#load
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Grab a single random batch for visualization
images, labels = next(iter(loader))

# ──────────────────────────────────────────────────────────────
#  Visualization
# ──────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 8))

for i in range(BATCH_SIZE):
    plt.subplot(4, 4, i + 1)
    # Remove channel dimension for grayscale display
    plt.imshow(images[i].squeeze(), cmap="gray")
    plt.title(f"Label: {labels[i].item()}")
    plt.axis("off")

plt.suptitle("MNIST Training Samples", fontsize=14, fontweight="bold")
plt.tight_layout()

# Ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150)
plt.show()

print(f"✓ Figure saved to: {OUTPUT_PATH}")
