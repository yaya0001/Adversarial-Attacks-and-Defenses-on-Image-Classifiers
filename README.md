# Adversarial Attacks and Defenses on Image Classifiers

> Exploring the **accuracy–robustness trade-off** of lightweight CNN classifiers on **CIFAR-10** and **MNIST** through fundamental adversarial attacks (FGSM, PGD) and empirical adversarial training.

---

## Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup & Installation](#setup--installation)
  - [Linux / macOS](#linux--macos)
  - [Windows](#windows)
- [Running the Project](#running-the-project)
  - [Launch JupyterLab](#launch-jupyterlab)
  - [Workflow Order](#workflow-order)
- [Attacks Implemented](#attacks-implemented)
- [Defense Strategy](#defense-strategy)
- [Results & Outputs](#results--outputs)
- [Datasets](#datasets)
- [License](#license)

---

## Overview

Deep neural networks are remarkably accurate on clean data — but remarkably fragile against carefully crafted perturbations that are **invisible to the human eye**. This project provides a clean, reproducible framework to:

1. **Train** lightweight CNN classifiers on CIFAR-10 and MNIST from scratch.
2. **Attack** them with the two canonical white-box adversarial attacks: FGSM and PGD.
3. **Defend** them using empirical adversarial training and benchmark the accuracy–robustness trade-off.

Everything is implemented in **pure PyTorch**, structured as clean, modular Python scripts, and fully reproducible via pinned Conda environments on both Linux and Windows.

---

## Key Concepts

| Term | Description |
|---|---|
| **Adversarial Example** | An input image perturbed by a small, imperceptible noise that causes a model to misclassify it |
| **FGSM** | Fast Gradient Sign Method — single-step L∞ attack; fast to compute, used as a baseline |
| **PGD** | Projected Gradient Descent — multi-step iterative attack; stronger and considered the standard benchmark attack |
| **ε (epsilon)** | Maximum allowed perturbation magnitude; controls the attack strength |
| **Adversarial Training** | Re-training the model on adversarial examples to improve robustness — the most reliable empirical defense |
| **Accuracy–Robustness Trade-off** | Improving robustness via adversarial training typically comes at a cost to clean accuracy |

---

## Project Structure

```text
Adversarial-Attacks-and-Defenses-on-Image-Classifiers/
├── data/                       # Datasets (MNIST, CIFAR-10)
├── results/
│   └── checkpoints/            # Saved model weights
├── reports/                    # LaTeX reports and generated figures
│   ├── figures/
│   └── report.tex              # Final project report
├── src/                        # Core library code
│   ├── models/                 # CNN architectures
│   │   ├── cnn.py              # SimpleCNN (MNIST)
│   │   └── cifar_cnn.py        # CifarCNN (CIFAR-10)
│   └── attacks/                # Attack implementations
│       ├── fgsm.py             # Single-step attack
│       └── pgd.py              # Multi-step iterative attack
├── scripts/                    # Execution scripts
│   ├── training/               # Clean and adversarial training
│   │   ├── train.py            # Baseline MNIST training
│   │   ├── train_cifar.py      # Baseline CIFAR-10 training
│   │   ├── train_adversarial.py
│   │   └── train_adversarial_cifar.py
│   ├── evaluation/             # Full eval matrices & ablations
│   │   ├── evaluate_fgsm.py    # Standalone FGSM eval (MNIST)
│   │   ├── evaluate_pgd.py     # Standalone PGD eval (MNIST)
│   │   ├── evaluate_all.py     # Full evaluation matrix (MNIST)
│   │   ├── evaluate_all_cifar.py
│   │   ├── ablation_study.py
│   │   └── ablation_study_cifar.py
│   └── visualization/          # Plotting and image collages
│       ├── preview_data.py     # Data preview utility
│       ├── visualize_steps.py  # Attack progression collages
│       ├── visualize_cifar.py  # CIFAR-10 visualizations
│       ├── plot_results.py     # Result curves (MNIST)
│       └── plot_results_cifar.py
├── dl-project-linux.yml        # Conda environment — Linux / macOS
├── dl-project-windows.yml      # Conda environment — Windows
├── .gitignore                  # Git ignore rules
└── LICENSE                     # MIT License
```

> **Note:** CIFAR-10 will be downloaded automatically by `torchvision` on first run (if implemented). MNIST is downloaded automatically under `src/data/`.

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.11.0 |
| TorchVision | 0.26.0 |
| NumPy | 1.26.4 |
| Matplotlib | 3.8.4 |
| Seaborn | 0.13.2 |
| scikit-learn | 1.5.2 |
| tqdm | 4.66.5 |
| Pandas | 2.2.2 |
| Pillow | 10.4.0 |

All dependencies (with exact pinned versions) are declared in the Conda environment files.

**Hardware:** A CUDA-capable GPU is strongly recommended for CIFAR-10 adversarial training. MNIST experiments run comfortably on CPU.

---

## Setup & Installation

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Git installed

### 1. Clone the Repository

```bash
git clone https://github.com/osama-elkhuriby/Adversarial-Attacks-and-Defenses-on-Image-Classifiers.git
cd Adversarial-Attacks-and-Defenses-on-Image-Classifiers
```

---

### Linux / macOS

```bash
# Create the conda environment from the pinned spec
conda env create -f dl-project-linux.yml

# Activate it
conda activate dl-project

# Verify PyTorch is correctly installed
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

### Windows

```bash
# Create the conda environment from the pinned spec
conda env create -f dl-project-windows.yml

# Activate it
conda activate dl-project

# Verify PyTorch is correctly installed
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

> **Tip (GPU on Windows):** If you have an NVIDIA GPU and CUDA is not detected, visit [pytorch.org](https://pytorch.org/get-started/locally/) to install the appropriate CUDA-enabled PyTorch build for your driver version, then re-run the verification command.

---

## Running the Project

### Running the Scripts

With the environment activated, navigate to the `src/` directory and run the scripts in sequence:

```bash
cd src/
```

### Workflow Order

Follow this sequence for a complete end-to-end experiment:

```bash
# Step 1: Train baseline CNN classifier (MNIST)
python train.py

# Step 2: Run FGSM attack on the trained model (varies ε)
python evaluate_fgsm.py

# Step 3: Run PGD attack on the trained model (varies ε and steps)
python evaluate_pgd.py

# Step 4: Generate visualization figures
python visualize_steps.py
```

> **Note:** All scripts use a fixed random seed (42) and export their metrics to JSON files under `src/results/` to ensure full reproducibility.

---

## Attacks Implemented

### FGSM — Fast Gradient Sign Method

A single-step, gradient-based L∞ attack (Goodfellow et al., 2014):

```
x_adv = x + ε · sign(∇_x J(θ, x, y))
```

- **Strength:** Fast; good baseline for evaluating model sensitivity.
- **Weakness:** One step only; weaker than iterative attacks.
- **Key hyperparameter:** `epsilon` (ε) — try values in `[0.01, 0.1, 0.3]` for MNIST and `[1/255, 4/255, 8/255]` for CIFAR-10.

### PGD — Projected Gradient Descent

A multi-step iterative L∞ attack (Madry et al., 2018), the de-facto standard robustness benchmark:

```
x_(t+1) = Π_{x+S}[ x_t + α · sign(∇_x J(θ, x_t, y)) ]
```

where `Π` projects back into the ε-ball after each step.

- **Strength:** Much stronger than FGSM; widely considered the gold-standard white-box attack.
- **Key hyperparameters:** `epsilon` (ε), step size `alpha` (α), number of steps `n_steps`.

---

## Defense Strategy

### Adversarial Training (AT)

The model is re-trained by **augmenting each mini-batch with adversarial examples** generated on-the-fly:

```
min_θ  E_(x,y)~D [ max_{δ ∈ S} L(θ, x+δ, y) ]
```

Two variants are included:

| Variant | Attack used to generate training examples | Speed | Robustness |
|---|---|---|---|
| **FGSM-AT** | FGSM | Fast | Moderate |
| **PGD-AT** | PGD (7 steps) | Slower | Strong |

**Trade-off:** Adversarially trained models typically show a 2–10% drop in clean accuracy in exchange for significantly improved robustness (reduced accuracy under attack goes from near-0% to 40–80% depending on ε and dataset).

---

## Results & Outputs

All generated outputs are saved automatically:

| Output | Location | Description |
|---|---|---|
| Model checkpoints | `src/results/checkpoints/` | `.pt` files for baseline and adversarially trained models |
| Evaluation Metrics | `src/results/` | `.json` files containing accuracy numbers for clean and adversarial runs |
| Accuracy vs. ε curves | `src/reports/figures/` | PNG plots showing model accuracy as ε increases |
| Adversarial example grids | `src/reports/figures/` | Visual comparison of clean vs. perturbed images |

---

## Datasets

| Dataset | Classes | Image Size | Split | Source |
|---|---|---|---|---|
| **MNIST** | 10 (digits 0–9) | 28×28 grayscale | 50k train / 10k val / 10k test | Auto-downloaded to `src/data/MNIST/raw/` |
| **CIFAR-10** | 10 (objects) | 32×32 RGB | 50k train / 10k test | Auto-downloaded via `torchvision.datasets.CIFAR10` |

Datasets will be downloaded to the `src/data/` subdirectory automatically when you run the training scripts for the first time. No manual download required.

---

## References

- Goodfellow et al. — *Explaining and Harnessing Adversarial Examples* (ICLR 2015) — FGSM
- Madry et al. — *Towards Deep Learning Models Resistant to Adversarial Attacks* (ICLR 2018) — PGD & adversarial training
- LeCun et al. — *Gradient-Based Learning Applied to Document Recognition* (1998) — MNIST
- Krizhevsky — *Learning Multiple Layers of Features from Tiny Images* (2009) — CIFAR-10

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

*Built with PyTorch · Conda*
