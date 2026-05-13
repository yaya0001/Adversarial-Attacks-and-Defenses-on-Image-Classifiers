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

Everything is implemented in **pure PyTorch**, packaged inside Jupyter notebooks, and fully reproducible via pinned Conda environments on both Linux and Windows.

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

```
Adversarial-Attacks-and-Defenses-on-Image-Classifiers/
│
├── src/                        # All source code (notebooks & scripts)
│   ├── *.ipynb                 # Jupyter notebooks (training, attacks, defense, evaluation)
│   └── *.py                    # Supporting Python modules (models, attacks, utils)
│
├── data/
│   └── MNIST/
│       └── raw/                # Auto-downloaded MNIST binary files
│
├── reports/
│   └── figures/                # Generated plots (accuracy curves, adversarial examples, trade-off charts)
│
├── results/
│   └── checkpoints/            # Saved model weights (.pt files)
│
├── dl-project-linux.yml        # Conda environment — Linux / macOS
├── dl-project-windows.yml      # Conda environment — Windows
└── LICENSE                     # MIT License
```

> **Note:** CIFAR-10 is downloaded automatically by `torchvision` on first run. MNIST raw files are already included under `data/MNIST/raw/`.

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
| JupyterLab | 4.5.7 |
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

### Launch JupyterLab

With the environment activated, start JupyterLab from the project root:

```bash
jupyter lab
```

Your browser will open at `http://localhost:8888`. Navigate to the `src/` directory to find and open the notebooks.

---

### Workflow Order

Follow the notebooks in this sequence for a complete end-to-end experiment:

```
Step 1  →  Train baseline CNN classifiers (MNIST & CIFAR-10)
Step 2  →  Run FGSM attack on the trained models (vary ε)
Step 3  →  Run PGD attack on the trained models (vary ε and steps)
Step 4  →  Perform adversarial training (FGSM-AT or PGD-AT)
Step 5  →  Evaluate robustness: clean accuracy vs. adversarial accuracy
Step 6  →  Plot and compare results
```

> Open each notebook, read the markdown cells explaining the theory, then **run all cells top-to-bottom** (`Kernel → Restart & Run All`).

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
| Model checkpoints | `results/checkpoints/` | `.pt` files for baseline and adversarially trained models |
| Accuracy vs. ε curves | `reports/figures/` | PNG plots showing model accuracy as ε increases |
| Adversarial example grids | `reports/figures/` | Visual comparison of clean vs. perturbed images |
| Robustness summary tables | notebook output cells | Clean acc. vs. FGSM acc. vs. PGD acc. per model |

---

## Datasets

| Dataset | Classes | Image Size | Split | Source |
|---|---|---|---|---|
| **MNIST** | 10 (digits 0–9) | 28×28 grayscale | 60k train / 10k test | Included in `data/MNIST/raw/` |
| **CIFAR-10** | 10 (objects) | 32×32 RGB | 50k train / 10k test | Auto-downloaded via `torchvision.datasets.CIFAR10` on first run |

CIFAR-10 will be downloaded to a `data/` subdirectory automatically when you run the training notebook for the first time. No manual download required.

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

*Built with PyTorch · Conda · JupyterLab*
