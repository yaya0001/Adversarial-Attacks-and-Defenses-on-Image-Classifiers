"""
CIFAR-10 Results Plotting Script
===================================
Generates publication-quality figures from the CIFAR-10 JSON
evaluation files.

Generated figures (saved to reports/figures/):
    13  Accuracy vs. Epsilon under FGSM — CIFAR-10
    14  Accuracy vs. Epsilon under PGD  — CIFAR-10
    15  Robustness–Accuracy Trade-off   — CIFAR-10
    16  Training Curves Comparison      — CIFAR-10
    17  Ablation: PGD Steps             — CIFAR-10
    18  Ablation: Training Epochs       — CIFAR-10

Usage:
    python plot_results_cifar.py
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# ──────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────
FIGURE_DIR = "reports/figures"
RESULTS_DIR = "results"
DPI = 300

FULL_EVAL_PATH = os.path.join(RESULTS_DIR, "cifar10_full_evaluation.json")
BASELINE_METRICS_PATH = os.path.join(RESULTS_DIR, "cifar10_training_metrics.json")
FGSM_AT_METRICS_PATH = os.path.join(RESULTS_DIR, "cifar10_training_metrics_fgsm_at.json")
PGD_AT_METRICS_PATH = os.path.join(RESULTS_DIR, "cifar10_training_metrics_pgd_at.json")
ABLATION_PATH = os.path.join(RESULTS_DIR, "cifar10_ablation_results.json")

# Consistent styling
COLORS = {
    "Baseline": "#E74C3C",
    "FGSM-AT":  "#3498DB",
    "PGD-AT":   "#2ECC71",
}
MARKERS = {"Baseline": "o", "FGSM-AT": "s", "PGD-AT": "D"}

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 100,
})


def _save_figure(filename: str) -> None:
    """Save the current figure and close it."""
    path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_accuracy_vs_epsilon_fgsm(data: dict) -> None:
    """Figure 13: CIFAR-10 accuracy vs epsilon under FGSM."""
    plt.figure(figsize=(8, 5))

    for model_name in ["Baseline", "FGSM-AT", "PGD-AT"]:
        if model_name not in data:
            continue
        md = data[model_name]
        epsilons = [0.0] + [r["epsilon"] for r in md["fgsm_results"]]
        accuracies = [md["clean_accuracy"]] + \
                     [r["accuracy"] for r in md["fgsm_results"]]

        plt.plot(epsilons, accuracies,
                 marker=MARKERS[model_name],
                 color=COLORS[model_name],
                 linewidth=2, markersize=7,
                 label=model_name)

    plt.xlabel("Perturbation Magnitude (ε)")
    plt.ylabel("Accuracy (%)")
    plt.title("CIFAR-10: Model Accuracy Under FGSM Attack")
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.ylim(-2, 102)

    # Custom x-ticks showing fractions
    eps_values = [0, 2/255, 4/255, 8/255, 16/255]
    eps_labels = ["0", "2/255", "4/255", "8/255", "16/255"]
    plt.xticks(eps_values, eps_labels)

    plt.tight_layout()
    _save_figure("13_cifar10_accuracy_vs_epsilon_fgsm.png")


def plot_accuracy_vs_epsilon_pgd(data: dict) -> None:
    """Figure 14: CIFAR-10 accuracy vs epsilon under PGD."""
    plt.figure(figsize=(8, 5))

    for model_name in ["Baseline", "FGSM-AT", "PGD-AT"]:
        if model_name not in data:
            continue
        md = data[model_name]
        epsilons = [0.0] + [r["epsilon"] for r in md["pgd_results"]]
        accuracies = [md["clean_accuracy"]] + \
                     [r["accuracy"] for r in md["pgd_results"]]

        plt.plot(epsilons, accuracies,
                 marker=MARKERS[model_name],
                 color=COLORS[model_name],
                 linewidth=2, markersize=7,
                 label=model_name)

    plt.xlabel("Perturbation Magnitude (ε)")
    plt.ylabel("Accuracy (%)")
    plt.title("CIFAR-10: Model Accuracy Under PGD Attack")
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.ylim(-2, 102)

    eps_values = [0, 4/255, 8/255, 16/255]
    eps_labels = ["0", "4/255", "8/255", "16/255"]
    plt.xticks(eps_values, eps_labels)

    plt.tight_layout()
    _save_figure("14_cifar10_accuracy_vs_epsilon_pgd.png")


def plot_tradeoff_bar_chart(data: dict) -> None:
    """Figure 15: CIFAR-10 robustness-accuracy trade-off."""
    models = [m for m in ["Baseline", "FGSM-AT", "PGD-AT"] if m in data]
    if not models:
        return

    clean_accs = []
    fgsm_accs = []
    pgd_accs = []

    for m in models:
        clean_accs.append(data[m]["clean_accuracy"])
        # FGSM at ε=8/255
        fgsm_8 = [r["accuracy"] for r in data[m]["fgsm_results"]
                  if abs(r["epsilon"] - 8/255) < 0.001]
        fgsm_accs.append(fgsm_8[0] if fgsm_8 else 0.0)
        # PGD at ε=8/255
        pgd_8 = [r["accuracy"] for r in data[m]["pgd_results"]
                 if abs(r["epsilon"] - 8/255) < 0.001]
        pgd_accs.append(pgd_8[0] if pgd_8 else 0.0)

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width, clean_accs, width, label="Clean",
                   color="#2ECC71", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x, fgsm_accs, width, label="FGSM ε=8/255",
                   color="#3498DB", edgecolor="white", linewidth=0.8)
    bars3 = ax.bar(x + width, pgd_accs, width, label="PGD ε=8/255",
                   color="#E74C3C", edgecolor="white", linewidth=0.8)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CIFAR-10: Robustness–Accuracy Trade-off")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    _save_figure("15_cifar10_robustness_accuracy_tradeoff.png")


def plot_training_curves() -> None:
    """Figure 16: CIFAR-10 training loss curves comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics_files = {
        "Baseline": BASELINE_METRICS_PATH,
        "FGSM-AT": FGSM_AT_METRICS_PATH,
        "PGD-AT": PGD_AT_METRICS_PATH,
    }

    for model_name, path in metrics_files.items():
        if not os.path.exists(path):
            print(f"  ⚠ Missing {path} — skipping {model_name}")
            continue
        with open(path, "r") as f:
            metrics = json.load(f)

        epochs_data = metrics["epochs"]
        ep_nums = [e["epoch"] for e in epochs_data]
        losses = [e["avg_loss"] for e in epochs_data]

        ax.plot(ep_nums, losses,
                marker=MARKERS[model_name],
                color=COLORS[model_name],
                linewidth=2, markersize=5,
                label=model_name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Training Loss")
    ax.set_title("CIFAR-10: Training Loss — Baseline vs. Adversarial Training")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure("16_cifar10_training_curves.png")


def plot_ablation_pgd_steps(ablation_data: dict) -> None:
    """Figure 17: CIFAR-10 ablation — PGD inner steps."""
    steps_data = ablation_data.get("ablation_pgd_steps", [])
    if not steps_data:
        print("  ⚠ No PGD steps ablation data — skipping Figure 17")
        return

    steps = [d["pgd_steps"] for d in steps_data]
    clean_accs = [d["clean_accuracy"] for d in steps_data]
    robust_accs = [d["robust_accuracy"] for d in steps_data]

    x = np.arange(len(steps))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width/2, clean_accs, width, label="Clean Accuracy",
                   color="#2ECC71", edgecolor="white")
    bars2 = ax.bar(x + width/2, robust_accs, width,
                   label="Robust Acc. (PGD ε=8/255)",
                   color="#E74C3C", edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("PGD Inner Steps During Training")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CIFAR-10 Ablation: Effect of PGD Inner Steps")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in steps])
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    _save_figure("17_cifar10_ablation_pgd_steps.png")


def plot_ablation_epochs(ablation_data: dict) -> None:
    """Figure 18: CIFAR-10 ablation — training epochs."""
    epochs_data = ablation_data.get("ablation_epochs", [])
    if not epochs_data:
        print("  ⚠ No epochs ablation data — skipping Figure 18")
        return

    epochs = [d["epochs"] for d in epochs_data]
    clean_accs = [d["clean_accuracy"] for d in epochs_data]
    robust_accs = [d["robust_accuracy"] for d in epochs_data]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(epochs, clean_accs,
            marker="o", color="#2ECC71", linewidth=2, markersize=8,
            label="Clean Accuracy")
    ax.plot(epochs, robust_accs,
            marker="D", color="#E74C3C", linewidth=2, markersize=8,
            label="Robust Acc. (PGD ε=8/255)")

    for i, ep in enumerate(epochs):
        ax.annotate(f"{clean_accs[i]:.1f}%",
                    (ep, clean_accs[i]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)
        ax.annotate(f"{robust_accs[i]:.1f}%",
                    (ep, robust_accs[i]),
                    textcoords="offset points", xytext=(0, -15),
                    ha="center", fontsize=9)

    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CIFAR-10 Ablation: Effect of Training Epochs")
    ax.set_xticks(epochs)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    _save_figure("18_cifar10_ablation_epochs.png")


def main() -> None:
    """Generate all CIFAR-10 result figures."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    print("Generating CIFAR-10 result figures…\n")

    # Figures 13–15
    if os.path.exists(FULL_EVAL_PATH):
        with open(FULL_EVAL_PATH, "r") as f:
            eval_data = json.load(f)
        plot_accuracy_vs_epsilon_fgsm(eval_data)
        plot_accuracy_vs_epsilon_pgd(eval_data)
        plot_tradeoff_bar_chart(eval_data)
    else:
        print(f"  ⚠ {FULL_EVAL_PATH} not found — skipping Figures 13–15")

    # Figure 16
    plot_training_curves()

    # Figures 17–18
    if os.path.exists(ABLATION_PATH):
        with open(ABLATION_PATH, "r") as f:
            ablation_data = json.load(f)
        plot_ablation_pgd_steps(ablation_data)
        plot_ablation_epochs(ablation_data)
    else:
        print(f"  ⚠ {ABLATION_PATH} not found — skipping Figures 17–18")

    print(f"\n✓ All CIFAR-10 figures saved to: {FIGURE_DIR}/")


if __name__ == "__main__":
    main()
