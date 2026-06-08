# Development Log - Adversarial Attacks and Defenses on Image Classifiers
## Project 5 | Queen's University | Deep Learning Course

**Team:**
- Osama Elkhuribi
- Mohamed Abdelkhalek
- Mohamed Abdel Majid

---

## Week 1

### Session 1
**Work done:**
- Read and discussed the three core papers for the project:
  - Goodfellow et al. (2015) — FGSM
  - Kurakin et al. (2017) — Adversarial ML at Scale
  - Madry et al. (2018) — PGD and adversarial training
- Decided to use MNIST instead of CIFAR-10 for the baseline
  experiments. Reasoning: MNIST trains in minutes on CPU, which
  lets us run many epsilon sweeps without GPU. CIFAR-10 will be
  attempted as a stretch goal in Week 4 if time allows.
- Assigned roles consistent with the project instructions:
  - Osama → model architecture and training pipeline
  - Mohamed Abdelkhalek → attack implementations
  - Mohamed Abdel Majid → visualization, README, and reporting

**Decisions made:**
- Use PyTorch (all three members are more comfortable with it
  than TensorFlow).
- Store all metrics as JSON files so results are reproducible
  without rerunning training.
- Fix all random seeds (PyTorch, NumPy, Python random) to
  ensure reproducibility across machines.

**Issues:** None at this stage.

---

### Session 2
**Work done:**
- Osama set up the project directory structure:
src/models/
src/attacks/
results/checkpoints/
data/
- Created conda environment files for Linux and Windows to
  ensure cross-platform reproducibility.
- Osama began implementing the baseline CNN in `src/models/cnn.py`.
  Architecture decided: two convolutional layers (32 and 64
  filters, 3×3 kernels) followed by two fully connected layers
  (128 neurons → 10 output classes). MaxPool after each conv
  layer.
- Mohamed Abdelkhalek started reading the MNIST data loading
  pipeline using Torchvision.

**Decisions made:**
- Use `CrossEntropyLoss` with raw logits (no softmax in the
  forward pass) — PyTorch combines log-softmax and NLL internally,
  which avoids numerical instability.
- Split the 60,000 training images into 50,000 train /
  10,000 validation. Keep the original 10,000 test set
  completely held out.

**Issues:** None.

---

### Session 3
**Work done:**
- Osama completed `train.py`:
  - Adam optimizer, lr=1e-3, batch size=64
  - Fixed seeds: `torch.manual_seed(42)`, `numpy.random.seed(42)`
  - Saves checkpoint to `results/checkpoints/mnist_cnn.pth`
    after each epoch; keeps the final epoch checkpoint
  - Exports per-epoch loss, validation accuracy, and test accuracy
    to a JSON file for reproducibility
- First training run: 5 epochs on the 50,000-sample training
  split. Results:
  - Epoch 1: loss=0.2043, val=97.90%, test=98.31%
  - Epoch 5: loss=0.0225, val=98.44%, test=98.76%
  - Best test accuracy was at epoch 3: 98.87%
  - Final saved checkpoint: epoch 5, test accuracy 98.76%
- Mohamed Abdel Majid started the README draft.

**Decisions made:**
- Save the final epoch checkpoint rather than the best
  validation checkpoint. Rationale: we want to evaluate the
  converged model, not an early-stopped one, since our focus
  is on robustness analysis rather than maximizing clean accuracy.
- Keep training at 5 epochs — loss had converged and additional
  epochs showed no improvement.

**Issues:**
- Initial run failed because `tqdm` was not in the base conda
  environment. Fixed by adding it to `environment.yml`.

---

## Week 2

### Session 4
**Work done:**
- Mohamed Abdelkhalek implemented FGSM in `src/attacks/fgsm.py`.
  Formula: `x_adv = x + eps * sign(grad_x L(theta, x, y))`
  Implemented from scratch in PyTorch using `loss.backward()`
  and `x.grad.sign()`. Clamps output to [0, 1] to keep pixel
  values valid.
- Ran FGSM evaluation across epsilon values {0.05, 0.10, 0.20,
  0.30} on the held-out test set. Results:
  - ε=0.05: 94.45%
  - ε=0.10: 81.74%
  - ε=0.20: 25.74%
  - ε=0.30: 5.13%
- Results match the expected pattern from the literature —
  clean accuracy degrades monotonically with epsilon.

**Decisions made:**
- Evaluate attacks on the full 10,000-sample test set, not a
  subset, for statistically reliable accuracy estimates.
- Export per-epsilon results to JSON alongside the training
  metrics.

**Issues:**
- First FGSM attempt forgot to call `model.eval()` before
  evaluation, which left dropout (if any) active. Fixed.
  (Note: our model has no dropout, but the habit of calling
  `model.eval()` was enforced anyway for correctness.)
- Needed to set `requires_grad=True` on the input tensor
  explicitly before the forward pass. This was not obvious
  from the PyTorch docs at first.

---

### Session 5
**Work done:**
- Mohamed Abdelkhalek implemented PGD in `src/attacks/pgd.py`.
  Implemented as iterative FGSM with projection back onto the
  ε-ball after each step. Includes random initialization
  within B(x, ε) before the first iteration step.
  Formula per step:
  `x_adv = proj(x_adv + alpha * sign(grad L))`
- Ran PGD evaluation:
  - ε=0.10, α=0.01, 10 iters: 77.81%
  - ε=0.20, α=0.01, 20 iters: 2.33%
  - ε=0.30, α=0.01, 40 iters: 0.00%
- Confirmed PGD is stronger than FGSM at matched epsilon
  (e.g., 77.81% vs 81.74% at ε=0.10). This aligns with
  Madry et al.'s argument that iterative attacks find
  stronger adversarial examples.
- Mohamed Abdel Majid implemented `visualize_steps.py` to
  generate adversarial image collages showing the effect of
  increasing epsilon on digit appearance (Figures 2 and 3
  in the midterm report).

**Decisions made:**
- PGD iteration counts chosen so that total step budget
  (alpha × iters) exceeds epsilon slightly, relying on
  projection to enforce the constraint. This is standard
  practice and matches Madry et al.
- Alpha fixed at 0.01 across all epsilon settings for
  consistency and fair comparison.

**Issues:**
- PGD initially produced accuracy higher than FGSM at the
  same epsilon, which was suspicious. Root cause: missing
  the random initialization step. Adding uniform random
  noise from [-ε, ε] before the first PGD step fixed this
  and produced the expected stronger attack behavior.

---

### Session 6
**Work done:**
- Wrote and finalized the midterm report in IEEE two-column
  format covering: abstract, introduction, related work,
  methodology, preliminary results, planned work, and team
  contributions.
- Mohamed Abdel Majid configured reproducible Conda
  environments for both Linux and Windows and verified
  that training and evaluation scripts run cleanly in a
  fresh environment.
- Reviewed all JSON output files to confirm that Tables III,
  IV, and V in the report match the saved metrics exactly.

**Planned for Week 3:**
- Osama: implement FGSM-AT and PGD-AT adversarial training
  scripts
- Mohamed Abdelkhalek: design and run ablation studies
  (PGD steps, training epochs); generate accuracy-vs-epsilon
  curves
- Mohamed Abdel Majid: begin robustness–accuracy trade-off
  analysis; draft final report sections

**Issues:** None during this session.

---

## Week 3 — Adversarial Training & CIFAR-10 Extension

### Session 7
**Date:** 2026-06-02
**Work done:**
- Osama implemented `train_adversarial.py` to support FGSM-AT and PGD-AT (Madry et al. min-max formulation) on MNIST.
- Created `evaluate_all.py` to comprehensively test Baseline, FGSM-AT, and PGD-AT models across various FGSM and PGD epsilon settings.
- Mohamed Abdelkhalek created `ablation_study.py` to test the effect of PGD inner steps (3, 5, 7, 10) and training epochs (5, 10, 15) on robust accuracy.
- Mohamed Abdel Majid implemented `plot_results.py` to generate publication-quality figures from the JSON evaluation data.

**Decisions made:**
- Adversarial training is run for 10 epochs (vs 5 for baseline) to allow the model more time to adapt to perturbations.
- Decided to extend the project to CIFAR-10 (a stretch goal) to demonstrate robustness across datasets.

**Issues:** None.

---

### Session 8
**Date:** 2026-06-02

**Work done:**
- Designed a new `CifarCNN` architecture (3 conv layers + BN + dropout) to handle CIFAR-10's RGB images while remaining lightweight enough for CPU training.
- Replicated the entire experimental pipeline for CIFAR-10:
  - `train_cifar.py` (baseline)
  - `train_adversarial_cifar.py`
  - `evaluate_all_cifar.py`
  - `visualize_cifar.py`
  - `ablation_study_cifar.py`
  - `plot_results_cifar.py`
- Started background execution of MNIST FGSM-AT training.

**Decisions made:**
- Used standard CIFAR-10 attack parameters: ε = 8/255 (≈ 0.031) and α = 2/255.
- Reduced the scope of CIFAR-10 ablation studies to save CPU compute time.
- Initiated a project repository restructuring to separate scripts from the `src/` library modules.

**Issues:** None.

---

## Week 4 — [To be filled in as work progresses]