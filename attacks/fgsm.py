"""
FGSM (Fast Gradient Sign Method) Attack
========================================
Implementation of the one-step adversarial attack introduced by
Goodfellow et al. (2015), "Explaining and Harnessing Adversarial Examples".

The core idea:
    x_adv = x + ε · sign(∇_x L(θ, x, y))

The perturbation is computed in a *single* gradient step, making FGSM
extremely fast but relatively weak compared to iterative attacks like PGD.

Reference:
    https://arxiv.org/abs/1412.6572
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Generate adversarial examples using FGSM.

    Args:
        model:   The target classifier (must be in eval mode).
        images:  Clean input images of shape ``(B, C, H, W)``.
        labels:  Ground-truth labels of shape ``(B,)``.
        epsilon: Perturbation magnitude (ℓ∞ budget). Typical values for
                 MNIST are in the range [0.05, 0.3].

    Returns:
        Adversarial images of the same shape, clamped to [0, 1].
    """
    # Create a differentiable copy of the input (do NOT modify the original)
    images = images.clone().detach().requires_grad_(True)

    # Forward pass — compute classification loss
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    # Backward pass — compute gradient of loss w.r.t. input pixels
    model.zero_grad()
    loss.backward()

    # Craft adversarial examples: move each pixel by ε in the direction
    # that *increases* the loss (i.e., the sign of the gradient)
    perturbed_images = images + epsilon * images.grad.sign()

    # Clamp to valid pixel range [0, 1] to produce realistic images
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images