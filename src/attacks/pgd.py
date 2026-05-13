"""
PGD (Projected Gradient Descent) Attack
========================================
Implementation of the iterative adversarial attack introduced by
Madry et al. (2018), "Towards Deep Learning Models Resistant to
Adversarial Attacks".

PGD is essentially a multi-step FGSM that projects the perturbation
back into the ℓ∞ ball after each step:

    for t = 1, …, T:
        x_{t+1} = Π_{B(x, ε)} ( x_t + α · sign(∇_x L(θ, x_t, y)) )

where Π clips the total perturbation to ‖δ‖∞ ≤ ε and the pixel
values to [0, 1].

Following Madry et al., the attack is initialized with a uniform
random perturbation within the ε-ball to avoid poor local optima.

Reference:
    https://arxiv.org/abs/1706.06083
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.3,
    alpha: float = 0.01,
    iters: int = 40,
) -> torch.Tensor:
    """
    Generate adversarial examples using Projected Gradient Descent (PGD).

    Args:
        model:   The target classifier (must be in eval mode).
        images:  Clean input images of shape ``(B, C, H, W)``.
        labels:  Ground-truth labels of shape ``(B,)``.
        epsilon: Maximum ℓ∞ perturbation budget.
        alpha:   Step size for each iteration (typically ε / iters or small).
        iters:   Number of PGD iterations.

    Returns:
        Adversarial images of the same shape, clamped to [0, 1].
    """
    # Keep an unmodified reference to the original clean images
    original_images = images.clone().detach()

    # Random initialization within the ε-ball (Madry et al., 2018)
    perturbed_images = original_images + torch.empty_like(
        original_images
    ).uniform_(-epsilon, epsilon)
    perturbed_images = torch.clamp(perturbed_images, 0, 1).detach()

    for step in range(iters):
        # Enable gradient computation on the current adversarial images
        perturbed_images.requires_grad = True

        # Forward pass — compute classification loss
        outputs = model(perturbed_images)
        loss = F.cross_entropy(outputs, labels)

        # Backward pass — compute gradient of loss w.r.t. adversarial pixels
        model.zero_grad()
        loss.backward()

        # Take a signed gradient step (ascend the loss surface)
        adv_images = perturbed_images + alpha * perturbed_images.grad.sign()

        # Project back into the ε-ball around the original image (ℓ∞ constraint)
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)

        # Also clamp pixel values to the valid [0, 1] range
        perturbed_images = torch.clamp(original_images + eta, 0, 1).detach()

    return perturbed_images