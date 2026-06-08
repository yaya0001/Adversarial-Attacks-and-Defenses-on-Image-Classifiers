"""
PGD (Projected Gradient Descent) Attack:
Implementation of the iterative adversarial attack.
PGD is essentially a multi-step FGSM that projects the perturbation back
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
    Generate adversarial examples using PGD
    Args:
        model: The target classifier.
        images: Clean input images.
        labels: Ground-truth labels.
        epsilon: Maximum ℓ∞ perturbation budget.
        alpha: Step size for each iteration.
        iters: Number of PGD iterations.
    Returns:
        Adversarial images of the same shape, clamped to [0, 1].
    """
   
    original_images = images.clone().detach()

    # Random initialization
    perturbed_images = original_images + torch.empty_like(original_images).uniform_(-epsilon, epsilon)
    perturbed_images = torch.clamp(perturbed_images, 0, 1).detach()

    for step in range(iters):
        # Enable gradient computation
        perturbed_images.requires_grad = True

        # Forward pass : compute classification loss
        outputs = model(perturbed_images)
        loss = F.cross_entropy(outputs, labels)

        # Backward pass : compute gradient of loss w.r.t. adversarial pixels
        model.zero_grad()
        loss.backward()

        # Take a signed gradient step
        adv_images = perturbed_images + alpha * perturbed_images.grad.sign()

        # Project back into the ε-ball
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)

        # Also clamp pixel values to the valid [0, 1] rang
        perturbed_images = torch.clamp(original_images + eta, 0, 1).detach()

    return perturbed_images