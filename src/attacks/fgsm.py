"""
FGSM (Fast Gradient Sign Method) Attack:
Implementation of the one-step adversarial attack
The perturbation is computed in a single gradient step, making FGSM extremely fast but relatively weak compared to iterative attacks like PGD
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
        model:   The target classifier
        images:  Clean input images
        labels:  Ground-truth labels
        epsilon: Perturbation magnitude

    Returns:
        Adversarial images.
    """
    # Create a differentiable copy of the input
    images = images.clone().detach().requires_grad_(True)

    # Forward pass : compute classification loss
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    # Backward pass : compute gradient of loss w.r.t. input pixels
    model.zero_grad()
    loss.backward()

    # Craft adversarial examples: move each pixel by ε in the direction that increases the loss
    perturbed_images = images + epsilon * images.grad.sign()

    # Clamp to valid pixel range [0, 1] to produce realistic images
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images