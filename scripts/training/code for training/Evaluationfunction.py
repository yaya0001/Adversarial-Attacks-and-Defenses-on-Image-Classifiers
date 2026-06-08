
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    label: str = "Test",
) -> float:
    """
    Evaluate model accuracy on clean (unperturbed) data.

    Args:
        model:       Trained model to evaluate.
        data_loader: DataLoader providing the evaluation set.
        device:      Device on which to run inference.
        label:       Descriptive label printed alongside the result.

    Returns:
        Accuracy as a percentage (0–100).
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    print(f" {label} Accuracy: {acc:.2f}%")
    return acc
