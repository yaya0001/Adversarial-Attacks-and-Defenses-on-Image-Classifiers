"""
models : Neural network architectures.

Provides:
    - ``SimpleCNN``: A lightweight 2-layer CNN for MNIST classification For our DL project @ Queen's University.
    - ``CifarCNN``:  A 3-layer CNN with batch norm for CIFAR-10 classification.
"""

from .cnn import SimpleCNN
from .cifar_cnn import CifarCNN
