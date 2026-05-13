"""
attacks — Adversarial attack implementations.

Provides:
    - ``fgsm_attack``: Fast Gradient Sign Method (single-step)
    - ``pgd_attack``:  Projected Gradient Descent (iterative)
"""

from .fgsm import fgsm_attack
from .pgd import pgd_attack
