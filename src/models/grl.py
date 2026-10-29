"""
Gradient Reversal Layer (GRL) for Domain-Adversarial Training (DANN).
Used to align feature distributions across diverse deepfake generation methods
(e.g., GANs, Diffusion, Blending, FaceSwap).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse and scale gradient by -alpha
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer module.
    Multiplies the gradient by -alpha during backward pass to make the feature
    extractor domain-invariant.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def set_alpha(self, alpha: float):
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


def compute_dann_lambda(progress: float, gamma: float = 10.0, max_lambda: float = 1.0) -> float:
    """
    Dynamically ramps up lambda from 0 to max_lambda as training progresses:
    lambda(p) = max_lambda * (2 / (1 + exp(-gamma * p)) - 1)
    where p in [0, 1] is training progress.
    """
    p = np.clip(progress, 0.0, 1.0)
    lam = (2.0 / (1.0 + np.exp(-gamma * p))) - 1.0
    return float(max_lambda * lam)
