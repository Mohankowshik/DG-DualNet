"""
Loss functions for Domain-Generalized Deepfake Detection.
Includes:
- Binary Cross Entropy / Focal Loss for Fake Classification
- Domain Classification Cross Entropy for Adversarial Alignment
- Combined Total Objective: L_total = L_cls + lambda * L_domain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss to prevent easy background samples from dominating gradient updates.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_weight = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
            focal_loss = alpha_weight * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class DomainAdversarialLoss(nn.Module):
    """
    Joint objective combining task detection loss and domain-adversarial loss:
    L_total = L_cls(logits_cls, y) + lambda_dann * L_domain(logits_domain, d)
    """
    def __init__(self, use_focal: bool = True, domain_weight: float = 1.0):
        super().__init__()
        self.cls_criterion = FocalLoss(gamma=2.0) if use_focal else nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        self.domain_weight = domain_weight

    def forward(
        self,
        cls_logits: torch.Tensor,
        domain_logits: torch.Tensor,
        labels: torch.Tensor,
        domain_labels: torch.Tensor,
        current_lambda: float = 1.0
    ):
        loss_cls = self.cls_criterion(cls_logits, labels)
        loss_domain = self.domain_criterion(domain_logits, domain_labels)
        total_loss = loss_cls + (self.domain_weight * current_lambda) * loss_domain

        return {
            "total_loss": total_loss,
            "loss_cls": loss_cls,
            "loss_domain": loss_domain
        }
