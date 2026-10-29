"""
Evaluation and Cross-Domain Benchmarking Metrics.
Calculates Accuracy, AUC-ROC, Equal Error Rate (EER), F1-Score,
and Cross-Generator Generalization Gap.
"""

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Computes Equal Error Rate (EER) where False Positive Rate == False Negative Rate."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    # Find point where |fpr - fnr| is minimized
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates model across a dataloader.
    Returns comprehensive metrics dictionary.
    """
    model.eval()
    all_targets = []
    all_domains = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["rgb"].to(device)
            freq = batch["freq"].to(device)
            labels = batch["label"].cpu().numpy()
            domains = batch["domain"].cpu().numpy()

            out = model(rgb, freq)
            probs = out["cls_probs"].cpu().numpy()  # (B, 2)
            fake_probs = probs[:, 1]
            preds = np.argmax(probs, axis=1)

            all_targets.extend(labels)
            all_domains.extend(domains)
            all_probs.extend(fake_probs)
            all_preds.extend(preds)

    all_targets = np.array(all_targets)
    all_domains = np.array(all_domains)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    acc = float(accuracy_score(all_targets, all_preds))
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary", zero_division=0)

    # AUC and EER
    if len(np.unique(all_targets)) > 1:
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        roc_auc = float(auc(fpr, tpr))
        eer = compute_eer(all_targets, all_probs)
    else:
        roc_auc = 1.0 if acc == 1.0 else 0.5
        eer = 0.0
        fpr, tpr = np.array([0, 1]), np.array([0, 1])

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1]).tolist()

    return {
        "accuracy": acc,
        "auc": roc_auc,
        "eer": eer,
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "confusion_matrix": cm,
        "total_samples": len(all_targets),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    }
