"""
Domain-Generalized Deepfake Detection Trainer.
Implements the training loop with:
- Dual-stream optimization
- Dynamic Gradient Reversal Layer schedule
- Leave-One-Domain-Out validation tracking
- Checkpoint persistence
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from ..models.grl import compute_dann_lambda
from ..training.loss import DomainAdversarialLoss
from ..training.evaluate import evaluate_model


class DGTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_ood_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        save_dir: str = "weights",
        max_dann_lambda: float = 1.0,
        dann_gamma: float = 10.0
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_ood_loader = test_ood_loader
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=1e-6)
        self.criterion = DomainAdversarialLoss(use_focal=True, domain_weight=0.5)

        self.max_dann_lambda = max_dann_lambda
        self.dann_gamma = dann_gamma
        self.history = []

    def train_epoch(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        cls_loss_sum = 0.0
        dom_loss_sum = 0.0
        correct_cls = 0
        total_samples = 0

        num_batches = len(self.train_loader)

        for step, batch in enumerate(self.train_loader):
            rgb = batch["rgb"].to(self.device)
            freq = batch["freq"].to(self.device)
            labels = batch["label"].to(self.device)
            domains = batch["domain"].to(self.device)

            # Dynamic DANN lambda calculation
            global_step = epoch * num_batches + step
            total_steps = total_epochs * num_batches
            progress = global_step / max(total_steps, 1)
            current_lambda = compute_dann_lambda(progress, gamma=self.dann_gamma, max_lambda=self.max_dann_lambda)

            # Update GRL alpha in model
            self.model.set_grl_alpha(current_lambda)

            self.optimizer.zero_grad()
            outputs = self.model(rgb, freq)

            loss_dict = self.criterion(
                cls_logits=outputs["cls_logits"],
                domain_logits=outputs["domain_logits"],
                labels=labels,
                domain_labels=domains,
                current_lambda=current_lambda
            )

            loss = loss_dict["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # Tracking
            batch_sz = rgb.size(0)
            total_loss += loss.item() * batch_sz
            cls_loss_sum += loss_dict["loss_cls"].item() * batch_sz
            dom_loss_sum += loss_dict["loss_domain"].item() * batch_sz

            preds = torch.argmax(outputs["cls_probs"], dim=-1)
            correct_cls += (preds == labels).sum().item()
            total_samples += batch_sz

        self.scheduler.step()

        return {
            "train_loss": total_loss / max(total_samples, 1),
            "train_cls_loss": cls_loss_sum / max(total_samples, 1),
            "train_dom_loss": dom_loss_sum / max(total_samples, 1),
            "train_acc": correct_cls / max(total_samples, 1),
            "dann_lambda": current_lambda
        }

    def fit(self, epochs: int = 10, save_best_name: str = "dg_dualnet_best.pt") -> Dict[str, Any]:
        best_val_auc = -1.0
        best_checkpoint_path = os.path.join(self.save_dir, save_best_name)

        print(f"Starting Domain-Generalized Training for {epochs} epochs on device: {self.device}")

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(epoch, epochs)
            val_metrics = evaluate_model(self.model, self.val_loader, self.device)

            ood_metrics = {}
            if self.test_ood_loader is not None and len(self.test_ood_loader) > 0:
                ood_metrics = evaluate_model(self.model, self.test_ood_loader, self.device)

            epoch_log = {
                "epoch": epoch,
                **train_metrics,
                "val_acc": val_metrics["accuracy"],
                "val_auc": val_metrics["auc"],
                "val_eer": val_metrics["eer"],
                "ood_acc": ood_metrics.get("accuracy", 0.0),
                "ood_auc": ood_metrics.get("auc", 0.0),
                "ood_eer": ood_metrics.get("eer", 0.0)
            }
            self.history.append(epoch_log)

            print(
                f"[Epoch {epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_metrics['train_loss']:.4f} | Train Acc: {train_metrics['train_acc']:.3f} | "
                f"Val AUC: {val_metrics['auc']:.3f} | Val Acc: {val_metrics['accuracy']:.3f} | "
                f"OOD Target AUC: {ood_metrics.get('auc', 0.0):.3f}"
            )

            # Checkpoint best model on validation AUC
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "ood_metrics": ood_metrics,
                    "history": self.history
                }, best_checkpoint_path)

        # Save history log
        history_path = os.path.join(self.save_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return {
            "best_val_auc": best_val_auc,
            "best_checkpoint": best_checkpoint_path,
            "history": self.history
        }
