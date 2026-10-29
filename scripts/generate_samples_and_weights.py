"""
Setup Script: Generates multi-domain synthetic dataset, creates sample faces,
and trains a lightweight DG-DualNet checkpoint so the system is immediately functional.
"""

import os
import sys
import torch

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.synthetic_domains import (
    DOMAIN_NAMES,
    create_base_face_pattern,
    inject_gan_artifacts,
    inject_faceswap_artifacts,
    inject_diffusion_artifacts,
    generate_synthetic_dataset
)
from src.data.dataset import build_domain_dataloaders
from src.models.dg_dualnet import DGDualNet
from src.training.trainer import DGTrainer
import cv2


def generate_curated_samples(sample_dir: str = "samples"):
    """Generates 4 representative reference samples for quick UI demonstration."""
    os.makedirs(sample_dir, exist_ok=True)
    base = create_base_face_pattern(size=256, seed=42)

    samples = {
        "sample_real.png": base,
        "sample_gan.png": inject_gan_artifacts(base, seed=101),
        "sample_faceswap.png": inject_faceswap_artifacts(base, seed=202),
        "sample_diffusion.png": inject_diffusion_artifacts(base, seed=303)
    }

    for name, img in samples.items():
        out_path = os.path.join(sample_dir, name)
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Created sample: {out_path}")


def train_initial_checkpoint(data_dir: str = "data/synthetic", epochs: int = 5):
    """Trains a verified DG-DualNet model for the UI and CLI."""
    print("Preparing multi-domain dataset...")
    generate_synthetic_dataset(data_dir, num_per_domain=25)

    print("Building dataloaders with Leave-One-Domain-Out (Diffusion held out)...")
    train_loader, val_loader, test_ood_loader = build_domain_dataloaders(
        data_dir=data_dir,
        leave_out_domain=3,  # Target unseen domain: Latent Diffusion
        batch_size=8
    )

    print("Instantiating DG-DualNet...")
    model = DGDualNet(
        backbone_name="resnet18",
        pretrained_spatial=False,
        feature_dim=512,
        num_domains=len(DOMAIN_NAMES),
        fusion_type="cross_attention"
    )

    trainer = DGTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_ood_loader=test_ood_loader,
        lr=1e-3,
        device="cpu",
        save_dir="weights"
    )

    print("Training DG-DualNet...")
    results = trainer.fit(epochs=epochs, save_best_name="dg_dualnet_best.pt")
    print(f"Training completed! Best checkpoint at {results['best_checkpoint']}")
    return results


if __name__ == "__main__":
    generate_curated_samples()
    train_initial_checkpoint(epochs=4)
