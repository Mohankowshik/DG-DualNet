"""
Multi-Domain Dataset Loader and Cross-Domain Splitter.
Supports Leave-One-Domain-Out (LODO) generalization protocols:
Trains on source domains, evaluates generalization to unseen target generator domains.
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple

from .frequency_transforms import get_frequency_input
from .synthetic_domains import DOMAIN_NAMES, generate_synthetic_dataset


class MultiDomainDeepfakeDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, any]],
        transform=None,
        compute_freq_online: bool = True
    ):
        """
        Args:
            samples: List of dicts with keys: 'path', 'label' (0=Real, 1=Fake), 'domain' (int)
            transform: torchvision transforms or custom augmentations
            compute_freq_online: If True, computes frequency tensor dynamically
        """
        self.samples = samples
        self.transform = transform
        self.compute_freq_online = compute_freq_online

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        img_bgr = cv2.imread(item["path"])
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found at {item['path']}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1] torch tensor (C, H, W)
        tensor_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        # Frequency representation
        if self.compute_freq_online:
            with torch.no_grad():
                tensor_freq = get_frequency_input(tensor_rgb.unsqueeze(0)).squeeze(0)
        else:
            tensor_freq = torch.zeros_like(tensor_rgb)

        return {
            "rgb": tensor_rgb,
            "freq": tensor_freq,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "domain": torch.tensor(item["domain"], dtype=torch.long),
            "path": item["path"]
        }


def build_domain_dataloaders(
    data_dir: str,
    leave_out_domain: int = 3,  # Leave out Domain 3 (Diffusion) as unseen target domain
    batch_size: int = 8,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds (train_loader, val_in_domain_loader, test_out_of_domain_loader)
    following the Domain Generalization evaluation protocol.
    """
    # Ensure data directory exists and has samples
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        generate_synthetic_dataset(data_dir, num_per_domain=30)

    all_samples = []
    for domain_id, name in DOMAIN_NAMES.items():
        domain_folder = os.path.join(data_dir, name)
        if not os.path.exists(domain_folder):
            continue
        patterns = [os.path.join(domain_folder, ext) for ext in ["*.png", "*.jpg", "*.jpeg"]]
        files = []
        for p in patterns:
            files.extend(glob.glob(p))

        label = 0 if domain_id == 0 else 1  # 0 = Real, 1 = Fake
        for f in files:
            all_samples.append({
                "path": f,
                "label": label,
                "domain": domain_id
            })

    # Partition by domain
    train_samples = []
    val_in_domain_samples = []
    test_ood_samples = []

    np.random.seed(42)
    for s in all_samples:
        if s["domain"] == leave_out_domain:
            test_ood_samples.append(s)
        else:
            # 80/20 train/val split for training domains
            if np.random.rand() < 0.8:
                train_samples.append(s)
            else:
                val_in_domain_samples.append(s)

    train_ds = MultiDomainDeepfakeDataset(train_samples)
    val_ds = MultiDomainDeepfakeDataset(val_in_domain_samples)
    test_ood_ds = MultiDomainDeepfakeDataset(test_ood_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ood_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
