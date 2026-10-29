"""
Frequency Stream Network for Forensic Spectral Analysis.
Processes multi-channel frequency representations (2D-FFT log spectrum,
Laplacian residuals, and high-frequency noise statistics) to detect generator-agnostic traces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FrequencyConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class FrequencyStream(nn.Module):
    """
    Dedicated Frequency CNN architecture designed to capture spectral anomalies,
    radial decay inconsistencies, and periodic grid artifacts typical in GAN/Diffusion decoders.
    """
    def __init__(self, in_channels: int = 3, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual stages
        self.stage1 = FrequencyConvBlock(64, 64, stride=1)
        self.stage2 = FrequencyConvBlock(64, 128, stride=2)
        self.stage3 = FrequencyConvBlock(128, 256, stride=2)
        self.stage4 = FrequencyConvBlock(256, 512, stride=2)

        # Global Spectral Pooling: Combines standard AvgPool with Spectral Variance
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projector = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

    def extract_frequency_maps(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Frequency input tensor (B, 3, H, W)
        Returns:
            feat_map: (B, 512, H/32, W/32) frequency activation map
            features: (B, feature_dim) pooled frequency vector
        """
        feat_map = self.extract_frequency_maps(x)
        pooled = self.pool(feat_map).flatten(1)
        feat_vec = self.projector(pooled)
        return feat_map, feat_vec
