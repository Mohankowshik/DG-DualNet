"""
DG-DualNet: Domain-Generalized Dual-Stream Deepfake Detection Network.
Combines:
1. Spatial RGB Stream (ResNet backbone)
2. Frequency Spectral Stream (2D-FFT + High-frequency residuals)
3. Cross-Modal Attention & Gated Fusion
4. Primary Fake Classification Head (Binary: Real vs Fake)
5. Domain-Adversarial Head with Gradient Reversal Layer (GRL)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .spatial_stream import SpatialStream
from .frequency_stream import FrequencyStream
from .fusion import CrossModalAttention, GatedFusion
from .grl import GradientReversalLayer
from ..data.frequency_transforms import get_frequency_input


class DGDualNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained_spatial: bool = False,
        feature_dim: int = 512,
        num_domains: int = 4,  # e.g., Real, GAN, FaceSwap, Diffusion
        fusion_type: str = "cross_attention",
        grl_alpha: float = 1.0
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_domains = num_domains
        self.fusion_type = fusion_type

        # Streams
        self.spatial_stream = SpatialStream(
            backbone_name=backbone_name,
            pretrained=pretrained_spatial,
            feature_dim=feature_dim
        )
        self.frequency_stream = FrequencyStream(
            in_channels=3,
            feature_dim=feature_dim
        )

        # Fusion
        if fusion_type == "cross_attention":
            self.fusion = CrossModalAttention(embed_dim=feature_dim)
        else:
            self.fusion = GatedFusion(feature_dim=feature_dim)

        # Primary Task Head (Binary Real vs Fake)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim // 2, 2)  # Logits: [Real, Fake]
        )

        # Domain Adversarial Head (GRL + Domain Classifier)
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim // 2, num_domains)  # Logits for generator domains
        )

    def set_grl_alpha(self, alpha: float):
        """Update GRL alpha coefficient dynamically during training."""
        self.grl.set_alpha(alpha)

    def extract_features(self, rgb_images: torch.Tensor, freq_images: Optional[torch.Tensor] = None):
        """
        Extracts spatial, frequency, and fused representations.
        If freq_images is not provided, computes on the fly via get_frequency_input.
        """
        if freq_images is None:
            freq_images = get_frequency_input(rgb_images)

        feat_spatial_map, feat_spatial_vec = self.spatial_stream(rgb_images)
        feat_freq_map, feat_freq_vec = self.frequency_stream(freq_images)

        if self.fusion_type == "cross_attention":
            fused_feat = self.fusion(feat_spatial_vec, feat_freq_vec)
            stream_weights = None
        else:
            fused_feat, stream_weights = self.fusion(feat_spatial_vec, feat_freq_vec)

        return {
            "spatial_map": feat_spatial_map,
            "freq_map": feat_freq_map,
            "spatial_vec": feat_spatial_vec,
            "freq_vec": feat_freq_vec,
            "fused_feat": fused_feat,
            "stream_weights": stream_weights
        }

    def forward(
        self,
        rgb_images: torch.Tensor,
        freq_images: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Returns dictionary containing:
        - 'cls_logits': (B, 2) classification logits for Real (0) vs Fake (1)
        - 'domain_logits': (B, num_domains) domain logits passed through GRL
        - 'fused_feat': (B, feature_dim) domain-invariant fused embeddings
        - 'spatial_map': feature maps for Grad-CAM
        - 'cls_probs': (B, 2) softmax probabilities
        """
        extracted = self.extract_features(rgb_images, freq_images)
        fused = extracted["fused_feat"]

        # Classification logits
        cls_logits = self.classifier(fused)
        cls_probs = torch.softmax(cls_logits, dim=-1)

        # Domain adversarial branch
        reversed_feat = self.grl(fused)
        domain_logits = self.domain_classifier(reversed_feat)

        return {
            "cls_logits": cls_logits,
            "cls_probs": cls_probs,
            "domain_logits": domain_logits,
            "fused_feat": fused,
            "spatial_map": extracted["spatial_map"],
            "freq_map": extracted["freq_map"],
            "stream_weights": extracted["stream_weights"]
        }
