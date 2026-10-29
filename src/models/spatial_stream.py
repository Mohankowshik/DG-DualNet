"""
Spatial Stream Backbone for Deepfake Feature Extraction.
Extracts facial semantic features, blending artifacts, boundary discontinuities,
and lighting inconsistencies from RGB input images.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


class SpatialStream(nn.Module):
    """
    RGB Spatial Backbone with feature extraction and Grad-CAM hooks.
    Uses ResNet backbone with customizable depth and feature projection.
    """
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = False, feature_dim: int = 512):
        super().__init__()
        self.backbone_name = backbone_name
        self.feature_dim = feature_dim

        # Load backbone
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            try:
                base = models.resnet18(weights=weights)
            except Exception:
                base = models.resnet18(weights=None)
            in_features = base.fc.in_features
            # Extract conv layers up to layer4
            self.initial = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool
            )
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4  # Saliency hook target for Grad-CAM
        elif backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            try:
                base = models.resnet34(weights=weights)
            except Exception:
                base = models.resnet34(weights=None)
            in_features = base.fc.in_features
            self.initial = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            try:
                base = models.resnet50(weights=weights)
            except Exception:
                base = models.resnet50(weights=None)
            in_features = base.fc.in_features
            self.initial = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

    def extract_spatial_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Runs forward pass up to layer4 returning feature maps (B, C, H', W')."""
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            feature_map: (B, C_backbone, H/32, W/32) spatial activation map
            features: (B, feature_dim) pooled feature vector
        """
        feat_map = self.extract_spatial_maps(x)
        pooled = self.pool(feat_map).flatten(1)
        feat_vec = self.projector(pooled)
        return feat_map, feat_vec
