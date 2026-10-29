"""
Cross-Modal Feature Fusion for Spatial and Frequency Streams.
Implements Cross-Attention and Gated Dynamic Fusion to balance spatial and spectral cues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Bidirectional Cross-Attention between Spatial and Frequency representations.
    Allows spatial clues to guide frequency inspection, and vice versa.
    """
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_spatial_to_freq = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_freq_to_spatial = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_s = nn.LayerNorm(embed_dim)
        self.norm_f = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, feat_spatial: torch.Tensor, feat_freq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_spatial: (B, D)
            feat_freq: (B, D)
        Returns:
            fused: (B, D) fused representation
        """
        # Treat as sequences of length 1 for multihead attention
        s_seq = feat_spatial.unsqueeze(1)  # (B, 1, D)
        f_seq = feat_freq.unsqueeze(1)     # (B, 1, D)

        # Cross attention
        s_attended, _ = self.attn_spatial_to_freq(query=s_seq, key=f_seq, value=f_seq)
        f_attended, _ = self.attn_freq_to_spatial(query=f_seq, key=s_seq, value=s_seq)

        s_out = self.norm_s(s_seq + s_attended).squeeze(1)  # (B, D)
        f_out = self.norm_f(f_seq + f_attended).squeeze(1)  # (B, D)

        # Gated MLP fusion
        combined = torch.cat([s_out, f_out], dim=-1)
        fused = self.mlp(combined)
        return self.norm_out(fused + (feat_spatial + feat_freq) * 0.5)


class GatedFusion(nn.Module):
    """
    Dynamic gated fusion: computes a per-sample gating coefficient alpha in [0, 1]
    representing the reliance on spatial vs frequency evidence.
    """
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 2),
            nn.Softmax(dim=-1)
        )
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_spatial: torch.Tensor, feat_freq: torch.Tensor):
        combined = torch.cat([feat_spatial, feat_freq], dim=-1)
        weights = self.gate(combined)  # (B, 2)
        w_s = weights[:, 0:1]
        w_f = weights[:, 1:2]
        fused = w_s * feat_spatial + w_f * feat_freq
        return self.proj(fused), weights
