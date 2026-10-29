"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Dual-Stream Deepfake Explanations.
Produces spatial attention heatmaps identifying artifacts on eyes, blending seams,
and mouth boundaries.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class DualStreamGradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()

        # Target layer defaults to spatial_stream.layer4
        if target_layer is None:
            self.target_layer = self.model.spatial_stream.layer4
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        rgb_tensor: torch.Tensor,
        freq_tensor: Optional[torch.Tensor] = None,
        target_class: int = 1  # 1 = Fake
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generates Grad-CAM spatial heatmap for input sample.
        Args:
            rgb_tensor: (1, 3, H, W)
            freq_tensor: (1, 3, H, W) or None
            target_class: index to explain (0=Real, 1=Fake)
        Returns:
            heatmap_normalized: (H, W) in [0, 1]
            overlay_bgr: (H, W, 3) BGR image with heatmap overlay
            fake_prob: float probability
        """
        self.model.zero_grad()

        # Forward pass
        outputs = self.model(rgb_tensor, freq_tensor)
        cls_logits = outputs["cls_logits"]
        fake_prob = float(outputs["cls_probs"][0, target_class].item())

        # Target score
        score = cls_logits[0, target_class]
        score.backward(retain_graph=True)

        # Global average pool of gradients
        gradients = self.gradients  # (1, C, H', W')
        activations = self.activations  # (1, C, H', W')

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive influence

        # Upsample to match original image size
        _, _, h, w = rgb_tensor.shape
        cam_upsampled = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam_np = cam_upsampled.squeeze().detach().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            heatmap_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            heatmap_norm = np.zeros_like(cam_np)

        # Generate colored overlay
        heatmap_uint8 = np.uint8(255 * heatmap_norm)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        orig_img = (rgb_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        orig_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

        overlay = cv2.addWeighted(orig_bgr, 0.55, heatmap_color, 0.45, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        return heatmap_norm, overlay_rgb, fake_prob
