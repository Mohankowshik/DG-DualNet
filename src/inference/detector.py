"""
Inference Detector Engine for Domain-Generalized Deepfake Detection.
Integrates face extraction, dual-stream feature processing, Grad-CAM visualization,
spectral analysis, and structured reporting.
"""

import os
import io
import base64
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any, Optional

from ..models.dg_dualnet import DGDualNet
from ..data.face_extractor import FaceExtractor
from ..data.frequency_transforms import get_frequency_input
from ..explainability.gradcam import DualStreamGradCAM
from ..explainability.spectral_viz import (
    render_2d_fft_image,
    render_radial_profile_chart,
    render_high_pass_residual
)
from ..data.synthetic_domains import DOMAIN_NAMES


def numpy_to_base64_png(img_rgb: np.ndarray) -> str:
    """Encodes RGB numpy array to base64 data URI."""
    pil_img = Image.fromarray(img_rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


class DeepfakeDetector:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        backbone: str = "resnet18",
        device: str = "cpu"
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = DGDualNet(
            backbone_name=backbone,
            pretrained_spatial=False,
            feature_dim=512,
            num_domains=len(DOMAIN_NAMES),
            fusion_type="cross_attention"
        ).to(self.device)

        # Load weights if available
        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded weights from {weights_path}")
        else:
            print("Initialized DG-DualNet with default initialized weights.")

        self.model.eval()
        self.face_extractor = FaceExtractor(target_size=(256, 256))
        self.gradcam = DualStreamGradCAM(self.model)

    def analyze_image(
        self,
        image_input: Any,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Analyzes an image and returns full forensic diagnostic analysis.
        """
        # 1. Face Extraction
        extracted = self.face_extractor.extract_face(image_input)
        face_rgb = extracted["face_rgb"]
        bbox = extracted["bbox"]
        detected = extracted["detected"]

        # 2. Tensor preparation
        tensor_rgb = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
        tensor_rgb = tensor_rgb.unsqueeze(0).to(self.device)

        with torch.no_grad():
            tensor_freq = get_frequency_input(tensor_rgb)

        # 3. Model Inference
        outputs = self.model(tensor_rgb, tensor_freq)
        cls_probs = outputs["cls_probs"][0].detach().cpu().numpy()
        prob_real = float(cls_probs[0])
        prob_fake = float(cls_probs[1])

        # Domain head outputs (generation method likelihoods)
        dom_logits = outputs["domain_logits"][0].detach().cpu().numpy()
        dom_probs = np.exp(dom_logits - np.max(dom_logits))
        dom_probs = dom_probs / np.sum(dom_probs)

        domain_breakdown = {
            DOMAIN_NAMES.get(idx, f"Domain_{idx}"): float(prob)
            for idx, prob in enumerate(dom_probs)
        }

        # 4. Grad-CAM Spatial Heatmap
        _, overlay_rgb, _ = self.gradcam.generate_heatmap(tensor_rgb, tensor_freq, target_class=1)

        # 5. Frequency & Spectral Visualizations
        fft_spectrum_rgb = render_2d_fft_image(face_rgb)
        radial_chart_b64 = render_radial_profile_chart(face_rgb, title="Azimuthal Frequency Signature")
        residual_rgb = render_high_pass_residual(face_rgb)

        # Base64 encodings for UI
        face_b64 = numpy_to_base64_png(face_rgb)
        overlay_b64 = numpy_to_base64_png(overlay_rgb)
        fft_b64 = numpy_to_base64_png(fft_spectrum_rgb)
        residual_b64 = numpy_to_base64_png(residual_rgb)

        # Decision
        is_fake = prob_fake >= threshold
        verdict = "DEEPFAKE" if is_fake else "REAL"
        confidence = prob_fake if is_fake else prob_real

        # Forensic indicators
        indicators = []
        if prob_fake > 0.6:
            indicators.append("Strong multi-modal forensic anomaly detected.")
        if dom_breakdown_max := max(domain_breakdown, key=domain_breakdown.get):
            if dom_breakdown_max != "Pristine_Real" and prob_fake > 0.4:
                indicators.append(f"Spectral signature closely resembles {dom_breakdown_max}.")
        if not detected:
            indicators.append("No explicit frontal face detected; analyzed center crop.")

        return {
            "verdict": verdict,
            "is_fake": is_fake,
            "confidence": float(confidence),
            "prob_fake": prob_fake,
            "prob_real": prob_real,
            "threshold": threshold,
            "face_detected": detected,
            "bbox": bbox,
            "domain_breakdown": domain_breakdown,
            "indicators": indicators,
            "visualizations": {
                "face_crop": face_b64,
                "gradcam_overlay": overlay_b64,
                "fft_spectrum": fft_b64,
                "radial_chart": radial_chart_b64,
                "high_pass_residual": residual_b64
            }
        }
