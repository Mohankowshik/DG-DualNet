"""
Procedural Multi-Domain Generator & Augmentor for Cross-Domain Deepfake Forensics.
Simulates generator-specific fingerprints across:
- Domain 0: Pristine Real Faces
- Domain 1: GAN-based Artifacts (checkerboard deconvolution, periodic spectral spikes)
- Domain 2: Blending / FaceSwap Artifacts (boundary seams, chromatic shifts, blur transitions)
- Domain 3: Diffusion-based Artifacts (latent noise residue, high-frequency smoothing, denoising drift)
"""

import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFilter
from typing import Tuple, Dict, List


DOMAIN_NAMES = {
    0: "Pristine_Real",
    1: "GAN_StyleGAN",
    2: "FaceSwap_Blending",
    3: "Latent_Diffusion"
}


def create_base_face_pattern(size: int = 256, seed: int = 42) -> np.ndarray:
    """Creates a stylized realistic facial structure pattern with natural organic gradients."""
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size, 3), dtype=np.float32)

    # Base skin tone gradient
    skin_base = np.array([235.0, 195.0, 175.0]) + rng.uniform(-15, 15, 3)
    y, x = np.ogrid[:size, :size]
    cy, cx = size // 2, size // 2

    # Face oval
    dist_oval = ((x - cx) / (size * 0.38)) ** 2 + ((y - cy) / (size * 0.46)) ** 2
    face_mask = np.clip(1.0 - dist_oval, 0.0, 1.0)
    face_mask = cv2.GaussianBlur(face_mask, (21, 21), 0)[:, :, None]

    # Facial features (eyes, nose, mouth)
    features = np.ones((size, size, 3), dtype=np.float32)

    # Eyes
    eye_y = int(size * 0.42)
    left_x, right_x = int(size * 0.36), int(size * 0.64)
    for ex in [left_x, right_x]:
        eye_dist = ((x - ex) / (size * 0.08)) ** 2 + ((y - eye_y) / (size * 0.045)) ** 2
        eye_m = np.clip(1.0 - eye_dist, 0.0, 1.0)[:, :, None]
        features -= eye_m * 0.55

    # Nose
    nose_y = int(size * 0.54)
    nose_dist = ((x - cx) / (size * 0.05)) ** 2 + ((y - nose_y) / (size * 0.08)) ** 2
    nose_m = np.clip(1.0 - nose_dist, 0.0, 1.0)[:, :, None]
    features -= nose_m * 0.2

    # Mouth
    mouth_y = int(size * 0.70)
    mouth_dist = ((x - cx) / (size * 0.14)) ** 2 + ((y - mouth_y) / (size * 0.04)) ** 2
    mouth_m = np.clip(1.0 - mouth_dist, 0.0, 1.0)[:, :, None]
    features -= mouth_m * 0.45

    # Combine
    bg = np.array([50.0, 55.0, 65.0])
    img = bg * (1.0 - face_mask) + (skin_base * features) * face_mask

    # Add realistic photographic sensor grain
    grain = rng.normal(0, 3.5, img.shape)
    img = np.clip(img + grain, 0.0, 255.0).astype(np.uint8)
    return img


def inject_gan_artifacts(face_img: np.ndarray, seed: int = 101) -> np.ndarray:
    """
    Injects periodic upsampling checkerboard patterns and spectral spikes characteristic
    of transposed convolutions in GANs (e.g. StyleGAN, ProGAN).
    """
    h, w, c = face_img.shape
    rng = np.random.RandomState(seed)
    artifact_img = face_img.astype(np.float32).copy()

    # 1. Periodic spatial checkerboard grid (stride-2 upsampling trace)
    y, x = np.ogrid[:h, :w]
    freq = rng.choice([4, 8, 16])
    grid = np.sin(2 * np.pi * x / freq) * np.cos(2 * np.pi * y / freq)
    grid = grid[:, :, None] * rng.uniform(4.0, 8.0)

    # 2. Subtle color channel de-correlation
    artifact_img[:, :, 0] += rng.normal(0, 2.0, (h, w))
    artifact_img[:, :, 2] += rng.normal(0, 2.0, (h, w))

    result = np.clip(artifact_img + grid, 0.0, 255.0).astype(np.uint8)
    return result


def inject_faceswap_artifacts(face_img: np.ndarray, seed: int = 202) -> np.ndarray:
    """
    Injects face-swap and blending boundaries (e.g., DeepFakes, Face2Face, FaceSwap).
    Simulates Poisson blending boundary mismatch and localized Gaussian blur.
    """
    h, w, c = face_img.shape
    rng = np.random.RandomState(seed)
    result = face_img.copy()

    # Elliptical inner face swap region
    cy, cx = int(h * 0.55), int(w * 0.5)
    rx, ry = int(w * 0.28), int(h * 0.32)

    # Inner warped crop
    y, x = np.ogrid[:h, :w]
    dist = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
    inner_mask = (dist < 1.0).astype(np.float32)

    # Create seam ring around boundary
    seam_mask = ((dist >= 0.85) & (dist <= 1.15)).astype(np.float32)
    seam_blur = cv2.GaussianBlur(seam_mask, (15, 15), 0)[:, :, None]

    # Apply color shift inside swap
    swapped = face_img.astype(np.float32)
    swapped[:, :, 0] = np.clip(swapped[:, :, 0] * rng.uniform(0.92, 1.08), 0, 255)
    swapped[:, :, 1] = np.clip(swapped[:, :, 1] * rng.uniform(0.95, 1.05), 0, 255)

    # Smooth the inner swap slightly
    swapped = cv2.GaussianBlur(swapped, (3, 3), 0)

    # Boundary seam with subtle edge discoloration
    blended = face_img.astype(np.float32) * (1.0 - seam_blur) + swapped * seam_blur
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def inject_diffusion_artifacts(face_img: np.ndarray, seed: int = 303) -> np.ndarray:
    """
    Injects latent diffusion artifacts (e.g. Stable Diffusion / Midjourney).
    Features: oversmoothed skin textures, high-frequency spectral attenuation,
    and subtle structural latent noise drift.
    """
    h, w, c = face_img.shape
    rng = np.random.RandomState(seed)

    # Bilateral smoothing (preserves strong edges while eliminating micro-pore organic grain)
    smoothed = cv2.bilateralFilter(face_img, d=9, sigmaColor=75, sigmaSpace=75)

    # Add latent space residual noise
    latent_noise = rng.normal(0, 2.5, (h // 4, w // 4, c)).astype(np.float32)
    upscaled_noise = cv2.resize(latent_noise, (w, h), interpolation=cv2.INTER_CUBIC)

    # Mild high-frequency attenuation
    diff_img = smoothed.astype(np.float32) + upscaled_noise
    return np.clip(diff_img, 0.0, 255.0).astype(np.uint8)


def generate_synthetic_dataset(output_dir: str, num_per_domain: int = 25):
    """Generates synthetic faces across all 4 domains for initial training & verification."""
    os.makedirs(output_dir, exist_ok=True)
    for domain_id, name in DOMAIN_NAMES.items():
        domain_dir = os.path.join(output_dir, name)
        os.makedirs(domain_dir, exist_ok=True)

        for i in range(num_per_domain):
            seed = domain_id * 1000 + i
            base = create_base_face_pattern(size=256, seed=seed)

            if domain_id == 0:
                img = base  # Real
            elif domain_id == 1:
                img = inject_gan_artifacts(base, seed=seed)
            elif domain_id == 2:
                img = inject_faceswap_artifacts(base, seed=seed)
            elif domain_id == 3:
                img = inject_diffusion_artifacts(base, seed=seed)

            file_path = os.path.join(domain_dir, f"sample_{i:04d}.png")
            cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"Synthesized {num_per_domain * 4} multi-domain faces in '{output_dir}'.")
