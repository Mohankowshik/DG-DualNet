"""
Frequency-Domain Transformations and Spectral Analysis for Deepfake Forensics.
Provides:
- 2D Fast Fourier Transform (Log-Magnitude Spectrum)
- High-Pass Filtering & Azimuthal (Radial) Spectral Profiles
- Spatial Rich Model (SRM) High-Frequency Residual Filters
- Discrete Cosine Transform (DCT) utilities
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union


def compute_2d_fft(image_tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes 2D Fast Fourier Transform log-magnitude spectrum for an RGB image tensor.
    Args:
        image_tensor: Tensor of shape (B, C, H, W) or (C, H, W) in range [0, 1] or [-1, 1].
        eps: Small constant to avoid log(0).
    Returns:
        Magnitude spectrum normalized to [0, 1] with shape matching input spatial dims.
    """
    is_batched = image_tensor.ndim == 4
    if not is_batched:
        image_tensor = image_tensor.unsqueeze(0)

    # Convert to grayscale luminance channel if 3 channels: Y = 0.299R + 0.587G + 0.114B
    if image_tensor.shape[1] == 3:
        weights = torch.tensor([0.299, 0.587, 0.114], device=image_tensor.device).view(1, 3, 1, 1)
        gray = torch.sum(image_tensor * weights, dim=1, keepdim=True)
    else:
        gray = image_tensor

    # 2D FFT with centered low frequencies
    fft2 = torch.fft.fft2(gray, dim=(-2, -1))
    fft2_shifted = torch.fft.fftshift(fft2, dim=(-2, -1))
    magnitude = torch.abs(fft2_shifted)
    log_magnitude = torch.log(magnitude + eps)

    # Min-max normalization across spatial dimensions per sample
    b, c, h, w = log_magnitude.shape
    log_min = log_magnitude.view(b, c, -1).min(dim=-1)[0].view(b, c, 1, 1)
    log_max = log_magnitude.view(b, c, -1).max(dim=-1)[0].view(b, c, 1, 1)
    norm_magnitude = (log_magnitude - log_min) / (log_max - log_min + eps)

    if not is_batched:
        norm_magnitude = norm_magnitude.squeeze(0)
    return norm_magnitude


def compute_radial_profile(fft_magnitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes azimuthal (radial) average of 2D spectrum to characterize high-frequency decay.
    Generative models (GANs, Diffusion) exhibit characteristic spectral spikes/dips.
    Args:
        fft_magnitude: 2D numpy array (H, W).
    Returns:
        radii: 1D array of radius values.
        profile: 1D array of mean magnitude at each radius.
    """
    h, w = fft_magnitude.shape
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.hypot(x - center_x, y - center_y).astype(int)

    # Bin by radius
    max_r = min(center_y, center_x)
    radial_sum = np.bincount(r.ravel(), weights=fft_magnitude.ravel())
    radial_count = np.bincount(r.ravel())

    # Limit to valid disc
    radii = np.arange(max_r)
    profile = (radial_sum[:max_r] / np.maximum(radial_count[:max_r], 1))
    return radii, profile


class SRMConv2d(nn.Module):
    """
    Spatial Rich Model (SRM) fixed high-pass filter bank.
    Captures pixel-level residual noise from upsampling, blending boundaries, and deconvolution.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # 3 standard 5x5 SRM forensic kernels
        filter1 = np.array([
            [0,  0,   0,  0,  0],
            [0, -1,   2, -1,  0],
            [0,  2,  -4,  2,  0],
            [0, -1,   2, -1,  0],
            [0,  0,   0,  0,  0]
        ], dtype=np.float32) / 4.0

        filter2 = np.array([
            [-1,  2,  -2,  2, -1],
            [ 2, -6,   8, -6,  2],
            [-2,  8, -12,  8, -2],
            [ 2, -6,   8, -6,  2],
            [-1,  2,  -2,  2, -1]
        ], dtype=np.float32) / 12.0

        filter3 = np.array([
            [0,  0,  0,  0, 0],
            [0,  0,  1,  0, 0],
            [0,  1, -4,  1, 0],
            [0,  0,  1,  0, 0],
            [0,  0,  0,  0, 0]
        ], dtype=np.float32) / 4.0

        filters = np.stack([filter1, filter2, filter3], axis=0)  # (3, 5, 5)
        # Expand for in_channels (e.g. 3 RGB -> 3 SRM channels)
        weight = torch.tensor(filters).unsqueeze(1)  # (3, 1, 5, 5)
        if in_channels == 3:
            # Replicate or average over RGB
            weight = weight.repeat(1, 3, 1, 1) / 3.0

        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Non-trainable high-pass residual filter
        return F.conv2d(x, self.weight, padding=2)


def get_frequency_input(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Combines 2D-FFT log spectrum with High-Pass / Gradient residuals to form a 3-channel
    frequency-domain representation ready for the frequency CNN backbone:
    Channel 0: 2D-FFT Log Magnitude
    Channel 1: High-Frequency Laplace Residual
    Channel 2: Sobel Edge Magnitude
    """
    b, c, h, w = rgb_tensor.shape
    device = rgb_tensor.device

    # Channel 0: 2D-FFT
    fft_spec = compute_2d_fft(rgb_tensor)  # (B, 1, H, W)

    # Luminance grayscale
    if c == 3:
        w_lum = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
        gray = torch.sum(rgb_tensor * w_lum, dim=1, keepdim=True)
    else:
        gray = rgb_tensor

    # Channel 1: High-Pass Laplacian kernel
    lap_kernel = torch.tensor([
        [0.0, -1.0, 0.0],
        [-1.0, 4.0, -1.0],
        [0.0, -1.0, 0.0]
    ], device=device).view(1, 1, 3, 3)
    lap_res = torch.abs(F.conv2d(gray, lap_kernel, padding=1))
    lap_res = torch.clamp(lap_res * 2.0, 0.0, 1.0)

    # Channel 2: Sobel gradient magnitude
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    sobel_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
    sobel_mag = torch.clamp(sobel_mag / 2.0, 0.0, 1.0)

    freq_tensor = torch.cat([fft_spec, lap_res, sobel_mag], dim=1)  # (B, 3, H, W)
    return freq_tensor
