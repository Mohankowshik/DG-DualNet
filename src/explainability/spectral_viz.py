"""
Spectral Visualization and Forensic Analysis Charts.
Renders:
1. 2D Fast Fourier Transform (FFT) Magnitude Spectra
2. Azimuthal (Radial) Frequency Profile Plots
3. Forensic High-Frequency Residual maps
"""

import io
import base64
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple, Dict

from ..data.frequency_transforms import compute_radial_profile


def render_2d_fft_image(image_rgb: np.ndarray) -> np.ndarray:
    """
    Computes and renders 2D-FFT log-magnitude spectrum with 'inferno' colormap.
    Returns RGB uint8 image (H, W, 3).
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-7)

    # Normalize to 0..255
    norm_mag = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Apply colormap (INFERNO or VIRIDIS)
    color_spectrum = cv2.applyColorMap(norm_mag, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(color_spectrum, cv2.COLOR_BGR2RGB)


def render_radial_profile_chart(image_rgb: np.ndarray, title: str = "Azimuthal Radial Spectrum") -> str:
    """
    Plots the radial average of the FFT magnitude spectrum.
    Returns base64 encoded PNG string.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.log(np.abs(dft_shift) + 1e-7)

    radii, profile = compute_radial_profile(magnitude_spectrum)

    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=120)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    ax.plot(radii, profile, color='#38bdf8', linewidth=2.0, label='Sample Spectrum')

    # Stylize
    ax.set_title(title, color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Spatial Frequency Radius (r)', color='#94a3b8', fontsize=9)
    ax.set_ylabel('Mean Log-Magnitude', color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#334155')
    ax.grid(True, linestyle='--', alpha=0.3, color='#475569')
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def render_high_pass_residual(image_rgb: np.ndarray) -> np.ndarray:
    """Computes high-frequency Laplace residual map."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian_abs = np.abs(laplacian)
    norm_res = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_res = cv2.applyColorMap(norm_res, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(color_res, cv2.COLOR_BGR2RGB)
