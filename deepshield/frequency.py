"""
BlurGuard Frequency Regularization for DeepShield
===================================================
Implements the core BlurGuard insight: adversarial perturbations optimized
only for imperceptibility (L∞-ball) produce noise that is easily detectable
— and removable — in the frequency domain via JPEG or diffusion-based
purification.

Solution: shape the perturbation's power spectrum to match the natural 1/f²
distribution of the original image, making it indistinguishable to any
frequency-domain purification method.

Key components:
  - RAPSD: Radially Averaged Power Spectral Density
  - L_freq: Power spectrum regularization loss (Eq. 8 in BlurGuard paper)
  - Adaptive Gaussian blur: post-processing to realign spectrum of any perturbation

Reference: "BlurGuard: A Simple Approach for Robustifying Image Protection
Against AI-Powered Editing", Kim et al., NeurIPS 2025.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# RAPSD: Radially Averaged Power Spectral Density
# ──────────────────────────────────────────────────────────────────────────────

def compute_rapsd(x: torch.Tensor, n_bands: int = 16) -> torch.Tensor:
    """
    Compute Radially Averaged Power Spectral Density (RAPSD).

    Partitions 2D FFT frequency space into B radial bands and computes the
    average power (squared magnitude of FFT coefficients) in each band.
    Naturally follows a 1/f² law for photographic images.

    Args:
        x: Single image tensor [C, H, W] in any range.
        n_bands: Number of radial frequency bands.

    Returns:
        RAPSD vector of shape [n_bands], values > 0 (ε-clipped for log safety).
    """
    # Average channels → grayscale for frequency analysis
    x_gray = x.mean(dim=0)  # [H, W]
    H, W = x_gray.shape

    # 2D FFT → power spectrum
    f = torch.fft.fft2(x_gray)
    power = f.abs() ** 2

    # Shift DC component to center
    power = torch.fft.fftshift(power)

    # Build radial distance map from center
    cy, cx = H // 2, W // 2
    y_coords = torch.arange(H, device=x.device, dtype=torch.float32) - cy
    x_coords = torch.arange(W, device=x.device, dtype=torch.float32) - cx
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    radius = torch.sqrt(xx ** 2 + yy ** 2)

    max_r = float(min(cy, cx))
    band_edges = torch.linspace(0.0, max_r, n_bands + 1, device=x.device)

    rapsd = torch.zeros(n_bands, device=x.device, dtype=torch.float32)
    for b in range(n_bands):
        mask = (radius >= band_edges[b]) & (radius < band_edges[b + 1])
        if mask.any():
            rapsd[b] = power[mask].mean()

    return rapsd.clamp(min=1e-10)  # prevent log(0)


def frequency_reg_loss(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    n_bands: int = 16,
) -> torch.Tensor:
    """
    BlurGuard power spectrum regularization loss (Eq. 8 in paper).

    Minimizes the maximum log-ratio between the RAPSD of the protected image
    and the original across all frequency bands:

        L_freq(x̂, x) = ‖log(RAPSD(x̂) / RAPSD(x))‖_∞

    A small L_freq means the protected image's frequency fingerprint is
    close to the original — making purification methods unable to distinguish
    the perturbation from natural image content.

    Args:
        x_adv: Protected image [1, C, H, W].
        x_orig: Original image [1, C, H, W]. Same range as x_adv.
        n_bands: Number of radial frequency bands.

    Returns:
        Scalar loss value.
    """
    rapsd_orig = compute_rapsd(x_orig[0], n_bands)
    rapsd_adv = compute_rapsd(x_adv[0], n_bands)

    log_ratio = (torch.log(rapsd_adv) - torch.log(rapsd_orig)).abs()
    return log_ratio.max()


# ──────────────────────────────────────────────────────────────────────────────
# Learnable Gaussian blur for spectrum alignment
# ──────────────────────────────────────────────────────────────────────────────

def _make_gaussian_kernel(sigma: float, kernel_size: int, device: torch.device) -> torch.Tensor:
    """Build a 2D Gaussian kernel for a given sigma."""
    coords = torch.arange(kernel_size, device=device, dtype=torch.float32)
    coords -= kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2 + 1e-8))
    g /= g.sum()
    kernel = g.outer(g)
    return kernel


def apply_gaussian_blur_to_perturbation(
    perturbation: torch.Tensor,
    sigma: float,
    kernel_size: int = 11,
) -> torch.Tensor:
    """
    Apply Gaussian blur to a perturbation tensor.

    Blurring acts as a low-pass filter that removes the high-frequency
    components of the perturbation that JPEG compression would strip anyway.
    By removing them proactively and concentrating energy in lower frequencies,
    the perturbation becomes more robust to format conversion.

    Args:
        perturbation: Tensor [1, C, H, W].
        sigma: Gaussian blur std — higher = more blurring = lower frequencies.
        kernel_size: Kernel size (odd number).

    Returns:
        Blurred perturbation, same shape.
    """
    C = perturbation.shape[1]
    kernel = _make_gaussian_kernel(sigma, kernel_size, perturbation.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).expand(C, 1, -1, -1)
    padding = kernel_size // 2
    return F.conv2d(perturbation, kernel, padding=padding, groups=C)


def adaptive_blur_alignment(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    n_bands: int = 16,
    sigma_search: Tuple[float, float] = (0.1, 5.0),
    n_steps: int = 50,
    lr: float = 0.1,
) -> torch.Tensor:
    """
    Apply BlurGuard-style adaptive Gaussian blurring to align the perturbation's
    frequency spectrum with the original image.

    Optimizes blur intensity σ to minimize L_freq while keeping the perturbation
    as sharp as possible (i.e., uses minimal blurring that achieves alignment).

    This is run as a post-processing step after PGD converges.

    Args:
        x_adv: Protected image [1, C, H, W].
        x_orig: Original image [1, C, H, W].
        n_bands: Frequency bands for RAPSD.
        sigma_search: Search range for σ (log-space).
        n_steps: Gradient steps for σ optimization.
        lr: Learning rate for σ optimization.

    Returns:
        Frequency-aligned protected image, same shape as x_adv.
    """
    perturbation = (x_adv - x_orig).detach()

    # Optimize log(σ) for numerical stability
    log_sigma = torch.tensor(
        math.log((sigma_search[0] + sigma_search[1]) / 2),
        device=x_adv.device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([log_sigma], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        sigma_val = log_sigma.exp().clamp(sigma_search[0], sigma_search[1])

        # Apply blur with current sigma
        blurred = apply_gaussian_blur_to_perturbation(
            perturbation, sigma=sigma_val.item(), kernel_size=11
        )
        x_candidate = (x_orig + blurred).clamp(-1.0, 1.0)

        loss = frequency_reg_loss(x_candidate, x_orig, n_bands)
        loss.backward()
        optimizer.step()

    # Apply final optimal blur
    with torch.no_grad():
        sigma_final = log_sigma.exp().clamp(sigma_search[0], sigma_search[1]).item()
        blurred_final = apply_gaussian_blur_to_perturbation(
            perturbation, sigma=sigma_final, kernel_size=11
        )
        x_aligned = (x_orig + blurred_final).clamp(-1.0, 1.0)

    return x_aligned, sigma_final
