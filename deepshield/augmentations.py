"""
EOT Augmentation Suite for DeepShield
======================================
Implements Expectation over Transformations (EOT) augmentations that simulate
the preprocessing nudifier platforms apply to uploaded images before running
their models. By optimizing perturbations across these augmentations, we ensure
protection survives real-world format conversions and compressions.

Key augmentations:
  - JPEG compression (quality 40-95): primary stripping attack nudifiers use
  - Random resizing: many platforms downsample before inference
  - Gaussian blur: some platforms apply mild denoising

Uses Straight-Through Estimator (STE) for non-differentiable ops (JPEG).
"""

import io
import random
from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────────────

_to_pil = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()


def _tensor_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] float → [0,255] uint8 (for PIL conversion)."""
    return ((x.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0).byte()


def _uint8_to_float(x: torch.Tensor) -> torch.Tensor:
    """[0,255] uint8 → [-1,1] float."""
    return x.float() / 255.0 * 2.0 - 1.0


# ──────────────────────────────────────────────────────────────────────────────
# JPEG compression (non-differentiable, wrapped with STE)
# ──────────────────────────────────────────────────────────────────────────────

def jpeg_compress(x: torch.Tensor, quality: int) -> torch.Tensor:
    """
    Apply JPEG compression to a batch of images.

    Args:
        x: Tensor of shape [B, C, H, W] in range [-1, 1].
        quality: JPEG quality factor in [1, 95].

    Returns:
        JPEG-compressed tensor in range [-1, 1], same shape as x.
        NOTE: detached — gradients do NOT flow through this operation.
              Use jpeg_compress_ste() for gradient-compatible version.
    """
    x_uint8 = _tensor_to_uint8(x.detach().cpu())
    results = []
    for i in range(x_uint8.shape[0]):
        img = _to_pil(x_uint8[i])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        compressed = Image.open(buf).convert("RGB")
        results.append(_to_tensor(compressed) * 2.0 - 1.0)
    return torch.stack(results).to(x.device)


def jpeg_compress_ste(x: torch.Tensor, quality: int) -> torch.Tensor:
    """
    JPEG compression with Straight-Through Estimator for gradient flow.

    Forward pass: applies real JPEG compression (lossy, non-differentiable).
    Backward pass: gradient flows through x as if JPEG were identity (STE).

    This lets PGD push the perturbation in directions that remain effective
    even after JPEG compression strips high-frequency noise components.
    """
    x_compressed = jpeg_compress(x, quality)
    # STE: forward uses compressed, backward uses original gradient path
    return x + (x_compressed - x).detach()


# ──────────────────────────────────────────────────────────────────────────────
# Resize augmentation
# ──────────────────────────────────────────────────────────────────────────────

def random_resize_ste(
    x: torch.Tensor,
    scale_range: tuple = (0.85, 1.0),
) -> torch.Tensor:
    """
    Randomly downsample and upsample to simulate platform resizing.
    Uses STE for gradient flow.
    """
    B, C, H, W = x.shape
    scale = random.uniform(*scale_range)
    if abs(scale - 1.0) < 0.01:
        return x  # no-op
    H_small = max(int(H * scale), 32)
    W_small = max(int(W * scale), 32)

    # Downsample (non-differentiable via PIL round-trip)
    x_small = F.interpolate(x.detach(), size=(H_small, W_small), mode="bilinear", align_corners=False)
    # Upsample back to original size
    x_restored = F.interpolate(x_small, size=(H, W), mode="bilinear", align_corners=False)

    # STE: gradient flows through original x
    return x + (x_restored - x).detach()


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian blur augmentation
# ──────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 2D Gaussian kernel."""
    coords = torch.arange(kernel_size, device=device).float() - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.outer(g)
    return kernel


def gaussian_blur(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Apply Gaussian blur to a batch of images."""
    B, C, H, W = x.shape
    kernel = _gaussian_kernel(kernel_size, sigma, x.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).expand(C, 1, -1, -1)
    padding = kernel_size // 2
    return F.conv2d(x, kernel, padding=padding, groups=C)


# ──────────────────────────────────────────────────────────────────────────────
# Composite EOT augmentation sampler
# ──────────────────────────────────────────────────────────────────────────────

def webp_compress_ste(x: torch.Tensor, quality: int) -> torch.Tensor:
    """
    WebP compression with Straight-Through Estimator.

    Many nudifier platforms (clothoff.net, undressai.tools) transcode uploads
    to WebP before inference. Including WebP in EOT forces the perturbation
    to survive this format conversion in addition to JPEG.

    Args:
        x: Tensor [B, C, H, W] in [-1, 1].
        quality: WebP quality in [1, 95].

    Returns:
        WebP-compressed tensor, same shape, STE for gradient flow.
    """
    x_uint8 = _tensor_to_uint8(x.detach().cpu())
    results = []
    for i in range(x_uint8.shape[0]):
        img = _to_pil(x_uint8[i])
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=quality)
        buf.seek(0)
        compressed = Image.open(buf).convert("RGB")
        results.append(_to_tensor(compressed) * 2.0 - 1.0)
    x_compressed = torch.stack(results).to(x.device)
    return x + (x_compressed - x).detach()


def color_jitter_ste(
    x: torch.Tensor,
    brightness: float = 0.15,
    contrast: float = 0.15,
    saturation: float = 0.1,
) -> torch.Tensor:
    """
    Random brightness/contrast/saturation jitter with STE.

    Nudifier platforms often apply image normalization (gamma correction,
    brightness/contrast adjustment) before inference. Including color jitter
    forces the perturbation to survive these transformations.

    All values are sampled uniformly in [-factor, +factor].
    """
    # Work in [0, 1] range
    x_01 = (x.detach().clamp(-1, 1) + 1.0) / 2.0

    # Brightness: multiply by random factor
    b = 1.0 + random.uniform(-brightness, brightness)
    x_01 = (x_01 * b).clamp(0, 1)

    # Contrast: scale around mean
    if contrast > 0:
        c = 1.0 + random.uniform(-contrast, contrast)
        mean = x_01.mean(dim=(-2, -1), keepdim=True)
        x_01 = ((x_01 - mean) * c + mean).clamp(0, 1)

    # Saturation: interpolate toward grayscale
    if saturation > 0:
        s = 1.0 + random.uniform(-saturation, saturation)
        gray = x_01.mean(dim=1, keepdim=True)
        x_01 = (s * x_01 + (1 - s) * gray).clamp(0, 1)

    x_jittered = x_01 * 2.0 - 1.0
    return x + (x_jittered - x).detach()


def apply_eot_augmentation(
    x: torch.Tensor,
    jpeg_qualities: List[int] = (40, 50, 60, 70, 80, 90),
    resize_prob: float = 0.5,
    blur_prob: float = 0.2,
    webp_prob: float = 0.3,
    color_jitter_prob: float = 0.4,
) -> torch.Tensor:
    """
    Apply a random composition of augmentations to simulate nudifier preprocessing.

    Augmentation pipeline (matches real nudifier preprocessing):
      1. Format compression — JPEG always, WebP with probability webp_prob
         (clothoff.net, undressai.tools transcode uploads to WebP)
      2. Resize — simulates aggressive platform downscaling (0.5-1.0x)
      3. Color jitter — brightness/contrast/saturation normalization
      4. Mild Gaussian blur — denoising preprocessing

    All operations use STE so gradients flow back through x.

    Args:
        x: Input tensor [B, C, H, W] in [-1, 1].
        jpeg_qualities: Pool of JPEG quality values to sample from.
        resize_prob: Probability of random resize augmentation.
        blur_prob: Probability of Gaussian blur.
        webp_prob: Probability of additionally applying WebP compression.
        color_jitter_prob: Probability of color jitter augmentation.

    Returns:
        Augmented tensor, same shape as x, gradients flow through x.
    """
    # 1. JPEG compression — always applied (primary attack vector)
    quality = random.choice(jpeg_qualities)
    x = jpeg_compress_ste(x, quality)

    # 2. WebP — many platforms transcode to WebP before inference
    if random.random() < webp_prob:
        webp_q = random.randint(60, 90)
        x = webp_compress_ste(x, webp_q)

    # 3. Resize — simulates platform downscaling (wider range: 0.5-1.0x)
    if random.random() < resize_prob:
        x = random_resize_ste(x, scale_range=(0.5, 1.0))

    # 4. Color jitter — brightness/contrast normalization
    if random.random() < color_jitter_prob:
        x = color_jitter_ste(x)

    # 5. Mild Gaussian blur — denoising preprocessing
    if random.random() < blur_prob:
        sigma = random.uniform(0.3, 0.8)
        x = gaussian_blur(x, kernel_size=3, sigma=sigma)

    return x
