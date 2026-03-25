"""
Adversarial Loss Functions for DeepShield
==========================================
Multiple loss strategies for disrupting AI nudifiers and deepfake tools.

Loss hierarchy (most → least impactful for nudifiers):
  1. encoder_loss: Pushes VAE latent to a gray/noise target (fast, primary)
  2. denoising_loss: Maximizes UNet denoising error (stronger, slower)
  3. ensemble_loss: Average of encoder + denoising across multiple models

The encoder loss disrupts how the VAE encodes the image — critical because
ALL diffusion-based nudifiers (Stable Diffusion variants) must encode the
input through a VAE before inpainting/editing. A corrupted latent forces
the model to generate from garbage, producing incoherent outputs.
"""

import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Encoder attack (primary loss — fast)
# ──────────────────────────────────────────────────────────────────────────────

def get_gray_target(vae, x: torch.Tensor) -> torch.Tensor:
    """
    Encode a uniform gray image as the attack target.

    The goal is to push the protected image's VAE latent toward this gray
    target — as far from the real content as possible. This forces any
    diffusion model to generate from a corrupted, content-free latent.

    Using gray (rather than random noise) as target ensures consistent,
    stable optimization throughout PGD.
    """
    gray = torch.zeros_like(x)  # uniform gray in [-1, 1] space
    with torch.no_grad():
        z_target = vae.encode(gray).latent_dist.mean
    return z_target


def encoder_loss(
    x_adv: torch.Tensor,
    vae,
    z_target: torch.Tensor,
) -> torch.Tensor:
    """
    VAE encoder attack loss.

    Minimizing this pushes the protected image's latent representation
    far from its true content and toward the gray target.

    L_enc = MSE(E(x̂), z_target)

    Args:
        x_adv: Protected image [1, C, H, W] in [-1, 1].
        vae: Stable Diffusion VAE encoder.
        z_target: Gray image latent, precomputed via get_gray_target().

    Returns:
        Scalar loss.
    """
    z_adv = vae.encode(x_adv).latent_dist.mean
    return F.mse_loss(z_adv, z_target)


# ──────────────────────────────────────────────────────────────────────────────
# Denoising attack (stronger loss — targets full diffusion pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def denoising_loss(
    x_adv: torch.Tensor,
    vae,
    unet,
    noise_scheduler,
    text_embeddings: torch.Tensor,
    timesteps: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    AdvDM-style denoising loss: maximize UNet denoising error.

    Instead of just corrupting the encoder, this disrupts the entire
    denoising pipeline — the UNet will predict the wrong noise, causing
    the reverse diffusion process to diverge.

    L_denoise = -E_t[MSE(ε_θ(z_t(x̂), t, c), ε)]

    Maximizing this (minimizing negative) causes maximum denoising error,
    making inpainting/img2img completely fail.

    Args:
        x_adv: Protected image [1, C, H, W] in [-1, 1].
        vae: SD VAE.
        unet: SD UNet.
        noise_scheduler: DDPM/DDIM scheduler.
        text_embeddings: Text conditioning [1, seq_len, dim].
        timesteps: Specific timesteps to attack. Defaults to uniform sample.

    Returns:
        Scalar loss (negative → we maximize denoising error).
    """
    if timesteps is None:
        # Sample random timesteps covering early-to-mid diffusion (most impactful)
        timesteps = random.choices([100, 200, 300, 400, 500, 600, 700, 800], k=4)

    # Encode image to latent space
    with torch.no_grad():
        posterior = vae.encode(x_adv)
    z0 = posterior.latent_dist.sample() * vae.config.scaling_factor

    total_loss = torch.tensor(0.0, device=x_adv.device)
    for t in timesteps:
        t_tensor = torch.tensor([t], device=x_adv.device, dtype=torch.long)
        noise = torch.randn_like(z0)

        # Add noise at timestep t
        z_noisy = noise_scheduler.add_noise(z0, noise, t_tensor)

        # Predict noise with UNet
        noise_pred = unet(
            z_noisy,
            t_tensor,
            encoder_hidden_states=text_embeddings,
        ).sample

        # Maximize MSE between predicted and actual noise
        total_loss = total_loss - F.mse_loss(noise_pred, noise)

    return total_loss / len(timesteps)


# ──────────────────────────────────────────────────────────────────────────────
# Combined loss
# ──────────────────────────────────────────────────────────────────────────────

def combined_loss(
    x_adv: torch.Tensor,
    vae,
    z_target: torch.Tensor,
    unet=None,
    noise_scheduler=None,
    text_embeddings=None,
    encoder_weight: float = 1.0,
    denoising_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Combined adversarial loss: encoder attack + optional denoising attack.

    Using both losses simultaneously attacks the diffusion pipeline at two
    points: the input encoding and the denoising process itself. This improves
    black-box transfer to proprietary nudifier models.

    Args:
        x_adv: Protected image.
        vae: SD VAE.
        z_target: Gray target latent.
        unet: SD UNet (optional, for denoising loss).
        noise_scheduler: SD scheduler (optional).
        text_embeddings: Text conditioning (optional).
        encoder_weight: Weight for encoder loss.
        denoising_weight: Weight for denoising loss.

    Returns:
        Dict with 'total', 'encoder', and optionally 'denoising' losses.
    """
    losses = {}

    enc_loss = encoder_loss(x_adv, vae, z_target)
    losses["encoder"] = enc_loss
    total = encoder_weight * enc_loss

    if unet is not None and noise_scheduler is not None and text_embeddings is not None:
        den_loss = denoising_loss(x_adv, vae, unet, noise_scheduler, text_embeddings)
        losses["denoising"] = den_loss
        total = total + denoising_weight * den_loss

    losses["total"] = total
    return losses
