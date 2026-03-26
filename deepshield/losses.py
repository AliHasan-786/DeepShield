"""
Adversarial Loss Functions for DeepShield
==========================================
Multiple loss strategies for disrupting AI nudifiers and deepfake tools.

Loss hierarchy (most → least impactful for nudifiers):
  1. encoder_loss: Pushes VAE latent to a gray/noise target (fast, primary)
  2. denoising_loss: Maximizes UNet denoising error (stronger, slower)
  3. lpips_quality_loss: Constrains perceptual distortion (image quality)
  4. ensemble_loss: Average of encoder + denoising across multiple models

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

def _match_module_input(x: torch.Tensor, module) -> torch.Tensor:
    """Cast input tensors to the module parameter dtype/device."""
    ref = next(module.parameters())
    if x.device != ref.device or x.dtype != ref.dtype:
        return x.to(device=ref.device, dtype=ref.dtype)
    return x

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

    L_enc = MSE(E(x̂), z_target) / latent_channels

    Normalized by the number of latent channels so that 16ch VAEs
    (Flux, SD 3.5) contribute equally to the gradient as 4ch VAEs
    (SD 1.x/2.x/XL). Without normalization, 16ch models would
    produce ~4× larger MSE and dominate the ensemble gradient.

    Args:
        x_adv: Protected image [1, C, H, W] in [-1, 1].
        vae: Stable Diffusion VAE encoder.
        z_target: Gray image latent, precomputed via get_gray_target().

    Returns:
        Scalar loss (channel-normalized).
    """
    x_adv = _match_module_input(x_adv, vae)
    z_adv = vae.encode(x_adv).latent_dist.mean
    n_channels = z_adv.shape[1]
    return F.mse_loss(z_adv, z_target) / n_channels


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
    x_adv = _match_module_input(x_adv, vae)
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


# ──────────────────────────────────────────────────────────────────────────────
# Perceptual quality loss (LPIPS)
# ──────────────────────────────────────────────────────────────────────────────

_lpips_net = None


def load_lpips(device: torch.device):
    """Lazy-load LPIPS network (VGG-based, lightweight).

    LPIPS measures perceptual distance between two images using deep features.
    It naturally weights regions by their perceptual importance — textured
    areas (hair, clothing) tolerate more perturbation than smooth areas (skin).
    This automatically concentrates adversarial noise where it's least visible.
    """
    global _lpips_net
    if _lpips_net is not None:
        return _lpips_net

    try:
        import lpips
        _lpips_net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
        for p in _lpips_net.parameters():
            p.requires_grad_(False)
        print("[DeepShield] LPIPS perceptual model loaded (VGG).")
    except ImportError:
        print("[DeepShield] WARNING: lpips package not installed. "
              "Run 'pip install lpips' for perceptual quality loss. Falling back to MSE.")
        _lpips_net = None

    return _lpips_net


def perceptual_quality_loss(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    LPIPS perceptual distance between adversarial and original image.

    Minimizing this keeps the protected image perceptually close to the
    original, concentrating perturbation in regions where the human visual
    system is least sensitive (textures, high-frequency areas).

    Falls back to pixel-space MSE if LPIPS is not installed.

    Args:
        x_adv: Protected image [1, C, H, W] in [-1, 1].
        x_orig: Original image [1, C, H, W] in [-1, 1].
        device: Torch device.

    Returns:
        Scalar perceptual distance.
    """
    net = load_lpips(device)
    if net is not None:
        return net(x_adv, x_orig).mean()
    else:
        # Fallback: simple MSE (less effective but works without lpips package)
        return F.mse_loss(x_adv, x_orig)
