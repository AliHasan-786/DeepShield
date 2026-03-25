"""
DeepShield Enhanced Protection Engine
======================================
Main protection module combining:
  1. EOT-PGD (Expectation over Transformations + Projected Gradient Descent)
     → Makes perturbations survive JPEG compression and platform preprocessing
  2. BlurGuard Frequency Regularization
     → Aligns perturbation spectrum with natural image distribution, defeating
        diffusion-based purification methods
  3. Encoder + Denoising ensemble loss
     → Attacks the diffusion pipeline at multiple points for better transfer
        to black-box nudifier models

Why each component matters:
  - nudifiers like clothoff.net re-encode uploaded images (JPEG compression
    strips naive adversarial noise) → EOT-PGD solves this
  - Standard PGD produces high-frequency noise detectable in freq domain →
    BlurGuard frequency reg solves this
  - Perturbations tuned to SD v1.4 don't transfer to proprietary models →
    ensemble + stronger epsilon helps transfer
"""

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .augmentations import apply_eot_augmentation
from .frequency import adaptive_blur_alignment, frequency_reg_loss
from .losses import combined_loss, get_gray_target


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProtectionConfig:
    """
    Hyperparameters for the DeepShield protection pipeline.

    Tuned for maximum effectiveness against 2026-era nudifiers while keeping
    visual degradation acceptable for a demo/proof-of-concept.
    """

    # --- Perturbation budget ---
    epsilon: float = 20 / 255
    """L∞ perturbation budget. 16/255 = BlurGuard default (too weak for clothoff).
    20/255 is slightly more visible but dramatically more robust. For demo: 24/255."""

    # --- PGD optimization ---
    num_steps: int = 300
    """Number of PGD iterations. More steps = stronger protection. 200 minimum, 400 ideal."""

    step_size: float = 1.5 / 255
    """PGD step size per iteration. ~epsilon/100 is a good heuristic."""

    # --- EOT augmentation ---
    n_eot: int = 8
    """Number of augmentation samples averaged per PGD step.
    Higher = more robust but slower. 8 is a good balance for demo."""

    jpeg_qualities: List[int] = field(
        default_factory=lambda: [40, 50, 60, 70, 80, 90]
    )
    """JPEG quality values to sample during EOT.
    Include low values (40-60) to harden against aggressive compression."""

    resize_prob: float = 0.4
    """Probability of random resize augmentation per EOT sample."""

    # --- Loss weights ---
    freq_lambda: float = 8.0
    """Weight for BlurGuard frequency regularization loss.
    Higher = perturbation looks more natural but may be weaker adversarially."""

    use_denoising_loss: bool = False
    """Whether to add UNet denoising loss. Stronger but requires loading full UNet."""

    denoising_weight: float = 0.3
    """Weight for denoising loss relative to encoder loss."""

    # --- Model ---
    model_id: str = "runwayml/stable-diffusion-v1-5"
    """Surrogate Stable Diffusion model. The VAE architecture is nearly identical
    across SD v1.x-v2.x models, so perturbations transfer well."""

    image_size: int = 512
    """Processing resolution. Images are resized here, then scaled back."""

    # --- Frequency alignment post-processing ---
    apply_freq_alignment: bool = True
    """Whether to run BlurGuard adaptive blur alignment as a post-processing step."""

    freq_align_steps: int = 60
    """Gradient steps for sigma optimization in frequency alignment."""

    # --- Misc ---
    device: str = "cuda"
    seed: int = 42
    dtype: torch.dtype = torch.float32


# ──────────────────────────────────────────────────────────────────────────────
# Image utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_image_as_tensor(
    path: str,
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Load any image format (PNG, JPEG, WebP, BMP, etc.) as a [-1, 1] tensor.

    Returns:
        (tensor [1, C, H, W], original_size (W, H))
    """
    img = Image.open(path).convert("RGB")
    orig_size = img.size  # (W, H)
    img_resized = img.resize((size, size), Image.LANCZOS)
    tensor = transforms.ToTensor()(img_resized)  # [C, H, W] in [0, 1]
    tensor = tensor * 2.0 - 1.0  # → [-1, 1]
    return tensor.unsqueeze(0).to(device=device, dtype=dtype), orig_size


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Convert a [1, C, H, W] tensor in [-1, 1] to a PIL RGB image."""
    x_np = ((x[0].detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0)
    return transforms.ToPILImage()(x_np)


def save_protected_image(
    protected_tensor: torch.Tensor,
    orig_size: Tuple[int, int],
    output_path: str,
    also_save_jpeg: bool = True,
) -> str:
    """
    Save the protected image as PNG (lossless).

    CRITICAL: Always save the protected image as PNG. Saving as JPEG would
    strip the adversarial perturbation immediately — the whole point is to
    distribute the PNG and have the protection survive if/when a platform
    re-encodes it to JPEG.

    Also saves a JPEG copy for robustness verification.
    """
    pil_img = tensor_to_pil(protected_tensor)

    # Scale back to original dimensions
    W_orig, H_orig = orig_size
    if pil_img.size != (W_orig, H_orig):
        pil_img = pil_img.resize((W_orig, H_orig), Image.LANCZOS)

    # Force PNG output
    if not output_path.lower().endswith(".png"):
        output_path = os.path.splitext(output_path)[0] + ".png"

    pil_img.save(output_path, format="PNG")
    print(f"[DeepShield] Protected PNG saved → {output_path}")

    if also_save_jpeg:
        jpeg_path = os.path.splitext(output_path)[0] + "_jpeg_robustness_test.jpg"
        pil_img.save(jpeg_path, format="JPEG", quality=85)
        print(f"[DeepShield] JPEG robustness test copy → {jpeg_path}")
        print(f"[DeepShield] TIP: Upload both PNG and JPEG versions to clothoff.net to verify robustness.")

    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_vae(model_id: str, device: torch.device, dtype: torch.dtype):
    """Load just the VAE from a Stable Diffusion model checkpoint."""
    from diffusers import AutoencoderKL
    print(f"[DeepShield] Loading VAE from '{model_id}'...")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print("[DeepShield] VAE loaded and frozen.")
    return vae


def load_unet_and_scheduler(model_id: str, device: torch.device, dtype: torch.dtype):
    """Load UNet + scheduler for denoising loss (optional, heavier)."""
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    print(f"[DeepShield] Loading UNet + scheduler from '{model_id}'...")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)
    unet = unet.to(device).eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder = text_encoder.to(device).eval()

    # Neutral text conditioning (empty prompt)
    tokens = tokenizer([""], return_tensors="pt", padding="max_length", max_length=77)
    with torch.no_grad():
        text_embeddings = text_encoder(tokens.input_ids.to(device))[0]

    print("[DeepShield] UNet + scheduler loaded.")
    return unet, scheduler, text_embeddings


def build_runtime(cfg: ProtectionConfig) -> Dict[str, Any]:
    """
    Load and retain model state for repeated protection requests.

    This is intended for long-lived processes such as an API worker so the VAE
    and optional UNet do not need to be reloaded for every image.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if str(device) == "cuda" and not torch.cuda.is_available():
        print("[DeepShield] WARNING: CUDA not available, falling back to CPU (will be slow).")
        device = torch.device("cpu")

    vae = load_vae(cfg.model_id, device, cfg.dtype)

    unet, noise_scheduler, text_embeddings = None, None, None
    if cfg.use_denoising_loss:
        unet, noise_scheduler, text_embeddings = load_unet_and_scheduler(
            cfg.model_id, device, cfg.dtype
        )

    return {
        "device": device,
        "vae": vae,
        "unet": unet,
        "noise_scheduler": noise_scheduler,
        "text_embeddings": text_embeddings,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core EOT-PGD protection loop
# ──────────────────────────────────────────────────────────────────────────────

def eot_pgd(
    x_orig: torch.Tensor,
    vae,
    z_target: torch.Tensor,
    cfg: ProtectionConfig,
    unet=None,
    noise_scheduler=None,
    text_embeddings=None,
) -> torch.Tensor:
    """
    EOT-augmented Projected Gradient Descent.

    For each PGD step:
      1. Sample n_eot random augmentations (JPEG + optional resize/blur)
      2. Compute adversarial loss on each augmented version
      3. Average gradients → direction robust to all augmentations
      4. Take gradient step + project to L∞ epsilon ball

    The resulting perturbation is effective even AFTER JPEG compression,
    because the optimization explicitly averaged over JPEG versions.

    Args:
        x_orig: Original image [1, C, H, W] in [-1, 1].
        vae: Frozen SD VAE.
        z_target: Gray image VAE latent (attack target).
        cfg: ProtectionConfig.
        unet: Optional UNet for denoising loss.
        noise_scheduler: Optional scheduler.
        text_embeddings: Optional text embeddings.

    Returns:
        delta: Optimized perturbation [1, C, H, W].
    """
    epsilon = cfg.epsilon
    step_size = cfg.step_size

    # Initialize delta with small random noise within epsilon ball
    delta = (torch.rand_like(x_orig) * 2.0 - 1.0) * epsilon * 0.1
    delta = delta.to(x_orig.device)

    pbar = tqdm(range(cfg.num_steps), desc="EOT-PGD")

    for step in range(cfg.num_steps):
        delta.requires_grad_(True)
        x_adv = (x_orig + delta).clamp(-1.0, 1.0)

        # ── EOT: accumulate gradients over multiple augmentations ──
        grad_accum = torch.zeros_like(delta)
        loss_log = {"total": 0.0, "encoder": 0.0}

        for _ in range(cfg.n_eot):
            # Apply random preprocessing augmentations (with STE for grad flow)
            x_aug = apply_eot_augmentation(
                x_adv,
                jpeg_qualities=cfg.jpeg_qualities,
                resize_prob=cfg.resize_prob,
            )

            # Adversarial loss
            loss_dict = combined_loss(
                x_aug,
                vae,
                z_target,
                unet=unet,
                noise_scheduler=noise_scheduler,
                text_embeddings=text_embeddings,
                encoder_weight=1.0,
                denoising_weight=cfg.denoising_weight if cfg.use_denoising_loss else 0.0,
            )
            adv_loss = loss_dict["total"]

            # Frequency regularization (BlurGuard) — computed on unaugmented x_adv
            # (we want the *protected PNG* to be freq-aligned, not the JPEG version)
            freq_loss = frequency_reg_loss(x_adv, x_orig)
            total_loss = adv_loss + cfg.freq_lambda * freq_loss

            # Accumulate gradients
            grad = torch.autograd.grad(total_loss, delta)[0]
            grad_accum = grad_accum + grad.detach()

            loss_log["total"] += total_loss.item()
            loss_log["encoder"] += loss_dict["encoder"].item()

        grad_accum = grad_accum / cfg.n_eot
        loss_avg = {k: v / cfg.n_eot for k, v in loss_log.items()}

        # ── PGD step with L2 gradient normalization (BlurGuard style) ──
        grad_norm = grad_accum.norm(2) + 1e-8
        delta = delta.detach() - step_size * (grad_accum / grad_norm)

        # ── Project to L∞ epsilon ball ──
        delta = delta.clamp(-epsilon, epsilon)

        # ── Ensure valid image range ──
        delta = (x_orig + delta).clamp(-1.0, 1.0) - x_orig

        pbar.set_description(
            f"Loss={loss_avg['total']:.4f} "
            f"| Enc={loss_avg['encoder']:.4f} "
            f"| δ_max={delta.abs().max().item()*255:.1f}/255"
        )

    return delta.detach()


# ──────────────────────────────────────────────────────────────────────────────
# Main public API
# ──────────────────────────────────────────────────────────────────────────────

def protect_image(
    input_path: str,
    output_path: str,
    cfg: Optional[ProtectionConfig] = None,
    runtime: Optional[Dict[str, Any]] = None,
    also_save_jpeg: bool = True,
) -> str:
    """
    Protect an image against AI nudifiers using DeepShield.

    Full pipeline:
      1. Load image (any format: PNG, JPEG, WebP, BMP, ...)
      2. Run EOT-PGD to generate a robust adversarial perturbation
      3. Apply BlurGuard adaptive frequency alignment (optional post-processing)
      4. Save as PNG (lossless — the JPEG robustness comes from EOT, not the format)

    Args:
        input_path: Path to input image (any format).
        output_path: Where to save the protected image (PNG extension enforced).
        cfg: ProtectionConfig. Uses sane defaults if None.

    Returns:
        Path to saved protected image.
    """
    if cfg is None:
        cfg = ProtectionConfig()

    # Reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = runtime["device"] if runtime else torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if not runtime and str(device) == "cuda" and not torch.cuda.is_available():
        print("[DeepShield] WARNING: CUDA not available, falling back to CPU (will be slow).")
        device = torch.device("cpu")

    print(f"[DeepShield] Using device: {device}")
    print(f"[DeepShield] Config: ε={cfg.epsilon*255:.1f}/255, steps={cfg.num_steps}, "
          f"n_eot={cfg.n_eot}, JPEG={cfg.jpeg_qualities}")

    # ── Load image ──
    x_orig, orig_size = load_image_as_tensor(input_path, cfg.image_size, device, cfg.dtype)
    print(f"[DeepShield] Loaded '{input_path}' ({orig_size[0]}×{orig_size[1]}) → resized to {cfg.image_size}×{cfg.image_size}")

    # ── Load models ──
    vae = runtime["vae"] if runtime else load_vae(cfg.model_id, device, cfg.dtype)
    z_target = get_gray_target(vae, x_orig)

    if runtime:
        unet = runtime["unet"]
        noise_scheduler = runtime["noise_scheduler"]
        text_embeddings = runtime["text_embeddings"]
    else:
        unet, noise_scheduler, text_embeddings = None, None, None
        if cfg.use_denoising_loss:
            unet, noise_scheduler, text_embeddings = load_unet_and_scheduler(
                cfg.model_id, device, cfg.dtype
            )

    # ── EOT-PGD ──
    print(f"\n[DeepShield] Starting EOT-PGD ({cfg.num_steps} steps × {cfg.n_eot} augmentations)...")
    delta = eot_pgd(
        x_orig, vae, z_target, cfg,
        unet=unet,
        noise_scheduler=noise_scheduler,
        text_embeddings=text_embeddings,
    )

    x_protected = (x_orig + delta).clamp(-1.0, 1.0)

    # ── BlurGuard adaptive frequency alignment ──
    if cfg.apply_freq_alignment:
        print("\n[DeepShield] Applying BlurGuard adaptive frequency alignment...")
        x_protected, sigma_used = adaptive_blur_alignment(
            x_protected,
            x_orig,
            n_steps=cfg.freq_align_steps,
        )
        print(f"[DeepShield] Optimal blur sigma: {sigma_used:.3f}")

    # ── Compute and report final perturbation stats ──
    final_delta = x_protected - x_orig
    l_inf = final_delta.abs().max().item() * 255
    l2 = final_delta.norm(2).item()
    print(f"\n[DeepShield] Final perturbation: L∞={l_inf:.2f}/255, L2={l2:.2f}")

    # ── Save ──
    out_path = save_protected_image(x_protected, orig_size, output_path, also_save_jpeg=also_save_jpeg)

    # Clean up request-scoped memory.
    del z_target, delta
    if not runtime:
        del vae
        if unet is not None:
            del unet, text_embeddings
        torch.cuda.empty_cache()

    return out_path
