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
from .losses import combined_loss, get_gray_target, perceptual_quality_loss


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
    """Primary surrogate Stable Diffusion model."""

    ensemble_model_ids: List[str] = field(default_factory=list)
    """Additional surrogate models for multi-VAE ensemble attack.
    Averaging gradients across multiple VAE architectures dramatically improves
    black-box transfer to unknown nudifier models like clothoff.net.
    Use ENSEMBLE_PRESETS below for curated configurations."""

    ensemble_weights: Optional[List[float]] = None
    """Per-model weights for the ensemble. If None, all models weighted equally.
    First weight is for the primary model_id, rest for ensemble_model_ids."""

    image_size: int = 512
    """Processing resolution. Images are resized here, then scaled back."""

    # --- Perceptual quality ---
    lpips_weight: float = 0.0
    """Weight for LPIPS perceptual quality loss. Higher = less visible perturbation
    but potentially weaker adversarial effect. Recommended: 2.0-5.0 for demos.
    Set to 0.0 to disable (original behavior)."""

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
# Ensemble presets
# ──────────────────────────────────────────────────────────────────────────────

ENSEMBLE_PRESETS = {
    "standard": {
        "desc": "3 VAEs — SD 1.5 + SD inpainting + SD 2.1 (~12GB VRAM)",
        "models": [
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
            "stabilityai/stable-diffusion-2-1",
        ],
    },
    "nudifier": {
        "desc": "5 VAEs — targets the full range of SD-based nudifier architectures (~16GB VRAM)",
        "models": [
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
            "stabilityai/stable-diffusion-2-inpainting",
            "stabilityai/sd-vae-ft-mse",              # fine-tuned VAE used by Realistic Vision, community NSFW models
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",  # SDXL inpainting
        ],
    },
    "nudifier-v2": {
        "desc": "7 VAEs — adds Flux + SD 3.5 for next-gen nudifier coverage (~20GB VRAM)",
        "models": [
            "stable-diffusion-v1-5/stable-diffusion-inpainting",  # SD 1.5 inpainting (most current nudifiers)
            "stabilityai/stable-diffusion-2-inpainting",           # SD 2.x inpainting
            "stabilityai/sd-vae-ft-mse",                           # Community NSFW VAE (Realistic Vision etc.)
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",   # SDXL inpainting
            "black-forest-labs/FLUX.1-schnell",                     # Flux VAE — same VAE as FLUX.1-dev, Apache 2.0 (commercial OK)
            "stabilityai/stable-diffusion-3.5-large",              # SD 3.5 VAE — latest Stability AI architecture
        ],
    },
    "max": {
        "desc": "8 VAEs — every architecture we can target (~24GB VRAM)",
        "models": [
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
            "stabilityai/stable-diffusion-2-inpainting",
            "stabilityai/sd-vae-ft-mse",
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "black-forest-labs/FLUX.1-dev",
            "stabilityai/stable-diffusion-3.5-large",
        ],
    },
}


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
    """Load just the VAE from a Stable Diffusion / Flux / SD3 model checkpoint.

    Handles:
      - Standard SD models (VAE in 'vae' subfolder)
      - Standalone VAEs at the top level (e.g. sd-vae-ft-mse)
      - Gated models (Flux, SD 3.5) — requires HuggingFace token
      - Different latent dimensions (4ch for SD 1.x/2.x/XL, 16ch for Flux/SD3)

    For gated models, set HF_TOKEN env var or run `huggingface-cli login`.
    """
    import os
    from diffusers import AutoencoderKL

    hf_token = os.environ.get("HF_TOKEN", None)
    print(f"[DeepShield] Loading VAE from '{model_id}'...")

    load_kwargs = {"torch_dtype": dtype}
    if hf_token:
        load_kwargs["token"] = hf_token

    try:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", **load_kwargs)
    except (OSError, ValueError, Exception) as e:
        try:
            # Some models have VAE at top level (e.g. sd-vae-ft-mse)
            vae = AutoencoderKL.from_pretrained(model_id, **load_kwargs)
        except Exception as e2:
            # If it's a gated model auth issue, give a helpful message
            err_str = str(e) + str(e2)
            if "401" in err_str or "403" in err_str or "gated" in err_str.lower():
                print(f"[DeepShield] WARNING: '{model_id}' is a gated model. "
                      f"Accept the license at https://huggingface.co/{model_id} "
                      f"and set HF_TOKEN env var. Skipping this model.")
                return None
            raise

    latent_ch = getattr(vae.config, "latent_channels", 4)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"[DeepShield] VAE loaded and frozen ({model_id}, {latent_ch}ch latent).")
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

    When ensemble_model_ids is configured, loads multiple VAEs for the
    multi-model ensemble attack that dramatically improves black-box transfer.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if str(device) == "cuda" and not torch.cuda.is_available():
        print("[DeepShield] WARNING: CUDA not available, falling back to CPU (will be slow).")
        device = torch.device("cpu")

    # Load primary VAE
    vae = load_vae(cfg.model_id, device, cfg.dtype)

    # Load ensemble VAEs (skip any that fail to load, e.g. gated models without auth)
    ensemble_vaes = []
    for eid in cfg.ensemble_model_ids:
        ev = load_vae(eid, device, cfg.dtype)
        if ev is not None:
            ensemble_vaes.append(ev)

    if ensemble_vaes:
        n_total = 1 + len(ensemble_vaes)
        print(f"[DeepShield] Ensemble loaded: {n_total} VAEs for multi-model attack.")

    unet, noise_scheduler, text_embeddings = None, None, None
    if cfg.use_denoising_loss:
        unet, noise_scheduler, text_embeddings = load_unet_and_scheduler(
            cfg.model_id, device, cfg.dtype
        )

    return {
        "device": device,
        "vae": vae,
        "ensemble_vaes": ensemble_vaes,
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
    ensemble_vaes: Optional[List] = None,
    ensemble_z_targets: Optional[List[torch.Tensor]] = None,
    ensemble_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    EOT-augmented Projected Gradient Descent with multi-model ensemble.

    For each PGD step:
      1. Sample n_eot random augmentations (JPEG + optional resize/blur)
      2. Compute adversarial loss on each augmented version ACROSS ALL VAEs
      3. Average gradients → direction that corrupts latents for all models
      4. Take gradient step + project to L∞ epsilon ball

    The multi-model ensemble averages gradients from multiple VAE encoders
    (SD v1.5, SD inpainting, SD 2.1, optionally SDXL). This finds a
    perturbation that disrupts the shared latent space structure common to
    all SD-family models — dramatically improving black-box transfer to
    unknown nudifier models like clothoff.net.

    Args:
        x_orig: Original image [1, C, H, W] in [-1, 1].
        vae: Primary frozen SD VAE.
        z_target: Gray image VAE latent (attack target) for primary VAE.
        cfg: ProtectionConfig.
        unet: Optional UNet for denoising loss.
        noise_scheduler: Optional scheduler.
        text_embeddings: Optional text embeddings.
        ensemble_vaes: Additional VAEs for ensemble attack.
        ensemble_z_targets: Gray targets for each ensemble VAE.
        ensemble_weights: Per-model weights. If None, uniform weighting.

    Returns:
        delta: Optimized perturbation [1, C, H, W].
    """
    epsilon = cfg.epsilon
    step_size = cfg.step_size

    # Build the full list of (vae, z_target, weight) tuples
    all_vaes = [vae]
    all_targets = [z_target]
    if ensemble_vaes and ensemble_z_targets:
        all_vaes.extend(ensemble_vaes)
        all_targets.extend(ensemble_z_targets)

    n_models = len(all_vaes)

    if ensemble_weights is not None and len(ensemble_weights) == n_models:
        weights = ensemble_weights
    else:
        weights = [1.0 / n_models] * n_models

    # Normalize weights to sum to 1
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    is_ensemble = n_models > 1
    mode_str = f"EOT-PGD (ensemble: {n_models} VAEs)" if is_ensemble else "EOT-PGD"

    # Initialize delta with small random noise within epsilon ball
    delta = (torch.rand_like(x_orig) * 2.0 - 1.0) * epsilon * 0.1
    delta = delta.to(x_orig.device)

    pbar = tqdm(range(cfg.num_steps), desc=mode_str)

    for step in range(cfg.num_steps):
        delta.requires_grad_(True)
        x_adv = (x_orig + delta).clamp(-1.0, 1.0)

        # ── EOT: accumulate gradients over augmentations × models ──
        grad_accum = torch.zeros_like(delta)
        loss_log = {"total": 0.0, "encoder": 0.0}

        for _ in range(cfg.n_eot):
            # Apply random preprocessing augmentations (with STE for grad flow)
            x_aug = apply_eot_augmentation(
                x_adv,
                jpeg_qualities=cfg.jpeg_qualities,
                resize_prob=cfg.resize_prob,
            )

            # ── Compute loss across all VAEs in ensemble ──
            eot_loss = torch.tensor(0.0, device=x_orig.device)
            enc_loss_log = 0.0

            for model_idx, (cur_vae, cur_target, w) in enumerate(
                zip(all_vaes, all_targets, weights)
            ):
                # For ensemble VAEs with different latent sizes (e.g. SDXL),
                # we may need to resize the input
                cur_input = x_aug
                if hasattr(cur_vae.config, 'sample_size') and cur_vae.config.sample_size != cfg.image_size:
                    target_size = cur_vae.config.sample_size or cfg.image_size
                    if target_size != x_aug.shape[-1]:
                        cur_input = F.interpolate(
                            x_aug, size=(target_size, target_size),
                            mode="bilinear", align_corners=False,
                        )

                loss_dict = combined_loss(
                    cur_input,
                    cur_vae,
                    cur_target,
                    # Only use UNet denoising on primary model
                    unet=unet if model_idx == 0 else None,
                    noise_scheduler=noise_scheduler if model_idx == 0 else None,
                    text_embeddings=text_embeddings if model_idx == 0 else None,
                    encoder_weight=1.0,
                    denoising_weight=cfg.denoising_weight if (cfg.use_denoising_loss and model_idx == 0) else 0.0,
                )

                eot_loss = eot_loss + w * loss_dict["total"]
                enc_loss_log += w * loss_dict["encoder"].item()

            # Frequency regularization (BlurGuard) — computed on unaugmented x_adv
            freq_loss = frequency_reg_loss(x_adv, x_orig)
            total_loss = eot_loss + cfg.freq_lambda * freq_loss

            # Perceptual quality constraint (LPIPS) — keeps protected image
            # visually close to original by penalizing perceptual distortion.
            # Naturally concentrates noise in textured regions (hair, fabric)
            # where it's less visible to the human eye.
            if cfg.lpips_weight > 0:
                lpips_loss = perceptual_quality_loss(x_adv, x_orig, x_orig.device)
                total_loss = total_loss + cfg.lpips_weight * lpips_loss

            # Accumulate gradients
            grad = torch.autograd.grad(total_loss, delta)[0]
            grad_accum = grad_accum + grad.detach()

            loss_log["total"] += total_loss.item()
            loss_log["encoder"] += enc_loss_log

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
    ensemble_info = ""
    if cfg.ensemble_model_ids:
        ensemble_info = f", ensemble={1 + len(cfg.ensemble_model_ids)} models"
    print(f"[DeepShield] Config: ε={cfg.epsilon*255:.1f}/255, steps={cfg.num_steps}, "
          f"n_eot={cfg.n_eot}, JPEG={cfg.jpeg_qualities}{ensemble_info}")

    # ── Load image ──
    x_orig, orig_size = load_image_as_tensor(input_path, cfg.image_size, device, cfg.dtype)
    print(f"[DeepShield] Loaded '{input_path}' ({orig_size[0]}×{orig_size[1]}) → resized to {cfg.image_size}×{cfg.image_size}")

    # ── Load models ──
    vae = runtime["vae"] if runtime else load_vae(cfg.model_id, device, cfg.dtype)
    z_target = get_gray_target(vae, x_orig)

    # Load ensemble VAEs and their gray targets
    if runtime:
        ensemble_vaes = runtime.get("ensemble_vaes", [])
        unet = runtime["unet"]
        noise_scheduler = runtime["noise_scheduler"]
        text_embeddings = runtime["text_embeddings"]
    else:
        ensemble_vaes = []
        for eid in cfg.ensemble_model_ids:
            ev = load_vae(eid, device, cfg.dtype)
            if ev is not None:
                ensemble_vaes.append(ev)
        unet, noise_scheduler, text_embeddings = None, None, None
        if cfg.use_denoising_loss:
            unet, noise_scheduler, text_embeddings = load_unet_and_scheduler(
                cfg.model_id, device, cfg.dtype
            )

    # Compute gray targets for each ensemble VAE
    ensemble_z_targets = []
    for ev in ensemble_vaes:
        ensemble_z_targets.append(get_gray_target(ev, x_orig))

    n_models = 1 + len(ensemble_vaes)
    if n_models > 1:
        print(f"[DeepShield] Ensemble attack: {n_models} VAEs "
              f"({cfg.model_id} + {cfg.ensemble_model_ids})")

    # ── EOT-PGD ──
    print(f"\n[DeepShield] Starting EOT-PGD ({cfg.num_steps} steps × {cfg.n_eot} augmentations × {n_models} models)...")
    delta = eot_pgd(
        x_orig, vae, z_target, cfg,
        unet=unet,
        noise_scheduler=noise_scheduler,
        text_embeddings=text_embeddings,
        ensemble_vaes=ensemble_vaes,
        ensemble_z_targets=ensemble_z_targets,
        ensemble_weights=cfg.ensemble_weights,
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
    del z_target, delta, ensemble_z_targets
    if not runtime:
        del vae
        for ev in ensemble_vaes:
            del ev
        if unet is not None:
            del unet, text_embeddings
        torch.cuda.empty_cache()

    return out_path
