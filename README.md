# DeepShield v0.3.0

**Adversarial image protection against AI nudifiers and deepfake tools.**

Protects your images so they look identical to the originals — but are impossible to process with state-of-the-art nudifiers like clothoff.net, undress.app, and similar tools.

Built on top of [BlurGuard (NeurIPS 2025)](https://github.com/jsu-kim/BlurGuard) with significant enhancements for robustness against real-world nudifier pipelines.

---

## What's new vs. baseline BlurGuard

| Feature | BlurGuard (baseline) | DeepShield v0.3 |
|---|---|---|
| JPEG robustness | Fails after compression | EOT-PGD survives JPEG q=40+ |
| Perturbation budget | 16/255 | **24/255** recommended (configurable up to 32) |
| Frequency alignment | Yes | Adaptive per-image |
| Denoising loss | No | Optional UNet loss |
| **Multi-model ensemble** | No | **Up to 6 VAEs for black-box transfer** |
| **Perceptual quality (LPIPS)** | No | **Hides noise in textures, away from skin** |
| **Ensemble presets** | No | **standard / nudifier / max** |
| Format support | PNG only | PNG, JPEG, WebP, BMP, TIFF |
| Batch processing | No | Yes |

---

## Changelog

### v0.3.0 — Multi-model ensemble + perceptual quality

- **Multi-model ensemble attack**: Averages adversarial gradients across multiple VAE encoders simultaneously. Instead of optimizing against a single SD v1.5 model, we attack 3-6 models at once (SD 1.5, SD inpainting, SD 2.1, SDXL inpainting, community NSFW VAEs). This finds a perturbation that corrupts the shared latent space structure across all SD-family models, dramatically improving black-box transfer to unknown nudifiers like clothoff.net.
- **Ensemble presets**: Three curated configurations (`standard`, `nudifier`, `max`) targeting different VRAM budgets and protection levels.
- **LPIPS perceptual quality loss**: Optional constraint that keeps protected images visually close to originals by penalizing perceptual distortion. Naturally concentrates noise in textured regions (hair, clothing) where it's invisible and away from smooth areas (skin) where it's obvious.
- **Nudifier-specific surrogate models**: Added `stabilityai/sd-vae-ft-mse` (used by Realistic Vision and community NSFW models), `stabilityai/stable-diffusion-2-inpainting`, and `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` as ensemble targets.
- **API server defaults to `nudifier` preset** with ensemble enabled out of the box.

### v0.2.0 — Initial DeepShield

- EOT-PGD for JPEG robustness
- BlurGuard frequency regularization
- Encoder + denoising loss
- FastAPI server + systemd deployment

---

## Why the baseline failed against clothoff.net

Three root causes:

1. **JPEG stripping**: Nudifier platforms re-encode uploaded images as JPEG before running their model. Standard adversarial noise lives in high-frequency space and is completely wiped out by JPEG compression at quality <= 85.

2. **Transfer gap**: BlurGuard optimizes against SD v1.4's VAE encoder. Clothoff.net uses a proprietary fine-tuned model — perturbations don't transfer.

3. **Insufficient budget**: The default epsilon=16/255 provides minimal adversarial signal for black-box transfer.

## How DeepShield fixes this

### Multi-Model Ensemble Attack (v0.3)
Instead of attacking one model, we attack multiple VAE encoders simultaneously and average their gradients. This finds a universal perturbation direction that corrupts the latent space of any SD-family model:

```
For each PGD step:
  For each EOT augmentation:
    For each VAE in ensemble (SD1.5, SD-inpaint, SD2.1, SDXL, ...):
      grad_i = w_i * nabla_delta L(VAE_i.encode(JPEG(x+delta)))
    grad = mean(sum(grad_i))
  delta <- delta - step_size * grad / ||grad||_2
  delta <- clip(delta, -epsilon, epsilon)
```

### EOT-PGD (Expectation over Transformations)
During each PGD step, we apply random JPEG compression (quality 40-90) to the current adversarial image before computing the loss. This forces the optimizer to find a perturbation that remains effective **even after JPEG stripping**.

### BlurGuard Frequency Regularization
Shapes the perturbation's power spectrum to follow the natural 1/f^2 distribution of the image — making it impossible to distinguish from natural variation and impossible to remove with frequency-domain purification.

### LPIPS Perceptual Quality Loss (v0.3)
Constrains the perturbation so it's concentrated in regions where the human visual system is least sensitive (textures, high-frequency areas like hair and clothing). Smooth areas like skin stay clean.

### Encoder + Denoising Loss Ensemble
Attacks both the VAE encoder (fast, primary) and the UNet denoiser (optional, stronger). Disrupting both points of the diffusion pipeline improves transfer.

---

## Quick Start

### Install dependencies
```bash
pip install -r requirements_deepshield.txt
```

### Protect a single image
```bash
# Recommended settings for clothoff.net (GPU required, ~8-12 min on A10G)
python run_protection.py --input photo.jpg --output photo_protected.png \
    --ensemble nudifier --epsilon 24 --steps 400 --n-eot 10 --lpips-weight 2.0

# Quick test (faster, weaker)
python run_protection.py --input photo.jpg --output photo_protected.png \
    --ensemble standard

# Maximum protection (don't care about subtle artifacts)
python run_protection.py --input photo.jpg --output photo_protected.png \
    --ensemble max --epsilon 28 --steps 400

# CPU fallback (slow — ~2+ hours)
python run_protection.py --input photo.jpg --output photo_protected.png \
    --device cpu --steps 150 --n-eot 4
```

### Batch protect a folder
```bash
python run_protection.py --input-dir ./photos/ --output-dir ./protected/ \
    --ensemble nudifier --lpips-weight 2.0
```

### In Python
```python
from deepshield import protect_image, ProtectionConfig, ENSEMBLE_PRESETS

cfg = ProtectionConfig(
    epsilon=24/255,
    num_steps=400,
    n_eot=10,
    ensemble_model_ids=ENSEMBLE_PRESETS["nudifier"]["models"],
    lpips_weight=2.0,
    freq_lambda=8.0,
)

protect_image("photo.jpg", "photo_protected.png", cfg)
```

---

## Ensemble Presets

| Preset | VAEs | VRAM | Best for |
|---|---|---|---|
| `standard` | 3 (SD1.5 + inpainting + SD2.1) | ~12GB | Quick protection, limited VRAM |
| `nudifier` | 5 (+ SD2.0-inpaint + ft-mse VAE + SDXL-inpaint) | ~16GB | **Recommended for clothoff.net** |
| `max` | 6 (+ SDXL base) | ~20GB | Maximum coverage |

### How nudifiers work and why these models break them

Malicious AI nudifiers (clothoff.net, undress.app, etc.) all follow the same pipeline:

```
User uploads photo
  → Platform re-encodes to JPEG
  → AI segments the clothing region
  → Stable Diffusion inpainting replaces clothing with generated skin
  → Output returned to user
```

The critical step is **inpainting**: the image is encoded through a VAE (Variational Autoencoder) into a latent representation, then a UNet generates new content in the masked region. DeepShield corrupts the VAE encoding so the latent representation is garbage — the inpainting model has nothing coherent to work with and produces distorted, unusable output.

The problem is we don't know *exactly* which model each nudifier uses. Research shows they are typically fine-tuned from Stable Diffusion inpainting checkpoints, often using community NSFW models like Realistic Vision as a base. By attacking multiple VAEs simultaneously and averaging their gradients, we find a perturbation that corrupts the **shared latent space structure** common to all SD-family models — so even if clothoff.net uses a model we didn't specifically target, the perturbation still transfers.

**Models in the `nudifier` preset and why each one matters:**

| Model | What it covers | Why it matters |
|---|---|---|
| `runwayml/stable-diffusion-v1-5` | SD 1.5 base VAE | Foundation model — most nudifiers are fine-tuned from this |
| `stable-diffusion-v1-5/stable-diffusion-inpainting` | SD 1.5 inpainting VAE | The exact inpainting architecture nudifiers use to replace clothing. Slightly different VAE weights from base SD 1.5 due to inpainting fine-tuning |
| `stabilityai/stable-diffusion-2-inpainting` | SD 2.x inpainting VAE | Genuinely different VAE architecture from SD 1.x — covers nudifiers built on newer SD 2.x models |
| `stabilityai/sd-vae-ft-mse` | Fine-tuned community VAE | Used by Realistic Vision and most community NSFW/photorealistic models. Many nudifiers use this VAE variant for better skin/body quality |
| `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | SDXL inpainting VAE | Completely different VAE architecture (larger latent space). Covers next-gen nudifier tools upgrading to SDXL |

**Key insight**: We don't train any models. These are all pre-trained open-source models loaded as-is from HuggingFace. We just use their VAE encoders as surrogate targets during adversarial optimization. The more diverse the surrogate set, the better the perturbation transfers to unknown black-box nudifiers.

---

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `--ensemble` | off | Preset: `standard`, `nudifier`, or `max` |
| `--ensemble-models` | — | Custom model IDs (overrides preset) |
| `--epsilon` | `20` | Perturbation budget (0-255 units). **24 recommended for clothoff** |
| `--steps` | `300` | PGD iterations. 200 minimum, **400 for ensemble** |
| `--n-eot` | `8` | EOT samples per step. **10 recommended** |
| `--jpeg-qualities` | `40 50 60 70 80 90` | JPEG quality range for EOT |
| `--lpips-weight` | `0.0` | Perceptual quality. **2.0-3.0 for demos**, 0 for max protection |
| `--freq-lambda` | `8.0` | Frequency regularization strength |
| `--use-denoising-loss` | off | Adds UNet denoising loss (needs ~8GB extra VRAM) |
| `--device` | `cuda` | cuda / cpu / mps |
| `--dtype` | `float32` | Use `float16` to save VRAM |

### Recommended configurations

**Demo (good quality + protection):**
```bash
--ensemble nudifier --epsilon 20 --lpips-weight 3.0 --steps 300
```

**Production (break clothoff.net):**
```bash
--ensemble nudifier --epsilon 24 --steps 400 --n-eot 10 --lpips-weight 2.0
```

**Maximum (don't care about image quality):**
```bash
--ensemble max --epsilon 28 --steps 400 --n-eot 10 --lpips-weight 0
```

---

## EC2 Deployment (A10G recommended)

### Hardware requirements

| Preset | Min VRAM | Recommended instance |
|---|---|---|
| `standard` | 12GB | g5.xlarge (24GB A10G) |
| `nudifier` | 16GB | g5.xlarge (24GB A10G) |
| `max` | 20GB | g5.xlarge (24GB A10G) |

### Run the API server
```bash
pip install -r requirements_deepshield.txt

# Configure (these are the recommended production defaults)
export DEEPSHIELD_ENSEMBLE_PRESET=nudifier
export DEEPSHIELD_EPSILON=24
export DEEPSHIELD_STEPS=400
export DEEPSHIELD_N_EOT=10
export DEEPSHIELD_LPIPS_WEIGHT=2.0

# Start
./scripts/run_api.sh
```

### Endpoints
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/process -F image=@photo.jpg -o protected.png
```

### Run via systemd
```bash
sudo cp deploy/deepshield-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable deepshield-api
sudo systemctl start deepshield-api
```

---

## Known nudifiers tested against
- clothoff.net
- undress.app
- undressai.tools
- nudify.online

---

## Repo structure
```
DeepShield/
├── deepshield/              # Enhanced protection pipeline
│   ├── protect.py           # EOT-PGD engine + multi-model ensemble + presets
│   ├── losses.py            # Encoder + denoising + LPIPS perceptual losses
│   ├── frequency.py         # BlurGuard power spectrum regularization
│   └── augmentations.py     # EOT augmentation suite (JPEG, resize, blur)
├── run_protection.py        # CLI entry point
├── api_server.py            # FastAPI server for EC2 deployment
├── requirements_deepshield.txt
├── deploy/                  # systemd service files
├── BlurGuard/               # Original BlurGuard codebase (NeurIPS 2025)
└── scripts/                 # Deployment helpers
```

---

## References

- **BlurGuard** (NeurIPS 2025): Kim et al., "BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Editing"
- **Universal Image Immunization** (Feb 2026): Lee et al., "Universal Image Immunization against Diffusion-based Image Editing via Semantic Injection"
- **PhotoGuard**: Salman et al., "Raising the Cost of Malicious AI-Powered Image Editing" (2023)
- **EOT**: Athalye et al., "Synthesizing Robust Adversarial Examples" (2018)
- **LPIPS**: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (2018)
