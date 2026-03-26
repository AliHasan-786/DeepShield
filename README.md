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
| **Multi-model ensemble** | No | **Up to 8 VAEs (SD 1.x, 2.x, SDXL, Flux, SD 3.5)** |
| **Perceptual quality (LPIPS)** | No | **Hides noise in textures, away from skin** |
| **Ensemble presets** | No | **standard / nudifier / nudifier-v2 / max** |
| Format support | PNG only | PNG, JPEG, WebP, BMP, TIFF |
| Batch processing | No | Yes |

---

## Changelog

### v0.3.0 — Multi-model ensemble + perceptual quality

- **Multi-model ensemble attack**: Averages adversarial gradients across up to 8 VAE encoders simultaneously — SD 1.5, SD inpainting, SD 2.x, SDXL, Flux, and SD 3.5. Covers both current-gen (SD-based) and next-gen (Flux/SD3.5-based) nudifier architectures.
- **Four ensemble presets**: `standard` (3 VAEs), `nudifier` (5 VAEs), `nudifier-v2` (7 VAEs, recommended), `max` (8 VAEs) targeting different VRAM budgets.
- **Next-gen model coverage**: Added Flux (FLUX.1-schnell, Apache 2.0) and SD 3.5 Large VAEs — the architectures nudifiers are actively migrating to in 2025-2026.
- **LPIPS perceptual quality loss**: Optional constraint that concentrates noise in textured regions (hair, clothing) and away from smooth areas (skin) for better visual quality.
- **Nudifier-specific surrogate models**: Added `stabilityai/sd-vae-ft-mse` (Realistic Vision / community NSFW VAE), `stabilityai/stable-diffusion-2-inpainting`, and `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`.
- **Graceful gated model handling**: If a HuggingFace-gated model (SD 3.5) can't load, it's skipped with a warning instead of crashing.
- **All models commercially licensable**: Flux-schnell (Apache 2.0), SD 3.5/SDXL (Stability Community License, free under 1M revenue), SD 1.x/2.x (CreativeML Open RAIL-M).
- **API server defaults to `nudifier` preset** with ensemble enabled out of the box. Set `DEEPSHIELD_ENSEMBLE_PRESET=nudifier-v2` for next-gen coverage.

### v0.3.1 — Stability & 16ch VAE support

- **Flux + SD 3.5 now fully working**: Fixed VAE validation to accept 16-channel latent VAEs (Flux, SD 3.5) alongside standard 4-channel VAEs (SD 1.x/2.x/XL).
- **Channel-normalized encoder loss**: MSE loss is normalized by latent channel count so 16ch and 4ch models contribute equally to the gradient.
- **Quick presets**: `--preset demo` and `--preset strong` for one-flag configuration.
- **OOM resilience**: PGD loop catches CUDA OOM and returns partial protection instead of crashing. Periodic VRAM cleanup every 50 steps.
- **Smoke test**: `scripts/smoke_test.py` for verifying deployment before serving.
- **Fixed sigma optimization**: Replaced broken gradient-based blur alignment with robust grid search.
- **Fixed EOT loop**: Single autograd.grad call instead of per-augmentation (faster, less VRAM).
- **API dtype support**: `DEEPSHIELD_DTYPE=float16` env var to halve VRAM usage.

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

### Run smoke test (do this first on EC2!)
```bash
python scripts/smoke_test.py --ensemble nudifier-v2
```
Verifies all VAEs load, encodes work, gradients flow, and LPIPS loads. Takes <30 seconds on GPU.

### Protect a single image
```bash
# Using presets (simplest):
python run_protection.py --input photo.jpg --output photo_protected.png --preset demo
python run_protection.py --input photo.jpg --output photo_protected.png --preset strong

# Preset details:
#   demo:   nudifier ensemble, ε=20, 300 steps, LPIPS=2.0 (good image quality)
#   strong: nudifier-v2 ensemble (incl. Flux+SD3.5), ε=24, 400 steps, LPIPS=1.0

# Or manual configuration:
python run_protection.py --input photo.jpg --output photo_protected.png \
    --ensemble nudifier-v2 --epsilon 24 --steps 400 --n-eot 10 --lpips-weight 2.0

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
    ensemble_model_ids=ENSEMBLE_PRESETS["nudifier-v2"]["models"],
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
| `nudifier` | 5 (+ SD2.0-inpaint + ft-mse VAE + SDXL-inpaint) | ~16GB | Current-gen SD-based nudifiers |
| `nudifier-v2` | 7 (+ Flux VAE + SD 3.5 VAE) | ~20GB | **Recommended — covers next-gen nudifiers** |
| `max` | 8 (+ SDXL base) | ~24GB | Every architecture, fits on A10G |

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

**Models in the `nudifier-v2` preset and why each one matters:**

| Model | Architecture | Latent | Why it matters |
|---|---|---|---|
| `runwayml/stable-diffusion-v1-5` | SD 1.5 | 4ch | Foundation model — most current nudifiers are fine-tuned from this |
| `stable-diffusion-v1-5/stable-diffusion-inpainting` | SD 1.5 inpaint | 4ch | The exact inpainting architecture nudifiers use to replace clothing |
| `stabilityai/stable-diffusion-2-inpainting` | SD 2.x inpaint | 4ch | Different VAE weights from SD 1.x — covers SD 2.x-based nudifiers |
| `stabilityai/sd-vae-ft-mse` | Community VAE | 4ch | Used by Realistic Vision + community NSFW models for better skin quality |
| `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | SDXL inpaint | 4ch | Larger architecture — covers SDXL-based nudifier tools |
| `black-forest-labs/FLUX.1-schnell` | **Flux** (2024) | **16ch** | **Next-gen architecture** from ex-Stability AI team. Same VAE as FLUX.1-dev but Apache 2.0 licensed (commercial OK). CivitAI already has Flux-based NSFW models ("Fluxed Up", "CHROMA"). Nudifiers are migrating here. |
| `stabilityai/stable-diffusion-3.5-large` | **SD 3.5 MMDiT** (2025) | **16ch** | **Latest Stability AI model.** Different VAE from all prior SD versions. Covers the newest generation of tools. |

**Key insight**: We don't train any models. These are all pre-trained open-source models loaded as-is from HuggingFace. We just use their VAE encoders as surrogate targets during adversarial optimization. The more diverse the surrogate set, the better the perturbation transfers to unknown black-box nudifiers.

**Note on gated models**: SD 3.5 is gated on HuggingFace — accept the license at https://huggingface.co/stabilityai/stable-diffusion-3.5-large and set `HF_TOKEN` env var. FLUX.1-schnell is Apache 2.0 (no approval needed). If a gated model can't load, DeepShield skips it gracefully and continues with the remaining models.

**Commercial license summary**: All models in `nudifier-v2` are commercially usable. Flux-schnell is Apache 2.0. SD 3.5 and SDXL are Stability AI Community License (free under 1M monthly revenue). SD 1.x/2.x are CreativeML Open RAIL-M (commercial OK).

---

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `--ensemble` | off | Preset: `standard`, `nudifier`, `nudifier-v2`, or `max` |
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

**Production (break clothoff.net — recommended):**
```bash
--ensemble nudifier-v2 --epsilon 24 --steps 400 --n-eot 10 --lpips-weight 2.0
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
| `nudifier-v2` | 20GB | g5.xlarge (24GB A10G) |
| `max` | 24GB | g5.xlarge (24GB A10G) |

### Setup and verify
```bash
# Install dependencies
pip install -r requirements_deepshield.txt

# Set HuggingFace token (required for SD 3.5 gated model)
export HF_TOKEN=hf_your_token_here

# Run smoke test FIRST — verifies all VAEs load and gradients flow
python scripts/smoke_test.py --ensemble nudifier-v2

# Quick test with a real image
python run_protection.py --input test.jpg --output protected.png --preset strong
```

### Run the API server
```bash
# Configure (these are the recommended production defaults)
export HF_TOKEN=hf_your_token_here
export DEEPSHIELD_ENSEMBLE_PRESET=nudifier-v2
export DEEPSHIELD_EPSILON=24
export DEEPSHIELD_STEPS=400
export DEEPSHIELD_N_EOT=10
export DEEPSHIELD_LPIPS_WEIGHT=2.0

# If tight on VRAM, use float16 (~halves memory usage)
# export DEEPSHIELD_DTYPE=float16

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
├── scripts/
│   ├── run_api.sh           # API launch script
│   └── smoke_test.py        # Pre-deployment verification
├── BlurGuard/               # Original BlurGuard codebase (NeurIPS 2025)
```

---

## Troubleshooting

**CUDA out of memory**
- Try `--dtype float16` to halve VRAM usage
- Use a smaller ensemble: `--ensemble nudifier` instead of `nudifier-v2`
- Reduce `--n-eot` from 10 to 6
- Reduce `--image-size` from 512 to 384

**SD 3.5 won't load ("gated model")**
- Accept the license at https://huggingface.co/stabilityai/stable-diffusion-3.5-large
- Set `export HF_TOKEN=hf_your_token_here`
- DeepShield will skip it gracefully if it can't load — protection still works

**Models not found on HuggingFace (404)**
- Some Stability AI models have been removed (EU AI Act compliance)
- DeepShield has community mirror fallbacks for SD 2.x models
- If a model is missing, it's skipped and protection continues with remaining VAEs

**Slow on CPU**
- CPU mode works but is 50-100× slower than GPU
- Reduce `--steps 100 --n-eot 4` for reasonable CPU times (~30-60 min)

---

## References

- **BlurGuard** (NeurIPS 2025): Kim et al., "BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Editing"
- **Universal Image Immunization** (Feb 2026): Lee et al., "Universal Image Immunization against Diffusion-based Image Editing via Semantic Injection"
- **PhotoGuard**: Salman et al., "Raising the Cost of Malicious AI-Powered Image Editing" (2023)
- **EOT**: Athalye et al., "Synthesizing Robust Adversarial Examples" (2018)
- **LPIPS**: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (2018)
