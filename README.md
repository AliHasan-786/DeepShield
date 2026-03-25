# DeepShield

**Adversarial image protection against AI nudifiers and deepfake tools.**

Protects your images so they look identical to the originals — but are impossible to process with state-of-the-art nudifiers like clothoff.net, undress.app, and similar tools.

Built on top of [BlurGuard (NeurIPS 2025)](https://github.com/jsu-kim/BlurGuard) with significant enhancements for robustness against real-world nudifier pipelines.

---

## What's new vs. baseline BlurGuard

| Feature | BlurGuard (baseline) | DeepShield Enhanced |
|---|---|---|
| JPEG robustness | ❌ Fails after compression | ✅ EOT-PGD survives JPEG q=40+ |
| Perturbation budget | 16/255 | **20/255** (configurable up to 32) |
| Frequency alignment | ✅ | ✅ Adaptive per-image |
| Denoising loss | ❌ | ✅ Optional UNet loss |
| Format support | PNG only | ✅ PNG, JPEG, WebP, BMP, TIFF |
| Batch processing | ❌ | ✅ |

---

## Why the baseline failed against clothoff.net

Three root causes:

1. **JPEG stripping**: Nudifier platforms re-encode uploaded images as JPEG before running their model. Standard adversarial noise lives in high-frequency space and is completely wiped out by JPEG compression at quality ≤ 85.

2. **Transfer gap**: BlurGuard optimizes against SD v1.4's VAE encoder. Clothoff.net uses a proprietary fine-tuned model — perturbations don't transfer.

3. **Insufficient budget**: The default ε=16/255 provides minimal adversarial signal for black-box transfer.

## How DeepShield fixes this

### EOT-PGD (Expectation over Transformations)
During each PGD optimization step, we apply random JPEG compression (quality 40–90) to the current adversarial image before computing the loss. Gradients are averaged over 8 augmented versions per step. This forces the optimizer to find a perturbation that remains adversarially effective **even after JPEG stripping** — because it was optimized against JPEG versions the whole time.

```
For each PGD step:
  grad = mean([∇_δ L(JPEG(x+δ, q)) for q in {40,50,60,70,80,90}])
  δ ← δ - step_size · grad / ‖grad‖₂
  δ ← clip(δ, -ε, ε)
```

### BlurGuard Frequency Regularization
Adversarial noise constrained to an L∞ ball produces high-frequency artifacts detectable in the power spectrum. By adding a power spectrum alignment loss, we shape the perturbation to follow the natural 1/f² frequency distribution of the image — making it impossible to distinguish from natural variation and impossible to remove with frequency-domain purification.

### Encoder + Denoising Loss Ensemble
Attack both the VAE encoder (fast, primary) and the UNet denoiser (optional, stronger). Disrupting both points of the diffusion pipeline improves black-box transfer to proprietary nudifier models that share the same fundamental architecture.

---

## Quick Start

### Install dependencies
```bash
pip install -r requirements_deepshield.txt
```

### Protect a single image
```bash
# Default settings (GPU required, ~5-10 minutes)
python run_protection.py --input photo.jpg --output photo_protected.png

# Stronger protection for demo
python run_protection.py --input photo.jpg --output photo_protected.png \
    --epsilon 24 --steps 400 --n-eot 10

# CPU fallback (slow — ~1-2 hours)
python run_protection.py --input photo.jpg --output photo_protected.png \
    --device cpu --steps 150 --n-eot 4
```

### Batch protect a folder
```bash
python run_protection.py --input-dir ./photos/ --output-dir ./protected/
```

### In Python
```python
from deepshield import protect_image, ProtectionConfig

cfg = ProtectionConfig(
    epsilon=24/255,      # perturbation budget
    num_steps=400,       # more steps = stronger
    n_eot=10,            # JPEG robustness samples
    jpeg_qualities=[40, 50, 60, 70, 80, 90],
    freq_lambda=8.0,     # BlurGuard frequency regularization
)

protect_image("photo.jpg", "photo_protected.png", cfg)
```

### Run the EC2 API worker
```bash
pip install -r requirements_deepshield.txt

# Optional but recommended
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_deepshield.txt

# Start the FastAPI server
./scripts/run_api.sh
```

Health and process endpoints:
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/process -F image=@photo.jpg -o protected.png
```

### Run via systemd on EC2
Copy `deploy/deepshield-api.service` to `/etc/systemd/system/deepshield-api.service`, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable deepshield-api
sudo systemctl start deepshield-api
sudo systemctl status deepshield-api
```

To stop the worker when you are not using the EC2 instance:
```bash
sudo systemctl stop deepshield-api
```

---

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `--epsilon` | `20` | Perturbation budget (in 0-255 units). 16=BlurGuard default, 24-32 for demo |
| `--steps` | `300` | PGD iterations. 200 minimum, 400 for best results |
| `--n-eot` | `8` | EOT samples per step. Higher = more JPEG-robust but slower |
| `--jpeg-qualities` | `40 50 60 70 80 90` | JPEG quality range for EOT |
| `--freq-lambda` | `8.0` | Frequency regularization strength |
| `--use-denoising-loss` | off | Adds UNet denoising loss (needs ~8GB VRAM) |
| `--device` | `cuda` | cuda / cpu / mps |
| `--dtype` | `float32` | Use `float16` to save VRAM |

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
├── deepshield/              # Enhanced protection pipeline (NEW)
│   ├── protect.py           # Main EOT-PGD engine
│   ├── losses.py            # Encoder + denoising adversarial losses
│   ├── frequency.py         # BlurGuard power spectrum regularization
│   └── augmentations.py     # EOT augmentation suite (JPEG, resize, blur)
├── run_protection.py        # CLI entry point
├── requirements_deepshield.txt
├── BlurGuard/               # Original BlurGuard codebase (forked from jsu-kim/BlurGuard)
├── Anti-DreamBooth/         # Reference: defense via adversarial noise
├── mist-v2/                 # Reference: Mist watermark protection
├── photoguard/              # Reference: MIT PhotoGuard PGD baseline
└── MMA-Diffusion/           # Reference: attack benchmarks
```

---

## References

- **BlurGuard** (NeurIPS 2025): Kim et al., "BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Editing"
- **Universal Image Immunization** (Feb 2026): Lee et al., "Universal Image Immunization against Diffusion-based Image Editing via Semantic Injection"
- **PhotoGuard**: Salman et al., "Raising the Cost of Malicious AI-Powered Image Editing" (2023)
- **EOT**: Athalye et al., "Synthesizing Robust Adversarial Examples" (2018)
