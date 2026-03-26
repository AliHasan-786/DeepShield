# DeepShield — EC2 Setup & Debugging Session Handoff

**Date:** 2026-03-26
**Branch:** `feature/ensemble-v0.3`
**Prepared for:** Any teammate picking this up cold

---

## 1. Infrastructure

| Detail | Value |
|---|---|
| AWS Account | Amy Chen — 1034-9218-9547 |
| Instance ID | i-06878cb29bd6b68e3 |
| Instance Type | g5.xlarge (NVIDIA A10G, 24GB VRAM) |
| Public IP | 34.207.170.244 |
| AMI | Deep Learning Base AMI with Single CUDA (Ubuntu 22.04) — ami-0296250c0b9cc776b |
| CUDA Version | 13.0 |
| NVIDIA Driver | 580.126.09 |
| Key Pair | deepshield-key.pem (saved to ~/Downloads/ on Ali's Mac) |

**SSH into the instance:**
```bash
ssh -i ~/Downloads/deepshield-key.pem ubuntu@34.207.170.244
```

**Cost:** ~$1/hr. Stop the instance via the EC2 console when not in use. Do not leave it running overnight.

---

## 2. Repo & Environment Setup

**What was done:**
- Cloned `https://github.com/AliHasan-786/DeepShield.git` into `~/DeepShield`
- Checked out the active development branch: `feature/ensemble-v0.3`
- Installed Python dependencies: `pip install -r requirements_deepshield.txt`

**Two environment variables are required on every login.** Both are already written to `~/.bashrc` on the instance, so they should load automatically. If they don't, set them manually:

```bash
# Fix CUDA library path (prevents cuBLAS conflict — see Bug #1 below)
export LD_LIBRARY_PATH=$(find /home/ubuntu/.local/lib/python3.10/site-packages/nvidia -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH

# HuggingFace token (required to pull gated models)
export HF_TOKEN=<token>
```

**Security note:** The HF token was shared in plaintext during this session. Regenerate it at https://huggingface.co/settings/tokens before the next run and update `~/.bashrc` on the instance.

**Use `python3`, not `python`,** on this instance. The bare `python` command is not aliased.

---

## 3. Bugs Found and Fixed

All fixes are on `feature/ensemble-v0.3`. The `main` branch does not have them.

### Bug 1 — cuBLAS library conflict
**Symptom:** Import errors or crashes referencing cuBLAS on startup.
**Cause:** CUDA 13.0 system libraries conflicted with the CUDA libraries bundled with pip-installed PyTorch.
**Fix:** Set `LD_LIBRARY_PATH` to point to the pip-managed nvidia lib dirs (see Section 2). Added permanently to `~/.bashrc`.

---

### Bug 2 — stabilityai models returning 404
**Symptom:** `OSError` or HTTP 404 when loading `stabilityai/stable-diffusion-2-inpainting`.
**Cause:** Stability AI deprecated all SD 2.x models from HuggingFace in 2026 to comply with the EU AI Act.
**Fix:** Teammate added community mirror fallbacks via the `sd2-community` org and updated error handling to catch 404s gracefully. No action needed — the fallback is automatic.

---

### Bug 3 — SDXL VAE latent size mismatch
**Symptom:** Shape mismatch error when `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` VAE was used alongside SD 1.5 VAEs.
**Cause:** The SDXL VAE produces 32x32 latents vs. SD 1.5's 64x64 (different downsample factor).
**Fix:** Teammate added `_validate_vae_compatibility()`, which runs a real test-encode forward pass to check the actual downsample factor rather than inferring it from config block counts.

---

### Bug 4 — VAE compatibility check using wrong formula
**Symptom:** The original `_validate_vae_compatibility()` accepted incompatible VAEs without error.
**Cause:** The check used `2 ** n_down` (= 16 for SD 1.5's 4 blocks) instead of testing the actual output shape.
**Fix:** Replaced the formula-based check with the real forward pass approach described in Bug 3.

---

### Bug 5 — `autograd.grad` called inside EOT loop
**Symptom:** Gradient computation failed after the first EOT iteration with a "graph freed" error.
**Cause:** `autograd.grad` was called inside the `n_eot` loop. PyTorch frees the computation graph after the first `.grad()` call, so all subsequent iterations had no graph to differentiate.
**Fix:** EOT losses are now accumulated into `total_eot_loss` across all loop iterations. `autograd.grad` is called once, outside the loop.

---

### Bug 6 — tqdm progress bar never updating
**Symptom:** Progress bar showed 0% for the entire run.
**Cause:** The loop used `for step in range(cfg.num_steps)` and updated `pbar` manually, but `pbar` was never actually iterated over.
**Fix:** Changed the loop to `for step in pbar` so tqdm advances correctly on each iteration.

---

### Bug 7 — sigma gradient detached in `adaptive_blur_alignment`
**Symptom:** Sigma optimization was silently a no-op — the blur alignment term was not learning.
**Cause:** `sigma_val.item()` converted the sigma tensor to a Python scalar, detaching it from the autograd graph.
**Fix:** Teammate patched this in the latest push to `feature/ensemble-v0.3`. Confirm you have the latest commit before running.

---

## 4. Current Status

**What works:**
All 5 VAEs in the `nudifier` ensemble preset load successfully:

| Model | Role |
|---|---|
| `runwayml/stable-diffusion-v1-5` | Primary surrogate |
| `stable-diffusion-v1-5/stable-diffusion-inpainting` | Inpainting surrogate |
| `sd2-community/stable-diffusion-2-inpainting` | SD 2.x fallback (replaces deprecated stabilityai model) |
| `stabilityai/sd-vae-ft-mse` | Fine-tuned VAE surrogate |
| `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | SDXL surrogate |

**Blocking issue — Run 1 (float32):**
CUDA OOM when running `float32` with 5 VAEs and 8 EOT samples. The A10G has 24GB VRAM and float32 exceeds it at this configuration.

**Blocking issue — Run 2 (float16):**
Dtype mismatch crash: `RuntimeError: Input type (float) and bias type (c10::Half) should be the same`. The VAEs are loaded in float16 but the input image tensor `x_adv` is still float32. In `encoder_loss` (losses.py line 67), `x_adv` must be cast to match the VAE's dtype before calling `vae.encode()`.

**Fix needed in `losses.py`:**
In `encoder_loss`, change:
```python
z_adv = vae.encode(x_adv).latent_dist.mean
```
to:
```python
z_adv = vae.encode(x_adv.to(next(vae.parameters()).dtype)).latent_dist.mean
```
Same fix may be needed in `denoising_loss` if that path is ever used.

---

## 5. Next Steps

**Immediate — unblock the OOM:**

Run with `--dtype float16` to halve VRAM usage:

```bash
export LD_LIBRARY_PATH=$(find /home/ubuntu/.local/lib/python3.10/site-packages/nvidia -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH
export HF_TOKEN=<token>
cd ~/DeepShield
python3 run_protection.py \
  --input test1_original.png \
  --output test1_protected_v3.png \
  --ensemble nudifier \
  --dtype float16
```

**Download result to Mac:**

```bash
scp -i ~/Downloads/deepshield-key.pem ubuntu@34.207.170.244:~/DeepShield/test1_protected_v3.png ~/Downloads/
```

**After a successful run:**
- Test the output against clothoff.net and undress.app to evaluate transfer
- If protection holds, run on the full test set (test2, Bo1, Bo2 variants)
- If OOM persists even in float16, reduce `--n_eot` from 8 to 4 as a fallback

**Security:**
- Regenerate the HF token and update `~/.bashrc` on the instance
- Stop the EC2 instance when not actively running experiments

---

## 6. Branch Notes

| Branch | Description |
|---|---|
| `feature/ensemble-v0.3` | Active development. All 7 bug fixes applied. Use this. |
| `main` | Single-VAE pipeline. Weaker protection. Do not use for testing. |

The eventual goal is to merge `feature/ensemble-v0.3` into `main` once protection quality is validated against the target deepfake tools.
