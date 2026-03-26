#!/usr/bin/env python3
"""
DeepShield Smoke Test
=====================
Quick self-test to verify everything works before serving real images.
Run this on your EC2 instance after deploying.

Usage:
  python scripts/smoke_test.py                        # default: nudifier preset
  python scripts/smoke_test.py --ensemble nudifier-v2 # test with Flux + SD 3.5
  python scripts/smoke_test.py --device cpu            # CPU-only test

Exit code 0 = all good, non-zero = something broke.
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="DeepShield smoke test")
    parser.add_argument("--ensemble", type=str, default="nudifier",
                        choices=["standard", "nudifier", "nudifier-v2", "max"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])
    args = parser.parse_args()

    print("=" * 60)
    print("  DeepShield Smoke Test")
    print("=" * 60)
    print(f"  Ensemble: {args.ensemble}")
    print(f"  Device:   {args.device}")
    print(f"  Dtype:    {args.dtype}")
    print("=" * 60)

    errors = []

    # ── Step 1: Import check ──
    print("\n[1/6] Checking imports...")
    try:
        import torch
        import numpy as np
        from deepshield.protect import (
            ProtectionConfig, ENSEMBLE_PRESETS, load_vae,
            build_runtime, eot_pgd
        )
        from deepshield.losses import get_gray_target, combined_loss, encoder_loss
        from deepshield.frequency import frequency_reg_loss, adaptive_blur_alignment
        from deepshield.augmentations import apply_eot_augmentation
        print("  OK — all imports successful")
    except ImportError as e:
        print(f"  FAIL — {e}")
        errors.append(f"Import: {e}")
        print(f"\nSMOKE TEST FAILED: {len(errors)} error(s)")
        sys.exit(1)

    # ── Step 2: Load VAEs ──
    print(f"\n[2/6] Loading VAEs ({args.ensemble} preset)...")
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("  WARNING: CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    preset = ENSEMBLE_PRESETS[args.ensemble]
    all_model_ids = ["runwayml/stable-diffusion-v1-5"] + preset["models"]

    loaded_vaes = []
    for mid in all_model_ids:
        vae = load_vae(mid, device, dtype)
        if vae is not None:
            loaded_vaes.append((mid, vae))

    n_loaded = len(loaded_vaes)
    n_total = len(all_model_ids)
    if n_loaded == 0:
        errors.append("No VAEs loaded at all")
        print(f"  FAIL — 0/{n_total} VAEs loaded")
    elif n_loaded < n_total:
        print(f"  WARN — {n_loaded}/{n_total} VAEs loaded ({n_total - n_loaded} skipped)")
    else:
        print(f"  OK — {n_loaded}/{n_total} VAEs loaded")

    if device_str == "cuda":
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"  VRAM used: {mem:.1f} GB")

    # ── Step 3: Test encode + gray target ──
    print("\n[3/6] Testing encode + gray target generation...")
    test_img = torch.randn(1, 3, 64, 64, device=device, dtype=dtype).clamp(-1, 1)
    try:
        for mid, vae in loaded_vaes:
            z_target = get_gray_target(vae, test_img)
            z_enc = vae.encode(test_img).latent_dist.mean
            assert z_target.shape == z_enc.shape, f"Shape mismatch: {z_target.shape} vs {z_enc.shape}"
        print(f"  OK — all {n_loaded} VAEs encode and produce matching shapes")
    except Exception as e:
        print(f"  FAIL — {e}")
        errors.append(f"Encode: {e}")

    # ── Step 4: Test one PGD step ──
    print("\n[4/6] Testing one PGD step (forward + backward)...")
    try:
        primary_vae = loaded_vaes[0][1]
        z_target = get_gray_target(primary_vae, test_img)

        delta = torch.zeros_like(test_img, requires_grad=True)
        x_adv = (test_img + delta).clamp(-1, 1)
        loss_dict = combined_loss(x_adv, primary_vae, z_target)
        grad = torch.autograd.grad(loss_dict["total"], delta)[0]
        assert grad.shape == test_img.shape, f"Grad shape mismatch: {grad.shape}"
        assert not torch.isnan(grad).any(), "NaN in gradients"
        print(f"  OK — loss={loss_dict['total'].item():.4f}, grad norm={grad.norm().item():.4f}")
    except Exception as e:
        print(f"  FAIL — {e}")
        errors.append(f"PGD step: {e}")

    # ── Step 5: Test frequency alignment ──
    print("\n[5/6] Testing frequency alignment...")
    try:
        x_adv_test = (test_img + torch.randn_like(test_img) * 0.05).clamp(-1, 1)
        freq_loss = frequency_reg_loss(x_adv_test, test_img)
        x_aligned, sigma = adaptive_blur_alignment(x_adv_test, test_img, n_steps=5)
        assert x_aligned.shape == test_img.shape
        print(f"  OK — freq_loss={freq_loss.item():.4f}, sigma={sigma:.3f}")
    except Exception as e:
        print(f"  FAIL — {e}")
        errors.append(f"Frequency: {e}")

    # ── Step 6: Test LPIPS (optional) ──
    print("\n[6/6] Testing LPIPS perceptual loss...")
    try:
        from deepshield.losses import perceptual_quality_loss
        lpips_loss = perceptual_quality_loss(x_adv_test, test_img, device)
        print(f"  OK — LPIPS loss={lpips_loss.item():.4f}")
    except ImportError:
        print("  SKIP — lpips package not installed (run: pip install lpips)")
    except Exception as e:
        print(f"  FAIL — {e}")
        errors.append(f"LPIPS: {e}")

    # ── Summary ──
    print("\n" + "=" * 60)
    if errors:
        print(f"  SMOKE TEST FAILED: {len(errors)} error(s)")
        for i, err in enumerate(errors, 1):
            print(f"    {i}. {err}")
        print("=" * 60)
        sys.exit(1)
    else:
        print(f"  SMOKE TEST PASSED")
        print(f"  {n_loaded} VAEs loaded, all checks green.")
        if device_str == "cuda":
            mem = torch.cuda.memory_allocated() / 1024**3
            mem_total = torch.cuda.get_device_properties(0).total_mem / 1024**3
            print(f"  VRAM: {mem:.1f} / {mem_total:.1f} GB")
        print(f"  Ready to protect images!")
        print("=" * 60)
        sys.exit(0)

    # Clean up
    del loaded_vaes, test_img
    if device_str == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
