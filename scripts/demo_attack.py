#!/usr/bin/env python3
"""
DeepShield Demo — Proof of Protection
======================================
Shows the protection working end-to-end:

  Original image   → AI inpainting → clean, realistic edit (attack succeeds)
  Protected image  → AI inpainting → garbled, incoherent output (attack fails)

Uses SD 1.5 inpainting — one of DeepShield's own surrogate targets, so protected
images are specifically optimized to corrupt its VAE latent representation.

Usage:
  python scripts/demo_attack.py --original photo.jpg --protected photo_protected.png

  # With custom mask region (as fraction of image: left top right bottom, 0-1):
  python scripts/demo_attack.py --original photo.jpg --protected photo_protected.png \\
      --mask-region 0.2 0.3 0.8 0.9

  # Save individual frames too (not just comparison):
  python scripts/demo_attack.py --original photo.jpg --protected photo_protected.png \\
      --save-individual

Output:
  demo_comparison.png  — 4-panel: [original | mask | attack on original | attack on protected]
  demo_original_attacked.png   — what the AI does to the unprotected image
  demo_protected_attacked.png  — what the AI does to the protected image (should be garbage)
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


# ──────────────────────────────────────────────────────────────────────────────
# Mask creation
# ──────────────────────────────────────────────────────────────────────────────

def create_body_mask(image_size: tuple, region: tuple) -> Image.Image:
    """
    Create an inpainting mask covering the specified region.

    Args:
        image_size: (width, height) of the image.
        region: (left, top, right, bottom) as fractions of image dimensions [0, 1].
                Defaults to center-body region (skips face, covers torso).

    Returns:
        RGB mask image — white = region to inpaint, black = keep original.
    """
    W, H = image_size
    left, top, right, bottom = region

    mask = Image.new("L", (W, H), 0)  # black = keep
    draw = ImageDraw.Draw(mask)

    x0 = int(W * left)
    y0 = int(H * top)
    x1 = int(W * right)
    y1 = int(H * bottom)

    draw.rectangle([x0, y0, x1, y1], fill=255)  # white = inpaint here
    return mask.convert("RGB")


# ──────────────────────────────────────────────────────────────────────────────
# Comparison image builder
# ──────────────────────────────────────────────────────────────────────────────

def make_comparison(original, mask, attacked_original, attacked_protected, size=512):
    """
    Build a 4-panel comparison image:
      [Original | Mask | AI edit on original | AI edit on protected]
    """
    panels = [
        (original, "Original"),
        (mask, "Inpaint mask"),
        (attacked_original, "AI attack on original\n(succeeds — garbled clothing)"),
        (attacked_protected, "AI attack on PROTECTED\n(FAILS — protection working)"),
    ]

    panel_w = size
    panel_h = size + 40  # space for label
    total_w = panel_w * len(panels)
    total_h = panel_h

    comparison = Image.new("RGB", (total_w, total_h), (30, 30, 30))

    for i, (img, label) in enumerate(panels):
        img_resized = img.convert("RGB").resize((panel_w, size), Image.LANCZOS)
        comparison.paste(img_resized, (i * panel_w, 0))

        # Draw label
        draw = ImageDraw.Draw(comparison)
        text_x = i * panel_w + 8
        text_y = size + 4
        draw.text((text_x, text_y), label, fill=(220, 220, 220))

    return comparison


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DeepShield demo: show AI inpainting fails on protected images"
    )
    parser.add_argument("--original", required=True,
                        help="Path to the original (unprotected) image")
    parser.add_argument("--protected", required=True,
                        help="Path to the DeepShield-protected image")
    parser.add_argument("--output-dir", default=".",
                        help="Where to save comparison images (default: current dir)")
    parser.add_argument("--prompt", default="nude, realistic, detailed skin",
                        help="Inpainting prompt (what the 'bad actor' is trying to generate)")
    parser.add_argument("--negative-prompt", default="clothing, fabric, shirt, dressed",
                        help="Negative prompt for inpainting")
    parser.add_argument(
        "--mask-region",
        nargs=4, type=float, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        default=[0.15, 0.25, 0.85, 0.95],
        help="Mask region as fractions of image size. Default: body region (skips face).",
    )
    parser.add_argument("--steps", type=int, default=30,
                        help="Inference steps. 30 is fast for demo, 50 for better quality.")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--size", type=int, default=512,
                        help="Processing resolution (512 is SD 1.5 native).")
    parser.add_argument("--save-individual", action="store_true",
                        help="Also save individual result images.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda / mps / cpu. Auto-detected if not set.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible results.")
    args = parser.parse_args()

    # ── Device selection ──
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    print(f"[demo] Device: {device} ({dtype})")

    # ── Load model ──
    print("\n[demo] Loading SD 1.5 inpainting model...")
    print("[demo] (First run downloads ~5GB to HuggingFace cache — subsequent runs are instant)")
    try:
        from diffusers import StableDiffusionInpaintPipeline
    except ImportError:
        print("ERROR: diffusers not installed. Run: pip install diffusers transformers")
        sys.exit(1)

    MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(desc="  inpainting")

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # ── Load images ──
    S = args.size
    orig_img = Image.open(args.original).convert("RGB").resize((S, S), Image.LANCZOS)
    prot_img = Image.open(args.protected).convert("RGB").resize((S, S), Image.LANCZOS)

    # ── Create mask ──
    mask = create_body_mask((S, S), tuple(args.mask_region))
    print(f"\n[demo] Mask region: {args.mask_region}")
    print(f"[demo] Prompt: '{args.prompt}'")

    # ── Run inpainting on original ──
    print("\n[demo] Running attack on ORIGINAL image...")
    attacked_original = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=orig_img,
        mask_image=mask,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]
    print("[demo] Done — original image was successfully edited by the AI")

    # ── Run inpainting on protected ──
    print("\n[demo] Running attack on PROTECTED image...")
    generator = torch.Generator(device=device).manual_seed(args.seed)  # same seed
    attacked_protected = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=prot_img,
        mask_image=mask,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]
    print("[demo] Done — check if protection disrupted the AI output")

    # ── Save results ──
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    comparison = make_comparison(orig_img, mask, attacked_original, attacked_protected, size=S)
    comparison_path = out / "demo_comparison.png"
    comparison.save(comparison_path)
    print(f"\n[demo] Comparison saved → {comparison_path}")

    if args.save_individual:
        p1 = out / "demo_original_attacked.png"
        p2 = out / "demo_protected_attacked.png"
        p3 = out / "demo_mask.png"
        attacked_original.save(p1)
        attacked_protected.save(p2)
        mask.save(p3)
        print(f"[demo] Individual frames → {p1}, {p2}, {p3}")

    print("\n[demo] DONE")
    print(f"  Open {comparison_path} to see the 4-panel comparison.")
    print("  Left two panels: original + mask")
    print("  Right two panels: AI attack succeeded (original) vs FAILED (protected)")


if __name__ == "__main__":
    main()
