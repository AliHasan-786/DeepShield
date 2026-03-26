#!/usr/bin/env python3
"""
DeepShield CLI — Protect images against AI nudifiers
=====================================================

Usage examples:
  # Quick protection (default settings, GPU required):
  python run_protection.py --input test1_original.png --output test1_protected_v2.png

  # Stronger protection for demo (higher epsilon, more steps):
  python run_protection.py --input photo.jpg --output photo_protected.png \
      --epsilon 24 --steps 400 --n-eot 10

  # CPU fallback (slow but works without GPU):
  python run_protection.py --input photo.jpg --output photo_protected.png \
      --device cpu --steps 150 --n-eot 4

  # Batch protect a folder:
  python run_protection.py --input-dir ./photos/ --output-dir ./protected/

  # With denoising loss for stronger protection (loads full UNet, needs ~8GB VRAM):
  python run_protection.py --input photo.jpg --output photo_protected.png \
      --use-denoising-loss --denoising-weight 0.4
"""

import argparse
import os
import sys
from pathlib import Path

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepShield: Protect images against AI nudifiers and deepfake tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input/Output ──
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i", type=str,
        help="Path to a single input image (any format: PNG, JPEG, WebP, etc.)",
    )
    input_group.add_argument(
        "--input-dir", type=str,
        help="Directory of images to protect (batch mode)",
    )

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output path for protected image (PNG extension enforced). "
             "Default: <input_name>_deepshield.png",
    )
    output_group.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for batch mode",
    )

    # ── Core protection parameters ──
    perf = parser.add_argument_group("Protection parameters")
    perf.add_argument(
        "--epsilon", type=float, default=20.0,
        help="Perturbation budget in [0,255] units. "
             "Default: 20 (stronger than BlurGuard's 16, better vs clothoff). "
             "Max recommended: 32 (slightly visible but very robust).",
    )
    perf.add_argument(
        "--steps", type=int, default=300,
        help="Number of PGD iterations. More = stronger. Default: 300.",
    )
    perf.add_argument(
        "--step-size", type=float, default=None,
        help="PGD step size in [0,255] units. Default: epsilon/100.",
    )
    perf.add_argument(
        "--n-eot", type=int, default=8,
        help="EOT augmentation samples per step (JPEG robustness). Default: 8.",
    )
    perf.add_argument(
        "--jpeg-qualities", type=int, nargs="+", default=[40, 50, 60, 70, 80, 90],
        help="JPEG quality values for EOT. Include low values for aggressive robustness. "
             "Default: 40 50 60 70 80 90.",
    )

    # ── Loss configuration ──
    loss_grp = parser.add_argument_group("Loss configuration")
    loss_grp.add_argument(
        "--freq-lambda", type=float, default=8.0,
        help="BlurGuard frequency regularization weight. Higher = more natural-looking "
             "perturbation but potentially weaker adversarial effect. Default: 8.0.",
    )
    loss_grp.add_argument(
        "--use-denoising-loss", action="store_true",
        help="Add UNet denoising loss (stronger, but requires ~8GB VRAM and is slower).",
    )
    loss_grp.add_argument(
        "--denoising-weight", type=float, default=0.3,
        help="Weight for denoising loss (only if --use-denoising-loss). Default: 0.3.",
    )

    # ── Model ──
    model_grp = parser.add_argument_group("Model configuration")
    model_grp.add_argument(
        "--model-id", type=str, default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace model ID for primary surrogate diffusion model. "
             "Default: runwayml/stable-diffusion-v1-5",
    )
    model_grp.add_argument(
        "--ensemble", type=str, nargs="?", const="standard", default=None,
        choices=["standard", "nudifier", "nudifier-v2", "max"],
        help="Enable multi-model ensemble attack using a preset. "
             "standard: 3 VAEs (~12GB). "
             "nudifier: 5 VAEs, SD-family models (~16GB). "
             "nudifier-v2: 7 VAEs, adds Flux + SD 3.5 for next-gen coverage (~20GB). "
             "max: 8 VAEs, every architecture (~24GB). "
             "Default preset if just '--ensemble' with no value: standard.",
    )
    model_grp.add_argument(
        "--ensemble-models", type=str, nargs="+", default=None,
        help="Custom ensemble model IDs (overrides --ensemble preset). "
             "Example: --ensemble-models stabilityai/stable-diffusion-2-1 "
             "stabilityai/stable-diffusion-xl-base-1.0",
    )
    model_grp.add_argument(
        "--image-size", type=int, default=512,
        help="Processing resolution. Default: 512.",
    )

    # ── Image quality ──
    quality_grp = parser.add_argument_group("Image quality")
    quality_grp.add_argument(
        "--lpips-weight", type=float, default=0.0,
        help="LPIPS perceptual quality weight. Higher = better image quality "
             "but potentially weaker protection. Recommended for demos: 3.0. "
             "Requires 'pip install lpips'. Default: 0.0 (off).",
    )

    # ── Presets ──
    preset_grp = parser.add_argument_group("Quick presets (override individual params)")
    preset_grp.add_argument(
        "--preset", type=str, default=None,
        choices=["demo", "strong"],
        help="Quick preset that sets sensible defaults. "
             "demo: nudifier ensemble, ε=20, 300 steps, LPIPS=2.0 (good image quality). "
             "strong: nudifier-v2 ensemble (incl. Flux+SD3.5), ε=24, 400 steps, LPIPS=1.0 (max protection). "
             "Individual params (--epsilon, --steps, etc.) override preset values.",
    )

    # ── Misc ──
    misc = parser.add_argument_group("Misc")
    misc.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"],
        help="Compute device. Default: cuda.",
    )
    misc.add_argument(
        "--no-freq-alignment", action="store_true",
        help="Skip BlurGuard adaptive frequency alignment post-processing.",
    )
    misc.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. Default: 42.",
    )
    misc.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "float16"],
        help="Computation dtype. float16 saves VRAM on GPU. Default: float32.",
    )

    return parser.parse_args()


def build_config(args):
    """Convert parsed CLI args to ProtectionConfig."""
    import torch
    from deepshield.protect import ENSEMBLE_PRESETS, ProtectionConfig

    # ── Apply preset defaults (individual flags override) ──
    _PRESETS = {
        "demo": {
            "ensemble": "nudifier",
            "epsilon": 20.0,
            "steps": 300,
            "n_eot": 8,
            "lpips_weight": 2.0,
        },
        "strong": {
            "ensemble": "nudifier-v2",
            "epsilon": 24.0,
            "steps": 400,
            "n_eot": 10,
            "lpips_weight": 1.0,
        },
    }

    if args.preset:
        p = _PRESETS[args.preset]
        print(f"[DeepShield] Using preset '{args.preset}'")
        # Only apply preset values if the user didn't explicitly set them
        if args.epsilon == 20.0:   # parser default
            args.epsilon = p["epsilon"]
        if args.steps == 300:      # parser default
            args.steps = p["steps"]
        if args.n_eot == 8:        # parser default
            args.n_eot = p["n_eot"]
        if args.lpips_weight == 0.0:  # parser default
            args.lpips_weight = p["lpips_weight"]
        if args.ensemble is None and args.ensemble_models is None:
            args.ensemble = p["ensemble"]

    epsilon = args.epsilon / 255.0
    step_size = (args.step_size / 255.0) if args.step_size else epsilon / 100.0
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # Determine ensemble model IDs
    ensemble_model_ids = []
    if args.ensemble_models:
        ensemble_model_ids = args.ensemble_models
    elif args.ensemble:
        preset = ENSEMBLE_PRESETS.get(args.ensemble, ENSEMBLE_PRESETS["standard"])
        ensemble_model_ids = preset["models"]
        print(f"[DeepShield] Using ensemble preset '{args.ensemble}': {preset['desc']}")

    return ProtectionConfig(
        epsilon=epsilon,
        step_size=step_size,
        num_steps=args.steps,
        n_eot=args.n_eot,
        jpeg_qualities=args.jpeg_qualities,
        freq_lambda=args.freq_lambda,
        use_denoising_loss=args.use_denoising_loss,
        denoising_weight=args.denoising_weight,
        model_id=args.model_id,
        ensemble_model_ids=ensemble_model_ids,
        lpips_weight=args.lpips_weight,
        image_size=args.image_size,
        device=args.device,
        apply_freq_alignment=not args.no_freq_alignment,
        seed=args.seed,
        dtype=dtype,
    )


def get_output_path(input_path: str, output_arg: str | None) -> str:
    """Derive output path from input if not specified."""
    if output_arg:
        return output_arg
    stem = Path(input_path).stem
    parent = Path(input_path).parent
    return str(parent / f"{stem}_deepshield.png")


def main():
    args = parse_args()

    # Add the repo root to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from deepshield.protect import protect_image

    cfg = build_config(args)

    # ── Single image mode ──
    if args.input:
        if not os.path.exists(args.input):
            print(f"ERROR: Input file not found: {args.input}")
            sys.exit(1)

        output = get_output_path(args.input, args.output)
        print(f"\n{'='*60}")
        print(f"  DeepShield Protection")
        print(f"{'='*60}")
        print(f"  Input:   {args.input}")
        print(f"  Output:  {output}")
        print(f"  ε:       {args.epsilon}/255")
        print(f"  Steps:   {args.steps}")
        print(f"  EOT:     {args.n_eot} samples × {args.jpeg_qualities} JPEG qualities")
        if cfg.ensemble_model_ids:
            print(f"  Ensemble: {1 + len(cfg.ensemble_model_ids)} models")
            for eid in cfg.ensemble_model_ids:
                print(f"    + {eid}")
        print(f"{'='*60}\n")

        protect_image(args.input, output, cfg)

    # ── Batch mode ──
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"ERROR: Input directory not found: {args.input_dir}")
            sys.exit(1)

        output_dir = Path(args.output_dir) if args.output_dir else input_dir / "protected"
        output_dir.mkdir(parents=True, exist_ok=True)

        images = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not images:
            print(f"ERROR: No supported images found in {input_dir}")
            sys.exit(1)

        print(f"\n[DeepShield] Batch mode: {len(images)} images → {output_dir}")
        for i, img_path in enumerate(images, 1):
            out_path = output_dir / f"{img_path.stem}_deepshield.png"
            print(f"\n[{i}/{len(images)}] Processing: {img_path.name}")
            try:
                protect_image(str(img_path), str(out_path), cfg)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        print(f"\n[DeepShield] Batch complete. Protected images saved to: {output_dir}")


if __name__ == "__main__":
    main()
