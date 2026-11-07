#!/usr/bin/env python3
"""Validate and cost a DeepShield transfer-evaluation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepshield.evaluation import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an evaluation manifest without loading ML models."
    )
    parser.add_argument("manifest")
    parser.add_argument(
        "--seconds-per-run",
        type=float,
        default=120.0,
        help="Planning estimate only; replace after a measured pilot.",
    )
    parser.add_argument("--gpu-hourly-usd", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.seconds_per_run <= 0 or args.gpu_hourly_usd < 0:
        parser.error("runtime must be positive and price must be nonnegative")

    manifest = load_manifest(args.manifest)
    hours = manifest.planned_run_count * args.seconds_per_run / 3600
    summary = {
        "manifest": str(Path(args.manifest)),
        "protocol_status": manifest.protocol_status,
        "planned_runs": manifest.planned_run_count,
        "estimated_gpu_hours": round(hours, 3),
        "estimated_gpu_cost_usd": round(hours * args.gpu_hourly_usd, 2),
        "warning": (
            "Planning estimate only; do not publish results from a draft "
            "protocol. Run a measured pilot before budget approval."
        ),
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
