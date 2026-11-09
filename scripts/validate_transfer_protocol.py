#!/usr/bin/env python3
"""Validate and plan the unexecuted current-transfer protocol without models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepshield.evaluation import load_transfer_protocol


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a DeepShield transfer protocol without executing it."
    )
    parser.add_argument("protocol")
    parser.add_argument(
        "--gpu-hourly-usd",
        type=float,
        default=None,
        help="Optional live price for arithmetic only; does not authorize spend.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.gpu_hourly_usd is not None and args.gpu_hourly_usd < 0:
        parser.error("gpu hourly price must be nonnegative")

    plan = load_transfer_protocol(args.protocol).plan(args.gpu_hourly_usd)
    payload = plan.to_dict()
    payload["warning"] = (
        "Dry-run planning only. Unrun cells are not zero; blockers prohibit execution."
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
