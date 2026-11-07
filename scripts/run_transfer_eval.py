#!/usr/bin/env python3
"""Run a manifest-pinned DeepShield black-box transfer evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepshield.evaluation import build_matrix_markdown, load_manifest
from deepshield.evaluation.adapters import DiffusersAttackAdapter
from deepshield.evaluation.metrics import FrozenMetricBackend
from deepshield.evaluation.runner import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output-dir", default="evaluation_outputs/pilot")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Run a calibration pilot; draft results are not publishable.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if manifest.protocol_status != "frozen" and not args.allow_draft:
        parser.error("manifest is draft; use --allow-draft only for calibration")
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    adapter = DiffusersAttackAdapter(args.device, args.dtype, args.cpu_offload)
    metrics = FrozenMetricBackend(args.device)
    results = run_evaluation(manifest, REPO_ROOT, output_dir, adapter, metrics)
    (output_dir / "MATRIX.md").write_text(
        build_matrix_markdown(results), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
