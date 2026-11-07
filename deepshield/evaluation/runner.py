"""Paired protected/unprotected evaluation orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .adapters import AttackAdapter
from .metrics import MetricBackend
from .preprocessing import jpeg_reencode
from .schema import EvaluationManifest, EvaluationResult


def run_evaluation(
    manifest: EvaluationManifest,
    repo_root: Path,
    output_dir: Path,
    adapter: AttackAdapter,
    metrics: MetricBackend,
) -> list[EvaluationResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[EvaluationResult] = []
    for pipeline in manifest.pipelines:
        size = (pipeline.width, pipeline.height)
        for sample in manifest.samples:
            original = _open(repo_root, sample.original_path)
            mask = _open(repo_root, sample.mask_path).convert("L").resize(size)
            identity_reference = _open(repo_root, sample.identity_reference_path)
            original_upload = jpeg_reencode(original, pipeline.jpeg_quality, size)
            for variant in sample.variants:
                protected = _open(repo_root, variant.path)
                protected_upload = jpeg_reencode(protected, pipeline.jpeg_quality, size)
                for seed in manifest.seeds:
                    run_dir = output_dir / sample.id / variant.id / pipeline.id / str(seed)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    attacked_original = adapter.run(
                        pipeline, original_upload, mask, sample.prompt, seed
                    )
                    attacked_protected = adapter.run(
                        pipeline, protected_upload, mask, sample.prompt, seed
                    )
                    original_path = run_dir / "unprotected_attack.png"
                    protected_path = run_dir / "protected_attack.png"
                    attacked_original.save(original_path)
                    attacked_protected.save(protected_path)
                    scores = metrics.score(
                        identity_reference,
                        original,
                        protected,
                        attacked_protected,
                        sample.prompt,
                    )
                    attack_succeeded = (
                        scores.identity_similarity is not None
                        and scores.identity_similarity
                        >= manifest.thresholds.identity_similarity_min
                        and scores.clip_edit_fidelity
                        >= manifest.thresholds.clip_edit_fidelity_min
                    )
                    result = EvaluationResult(
                        sample_id=sample.id,
                        variant_id=variant.id,
                        method=variant.method,
                        epsilon_255=variant.epsilon_255,
                        pipeline_id=pipeline.id,
                        architecture=pipeline.architecture,
                        seed=seed,
                        jpeg_quality=pipeline.jpeg_quality,
                        identity_similarity=scores.identity_similarity,
                        clip_edit_fidelity=scores.clip_edit_fidelity,
                        protection_lpips=scores.protection_lpips,
                        attack_succeeded=attack_succeeded,
                        original_output_path=str(original_path),
                        protected_output_path=str(protected_path),
                    )
                    results.append(result)
                    _append_jsonl(output_dir / "results.jsonl", result.to_dict())
        adapter.close()
    return results


def _open(root: Path, relative_path: str) -> Image.Image:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"asset path escapes repository root: {relative_path}") from exc
    image = Image.open(path)
    image.load()
    return image


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
