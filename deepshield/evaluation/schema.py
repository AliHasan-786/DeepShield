"""Strict, dependency-light schemas for transfer experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class ProtectionVariant:
    id: str
    path: str
    method: str
    epsilon_255: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProtectionVariant":
        return cls(
            id=_required_text(value, "id"),
            path=_required_text(value, "path"),
            method=_required_text(value, "method"),
            epsilon_255=_nonnegative_number(value, "epsilon_255"),
        )


@dataclass(frozen=True)
class Sample:
    id: str
    original_path: str
    mask_path: str
    prompt: str
    identity_reference_path: str
    variants: tuple[ProtectionVariant, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Sample":
        variants = tuple(
            ProtectionVariant.from_dict(item)
            for item in _required_list(value, "variants")
        )
        if not variants:
            raise ValueError("sample variants must not be empty")
        return cls(
            id=_required_text(value, "id"),
            original_path=_required_text(value, "original_path"),
            mask_path=_required_text(value, "mask_path"),
            prompt=_required_text(value, "prompt"),
            identity_reference_path=_required_text(
                value, "identity_reference_path"
            ),
            variants=variants,
        )


@dataclass(frozen=True)
class AttackPipeline:
    id: str
    architecture: str
    task: str
    model_id: str
    jpeg_quality: int
    steps: int
    guidance_scale: float
    width: int
    height: int
    strength: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AttackPipeline":
        task = _required_text(value, "task")
        if task not in {"inpainting", "img2img"}:
            raise ValueError("pipeline task must be 'inpainting' or 'img2img'")
        jpeg_quality = _integer(value, "jpeg_quality")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        return cls(
            id=_required_text(value, "id"),
            architecture=_required_text(value, "architecture"),
            task=task,
            model_id=_required_text(value, "model_id"),
            jpeg_quality=jpeg_quality,
            steps=_positive_integer(value, "steps"),
            guidance_scale=_nonnegative_number(value, "guidance_scale"),
            width=_positive_integer(value, "width"),
            height=_positive_integer(value, "height"),
            strength=_unit_number(value, "strength"),
        )


@dataclass(frozen=True)
class Thresholds:
    identity_similarity_min: float
    clip_edit_fidelity_min: float
    source: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Thresholds":
        identity = _unit_number(value, "identity_similarity_min")
        fidelity = _unit_number(value, "clip_edit_fidelity_min")
        return cls(identity, fidelity, _required_text(value, "source"))


@dataclass(frozen=True)
class EvaluationManifest:
    schema_version: int
    protocol_status: str
    seeds: tuple[int, ...]
    samples: tuple[Sample, ...]
    pipelines: tuple[AttackPipeline, ...]
    thresholds: Thresholds

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationManifest":
        if value.get("schema_version") != 1:
            raise ValueError("schema_version must be 1")
        protocol_status = _required_text(value, "protocol_status")
        if protocol_status not in {"draft", "frozen"}:
            raise ValueError("protocol_status must be 'draft' or 'frozen'")
        seeds = tuple(int(seed) for seed in _required_list(value, "seeds"))
        if not seeds or any(seed < 0 for seed in seeds):
            raise ValueError("seeds must contain nonnegative integers")
        samples = tuple(
            Sample.from_dict(item) for item in _required_list(value, "samples")
        )
        pipelines = tuple(
            AttackPipeline.from_dict(item)
            for item in _required_list(value, "pipelines")
        )
        if not samples or not pipelines:
            raise ValueError("samples and pipelines must not be empty")
        _require_unique("sample", [sample.id for sample in samples])
        _require_unique("pipeline", [pipeline.id for pipeline in pipelines])
        return cls(
            schema_version=1,
            protocol_status=protocol_status,
            seeds=seeds,
            samples=samples,
            pipelines=pipelines,
            thresholds=Thresholds.from_dict(
                _required_dict(value, "thresholds")
            ),
        )

    @property
    def planned_run_count(self) -> int:
        return sum(
            len(sample.variants) * len(self.pipelines) * len(self.seeds)
            for sample in self.samples
        )


@dataclass(frozen=True)
class EvaluationResult:
    sample_id: str
    variant_id: str
    method: str
    epsilon_255: float
    pipeline_id: str
    architecture: str
    seed: int
    jpeg_quality: int
    identity_similarity: Optional[float]
    clip_edit_fidelity: float
    protection_lpips: float
    attack_succeeded: bool
    original_output_path: str
    protected_output_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_manifest(path: str | Path) -> EvaluationManifest:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("manifest root must be an object")
    return EvaluationManifest.from_dict(value)


def _required_text(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return item


def _required_list(value: dict[str, Any], key: str) -> list[Any]:
    item = value.get(key)
    if not isinstance(item, list):
        raise ValueError(f"{key} must be a list")
    return item


def _required_dict(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"{key} must be an object")
    return item


def _integer(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"{key} must be an integer")
    return item


def _positive_integer(value: dict[str, Any], key: str) -> int:
    item = _integer(value, key)
    if item <= 0:
        raise ValueError(f"{key} must be positive")
    return item


def _nonnegative_number(value: dict[str, Any], key: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ValueError(f"{key} must be a number")
    if item < 0:
        raise ValueError(f"{key} must be nonnegative")
    return float(item)


def _unit_number(value: dict[str, Any], key: str) -> float:
    item = _nonnegative_number(value, key)
    if item > 1:
        raise ValueError(f"{key} must be between 0 and 1")
    return item


def _require_unique(kind: str, values: list[str]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{kind} ids must be unique")
