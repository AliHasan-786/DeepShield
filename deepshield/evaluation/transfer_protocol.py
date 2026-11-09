"""Validation and dry-run planning for the unexecuted transfer protocol."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StagePlan:
    id: str
    split: str
    image_slots: int
    editor_calls: int
    protection_jobs: int
    setup_gpu_hours: float
    estimated_gpu_hours: float


@dataclass(frozen=True)
class TransferProtocolPlan:
    protocol_id: str
    protocol_status: str
    evidence_state: str
    stages: tuple[StagePlan, ...]
    total_editor_calls: int
    total_protection_jobs: int
    estimated_gpu_hours: float
    estimated_gpu_cost_usd: float | None
    first_gate_stage_ids: tuple[str, ...]
    first_gate_gpu_hours: float
    first_gate_cost_usd: float | None
    execution_ready: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransferProtocol:
    value: dict[str, Any]

    @property
    def protocol_id(self) -> str:
        return self.value["protocol_id"]

    def plan(self, gpu_hourly_usd: float | None = None) -> TransferProtocolPlan:
        compute = self.value["compute"]
        assumptions = compute["planning_assumptions"]
        editor_seconds = float(assumptions["editor_seconds_per_call"])
        protection_seconds = float(assumptions["protection_seconds_per_job"])
        stages: list[StagePlan] = []
        for stage in self.value["stages"]:
            editor_calls = (
                stage["image_slots"]
                * stage["method_condition_count"]
                * len(stage["editor_ids"])
                * len(stage["preprocessing_ids"])
                * len(stage["seeds"])
            )
            protection_jobs = (
                stage["image_slots"] * stage["protection_condition_count"]
            )
            estimated_hours = (
                editor_calls * editor_seconds
                + protection_jobs * protection_seconds
            ) / 3600 + float(stage["setup_gpu_hours"])
            stages.append(
                StagePlan(
                    id=stage["id"],
                    split=stage["split"],
                    image_slots=stage["image_slots"],
                    editor_calls=editor_calls,
                    protection_jobs=protection_jobs,
                    setup_gpu_hours=float(stage["setup_gpu_hours"]),
                    estimated_gpu_hours=round(estimated_hours, 3),
                )
            )

        total_hours = round(sum(stage.estimated_gpu_hours for stage in stages), 3)
        first_gate_ids = tuple(compute["first_gate_proposal"]["stage_ids"])
        first_gate_hours = round(
            sum(stage.estimated_gpu_hours for stage in stages if stage.id in first_gate_ids),
            3,
        )
        blockers = tuple(_execution_blockers(self.value))
        return TransferProtocolPlan(
            protocol_id=self.protocol_id,
            protocol_status=self.value["protocol_status"],
            evidence_state=self.value["evidence_state"],
            stages=tuple(stages),
            total_editor_calls=sum(stage.editor_calls for stage in stages),
            total_protection_jobs=sum(stage.protection_jobs for stage in stages),
            estimated_gpu_hours=total_hours,
            estimated_gpu_cost_usd=(
                None if gpu_hourly_usd is None else round(total_hours * gpu_hourly_usd, 2)
            ),
            first_gate_stage_ids=first_gate_ids,
            first_gate_gpu_hours=first_gate_hours,
            first_gate_cost_usd=(
                None
                if gpu_hourly_usd is None
                else round(first_gate_hours * gpu_hourly_usd, 2)
            ),
            execution_ready=not blockers,
            blockers=blockers,
        )


def load_transfer_protocol(path: str | Path) -> TransferProtocol:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("protocol root must be an object")
    _validate(value)
    return TransferProtocol(value)


def _validate(value: dict[str, Any]) -> None:
    if value.get("schema_version") != 2:
        raise ValueError("schema_version must be 2")
    _text(value, "protocol_id")
    if value.get("protocol_status") not in {"draft-unrun", "frozen"}:
        raise ValueError("protocol_status must be 'draft-unrun' or 'frozen'")
    if value.get("evidence_state") != "unrun":
        raise ValueError("a preparation protocol must start with evidence_state='unrun'")
    _text(value, "review_cutoff")
    _text(value, "decision_use")
    _text(value, "question")
    _text(value, "hypothesis")
    _text(value, "contradiction_rule")

    methods = _object_list(value, "methods")
    method_ids = _unique_ids(methods, "method")
    kinds = {_text(method, "kind") for method in methods}
    if not {"control", "baseline", "candidate"}.issubset(kinds):
        raise ValueError("methods must include control, baseline, and candidate")
    for method in methods:
        _text(method, "license")
        _text(method, "source_url")
        _text(method, "source_revision")
        _text(method, "implementation_status")

    editors = _object_list(value, "editors")
    editor_ids = _unique_ids(editors, "editor")
    development = set()
    held_out = set()
    for editor in editors:
        split = _text(editor, "split")
        if split == "development":
            development.add(editor["id"])
        elif split == "held_out":
            held_out.add(editor["id"])
        else:
            raise ValueError("editor split must be development or held_out")
        revision = _text(editor, "revision")
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
            raise ValueError("editor revision must be a 40-character lowercase commit hash")
        for key in (
            "model_id",
            "license",
            "source_url",
            "availability_checked",
            "adapter_status",
        ):
            _text(editor, key)
    if not development or not held_out or development & held_out:
        raise ValueError("development and held_out editors must be nonempty and disjoint")

    preprocessing = _object_list(value, "preprocessing")
    preprocessing_ids = set(_unique_ids(preprocessing, "preprocessing"))
    for item in preprocessing:
        _text(item, "operation")
        _text(item, "implementation")

    stages = _object_list(value, "stages")
    stage_ids = _unique_ids(stages, "stage")
    for stage in stages:
        split = _text(stage, "split")
        if split not in {"development", "held_out"}:
            raise ValueError("stage split must be development or held_out")
        _positive_int(stage, "image_slots")
        _positive_int(stage, "method_condition_count")
        protection_count = _nonnegative_int(stage, "protection_condition_count")
        if protection_count >= stage["method_condition_count"]:
            raise ValueError("each stage must include at least one unprotected control")
        if not isinstance(stage.get("setup_gpu_hours"), (int, float)) or stage["setup_gpu_hours"] < 0:
            raise ValueError("setup_gpu_hours must be nonnegative")
        referenced_editors = _text_list(stage, "editor_ids")
        if not set(referenced_editors).issubset(editor_ids):
            raise ValueError("stage references an unknown editor")
        allowed = development if split == "development" else held_out
        if not set(referenced_editors).issubset(allowed):
            raise ValueError("stage editor does not match its split")
        referenced_preprocessing = _text_list(stage, "preprocessing_ids")
        if not set(referenced_preprocessing).issubset(preprocessing_ids):
            raise ValueError("stage references unknown preprocessing")
        seeds = stage.get("seeds")
        if not isinstance(seeds, list) or not seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
        ):
            raise ValueError("stage seeds must be nonnegative integers")

    asset_policy = _object(value, "asset_policy")
    _positive_int(asset_policy, "required_development_count")
    _positive_int(asset_policy, "required_held_out_count")
    for key in ("eligibility_rule", "inherited_asset_rule", "provenance_rule"):
        _text(asset_policy, key)
    assets = asset_policy.get("assets")
    if not isinstance(assets, list):
        raise ValueError("asset_policy.assets must be a list")
    seen_assets: set[str] = set()
    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("assets must be objects")
        asset_id = _text(asset, "id")
        if asset_id in seen_assets:
            raise ValueError("asset ids must be unique")
        seen_assets.add(asset_id)
        if asset.get("split") not in {"development", "held_out"}:
            raise ValueError("asset split must be development or held_out")
        if asset.get("authorized") is not True:
            raise ValueError("every listed asset must be explicitly authorized")
        for key in ("source_type", "provenance_url", "license_or_permission"):
            _text(asset, key)
        digest = _text(asset, "sha256")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("asset sha256 must be a 64-character lowercase digest")

    visual_budget = _object(value, "visual_budget")
    for key in ("status", "reference_rule", "selection_rule", "reporting_rule"):
        _text(visual_budget, key)
    if not isinstance(visual_budget.get("candidate_epsilon_255"), list):
        raise ValueError("candidate_epsilon_255 must be a list")

    metrics = _object(value, "metrics")
    primary = _object(metrics, "primary")
    for key in ("id", "status", "rubric", "failure_mode_rule"):
        _text(primary, key)
    if not isinstance(metrics.get("quality"), list) or not metrics["quality"]:
        raise ValueError("metrics.quality must be a nonempty list")
    _text(metrics, "automated_metric_warning")

    analysis = _object(value, "analysis")
    for key in (
        "unit",
        "primary_contrast",
        "uncertainty_rule",
        "multiplicity_rule",
        "missingness_rule",
        "stopping_rule",
    ):
        _text(analysis, key)

    outcomes = _object(value, "outcomes")
    states = set(_text_list(outcomes, "allowed_cell_states"))
    if not {"unrun", "measured", "failed", "unavailable", "missing"}.issubset(states):
        raise ValueError("allowed_cell_states is incomplete")
    if outcomes.get("initial_cell_state") != "unrun":
        raise ValueError("initial_cell_state must be 'unrun', never numeric zero")
    _text(outcomes, "zero_semantics")

    compute = _object(value, "compute")
    assumptions = _object(compute, "planning_assumptions")
    _positive_number(assumptions, "editor_seconds_per_call")
    _positive_number(assumptions, "protection_seconds_per_job")
    _text(assumptions, "provenance")
    first_gate = _object(compute, "first_gate_proposal")
    first_stage_ids = _text_list(first_gate, "stage_ids")
    if not set(first_stage_ids).issubset(stage_ids):
        raise ValueError("first gate references an unknown stage")
    _positive_number(first_gate, "max_gpu_hours")
    _positive_number(first_gate, "max_usd")
    _positive_number(first_gate, "max_download_gb")
    if first_gate.get("approval_status") != "not-approved":
        raise ValueError("preparation milestone cannot approve compute")

    interaction = _object(value, "eventual_interaction")
    for key in ("input", "observable_output", "evidence_source", "intended_insight"):
        _text(interaction, key)


def _execution_blockers(value: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if value["protocol_status"] != "frozen":
        blockers.append("protocol_not_frozen")
    policy = value["asset_policy"]
    development = sum(asset["split"] == "development" for asset in policy["assets"])
    held_out = sum(asset["split"] == "held_out" for asset in policy["assets"])
    if development < policy["required_development_count"] or held_out < policy["required_held_out_count"]:
        blockers.append("rights_cleared_assets_missing")
    if any(editor["adapter_status"] != "implemented-and-verified" for editor in value["editors"]):
        blockers.append("editor_adapter_not_verified")
    if any(method["implementation_status"] != "runnable-and-verified" for method in value["methods"]):
        blockers.append("method_implementation_not_verified")
    if value["visual_budget"]["status"] != "frozen":
        blockers.append("visual_budget_not_frozen")
    if value["metrics"]["primary"]["status"] != "ready":
        blockers.append("primary_metric_not_ready")
    if value["compute"]["first_gate_proposal"]["approval_status"] != "approved":
        blockers.append("compute_not_approved")
    return blockers


def _object(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"{key} must be an object")
    return item


def _object_list(value: dict[str, Any], key: str) -> list[dict[str, Any]]:
    items = value.get(key)
    if not isinstance(items, list) or not items or any(not isinstance(item, dict) for item in items):
        raise ValueError(f"{key} must be a nonempty list of objects")
    return items


def _unique_ids(items: list[dict[str, Any]], kind: str) -> set[str]:
    values = [_text(item, "id") for item in items]
    if len(values) != len(set(values)):
        raise ValueError(f"{kind} ids must be unique")
    return set(values)


def _text(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return item


def _text_list(value: dict[str, Any], key: str) -> list[str]:
    items = value.get(key)
    if not isinstance(items, list) or not items or any(
        not isinstance(item, str) or not item.strip() for item in items
    ):
        raise ValueError(f"{key} must be a nonempty list of strings")
    return items


def _positive_int(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return item


def _nonnegative_int(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise ValueError(f"{key} must be a nonnegative integer")
    return item


def _positive_number(value: dict[str, Any], key: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)) or item <= 0:
        raise ValueError(f"{key} must be a positive number")
    return float(item)
