"""Render the protection-vs-pipeline record without hiding negative cells."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from .schema import EvaluationResult


def build_matrix_markdown(results: list[EvaluationResult]) -> str:
    if not results:
        return (
            "# Protection vs. pipeline matrix\n\n"
            "No measured results are available. This is intentionally blank; "
            "dry runs and planned experiments are not evidence.\n"
        )

    pipelines = sorted({result.pipeline_id for result in results})
    variants = sorted(
        {(result.method, result.variant_id) for result in results}
    )
    grouped: dict[tuple[str, str, str], list[EvaluationResult]] = defaultdict(list)
    for result in results:
        grouped[(result.method, result.variant_id, result.pipeline_id)].append(
            result
        )

    lines = [
        "# Protection vs. pipeline matrix",
        "",
        "Each cell reports attack success rate across recorded seeds. Lower is "
        "better for the defense. `—` means not measured.",
        "",
        "| Protection variant | " + " | ".join(pipelines) + " |",
        "| --- | " + " | ".join("---:" for _ in pipelines) + " |",
    ]
    for method, variant_id in variants:
        cells = []
        for pipeline_id in pipelines:
            rows = grouped.get((method, variant_id, pipeline_id), [])
            if not rows:
                cells.append("—")
                continue
            success_rate = mean(float(row.attack_succeeded) for row in rows)
            cells.append(f"{success_rate:.0%} ({len(rows)} runs)")
        lines.append(
            f"| {method} / {variant_id} | " + " | ".join(cells) + " |"
        )
    lines.append("")
    lines.append(
        "Negative results are retained: a 100% cell means the protection failed "
        "for every recorded run."
    )
    lines.append("")
    return "\n".join(lines)
