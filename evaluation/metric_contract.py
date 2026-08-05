"""Canonical metric helpers shared by evaluation consumers."""

from __future__ import annotations

from typing import Any, Mapping

SEMANTIC_METRICS = (
    "grounding",
    "completeness",
    "actionability",
    "logical_order",
    "structure_quality",
    "step_distinctness",
)

def metric_score(name: str, result: Mapping[str, Any] | None) -> float:
    """Read a canonical 0-10 score without changing the metric payload."""
    value = result or {}
    raw = value.get("support_score", 0) if name == "grounding" else value.get("score", 0)
    try:
        score = float(raw)
    except (TypeError, ValueError):
        score = 0.0
    return round(max(0.0, min(10.0, score)), 2)


def metric_reason(result: Mapping[str, Any] | None) -> str:
    """Return the single canonical explanation exposed by a metric."""
    value = result or {}
    return str(value.get("reason") or value.get("score_rationale") or "").strip()


def canonical_scores(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """Return available stakeholder scores from their canonical payloads."""
    return {
        name: metric_score(name, metrics[name])
        for name in SEMANTIC_METRICS
        if name in metrics
    }


def validate_canonical_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Return objective contract errors without rewriting semantic results."""
    errors = []
    for name in SEMANTIC_METRICS:
        result = metrics.get(name)
        if not isinstance(result, Mapping):
            errors.append(f"Missing canonical metric: {name}")
            continue
        score = metric_score(name, result)
        verdict = str(result.get("verdict", "")).strip()
        expected_verdict = "PASS" if score >= 8 else "NEEDS_REVIEW" if score >= 5 else "FAIL"
        if verdict != expected_verdict:
            errors.append(
                f"{name} verdict {verdict or '(missing)'} does not match score {score}."
            )
    return errors
