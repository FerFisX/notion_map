"""Canonical roadmap readiness derived from dedicated metric verdicts."""

from __future__ import annotations

from typing import Any, Mapping

from evaluation.metric_contract import SEMANTIC_METRICS, metric_reason, metric_score


_LABELS = {
    "grounding": "Grounding",
    "completeness": "Completeness",
    "actionability": "Actionability",
    "logical_order": "Logical Order",
    "structure_quality": "Structure Quality",
    "step_distinctness": "Step Distinctness",
    "schema_validity": "Schema Validity",
}


def _issue(name: str, result: Mapping[str, Any], severity: str) -> dict[str, Any]:
    return {
        "metric": name,
        "label": _LABELS[name],
        "severity": severity,
        "score": metric_score(name, result),
        "reason": metric_reason(result),
    }


def roadmap_readiness(
    metrics: Mapping[str, Mapping[str, Any]],
    contract_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Return readiness without averaging away a canonical metric failure."""
    required = set(SEMANTIC_METRICS) | {"schema_validity"}
    missing = sorted(required - set(metrics))
    if missing:
        return {
            "status": "NOT_EVALUATED",
            "code": -1,
            "reason": f"Missing required metrics: {', '.join(missing)}",
            "reasons": [f"Missing required metrics: {', '.join(missing)}"],
            "issues": [],
        }

    critical = []
    review = []
    for name in SEMANTIC_METRICS:
        result = metrics[name]
        verdict = str(result.get("verdict", "")).strip().upper()
        if verdict == "FAIL":
            critical.append(_issue(name, result, "critical"))
        elif verdict == "NEEDS_REVIEW":
            review.append(_issue(name, result, "review"))
        if bool(result.get("manual_review_required")) and verdict == "PASS":
            technical_issue = _issue(name, result, "review")
            notes = result.get("normalization_notes", [])
            note = str(notes[-1]).strip() if isinstance(notes, list) and notes else ""
            technical_issue["reason"] = (
                "Evaluation output requires manual review"
                + (f": {note}" if note else ".")
            )
            review.append(technical_issue)

    schema = metrics["schema_validity"]
    if str(schema.get("verdict", "")).strip().upper() != "PASS":
        critical.append(_issue("schema_validity", schema, "critical"))

    for error in contract_errors or []:
        review.append({
            "metric": "metric_contract",
            "label": "Metric Contract",
            "severity": "review",
            "score": None,
            "reason": error,
        })

    if critical:
        status, code, issues = "FAIL", 0, critical + review
    elif review:
        status, code, issues = "NEEDS_REVIEW", 1, review
    else:
        status, code, issues = "READY", 2, []

    reasons = []
    for item in issues:
        detail = item["reason"] or f"score {item['score']}/10"
        reasons.append(f"{item['label']}: {detail}")
    if not reasons:
        reasons = ["All canonical metrics meet their readiness requirements."]
    return {
        "status": status,
        "code": code,
        "reason": "; ".join(reasons),
        "reasons": reasons,
        "issues": issues,
    }
