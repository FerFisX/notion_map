"""Deterministic validation of the roadmap technical schema contract.

This metric checks whether the generated roadmap has the minimum shape required
by downstream consumers such as reports, renderers, and evaluators. It does not
judge semantic quality, learning quality, or whether the number of steps fits
the user's question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_VALID_TYPES = {"inicio", "proceso", "decision", "fin"}
_REQUIRED_STEP_FIELDS = ("id", "label", "description", "type")


@dataclass
class _Check:
    name: str
    passed: bool
    detail: str
    weight: float = 1.0


class StructureValidator:
    """Validate the technical schema contract of a generated roadmap."""

    PASS_THRESHOLD = 7.0

    def validate(self, roadmap: dict[str, Any]) -> dict[str, Any]:
        raw_steps = roadmap.get("steps", [])
        steps = raw_steps if isinstance(raw_steps, list) else []
        checks: list[_Check] = []

        checks.append(_Check(
            "has_steps",
            bool(steps),
            f"{len(steps)} steps" if steps else "No steps found",
            weight=1.2,
        ))

        title = str(roadmap.get("title", "")).strip()
        checks.append(_Check(
            "has_title",
            bool(title),
            f"Title: '{title[:60]}'" if title else "Missing title",
        ))

        ids = [str(step.get("id", "")).strip() for step in steps if isinstance(step, dict)]
        duplicates = [id_ for id_ in set(ids) if id_ and ids.count(id_) > 1]
        checks.append(_Check(
            "unique_ids",
            not duplicates,
            "Unique IDs" if not duplicates else f"Duplicate IDs: {duplicates}",
        ))

        inicio_count = sum(1 for step in steps if isinstance(step, dict) and step.get("type") == "inicio")
        checks.append(_Check(
            "single_inicio",
            inicio_count == 1,
            f"'inicio' nodes: {inicio_count} (must be 1)",
            weight=1.2,
        ))

        fin_count = sum(1 for step in steps if isinstance(step, dict) and step.get("type") == "fin")
        checks.append(_Check(
            "single_fin",
            fin_count == 1,
            f"'fin' nodes: {fin_count} (must be 1)",
            weight=1.2,
        ))

        invalid_types = [
            f"'{step.get('label', '?')}' type='{step.get('type')}'"
            for step in steps
            if isinstance(step, dict) and step.get("type") not in _VALID_TYPES
        ]
        checks.append(_Check(
            "valid_types",
            not invalid_types,
            "All step types are valid" if not invalid_types
            else f"Invalid types: {invalid_types[:3]}",
        ))

        empty_fields = []
        for step in steps:
            if not isinstance(step, dict):
                empty_fields.append("non-dict step")
                continue
            for field in _REQUIRED_STEP_FIELDS:
                if not str(step.get(field, "")).strip():
                    empty_fields.append(f"step '{step.get('id', '?')}' field '{field}'")
        checks.append(_Check(
            "required_fields_present",
            not empty_fields,
            "Required fields are present" if not empty_fields
            else f"Missing or empty fields: {empty_fields[:3]}",
        ))

        key_points_bad = [
            f"'{step.get('label', '?')}' key_points={type(step.get('key_points')).__name__}"
            for step in steps
            if not isinstance(step, dict)
            or "key_points" not in step
            or not isinstance(step.get("key_points"), list)
        ]
        checks.append(_Check(
            "key_points_list",
            not key_points_bad,
            "All steps have key_points as a list" if not key_points_bad
            else f"Invalid key_points: {key_points_bad[:3]}",
        ))

        total_weight = sum(check.weight for check in checks)
        passed_weight = sum(check.weight for check in checks if check.passed)
        score = round((passed_weight / total_weight) * 10, 2) if total_weight else 0.0
        passed_count = sum(1 for check in checks if check.passed)

        return {
            "score": score,
            "verdict": "PASS" if score >= self.PASS_THRESHOLD else "FAIL",
            "passed": passed_count,
            "total_checks": len(checks),
            "pass_rate": round(passed_count / len(checks), 4) if checks else 0.0,
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "detail": check.detail,
                    "weight": check.weight,
                }
                for check in checks
            ],
            "violations": [check.detail for check in checks if not check.passed],
        }
