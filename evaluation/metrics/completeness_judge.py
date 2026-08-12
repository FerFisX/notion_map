"""Dedicated semantic judge for roadmap Completeness.

Completeness evaluates whether the roadmap covers the content required by the
user's request. It does not evaluate factual correctness, contextual support,
order, actionability, distinctness, or global structure quality.
"""

from __future__ import annotations

import json
from typing import Any

from src.llm_provider import get_judge_llm


_COVERAGE_STATES = {"covered", "partial", "missing"}
_IMPORTANCE_LEVELS = {"critical", "important", "supporting"}
_SOURCE_TYPES = {
    "explicit_expected",
    "explicit_scope",
    "ground_truth",
    "question_inferred",
}


_PROMPT_TEMPLATE = """\
You are an expert evaluator of technical learning and execution roadmaps.
Return ONLY valid JSON. Do not use markdown or extra text.

ORIGINAL QUESTION
{question}

REFINED QUESTION
{refined_question}

EXPLICIT SCOPE
{explicit_scope}

GROUND TRUTH
{ground_truth}

EXPECTED ELEMENTS
{expected_elements}

ROADMAP
{roadmap}

OBJECTIVE
Evaluate only Completeness: whether the roadmap covers the content materially
required to satisfy the user's request.

REFERENCE PRIORITY
1. Explicit scope requested by the user.
2. Provided expected elements.
3. Ground truth.
4. Original and refined questions.
5. Domain inference only when the previous sources leave a genuine gap.

When EXPECTED ELEMENTS are provided:
- return exactly one assessment for every provided expected element;
- preserve every provided element ID exactly;
- preserve each provided name and importance exactly; these fields define the
  controlled coverage universe and must not be reclassified;
- interpret wording semantically rather than requiring an exact label match;
- do not add unrelated best practices or expand the requested scope.

When EXPECTED ELEMENTS are empty:
- infer the smallest sufficient set of expected elements from the questions,
  explicit scope, and ground truth;
- assign stable IDs such as expected_1, expected_2, and so on;
- avoid inflating the universe with merely optional enhancements.

EXPECTED ELEMENT IMPORTANCE
- critical: without this element, the core user need is not resolved.
- important: the answer remains useful, but a meaningful requested phase,
  concept, practice, validation, or outcome is absent.
- supporting: useful depth or reinforcement that improves coverage but is not
  required for the core request.

COVERAGE STATES
- covered: the roadmap clearly represents the expected element.
- partial: the element is recognizable, but a meaningful part of its requested
  scope is absent.
- missing: the element is not represented.

Coverage requires identifiable subject matter, objective, phase, concept,
practice, or outcome. It does not require executable detail. A vague step can
cover an expected topic even when Actionability should score it poorly.

SCORE BANDS
- 8.0-10.0 / complete:
  The core need and all critical elements are covered. At most a supporting
  element or minor part of an important element is absent.
- 5.0-7.9 / needs review:
  The roadmap addresses the core need but omits or only partially covers a
  meaningful important element, or partially covers one critical element.
- 0.0-4.9 / incomplete:
  A critical element is missing, the central process is absent, or multiple
  important omissions prevent the roadmap from satisfying the request.

The overall score is NOT an arithmetic coverage percentage. Select its band
from the importance and practical impact of the coverage gaps, then choose a
value naturally inside that band.

MECE BOUNDARIES
Do NOT lower Completeness because:
- included content is factually wrong; that belongs to Accuracy;
- included content is unsupported by retrieved context; that belongs to
  Grounding;
- covered steps are in the wrong order; that belongs to Logical Order;
- covered steps are vague or non-executable; that belongs to Actionability;
- steps repeat work; that belongs to Step Distinctness and Step Overlap;
- framing, closure, granularity, navigation, or roadmap length is weak; that
  belongs to Structure Quality and Scope Fit;
- the technical JSON contract is invalid; that belongs to Schema Validity.

Do not require framing or closure merely because roadmaps usually contain them.
Count them only when they are materially required by the user's need, expected
elements, or explicit scope.

CONSISTENCY REQUIREMENTS
- Every evidence_step_id must be an exact roadmap step ID.
- A missing element must have no evidence step IDs.
- A covered element must identify at least one evidence step.
- Partial coverage must explain what is present and what remains absent.
- If ANY critical expected element is missing, the overall score MUST be in
  the 0.0-4.9 band. A roadmap cannot satisfy its core need without a critical
  element.
- If a critical element is partial but none is missing, the overall score
  normally belongs in the 5.0-7.9 band.
- A score of 8.0 or higher requires every critical element to be covered and
  no important element to be missing.
- The score, reason, importance, coverage states, strengths, and
  recommendations must describe the same assessment.

Before returning JSON, perform this final check in order:
1. List all elements marked missing.
2. Inspect their importance.
3. If at least one is critical, select 0.0-4.9.
4. Otherwise, select the band from the remaining partial or important gaps.
Do not describe a critical element as missing while returning NEEDS_REVIEW.

Expected JSON schema:
{{
  "completeness": {{
    "expected_elements": [
      {{
        "id": "<exact provided ID or inferred expected_N>",
        "name": "<concise expected element>",
        "importance": "critical|important|supporting",
        "source": "explicit_expected|explicit_scope|ground_truth|question_inferred",
        "coverage_status": "covered|partial|missing",
        "evidence_step_ids": ["<exact roadmap step ID>", "..."],
        "reason": "<case-specific coverage explanation>",
        "expected_phase": "<where it belongs conceptually or empty string>",
        "recommendation": "<how to close a partial/missing gap or empty string>"
      }}
    ],
    "score": <0-10>,
    "reason": "<short explanation supporting the score band>",
    "strengths": ["<covered strength>", "..."],
    "recommendations": ["<coverage recommendation>", "..."]
  }}
}}
"""


def _strip_markdown_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _parse_json_response(raw: str) -> dict[str, Any]:
    text = _strip_markdown_json(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Completeness response must be a JSON object")
    return parsed


def _normalize_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return round(max(0.0, min(10.0, score)), 2)


def _verdict(score: float) -> str:
    if score >= 8.0:
        return "PASS"
    if score >= 5.0:
        return "NEEDS_REVIEW"
    return "FAIL"


def _score_band(score: float) -> str:
    if score >= 8.0:
        return "8.0-10.0"
    if score >= 5.0:
        return "5.0-7.9"
    return "0.0-4.9"


def _roadmap_steps(roadmap: dict[str, Any]) -> list[dict[str, Any]]:
    raw_steps = roadmap.get("steps", [])
    if not isinstance(raw_steps, list):
        return []
    return [step for step in raw_steps if isinstance(step, dict)]


def _provided_elements(expected_elements: list[Any] | None) -> list[dict[str, str]]:
    normalized = []
    for index, item in enumerate(expected_elements or [], 1):
        if isinstance(item, dict):
            element_id = str(item.get("id", "")).strip() or f"expected_{index}"
            name = str(item.get("name", "")).strip()
            importance = str(item.get("importance", "important")).lower().strip()
        else:
            element_id = f"expected_{index}"
            name = str(item).strip()
            importance = "important"
        if not name:
            continue
        if importance not in _IMPORTANCE_LEVELS:
            importance = "important"
        normalized.append({
            "id": element_id,
            "name": name,
            "importance": importance,
        })
    return normalized


def _gap_severity(importance: str, status: str) -> str:
    if importance == "critical" and status == "missing":
        return "high"
    if importance == "critical" or (
        importance == "important" and status == "missing"
    ):
        return "medium"
    return "low"


def _fallback_result(
    expected_elements: list[dict[str, str]],
    error: Exception | None = None,
) -> dict[str, Any]:
    reason = "Completeness judge failed; manual review is required."
    if error:
        reason = f"{reason} Error: {type(error).__name__}"
    elements = [
        {
            **element,
            "source": "explicit_expected",
            "coverage_status": "missing",
            "evidence_step_ids": [],
            "reason": reason,
            "expected_phase": "",
            "recommendation": "Review this expected element manually.",
        }
        for element in expected_elements
    ]
    gaps = [
        {
            "expected_element_id": element["id"],
            "name": element["name"],
            "importance": element["importance"],
            "coverage_status": "missing",
            "severity": _gap_severity(element["importance"], "missing"),
            "explanation": reason,
            "expected_phase": "",
            "recommendation": "Review this expected element manually.",
            "evidence_step_ids": [],
        }
        for element in elements
    ]
    return {
        "completeness": {
            "score": 5.0,
            "verdict": "NEEDS_REVIEW",
            "score_band": "5.0-7.9",
            "reason": reason,
            "expected_elements": elements,
            "coverage_pct": 0.0,
            "covered_element_count": 0,
            "partial_element_count": 0,
            "missing_elements": gaps,
            "missing_element_count": len(gaps),
            "strengths": [],
            "recommendations": ["Review Completeness manually."],
            "normalization_notes": [],
            "manual_review_required": True,
        }
    }


def _normalize_result(
    result: dict[str, Any],
    roadmap: dict[str, Any],
    provided_elements: list[dict[str, str]],
) -> dict[str, Any]:
    raw = (
        result.get("completeness")
        if isinstance(result.get("completeness"), dict)
        else {}
    )
    step_ids = {
        str(step.get("id", "")).strip()
        for step in _roadmap_steps(roadmap)
        if str(step.get("id", "")).strip()
    }
    notes: list[str] = []
    manual_review_required = False
    raw_elements = (
        raw.get("expected_elements")
        if isinstance(raw.get("expected_elements"), list)
        else []
    )
    raw_by_id = {
        str(item.get("id", "")).strip(): item
        for item in raw_elements
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }
    reference_by_id = {item["id"]: item for item in provided_elements}
    if provided_elements:
        element_ids = [item["id"] for item in provided_elements]
        unknown = set(raw_by_id) - set(element_ids)
        if unknown:
            notes.append("Discarded expected elements outside the provided universe.")
            manual_review_required = True
    else:
        element_ids = list(raw_by_id)
        if not element_ids:
            notes.append("The judge did not define an expected coverage universe.")
            manual_review_required = True

    normalized_elements = []
    for element_id in element_ids:
        item = raw_by_id.get(element_id, {})
        reference = reference_by_id.get(element_id, {})
        if not item:
            notes.append(f"Added missing assessment for expected element {element_id}.")
            manual_review_required = True
        name = str(reference.get("name") or item.get("name") or element_id).strip()
        importance = str(
            reference.get("importance") or item.get("importance") or "important"
        ).lower().strip()
        if importance not in _IMPORTANCE_LEVELS:
            importance = "important"
            notes.append(f"Normalized importance for expected element {element_id}.")
            manual_review_required = True
        status = str(item.get("coverage_status", "missing")).lower().strip()
        if status not in _COVERAGE_STATES:
            status = "missing"
            notes.append(f"Normalized coverage status for expected element {element_id}.")
            manual_review_required = True
        source = str(item.get("source", "")).lower().strip()
        if source not in _SOURCE_TYPES:
            source = "explicit_expected" if provided_elements else "question_inferred"
        evidence = []
        for value in item.get("evidence_step_ids", []):
            step_id = str(value).strip()
            if step_id in step_ids and step_id not in evidence:
                evidence.append(step_id)
            elif step_id:
                notes.append(
                    f"Discarded unknown evidence step ID for expected element {element_id}."
                )
                manual_review_required = True
        if status == "missing":
            evidence = []
        elif status == "covered" and not evidence:
            status = "partial"
            notes.append(
                f"Downgraded expected element {element_id} to partial without evidence."
            )
            manual_review_required = True
        reason = str(item.get("reason", "")).strip()
        if not reason:
            reason = "The judge did not provide a coverage explanation."
            manual_review_required = True
        normalized_elements.append({
            "id": element_id,
            "name": name,
            "importance": importance,
            "source": source,
            "coverage_status": status,
            "evidence_step_ids": evidence,
            "reason": reason,
            "expected_phase": str(item.get("expected_phase", "")).strip(),
            "recommendation": str(item.get("recommendation", "")).strip(),
        })

    covered = [
        item for item in normalized_elements
        if item["coverage_status"] == "covered"
    ]
    partial = [
        item for item in normalized_elements
        if item["coverage_status"] == "partial"
    ]
    gaps = [
        {
            "expected_element_id": item["id"],
            "name": item["name"],
            "importance": item["importance"],
            "coverage_status": item["coverage_status"],
            "severity": _gap_severity(
                item["importance"],
                item["coverage_status"],
            ),
            "explanation": item["reason"],
            "expected_phase": item["expected_phase"],
            "recommendation": item["recommendation"],
            "evidence_step_ids": item["evidence_step_ids"],
        }
        for item in normalized_elements
        if item["coverage_status"] != "covered"
    ]
    total = len(normalized_elements)
    coverage_pct = (
        round((len(covered) + 0.5 * len(partial)) / total * 100, 2)
        if total
        else 0.0
    )

    score = _normalize_score(raw.get("score", 0))
    critical_missing = [
        item for item in gaps
        if item["importance"] == "critical"
        and item["coverage_status"] == "missing"
    ]
    critical_partial = [
        item for item in gaps
        if item["importance"] == "critical"
        and item["coverage_status"] == "partial"
    ]
    if not gaps and score < 8.0:
        notes.append("The score is below 8.0 despite complete expected coverage.")
        manual_review_required = True
    if critical_missing and score >= 5.0:
        notes.append("The score is 5.0 or higher despite a missing critical element.")
        manual_review_required = True
    if critical_partial and score >= 8.0:
        notes.append("The score is 8.0 or higher despite a partial critical element.")
        manual_review_required = True
    if score >= 8.0 and any(
        item["importance"] == "important"
        and item["coverage_status"] == "missing"
        for item in gaps
    ):
        notes.append("The score is 8.0 or higher despite a missing important element.")
        manual_review_required = True

    reason = str(raw.get("reason", "")).strip()
    if not reason:
        reason = "The judge did not provide an overall Completeness reason."
        manual_review_required = True
    strengths = (
        [str(item).strip() for item in raw.get("strengths", []) if str(item).strip()]
        if isinstance(raw.get("strengths"), list)
        else []
    )
    recommendations = (
        [
            str(item).strip()
            for item in raw.get("recommendations", [])
            if str(item).strip()
        ]
        if isinstance(raw.get("recommendations"), list)
        else []
    )

    return {
        "completeness": {
            "score": score,
            "verdict": _verdict(score),
            "score_band": _score_band(score),
            "reason": reason,
            "expected_elements": normalized_elements,
            "coverage_pct": coverage_pct,
            "covered_element_count": len(covered),
            "partial_element_count": len(partial),
            "missing_elements": gaps,
            "missing_element_count": len(gaps),
            "strengths": strengths,
            "recommendations": recommendations,
            "normalization_notes": notes,
            "manual_review_required": manual_review_required,
        }
    }


class CompletenessJudge:
    """Evaluate roadmap Completeness against an explicit or inferred universe."""

    def __init__(self, llm=None):
        self.llm = llm or get_judge_llm(temperature=0.0, max_tokens=4096)

    def _invoke(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content

    def evaluate(
        self,
        roadmap: dict[str, Any],
        question: str = "",
        refined_question: str = "",
        explicit_scope: str = "",
        ground_truth: str = "",
        expected_elements: list[Any] | None = None,
    ) -> dict[str, Any]:
        provided = _provided_elements(expected_elements)
        prompt = _PROMPT_TEMPLATE.format(
            question=question or "(not provided)",
            refined_question=refined_question or "(not provided)",
            explicit_scope=explicit_scope or "(not provided)",
            ground_truth=ground_truth or "(not provided)",
            expected_elements=json.dumps(provided, ensure_ascii=False, indent=2),
            roadmap=json.dumps(roadmap, ensure_ascii=False, indent=2),
        )
        try:
            raw = self._invoke(prompt)
            retried = False
            try:
                parsed = _parse_json_response(raw)
            except (json.JSONDecodeError, ValueError):
                retry_prompt = (
                    f"{prompt}\n\nRETRY REQUIREMENT\n"
                    "The previous response was not valid JSON. Return the complete JSON object only."
                )
                parsed = _parse_json_response(self._invoke(retry_prompt))
                retried = True
            normalized = _normalize_result(parsed, roadmap, provided)
            if retried:
                normalized["completeness"]["normalization_notes"].append(
                    "Retried once after an invalid JSON response."
                )
            return normalized
        except Exception as exc:
            return _fallback_result(provided, exc)
