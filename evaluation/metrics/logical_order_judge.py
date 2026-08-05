"""LLM-based semantic judge for roadmap logical order.

Logical Order evaluates dependency correctness only. It intentionally avoids
scoring completeness, grounding, actionability, step distinctness, schema
validity, or the roadmap's broader narrative flow.
"""

from __future__ import annotations

import json
from typing import Any

from src.llm_provider import get_judge_llm, invoke_llm_text


_PROMPT_TEMPLATE = """\
You are an expert evaluator of technical learning and execution roadmaps.
Return ONLY valid JSON. Do not use markdown or extra text.

USER QUESTION
{question}

CATEGORY
{category}

ROADMAP JSON
{roadmap}

CURRENT STEP POSITIONS
Lower positions execute before higher positions.
{current_order}

TASK
Evaluate LOGICAL ORDER only.

Logical Order asks whether the current sequence respects prerequisites, causal
dependencies, learning progression, and the timing of validation activities.

Evaluate exactly these four criteria:

1. Prerequisite order
   - Are required concepts, tools, credentials, inputs, or foundations
     introduced before a step depends on them?

2. Causal dependency order
   - Is an output created before another step consumes, transforms, tests, or
     publishes it?

3. Learning or execution progression
   - Does the sequence move from necessary foundations to application without
     requiring knowledge or state that only appears later?

4. Validation timing
   - Do testing, verification, monitoring, or final review steps occur after
     the result they validate exists?

SCORING PROCESS
Follow this order. Do not score before completing the dependency analysis.
1. Read the CURRENT STEP ORDER and map every step ID to its current position.
2. Identify only valid dependency violations.
3. Build the smallest correction plan that repairs all valid violations.
   Count independent step movements, not every dependency relationship restored
   by the same movement.
4. Determine the repair scope: no repair, one local repair, or structural
   reordering.
5. Select the score band from the repair scope.
6. Choose a numeric score inside that band.
7. Verify that the final score, criterion scores, reason, violations, and
   suggested order all describe the same result.

FINAL SCORE BANDS
- 8.0-10.0 — no repair required:
  No meaningful dependency violation exists. Independent steps may appear in
  any reasonable order. dependency_violations must be empty and
  suggested_order must equal CURRENT STEP ORDER.
- 5.0-7.9 — one local repair:
  One isolated or moderate dependency reversal exists, but moving one step or
  making one localized reorder repairs the sequence. This remains a local
  repair even when the misplaced step must move across several positions or
  that one move restores several related dependencies.
- 0.0-4.9 — structural reordering required:
  Two or more independent step movements, multiple unrelated root ordering
  defects, or a broadly reversed workflow requires substantial reordering.

The final score is NOT an average of the four criterion scores. Its band must
represent the most severe valid ordering defect and the scope of repair.

ONE-MOVE INVARIANT
Compare CURRENT STEP ORDER with suggested_order before selecting the score.
If suggested_order can be obtained by removing one existing step and inserting
that same step at another position while every other step keeps its relative
order, the correction is exactly one local repair. In that situation:
- score MUST be 5.0-7.9;
- the root violation severity MUST be low or medium, never high;
- do not classify the roadmap as structural reordering, regardless of how
  important the misplaced step is or how many positions it crosses.
The 0.0-4.9 band is allowed only when the correction cannot be achieved with
one such relocation because at least two independent movements are necessary
or the workflow is broadly reversed.

DEPENDENCY VIOLATIONS
Report only meaningful order problems. A stylistic preference is not a causal
dependency.

Report the minimal set of root ordering defects, not every downstream
consequence of the same misplaced step. When one step is too early or too late
relative to several steps and moving that single step to one position repairs
all relationships, report one representative root violation and classify the
roadmap as one local repair. Do not turn that one correction into separate
prerequisite, causal-output, learning-progression, and validation-timing
violations.

For every violation identify:
- dependent_step_id: the step currently placed too early;
- required_predecessor_step_id: the later step that must occur first;
- dependency_type:
  prerequisite | causal_output | learning_progression | validation_timing;
- severity: low | medium | high;
- explanation: why the dependent step requires the predecessor;
- suggested_fix: how to reorder the affected steps.

Use exact step IDs from the roadmap. Do not use labels in ID fields.

POSITION INVARIANT
A reported violation means the required predecessor is currently AFTER the
dependent step. In the current roadmap positions must satisfy:

  position(required_predecessor_step_id) > position(dependent_step_id)

Never report a pair when the required predecessor is already before the
dependent step. Before returning JSON, verify this invariant for every pair.
Example: if step_2 tries to use something created by the later step_3, report
dependent_step_id="step_2" and required_predecessor_step_id="step_3". The
corrected suggested order must then place step_3 before step_2.

SEVERITY CALIBRATION
Severity measures the breadth of the ordering defect and the complexity of
repair, not merely whether the dependent step can execute at its current
position.
- low: a minor ordering weakness with limited practical impact;
- medium: one isolated dependency reversal that a local move can repair;
- high: multiple independent root defects or a broadly reversed workflow that
  requires at least two independent movements and cannot be described as one
  isolated local correction.
Do not label a single adjacent prerequisite reversal as high solely because
the dependent step would fail before the move.
Do not label any single-step relocation as high solely because it restores
multiple prerequisites or moves across several positions.

REPAIR-SCOPE EXAMPLES
- A credential step placed immediately after the request that needs it is one
  local repair: move the credential before the request. Use 5.0-7.9.
- A validation step placed before the model, measures, and final artifact is
  one local repair when moving that validation step after the artifact repairs
  the entire sequence. Use 5.0-7.9 and report one validation-timing root defect.
- A roadmap with both a misplaced foundation and an independently misplaced
  deployment or validation step requires multiple movements. Use 0.0-4.9.

CRITERION SEPARATION
Assign each root ordering defect to its most specific primary criterion. Do
not penalize multiple criteria for the same causal fact unless independent
evidence supports each penalty.
- Missing credentials before an authenticated request primarily affects
  prerequisite_order.
- Testing an artifact before it exists primarily affects validation_timing.
- An output consumed before it is produced primarily affects
  causal_dependency_order.
- Advanced practice before required foundational learning primarily affects
  learning_progression.
Unrelated criteria should remain high when their own ordering logic is valid.
For one isolated root defect, score its primary criterion below 8.0 and keep
the other criteria between 8.0 and 10.0 unless you identify separate evidence
of another independent ordering defect.

If the order is valid:
- dependency_violations must be [];
- suggested_order must reproduce the current step ID order.

If the order is invalid, suggested_order must contain every step ID exactly
once in a corrected order. Every change in suggested_order must be supported
by at least one dependency_violations entry. Never return an empty violations
list together with a reordered suggested_order.

GUARDRAILS AGAINST METRIC OVERLAP
Do NOT lower Logical Order because:
- a necessary step is missing; that belongs to Completeness;
- a step is vague or difficult to execute; that belongs to Actionability;
- a claim lacks contextual support; that belongs to Grounding;
- two steps overlap; that belongs to Step Distinctness;
- framing, closure, granularity, or navigation is weak; that belongs to
  Structure Quality;
- the JSON contract is invalid; that belongs to Schema Validity.

An incomplete, vague, unsupported, or repetitive roadmap can still have a
correct relative order among the steps that are present.
Do not require an additional validation, monitoring, review, or completion
step. If testing happens after the result exists and release happens after
testing, validation timing is correct. Missing additional coverage belongs to
Completeness, not Logical Order.

CRITERION AND SCORE CONSISTENCY
Before returning JSON, verify all of the following:
- score 8.0-10.0 means no meaningful dependency violation remains and
  suggested_order is identical to the current order;
- score 5.0-7.9 means one isolated or moderate violation exists and the
  suggested order repairs it locally;
- score 0.0-4.9 means multiple independent root defects or a broadly reversed
  workflow requires at least two independent movements;
- every reported predecessor is currently after its dependent step and appears
  before that dependent step in suggested_order;
- dependency_type uses exactly one of the allowed values;
- a prerequisite violation prevents prerequisite_order from scoring 8.0 or
  higher;
- a causal-output violation prevents causal_dependency_order from scoring 8.0
  or higher;
- a learning-progression violation prevents learning_progression from scoring
  8.0 or higher;
- a validation-timing violation prevents validation_timing from scoring 8.0 or
  higher;
- multiple independent root defects or a high-severity structural failure
  require a final score below 5.0;
- the overall reason, criterion scores, violations, and suggested order support
  the same score band.
Do not return JSON until these conditions are internally consistent.

Expected JSON schema:
{{
  "logical_order": {{
    "dependency_violations": [
      {{
        "dependent_step_id": "<exact step ID>",
        "required_predecessor_step_id": "<exact step ID>",
        "dependency_type": "prerequisite|causal_output|learning_progression|validation_timing",
        "severity": "low|medium|high",
        "explanation": "<why the current order is invalid>",
        "suggested_fix": "<specific reordering recommendation>"
      }}
    ],
    "suggested_order": ["<step ID>", "<step ID>"],
    "criteria": {{
      "prerequisite_order": <0-10>,
      "causal_dependency_order": <0-10>,
      "learning_progression": <0-10>,
      "validation_timing": <0-10>
    }},
    "criteria_rationales": {{
      "prerequisite_order": "<case-specific rationale>",
      "causal_dependency_order": "<case-specific rationale>",
      "learning_progression": "<case-specific rationale>",
      "validation_timing": "<case-specific rationale>"
    }},
    "score": <0-10, choose only after completing all fields above>,
    "reason": "<short explanation supporting the final score band>",
    "recommendations": ["<recommendation>", "..."]
  }}
}}
"""


_CRITERIA = (
    "prerequisite_order",
    "causal_dependency_order",
    "learning_progression",
    "validation_timing",
)
_DEPENDENCY_TYPES = {
    "prerequisite",
    "causal_output",
    "learning_progression",
    "validation_timing",
}
_DEPENDENCY_TYPE_ALIASES = {
    "prerequisite_order": "prerequisite",
    "causal_dependency_order": "causal_output",
    "learning_order": "learning_progression",
    "validation_order": "validation_timing",
}
_SEVERITIES = {"low", "medium", "high"}


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
        raise ValueError("Logical order response must be a JSON object")
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


def _roadmap_step_ids(roadmap: dict[str, Any]) -> list[str]:
    steps = roadmap.get("steps", [])
    if not isinstance(steps, list):
        return []
    return [
        str(step.get("id", "")).strip()
        for step in steps
        if isinstance(step, dict) and str(step.get("id", "")).strip()
    ]


def _fallback_result(error: Exception | None = None) -> dict[str, Any]:
    reason = "Logical order judge failed; manual review is required."
    if error:
        reason = f"{reason} Error: {type(error).__name__}"
    return {
        "logical_order": {
            "score": 5.0,
            "verdict": "NEEDS_REVIEW",
            "score_band": "5.0-7.9",
            "reason": reason,
            "criteria": {name: 5.0 for name in _CRITERIA},
            "criteria_rationales": {name: reason for name in _CRITERIA},
            "has_valid_sequence": False,
            "dependency_violations": [],
            "dependency_violation_count": 0,
            "suggested_order": [],
            "issues": [],
            "recommendations": ["Review logical order manually."],
            "normalization_notes": [],
            "manual_review_required": True,
        }
    }


def _normalize_result(
    result: dict[str, Any],
    roadmap: dict[str, Any],
) -> dict[str, Any]:
    raw = result.get("logical_order") if isinstance(result.get("logical_order"), dict) else {}
    raw_criteria = raw.get("criteria") if isinstance(raw.get("criteria"), dict) else {}
    raw_rationales = (
        raw.get("criteria_rationales")
        if isinstance(raw.get("criteria_rationales"), dict)
        else {}
    )
    criteria = {name: _normalize_score(raw_criteria.get(name, 0)) for name in _CRITERIA}
    criteria_rationales = {
        name: str(raw_rationales.get(name, "")) for name in _CRITERIA
    }

    step_ids = _roadmap_step_ids(roadmap)
    known_ids = set(step_ids)
    positions = {step_id: idx for idx, step_id in enumerate(step_ids)}
    notes: list[str] = []
    manual_review_required = False
    raw_suggested_order = raw.get("suggested_order")
    suggested_order = (
        [str(value).strip() for value in raw_suggested_order]
        if isinstance(raw_suggested_order, list)
        else []
    )
    if len(suggested_order) != len(step_ids) or set(suggested_order) != known_ids:
        suggested_order = list(step_ids)
        manual_review_required = True
        notes.append(
            "Replaced an incomplete or invalid suggested order with the current step order."
        )

    suggested_positions = {
        step_id: idx for idx, step_id in enumerate(suggested_order)
    }
    violations: list[dict[str, Any]] = []
    raw_violations = (
        raw.get("dependency_violations")
        if isinstance(raw.get("dependency_violations"), list)
        else []
    )
    for item in raw_violations:
        if not isinstance(item, dict):
            notes.append("Ignored a non-object dependency violation.")
            manual_review_required = True
            continue
        dependent_id = str(item.get("dependent_step_id", "")).strip()
        predecessor_id = str(item.get("required_predecessor_step_id", "")).strip()
        dependency_type = str(item.get("dependency_type", "prerequisite")).lower().strip()
        dependency_type = _DEPENDENCY_TYPE_ALIASES.get(dependency_type, dependency_type)
        severity = str(item.get("severity", "medium")).lower().strip()
        if dependency_type not in _DEPENDENCY_TYPES:
            notes.append("Discarded a dependency violation with an unknown dependency type.")
            manual_review_required = True
            continue
        if severity not in _SEVERITIES:
            notes.append("Discarded a dependency violation with an unknown severity.")
            manual_review_required = True
            continue
        if dependent_id not in known_ids or predecessor_id not in known_ids:
            notes.append(
                "Discarded a dependency violation with an unknown step ID."
            )
            manual_review_required = True
            continue
        if positions[predecessor_id] <= positions[dependent_id]:
            notes.append(
                "Discarded a dependency violation whose predecessor is not after the dependent step."
            )
            manual_review_required = True
            continue
        if suggested_positions[predecessor_id] >= suggested_positions[dependent_id]:
            notes.append(
                "The suggested order does not correct a reported dependency violation."
            )
            manual_review_required = True
        explanation = str(item.get("explanation", "")).strip()
        suggested_fix = str(item.get("suggested_fix", "")).strip()
        if not explanation or not suggested_fix:
            notes.append("A dependency violation is missing its explanation or suggested fix.")
            manual_review_required = True
        violations.append({
            "dependent_step_id": dependent_id,
            "required_predecessor_step_id": predecessor_id,
            "dependency_type": dependency_type,
            "severity": severity,
            "explanation": explanation,
            "suggested_fix": suggested_fix,
        })

    if suggested_order != step_ids and not violations:
        notes.append(
            "The judge changed the suggested order without reporting a valid dependency violation."
        )
        manual_review_required = True
    if suggested_order == step_ids and violations:
        notes.append("The judge reported violations but preserved the current order.")
        manual_review_required = True

    score = _normalize_score(raw.get("score", 0))
    severe_or_multiple = (
        any(item["severity"] == "high" for item in violations)
        or len(violations) >= 2
    )
    if not violations and score < 8.0:
        notes.append("The score is below 8.0 even though no valid dependency violations remain.")
        manual_review_required = True
    elif violations and score >= 8.0:
        notes.append("The score is 8.0 or higher despite reported dependency violations.")
        manual_review_required = True
    elif severe_or_multiple and score >= 5.0:
        notes.append("The score is 5.0 or higher despite high-severity or multiple violations.")
        manual_review_required = True

    missing_rationales = [
        name for name, rationale in criteria_rationales.items() if not rationale.strip()
    ]
    if missing_rationales:
        notes.append(f"Missing criterion rationales: {', '.join(missing_rationales)}.")
        manual_review_required = True

    issues = [
        {
            "type": violation["dependency_type"],
            "severity": violation["severity"],
            "explanation": violation["explanation"],
        }
        for violation in violations
    ]
    recommendations = (
        [str(item) for item in raw.get("recommendations", [])]
        if isinstance(raw.get("recommendations"), list)
        else []
    )
    reason = str(raw.get("reason", "")).strip()
    if not reason:
        notes.append("The judge did not provide an overall reason.")
        manual_review_required = True

    return {
        "logical_order": {
            "score": score,
            "verdict": _verdict(score),
            "score_band": _score_band(score),
            "reason": reason,
            "criteria": criteria,
            "criteria_rationales": criteria_rationales,
            "has_valid_sequence": not violations,
            "dependency_violations": violations,
            "dependency_violation_count": len(violations),
            "suggested_order": suggested_order,
            "issues": issues,
            "recommendations": recommendations,
            "normalization_notes": notes,
            "manual_review_required": manual_review_required,
        }
    }


class LogicalOrderJudge:
    """Evaluate roadmap dependency order with a dedicated LLM judge."""

    def __init__(self, llm=None):
        self.llm = llm or get_judge_llm(temperature=0.0, max_tokens=2048)

    def _invoke(self, prompt: str, attempt: int = 1) -> str:
        return invoke_llm_text(
            self.llm,
            prompt,
            operation="Logical Order",
            attempt=attempt,
        )

    def evaluate(
        self,
        roadmap: dict[str, Any],
        question: str = "",
        category: str = "",
    ) -> dict[str, Any]:
        steps = roadmap.get("steps", [])
        current_order_lines = [
            (
                f"position={position} | id={str(step.get('id', '')).strip()} "
                f"| label={str(step.get('label', '')).strip()}"
            )
            for position, step in enumerate(steps, 1)
            if isinstance(step, dict)
        ] if isinstance(steps, list) else []
        prompt = _PROMPT_TEMPLATE.format(
            question=question or "(not provided)",
            category=category or "(not provided)",
            roadmap=json.dumps(roadmap, ensure_ascii=False, indent=2)[:6000],
            current_order="\n".join(current_order_lines) or "(no steps)",
        )
        try:
            raw = self._invoke(prompt, attempt=1)
            retried = False
            retry_reason = ""
            try:
                parsed = _parse_json_response(raw)
            except (json.JSONDecodeError, ValueError):
                retry_reason = "the previous response was not valid JSON"
                print(
                    "      [LLM] Logical Order returned invalid JSON; "
                    "starting one corrective retry.",
                    flush=True,
                )
                retry_prompt = (
                    f"{prompt}\n\nRETRY REQUIREMENT\n"
                    "The previous response was not valid JSON. Return the complete JSON object only."
                )
                retry_raw = self._invoke(retry_prompt, attempt=2)
                parsed = _parse_json_response(retry_raw)
                retried = True
            normalized = _normalize_result(parsed, roadmap)
            if (
                not retried
                and normalized["logical_order"].get("manual_review_required")
            ):
                notes = normalized["logical_order"].get("normalization_notes", [])
                retry_reason = "; ".join(str(note) for note in notes)
                print(
                    "      [LLM] Logical Order contract inconsistent; "
                    "starting one corrective retry.",
                    flush=True,
                )
                retry_prompt = (
                    f"{prompt}\n\nRETRY REQUIREMENT\n"
                    "The previous JSON violated the evaluation contract: "
                    f"{retry_reason}. Re-evaluate the CURRENT STEP POSITIONS, "
                    "ensure every predecessor is currently after its dependent "
                    "step, and make score, violations, and suggested_order "
                    "consistent. Return the complete JSON object only."
                )
                parsed = _parse_json_response(
                    self._invoke(retry_prompt, attempt=2)
                )
                normalized = _normalize_result(parsed, roadmap)
                retried = True
            if retried:
                normalized["logical_order"]["normalization_notes"].append(
                    f"Retried once after an incomplete response: {retry_reason}."
                )
            return normalized
        except Exception as exc:
            return _fallback_result(exc)
