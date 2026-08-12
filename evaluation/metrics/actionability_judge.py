"""Dedicated semantic judge for roadmap Actionability.

Actionability evaluates whether the steps that are present can be understood
and executed. It does not evaluate factual correctness, contextual support,
coverage, order, distinctness, or global roadmap structure.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.llm_provider import get_judge_llm, invoke_llm_text


_CRITERIA = (
    "identifiable_action",
    "execution_method",
    "starting_point",
    "expected_result",
    "execution_specificity",
)
_CRITERION_STATES = {"met", "partial", "missing", "not_applicable"}
_MISSING_ELEMENTS = {
    "action",
    "method",
    "input",
    "output",
    "specificity",
}
_MISSING_ELEMENT_ALIASES = {
    "tool": "method",
    "procedure": "method",
    "mechanism": "method",
    "source": "input",
    "starting_point": "input",
    "prerequisite": "input",
    "result": "output",
    "expected_result": "output",
    "validation": "output",
    "validation_criteria": "output",
    "verification": "output",
    "verification_criteria": "output",
    "acceptance_criteria": "output",
    "completion_criteria": "output",
    "done_criteria": "output",
    "success_criteria": "output",
    "clarity": "specificity",
    "detail": "specificity",
    "execution_specificity": "specificity",
}


_PROMPT_TEMPLATE = """\
You are an expert evaluator of technical learning and execution roadmaps.
Return ONLY valid JSON. Do not use markdown or extra text.

USER QUESTION
{question}

CATEGORY
{category}

ROADMAP
{roadmap}

OBJECTIVE
Evaluate only Actionability: whether a user can understand and execute each
existing step with sufficiently clear direction.

Read each step's label and description together. Evaluate each step according
to its role in the roadmap, not with a rigid checklist that requires every
field to be stated literally.

ACTIONABILITY CRITERIA
1. Identifiable action
   - The user can tell what they must do, decide, create, inspect, practice,
     configure, compare, or validate.
2. Execution method
   - The step provides a usable method, tool, concept, technique, or procedure
     when one is materially needed.
3. Starting point
   - Required input, artifact, source, prerequisite state, or object of work is
     identifiable when execution depends on one.
4. Expected result
   - The user can recognize the output, decision, validation, learning result,
     or done condition when one is materially needed.
5. Execution specificity
   - The combined label and description remove enough ambiguity to act without
     inventing the main procedure.

ROLE-AWARE EVALUATION
- A framing step may produce a scope, decision, requirement, or success target.
- A learning step may use a concept, example, exercise, comparison, or
  explanation as its method and understanding as its result.
- A build step usually needs an object, method, and expected artifact.
- A validation step usually needs an object to inspect and a success signal.
- A short step can be highly actionable when its domain action and result are
  unambiguous.
- Do not require click-by-click UI navigation when the step already names the
  application, menu, node, field, or object needed to perform the action.
- Assume the intended user has normal access to named files, workspaces,
  connections, tables, and tools, plus basic proficiency with the named
  technology. Access setup is a material blocker only when the step itself is
  about obtaining or configuring that access.
- Use "not_applicable" when a criterion is genuinely unnecessary for that
  step's role. Do not treat "not_applicable" as a weakness.
- Evaluate what the individual step is responsible for. Do not require an
  early planning or definition step to contain implementation formulas,
  configuration details, or validation procedures that later steps explicitly
  provide.
- A definition step such as "list the measures, periods, and required output"
  is actionable when that list is its intended result. Its method may be
  not_applicable; it does not need to implement the measures itself.
- For example, "write down the business measure, comparison periods, and
  required output" is already an executable requirements-definition step. Do
  not require DAX formulas, example values, or implementation syntax in that
  step when later steps own implementation.

SCORE BANDS
- 8.0-10.0 — actionable:
  The roadmap can be executed with little interpretation. Steps identify
  concrete actions and provide the method, starting point, result, or
  specificity that their roles materially require. Minor optional detail may
  be absent.
- 5.0-7.9 — needs clarification:
  The roadmap is usable, but multiple steps need local clarification or one
  important step omits a material method, input, result, or execution detail.
- 0.0-4.9 — not actionable as-is:
  Many steps or core steps are abstract, generic, or underspecified. The user
  must invent substantial parts of the procedure before acting.

The overall score is NOT an arithmetic average of the per-step or criterion
scores. Select its band from the practical clarification burden across the
roadmap, then choose a value naturally inside that band.

CLARIFICATION BURDEN
- Optional examples, extra safeguards, richer done criteria, or implementation
  detail owned by a later step do not make an otherwise executable step weak.
- A local clarification means the main action and procedure are already
  identifiable; the user only needs a bounded detail to proceed confidently.
- A step is not actionable when phrases such as "learn what is important",
  "review carefully", "apply suitable improvements", "use best practices",
  "make it better", or "complete the work" require the user to invent the main
  method and expected result.
- When many or core steps require the user to invent the main procedure, use
  the 0.0-4.9 band. Do not classify pervasive underspecification as a series of
  harmless local clarifications.
- Concrete-sounding labels do not make a roadmap executable by themselves.
  Labels such as "receive the request", "evaluate the request", "handle the
  decision", "notify stakeholders", or "monitor the workflow" only name
  desired functions. If their descriptions do not identify the mechanism,
  configuration, decision criteria, inputs, or observable result needed to
  implement them, they are not actionable as-is.
- If most or all steps name an intended function but omit the main method and
  execution specificity, use the 0.0-4.9 overall band even when the labels use
  action verbs.
- Validation, review, and finalization steps require an observable check,
  comparison, acceptance condition, or done signal. Phrases such as "inspect
  the data", "check that it works", or "finish when it looks acceptable" are
  materially ambiguous when no verification method or success condition is
  provided.
- When all steps are executable and at most one non-core step lacks only
  optional detail, use the 8.0-10.0 band.
- For a short roadmap of roughly three to eight steps, two or more materially
  weak steps normally require the 5.0-7.9 band even when the remaining build
  steps are strong.

PER-STEP SCORING
- 8.0-10.0 / ACTIONABLE: executable with little interpretation.
- 5.0-7.9 / NEEDS_CLARIFICATION: locally usable but materially ambiguous.
- 0.0-4.9 / NOT_ACTIONABLE: cannot be executed without inventing the main
  action, method, starting point, or expected result.

Use the full per-step scale:
- Score below 8.0 only when an ambiguity is material enough to prevent,
  significantly delay, or change execution. Do not lower a step merely because
  it could include a richer tutorial.
- UI click paths, examples, edge-case handling, and exhaustive item lists are
  optional when the application, operation, target object, and selection rule
  are already identifiable.
- Naming a menu and target columns is sufficient without explaining how to
  navigate the interface.
- Naming transformations and their target column is sufficient without example
  values when the operation is standard and unambiguous.
- A selection rule such as "columns with null values" is a valid starting point
  and does not require enumerating every matching column.
- "Select Publish, choose the named workspace, and confirm the report appears"
  is actionable without a tutorial for the Publish dialog.
- "Compare the displayed totals with the named approved source" is actionable
  without explaining how to obtain a source the step already identifies.
- "Create the named relationships and confirm they are active" is actionable
  without a tutorial for opening the relationship properties.
- "Use Get Data to load named tables from the specified connection" is
  actionable without restating standard connection-dialog steps.
- If both the main method and expected result must be invented, the step
  normally belongs below 5.0.
- If the action, method, and result are identifiable and only optional detail
  is absent, the step normally belongs at 8.0 or higher.

MECE BOUNDARIES
Do NOT lower Actionability because:
- an instruction is factually wrong or unsafe; that belongs to Accuracy;
- an instruction is unsupported by retrieved context; that belongs to
  Grounding;
- a necessary step or topic is missing; that belongs to Completeness;
- steps are in the wrong order; that belongs to Logical Order;
- two executable steps duplicate work; that belongs to Step Distinctness and
  Step Overlap;
- framing, closure, granularity, navigation, or roadmap length is weak; that
  belongs to Structure Quality.

A roadmap may be incomplete, unsupported, factually wrong, redundant, or
misordered and still receive a high Actionability score if every existing step
is clearly executable.

ORDER-ISOLATION INVARIANT
Evaluate every step as an independent instruction and assume its prerequisites
and normal tool access are available. Do not ask whether an input has already
been created at the step's current position. That is Logical Order.

For starting_point, check whether the instruction identifies the object of
work, source, artifact, or connection. A named table, column, measure, file,
workspace, connection, endpoint, or visual is an identifiable starting point
even if another roadmap step creates it later.

Examples of explicit methods:
- a complete DAX expression is a method; do not require instructions for where
  to type the formula when measure creation is already named;
- named relationship field pairs plus an instruction to activate or confirm
  them are a method;
- "Use Get Data to load named tables from the specified connection" is a
  method;
- creating a table with named fields and comparing its totals with a named
  source is a validation method.
- an HTTP Request node with method, endpoint, and attached credential is an
  executable request method; node-menu navigation, header examples, and the
  internal credential scheme are optional unless authentication setup is the
  step's main responsibility;
- an Edit Fields or mapping step that names the source and destination fields
  is executable without sample values or nested-payload edge cases;
- a Google Sheets append step with operation, target sheet, and field mapping
  is executable without file-permission setup instructions;
- running a workflow with a small response and confirming expected values in
  the destination is an executable validation method; richer error handling is
  optional.

Do not raise Actionability because the roadmap mentions relevant topics,
necessary areas, expected phases, or a complete-looking outline. Topic and
phase coverage belong to Completeness. A comprehensive list of vague stages is
still not actionable.

MATERIAL BLOCKERS VS OPTIONAL IMPROVEMENTS
For every step, separate:
- material_missing_elements: absent information that prevents or substantially
  changes execution of the step's main responsibility;
- optional_improvements: useful tutorials, examples, edge cases, safeguards,
  richer checks, or UI detail that would improve the instruction but is not
  needed to execute its main responsibility.

Only material_missing_elements may lower the per-step score below 8.0.
Optional improvements must not lower the score or create a weak step.
Do not describe the same absent detail as both a material blocker and an
optional improvement. If execution can proceed without it, it is optional and
must not appear in material_missing_elements.

CONSISTENCY REQUIREMENTS
- Return exactly one step_actionability entry for every roadmap step.
- Use exact roadmap step IDs.
- A per-step score below 8.0 must explain the material ambiguity and include at
  least one material_missing_element.
- A per-step score of 8.0 or higher must use an empty
  material_missing_elements list.
- Use only these material_missing_elements:
  action | method | input | output | specificity.
- Criterion states must be:
  met | partial | missing | not_applicable.
- Judge criteria independently. Do not mark all five criteria missing because
  one phrase is vague.
- Ignore the current sequence when judging starting_point and execution_method.
- Roadmap-level criterion scores must reflect the per-step evidence. A
  criterion that is met or not applicable in nearly every step should normally
  score 8.0 or higher. A criterion missing from most steps should normally
  score below 5.0.
- If every step scores below 8.0 and most steps require the user to invent the
  method and execution specificity, the overall score must be in 0.0-4.9.
- If all five roadmap-level criterion scores are below 5.0, the overall score
  must also be below 5.0.
- The overall score, overall reason, per-step diagnostics, criterion scores,
  strengths, and recommendations must describe the same evaluation.

Expected JSON schema:
{{
  "actionability": {{
    "step_actionability": [
      {{
        "step_id": "<exact step ID>",
        "score": <0-10>,
        "criteria": {{
          "identifiable_action": "met|partial|missing|not_applicable",
          "execution_method": "met|partial|missing|not_applicable",
          "starting_point": "met|partial|missing|not_applicable",
          "expected_result": "met|partial|missing|not_applicable",
          "execution_specificity": "met|partial|missing|not_applicable"
        }},
        "reason": "<case-specific explanation>",
        "material_missing_elements": ["action|method|input|output|specificity", "..."],
        "optional_improvements": ["<non-blocking improvement>", "..."],
        "recommendation": "<local clarification or empty string>"
      }}
    ],
    "criteria": {{
      "identifiable_action": <0-10>,
      "execution_method": <0-10>,
      "starting_point": <0-10>,
      "expected_result": <0-10>,
      "execution_specificity": <0-10>
    }},
    "criteria_rationales": {{
      "identifiable_action": "<roadmap-level rationale>",
      "execution_method": "<roadmap-level rationale>",
      "starting_point": "<roadmap-level rationale>",
      "expected_result": "<roadmap-level rationale>",
      "execution_specificity": "<roadmap-level rationale>"
    }},
    "score": <0-10>,
    "reason": "<short explanation supporting the score band>",
    "strengths": ["<strength>", "..."],
    "recommendations": ["<recommendation>", "..."]
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
        raise ValueError("Actionability response must be a JSON object")
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


def _step_status(score: float) -> str:
    if score >= 8.0:
        return "ACTIONABLE"
    if score >= 5.0:
        return "NEEDS_CLARIFICATION"
    return "NOT_ACTIONABLE"


def _fallback_step(step: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "step_id": str(step.get("id", "")).strip(),
        "label": str(step.get("label", "")).strip(),
        "score": 5.0,
        "status": "NEEDS_CLARIFICATION",
        "criteria": {name: "partial" for name in _CRITERIA},
        "reason": reason,
        "missing_elements": ["specificity"],
        "optional_improvements": [],
        "recommendation": "Review this step manually.",
    }


def _fallback_result(
    roadmap: dict[str, Any],
    error: Exception | None = None,
) -> dict[str, Any]:
    reason = "Actionability judge failed; manual review is required."
    if error:
        reason = f"{reason} Error: {type(error).__name__}"
    steps = [_fallback_step(step, reason) for step in _roadmap_steps(roadmap)]
    return {
        "actionability": {
            "score": 5.0,
            "verdict": "NEEDS_REVIEW",
            "score_band": "5.0-7.9",
            "reason": reason,
            "criteria": {name: 5.0 for name in _CRITERIA},
            "criteria_rationales": {name: reason for name in _CRITERIA},
            "step_actionability": steps,
            "actionability_density": 0.0,
            "actionable_step_count": 0,
            "weak_action_steps": steps,
            "weak_action_step_count": len(steps),
            "issues": [],
            "strengths": [],
            "recommendations": ["Review Actionability manually."],
            "normalization_notes": [],
            "manual_review_required": True,
        }
    }


def _resolve_step_id(
    raw_step_id: Any,
    step_by_id: dict[str, dict[str, Any]],
) -> str | None:
    """Resolve unambiguous formatting variants without changing judge content."""
    value = str(raw_step_id or "").strip()
    if value in step_by_id:
        return value

    lowered = value.casefold()
    by_label = {
        str(step.get("label", "")).strip().casefold(): step_id
        for step_id, step in step_by_id.items()
        if str(step.get("label", "")).strip()
    }
    if lowered in by_label:
        return by_label[lowered]

    match = re.fullmatch(r"(?:step[\s_-]*)?(\d+)", lowered)
    if match:
        position = int(match.group(1))
        ids = list(step_by_id)
        if 1 <= position <= len(ids):
            return ids[position - 1]
    return None


def _step_contract_issues(
    result: dict[str, Any],
    roadmap: dict[str, Any],
) -> list[str]:
    """Identify incomplete per-step coverage before normalization can mask it."""
    raw = result.get("actionability")
    if not isinstance(raw, dict):
        return ["missing actionability object"]
    entries = raw.get("step_actionability")
    if not isinstance(entries, list):
        return ["missing step_actionability list"]

    steps = _roadmap_steps(roadmap)
    step_by_id = {
        str(step.get("id", "")).strip(): step
        for step in steps
        if str(step.get("id", "")).strip()
    }
    resolved: list[str] = []
    unknown: list[str] = []
    for item in entries:
        if not isinstance(item, dict):
            unknown.append("<non-object>")
            continue
        raw_id = str(item.get("step_id", "")).strip()
        step_id = _resolve_step_id(raw_id, step_by_id)
        if step_id is None:
            unknown.append(raw_id or "<empty>")
        else:
            resolved.append(step_id)

    expected = list(step_by_id)
    missing = [step_id for step_id in expected if step_id not in resolved]
    duplicates = sorted({step_id for step_id in resolved if resolved.count(step_id) > 1})
    issues = []
    if missing:
        issues.append(f"missing step IDs: {', '.join(missing)}")
    if duplicates:
        issues.append(f"duplicate step IDs: {', '.join(duplicates)}")
    if unknown:
        issues.append(f"unknown step IDs: {', '.join(unknown)}")
    return issues


def _missing_step_ids(
    result: dict[str, Any],
    roadmap: dict[str, Any],
) -> list[str]:
    raw = result.get("actionability")
    entries = raw.get("step_actionability", []) if isinstance(raw, dict) else []
    steps = _roadmap_steps(roadmap)
    step_by_id = {
        str(step.get("id", "")).strip(): step
        for step in steps
        if str(step.get("id", "")).strip()
    }
    resolved = {
        step_id
        for item in entries if isinstance(item, dict)
        if (step_id := _resolve_step_id(item.get("step_id"), step_by_id))
    }
    return [step_id for step_id in step_by_id if step_id not in resolved]


def _missing_steps_retry_prompt(
    roadmap: dict[str, Any],
    missing_ids: list[str],
) -> str:
    missing_steps = [
        step for step in _roadmap_steps(roadmap)
        if str(step.get("id", "")).strip() in set(missing_ids)
    ]
    return f"""\
You are completing an Actionability evaluation whose roadmap-level result was
already produced. Return ONLY valid JSON and evaluate exactly these missing
steps: {missing_ids}.

For each step, assess identifiable action, execution method, starting point,
expected result, and execution specificity. Use score 8.0-10.0 when executable
with little interpretation, 5.0-7.9 when local clarification is needed, and
0.0-4.9 when the main procedure must be invented. Optional detail must not
lower the score. Use exact step IDs.

MISSING ROADMAP STEPS
{json.dumps(missing_steps, ensure_ascii=False, indent=2)}

Return exactly:
{{
  "actionability": {{
    "step_actionability": [
      {{
        "step_id": "<exact missing ID>",
        "score": <0-10>,
        "criteria": {{
          "identifiable_action": "met|partial|missing|not_applicable",
          "execution_method": "met|partial|missing|not_applicable",
          "starting_point": "met|partial|missing|not_applicable",
          "expected_result": "met|partial|missing|not_applicable",
          "execution_specificity": "met|partial|missing|not_applicable"
        }},
        "reason": "<short case-specific reason>",
        "material_missing_elements": ["action|method|input|output|specificity"],
        "optional_improvements": [],
        "recommendation": "<local recommendation or empty string>"
      }}
    ]
  }}
}}
"""


def _normalize_result(
    result: dict[str, Any],
    roadmap: dict[str, Any],
) -> dict[str, Any]:
    raw = (
        result.get("actionability")
        if isinstance(result.get("actionability"), dict)
        else {}
    )
    steps = _roadmap_steps(roadmap)
    step_by_id = {
        str(step.get("id", "")).strip(): step
        for step in steps
        if str(step.get("id", "")).strip()
    }
    notes: list[str] = []
    manual_review_required = False

    raw_entries = (
        raw.get("step_actionability")
        if isinstance(raw.get("step_actionability"), list)
        else []
    )
    entries_by_id: dict[str, dict[str, Any]] = {}
    for item in raw_entries:
        if not isinstance(item, dict):
            notes.append("Ignored a non-object step Actionability entry.")
            manual_review_required = True
            continue
        raw_step_id = str(item.get("step_id", "")).strip()
        step_id = _resolve_step_id(raw_step_id, step_by_id)
        if step_id is None:
            notes.append(
                f"Discarded a step Actionability entry with unknown step ID '{raw_step_id}'."
            )
            manual_review_required = True
            continue
        if step_id != raw_step_id:
            notes.append(f"Mapped step ID '{raw_step_id}' to '{step_id}'.")
        if step_id in entries_by_id:
            notes.append("Discarded a duplicate step Actionability entry.")
            manual_review_required = True
            continue
        entries_by_id[step_id] = item

    normalized_steps: list[dict[str, Any]] = []
    for step_id, step in step_by_id.items():
        item = entries_by_id.get(step_id)
        if item is None:
            normalized_steps.append(
                _fallback_step(
                    step,
                    "The judge did not return an Actionability evaluation for this step.",
                )
            )
            notes.append(f"Added a manual-review fallback for missing step {step_id}.")
            manual_review_required = True
            continue

        score = _normalize_score(item.get("score", 0))
        raw_criteria = (
            item.get("criteria") if isinstance(item.get("criteria"), dict) else {}
        )
        criteria: dict[str, str] = {}
        for name in _CRITERIA:
            state = str(raw_criteria.get(name, "")).lower().strip()
            if state not in _CRITERION_STATES:
                state = "partial"
                notes.append(
                    f"Normalized an invalid {name} state for step {step_id} to partial."
                )
                manual_review_required = True
            criteria[name] = state

        raw_missing_value = item.get("material_missing_elements")
        if not isinstance(raw_missing_value, list):
            raw_missing_value = item.get("missing_elements")
        raw_missing = raw_missing_value if isinstance(raw_missing_value, list) else []
        missing_elements = []
        for value in raw_missing:
            normalized = (
                str(value)
                .lower()
                .strip()
                .replace(" ", "_")
                .replace("-", "_")
            )
            normalized = _MISSING_ELEMENT_ALIASES.get(normalized, normalized)
            if normalized in _MISSING_ELEMENTS and normalized not in missing_elements:
                missing_elements.append(normalized)
            elif normalized not in _MISSING_ELEMENTS:
                notes.append(
                    f"Discarded unknown missing element '{normalized}' for step {step_id}."
                )
                manual_review_required = True

        if score >= 8.0 and missing_elements:
            notes.append(
                f"Removed missing elements from actionable step {step_id}."
            )
            missing_elements = []
        elif score < 8.0 and not missing_elements:
            missing_elements = ["specificity"]
            notes.append(
                f"Added a fallback missing element for weak step {step_id}."
            )
            manual_review_required = True

        reason = str(item.get("reason", "")).strip()
        recommendation = str(item.get("recommendation", "")).strip()
        optional_improvements = (
            [str(value).strip() for value in item.get("optional_improvements", []) if str(value).strip()]
            if isinstance(item.get("optional_improvements"), list)
            else []
        )
        if not reason:
            reason = "The judge did not provide a step-specific reason."
            notes.append(f"Missing Actionability reason for step {step_id}.")
            manual_review_required = True

        normalized_steps.append({
            "step_id": step_id,
            "label": str(step.get("label", "")).strip(),
            "score": score,
            "status": _step_status(score),
            "criteria": criteria,
            "reason": reason,
            "missing_elements": missing_elements,
            "optional_improvements": optional_improvements,
            "recommendation": recommendation,
        })

    weak_steps = [step for step in normalized_steps if step["score"] < 8.0]
    actionable_count = len(normalized_steps) - len(weak_steps)
    density = (
        round(actionable_count / len(normalized_steps), 4)
        if normalized_steps
        else 0.0
    )

    raw_criteria = raw.get("criteria") if isinstance(raw.get("criteria"), dict) else {}
    raw_rationales = (
        raw.get("criteria_rationales")
        if isinstance(raw.get("criteria_rationales"), dict)
        else {}
    )
    criteria = {name: _normalize_score(raw_criteria.get(name, 0)) for name in _CRITERIA}
    criteria_rationales = {
        name: str(raw_rationales.get(name, "")).strip() for name in _CRITERIA
    }
    missing_rationales = [
        name for name, rationale in criteria_rationales.items() if not rationale
    ]
    if missing_rationales:
        notes.append(
            f"Missing criterion rationales: {', '.join(missing_rationales)}."
        )
        manual_review_required = True

    score = _normalize_score(raw.get("score", 0))
    if not weak_steps and score < 8.0:
        notes.append(
            "The overall score is below 8.0 even though every step is actionable."
        )
        manual_review_required = True
    elif (
        score >= 8.0
        and weak_steps
        and (
            len(weak_steps) > 1
            or any(step["score"] < 5.0 for step in weak_steps)
        )
    ):
        notes.append(
            "The overall score is 8.0 or higher despite substantial weak Actionability coverage."
        )
        manual_review_required = True
    if normalized_steps and all(step["score"] < 5.0 for step in normalized_steps):
        if score >= 5.0:
            notes.append(
                "The overall score is 5.0 or higher although every step is not actionable."
            )
            manual_review_required = True

    reason = str(raw.get("reason", "")).strip()
    if not reason:
        reason = "The judge did not provide an overall Actionability reason."
        notes.append("Missing overall Actionability reason.")
        manual_review_required = True

    issues = [
        {
            "step_id": step["step_id"],
            "label": step["label"],
            "severity": "high" if step["score"] < 5.0 else "medium",
            "missing_elements": step["missing_elements"],
            "explanation": step["reason"],
        }
        for step in weak_steps
    ]
    strengths = (
        [str(item) for item in raw.get("strengths", [])]
        if isinstance(raw.get("strengths"), list)
        else []
    )
    recommendations = (
        [str(item) for item in raw.get("recommendations", [])]
        if isinstance(raw.get("recommendations"), list)
        else []
    )

    return {
        "actionability": {
            "score": score,
            "verdict": _verdict(score),
            "score_band": _score_band(score),
            "reason": reason,
            "criteria": criteria,
            "criteria_rationales": criteria_rationales,
            "step_actionability": normalized_steps,
            "actionability_density": density,
            "actionable_step_count": actionable_count,
            "weak_action_steps": weak_steps,
            "weak_action_step_count": len(weak_steps),
            "issues": issues,
            "strengths": strengths,
            "recommendations": recommendations,
            "normalization_notes": notes,
            "manual_review_required": manual_review_required,
        }
    }


class ActionabilityJudge:
    """Evaluate roadmap Actionability with per-step semantic diagnostics."""

    def __init__(self, llm=None):
        self.llm = llm or get_judge_llm(temperature=0.0, max_tokens=4096)

    def _invoke(self, prompt: str, attempt: int = 1) -> str:
        return invoke_llm_text(
            self.llm,
            prompt,
            operation="Actionability",
            attempt=attempt,
        )

    def evaluate(
        self,
        roadmap: dict[str, Any],
        question: str = "",
        category: str = "",
    ) -> dict[str, Any]:
        prompt = _PROMPT_TEMPLATE.format(
            question=question or "(not provided)",
            category=category or "(not provided)",
            roadmap=json.dumps(roadmap, ensure_ascii=False, indent=2)[:7000],
        )
        try:
            raw = self._invoke(prompt, attempt=1)
            retried = False
            retry_reason = ""
            try:
                parsed = _parse_json_response(raw)
            except (json.JSONDecodeError, ValueError):
                retry_reason = "the previous response was not valid JSON"
                parsed = None
            if parsed is not None:
                contract_issues = _step_contract_issues(parsed, roadmap)
                if contract_issues:
                    retry_reason = "; ".join(contract_issues)
            if retry_reason:
                expected_ids = [
                    str(step.get("id", "")).strip()
                    for step in _roadmap_steps(roadmap)
                    if str(step.get("id", "")).strip()
                ]
                print(
                    f"      [LLM] Actionability contract incomplete ({retry_reason}); "
                    "starting one corrective retry.",
                    flush=True,
                )
                missing_ids = _missing_step_ids(parsed, roadmap) if parsed else []
                targeted_retry = bool(parsed and missing_ids and all(
                    issue.startswith("missing step IDs:") for issue in contract_issues
                ))
                if targeted_retry:
                    retry_prompt = _missing_steps_retry_prompt(roadmap, missing_ids)
                    patch = _parse_json_response(self._invoke(retry_prompt, attempt=2))
                    patch_root = patch.get("actionability", {})
                    patch_entries = patch_root.get("step_actionability", [])
                    if not isinstance(patch_entries, list):
                        raise ValueError("Actionability retry did not return step entries")
                    parsed["actionability"]["step_actionability"].extend(patch_entries)
                else:
                    retry_prompt = (
                        f"{prompt}\n\nRETRY REQUIREMENT\n"
                        f"The previous response was incomplete because {retry_reason}. "
                        "Return valid JSON with exactly one step_actionability entry "
                        f"for each of these IDs, in this order: {expected_ids}. "
                        "Return the complete JSON object only."
                    )
                    parsed = _parse_json_response(self._invoke(retry_prompt, attempt=2))
                retried = True
            normalized = _normalize_result(parsed, roadmap)
            if retried:
                normalized["actionability"]["normalization_notes"].append(
                    f"Retried once after an incomplete response: {retry_reason}."
                )
            return normalized
        except Exception as exc:
            return _fallback_result(roadmap, exc)
