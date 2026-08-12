"""LLM-based semantic judge for roadmap structure quality.

This metric evaluates whether a roadmap is organized as a useful learning or
execution artifact. It is intentionally separate from the deterministic
StructureValidator, which checks schema validity.
"""

from __future__ import annotations

import json
import re
from typing import Any

from evaluation.structured_output import invoke_json_with_retry
from src.llm_provider import get_judge_llm


_PROMPT_TEMPLATE = """\
You are an expert evaluator of technical learning roadmaps.
Return ONLY valid JSON. Do not use markdown or extra text.

USER QUESTION
{question}

CATEGORY
{category}

ROADMAP JSON
{roadmap}

TASK
Evaluate STRUCTURE QUALITY only.

Structure Quality measures whether the roadmap is organized as a useful,
coherent, navigable learning or execution artifact.

Evaluate exactly these six criteria:
1. Goal framing:
   Evaluate whether the roadmap starts with a clear purpose and scope.
   - 8.0-10.0: the first step frames the goal, target user or audience, intended
     outcome, and success direction before execution starts.
   - 5.0-7.9: the goal is present but generic, partial, or missing important scope
     details.
   - 0.0-4.9: the roadmap starts mid-task or with tool usage and does not explain
     why the roadmap exists.

2. Closure quality:
   Evaluate whether the roadmap ends with a meaningful completion shape.
   - 8.0-10.0: the final step validates, applies, synthesizes, transfers, or defines
     clear done criteria for the roadmap result.
   - 5.0-7.9: the final step closes the roadmap, but with generic wording such as
     review, finish, adjust, or improve without concrete validation,
     application, synthesis, or done criteria.
   - 0.0-4.9: the roadmap stops without meaningful closure, or ends with vague
     future work such as continue improving, save the file, or stop working.

3. Granularity:
   Evaluate whether steps operate at a consistent and useful level of detail.
   - 8.0-10.0: steps have comparable scope and each step represents one clear phase
     or structural unit.
   - 5.0-7.9: the roadmap remains usable, but one or a few steps are noticeably too
     broad, too narrow, or at a different abstraction level than the rest.
   - 0.0-4.9: many steps are fragmented, mixed-level, or bundled in ways that make
     the roadmap hard to navigate.
   A step that bundles many responsibilities at once, such as objectives,
   boundaries, stakeholders, ownership, risks, governance, and success criteria,
   is too broad even if the topics are relevant.

4. Flow coherence:
   Evaluate whether the roadmap reads as one cohesive artifact.
   - 8.0-10.0: the steps have a clear structural arc, stable perspective, and
     understandable transitions between phases.
   - 5.0-7.9: the roadmap is mostly coherent, but includes abrupt jumps, mixed
     viewpoints, or weak transitions.
   - 0.0-4.9: the roadmap feels like disconnected notes or a loose checklist.
   Do not judge prerequisite correctness here; only judge whether the roadmap's
   shape is cohesive.

5. Structural usefulness:
   Evaluate whether the organization helps the user navigate and review the
   roadmap.
   - 8.0-10.0: the structure makes it clear where the user is, what each phase
     contributes, and how the roadmap reaches a usable result.
   - 5.0-7.9: the roadmap can be followed, but would benefit from clearer phase
     labels, outputs, review points, or scope boundaries.
   - 0.0-4.9: the structure is confusing enough that the user would need external
     explanation to understand how to use it.

6. Scope fit:
   Evaluate whether the roadmap length fits the complexity and scope of the
   user's question.
   First estimate the ideal number of steps for this specific question, then
   compare it with the actual number of steps.
   - 8.0-10.0: the actual step count is appropriate for the question's
     complexity. As a loose guide, it is usually within about 2 steps of the
     ideal count.
   - 5.0-7.9: the roadmap is somewhat too short or too long, but still usable.
     As a loose guide, it is usually within about 4 steps of the ideal count.
   - 0.0-4.9: the roadmap is clearly over-expanded or under-specified for the
     question. As a loose guide, it differs from the ideal by more than about 4
     steps, or the mismatch strongly harms usability.
   Use judgment: a difference of one step matters more in a short 5-step
   roadmap than in a complex 15-step roadmap. Do not apply the tolerance as a
   rigid rule.
   The scope_fit.reason field is the single source of truth for the Scope Fit
   explanation. If you report a poor_scope_fit issue, reuse that reason exactly
   instead of producing a separate explanation.

GUARDRAILS AGAINST METRIC OVERLAP
Do NOT evaluate:
- whether prerequisites are ordered correctly; that belongs to Logical Order.
- whether all topic content is covered; that belongs to Completeness.
- whether each individual step is executable; that belongs to Actionability.
- whether steps are mutually exclusive; that belongs to Step Distinctness.
- whether the JSON schema is technically valid; that belongs to Schema Validity.
- whether the roadmap has a fixed global step count; Scope Fit should judge
  length relative to the question, not a universal 6-12 rule.

You may mention these issues only when they affect the roadmap's overall shape
as a navigable artifact, but do not rescore those dimensions directly.

LANGUAGE OWNERSHIP
- Never describe a roadmap as "logically ordered", "correctly ordered", or as
  having ordered steps, valid prerequisites, or valid dependencies. Do not
  discuss step order. Those conclusions belong only to Logical Order.
- The words "logical" and "logically" are forbidden anywhere in the Structure
  Quality output. Use "cohesive", "navigable", or "clear structural arc" when
  describing shape and transitions.
- Describe positive flow only with structural language such as "clear arc",
  "cohesive phases", "stable perspective", or "understandable transitions".
- Strengths, reasons, rationales, issues, and recommendations must follow this
  vocabulary boundary. A cohesive-looking roadmap may still contain invalid
  causal dependencies that Structure Quality must not judge.

Use this scoring scale explicitly:
- 8.0-10.0: strong structure; the roadmap has clear framing, useful granularity,
  coherent shape, meaningful closure, and length that fits the question.
- 5.0-7.9: review needed; the roadmap is usable, but its framing, closure,
  granularity, scope fit, or overall shape needs improvement.
- 0.0-4.9: structurally not usable; the roadmap feels fragmented, poorly framed,
  badly scoped, over-expanded, under-specified, or lacks a usable
  learning/execution shape.

When choosing the score, first identify the closest band from the scale, then
choose the numeric value inside that band. The score must be understandable on
its own in a first reading.

SCORING CONSISTENCY GUIDANCE
- Do not give 8.0-10.0 if the roadmap has any medium-severity issue in a core
  structure dimension. Medium issues normally mean the roadmap is usable but
  needs review, not strong.
- Do not give 8.0-10.0 when the issues list contains weak_closure,
  weak_goal_framing, poor_granularity, fragmented_flow, or poor_scope_fit with
  medium or high severity.
- If the roadmap has multiple medium-severity issues across core dimensions,
  choose 5.0-7.9 unless the issues are clearly minor edge cases.
- If the roadmap combines weak goal framing, weak closure, and fragmented flow,
  the final score must be in 0.0-4.9. These three issues together mean the
  roadmap lacks a usable shape.
- Scope Fit must not rescue a roadmap with broken framing, closure, and flow.
  A roadmap can have an appropriate number of steps and still be structurally
  not usable.
- Keep the final score consistent with the criterion scores. If goal_framing
  and closure_quality are both below 5.0, the final score should normally be in
  0.0-4.9 because the roadmap lacks both a usable start and a usable end.
- If two or more of goal_framing, closure_quality, flow_coherence, and
  structural_usefulness are below 5.0, the final score must be in 0.0-4.9.
- Do not treat generic final steps such as "finish", "review", "adjust", or
  "make final changes" as strong closure unless they include concrete
  validation, application, synthesis, or done criteria.
- Do not reward relevance alone. A roadmap can contain relevant topics and still
  have weak structure.
- Do not reward verbosity alone. A roadmap can have many relevant-looking steps
  and still be over-expanded for a simple question.
- Do not punish concise roadmaps solely for being short if the question is
  narrow and the roadmap still reaches a complete, useful result.
- Prefer clear, explicit reasoning over generous assumptions.

Expected JSON schema:
{{
  "structure_quality": {{
    "score": <0-10>,
    "verdict": "PASS|NEEDS_REVIEW|FAIL",
    "score_band": "8.0-10.0|5.0-7.9|0.0-4.9",
    "score_rationale": "<why this score belongs to that band>",
    "reason": "<short explanation>",
    "criteria": {{
      "goal_framing": <0-10>,
      "closure_quality": <0-10>,
      "granularity": <0-10>,
      "flow_coherence": <0-10>,
      "structural_usefulness": <0-10>,
      "scope_fit": <0-10>
    }},
    "criteria_rationales": {{
      "goal_framing": "<why this roadmap received this goal_framing score>",
      "closure_quality": "<why this roadmap received this closure_quality score>",
      "granularity": "<why this roadmap received this granularity score>",
      "flow_coherence": "<why this roadmap received this flow_coherence score>",
      "structural_usefulness": "<why this roadmap received this structural_usefulness score>"
    }},
    "scope_fit": {{
      "ideal_step_count": <integer>,
      "actual_step_count": <integer>,
      "difference": <integer>,
      "reason": "<why the roadmap length fits or does not fit this question>"
    }},
    "strengths": ["<strength>", ...],
    "issues": [
      {{
        "type": "weak_goal_framing|weak_closure|poor_granularity|fragmented_flow|low_structural_usefulness|poor_scope_fit",
        "severity": "low|medium|high",
        "explanation": "<why this structure issue matters>"
      }}
    ],
    "recommendations": ["<recommendation>", ...]
  }}
}}
"""


_LANGUAGE_CORRECTION_PROMPT = """\
Rewrite only the textual fields in this Structure Quality JSON so they use
structural vocabulary. Preserve every number, enum, list item meaning, issue,
and recommendation. Do not reassess the roadmap or change any score.

Describe flow as a clear arc, cohesive phases, stable perspective, navigable
transitions, or fragmented progression. Do not discuss prerequisites, causal
dependencies, correct ordering, describe steps as ordered, discuss step order,
or use the words logical or logically.

Return ONLY the complete corrected JSON object:
{assessment}
"""


def _strip_markdown_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _parse_structure_response(raw: str) -> dict[str, Any]:
    parsed = json.loads(_strip_markdown_json(raw))
    if not isinstance(parsed, dict):
        raise ValueError("Structure Quality response must be a JSON object")
    quality = parsed.get("structure_quality")
    if not isinstance(quality, dict):
        raise ValueError("structure_quality must be an object")
    return parsed


def _structure_boundary_terms(parsed: dict[str, Any]) -> list[str]:
    quality = parsed.get("structure_quality") or {}
    serialized = json.dumps(quality, ensure_ascii=False).lower()
    forbidden_phrases = (
        "logically ordered",
        "correctly ordered",
        "prerequisite order",
        "dependency order",
        "causal dependency",
        "steps are ordered",
        "step order",
    )
    used = [phrase for phrase in forbidden_phrases if phrase in serialized]
    if re.search(r"\blogic(?:al|ally)\b", serialized):
        used.append("logical/logically")
    return sorted(set(used))


def _parse_structure_boundary_response(raw: str) -> dict[str, Any]:
    parsed = _parse_structure_response(raw)
    used = _structure_boundary_terms(parsed)
    if used:
        raise ValueError(
            "Structure Quality used Logical Order vocabulary: " + ", ".join(used)
        )
    return parsed


def _verdict(score: float) -> str:
    if score >= 8:
        return "PASS"
    if score >= 5:
        return "NEEDS_REVIEW"
    return "FAIL"


def _score_band(score: float) -> str:
    if score >= 8:
        return "8.0-10.0"
    if score >= 5:
        return "5.0-7.9"
    return "0.0-4.9"


def _normalize_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return round(max(0.0, min(10.0, score)), 2)


def _normalize_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _fallback_result(error: Exception | None = None) -> dict[str, Any]:
    reason = "Structure quality judge failed; manual review is required."
    if error:
        reason = f"{reason} Error: {type(error).__name__}"
    return {
        "structure_quality": {
            "score": 5.0,
            "verdict": "NEEDS_REVIEW",
            "score_band": "5.0-7.9",
            "score_rationale": reason,
            "reason": reason,
            "criteria": {
                "goal_framing": 5.0,
                "closure_quality": 5.0,
                "granularity": 5.0,
                "flow_coherence": 5.0,
                "structural_usefulness": 5.0,
                "scope_fit": 5.0,
            },
            "criteria_rationales": {
                "goal_framing": reason,
                "closure_quality": reason,
                "granularity": reason,
                "flow_coherence": reason,
                "structural_usefulness": reason,
                "scope_fit": reason,
            },
            "scope_fit": {
                "ideal_step_count": 0,
                "actual_step_count": 0,
                "difference": 0,
                "reason": reason,
            },
            "strengths": [],
            "issues": [],
            "recommendations": ["Review structure quality manually."],
        }
    }


def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    quality = result.get("structure_quality") or {}
    score = _normalize_score(quality.get("score", 0))

    raw_criteria = quality.get("criteria") or {}
    criteria = {
        "goal_framing": _normalize_score(raw_criteria.get("goal_framing", 0)),
        "closure_quality": _normalize_score(raw_criteria.get("closure_quality", 0)),
        "granularity": _normalize_score(raw_criteria.get("granularity", 0)),
        "flow_coherence": _normalize_score(raw_criteria.get("flow_coherence", 0)),
        "structural_usefulness": _normalize_score(raw_criteria.get("structural_usefulness", 0)),
        "scope_fit": _normalize_score(raw_criteria.get("scope_fit", 0)),
    }
    core_scores = [
        criteria["goal_framing"],
        criteria["closure_quality"],
        criteria["flow_coherence"],
        criteria["structural_usefulness"],
    ]
    consistency_notes = []
    if score >= 5.0 and criteria["goal_framing"] < 5.0 and criteria["closure_quality"] < 5.0:
        score = 4.9
        consistency_notes.append(
            "Adjusted final score to 0.0-4.9 because goal framing and closure quality are both below 5.0."
        )
    elif score >= 5.0 and sum(1 for value in core_scores if value < 5.0) >= 2:
        score = 4.9
        consistency_notes.append(
            "Adjusted final score to 0.0-4.9 because multiple core structure criteria are below 5.0."
        )
    raw_scope_fit = quality.get("scope_fit") if isinstance(quality.get("scope_fit"), dict) else {}
    raw_rationales = (
        quality.get("criteria_rationales")
        if isinstance(quality.get("criteria_rationales"), dict)
        else {}
    )
    criteria_rationales = {
        "goal_framing": str(raw_rationales.get("goal_framing", "")),
        "closure_quality": str(raw_rationales.get("closure_quality", "")),
        "granularity": str(raw_rationales.get("granularity", "")),
        "flow_coherence": str(raw_rationales.get("flow_coherence", "")),
        "structural_usefulness": str(raw_rationales.get("structural_usefulness", "")),
        # Scope Fit has one canonical explanation: scope_fit.reason. Keeping it
        # here preserves the normalized result shape for existing consumers.
        "scope_fit": str(raw_scope_fit.get("reason", "")),
    }

    strengths = quality.get("strengths") if isinstance(quality.get("strengths"), list) else []
    raw_issues = quality.get("issues") if isinstance(quality.get("issues"), list) else []
    scope_fit_reason = str(raw_scope_fit.get("reason", ""))
    issues = []
    for issue in raw_issues:
        if not isinstance(issue, dict):
            issues.append(issue)
            continue
        normalized_issue = dict(issue)
        if normalized_issue.get("type") == "poor_scope_fit" and scope_fit_reason:
            normalized_issue["explanation"] = scope_fit_reason
        issues.append(normalized_issue)
    recommendations = (
        quality.get("recommendations")
        if isinstance(quality.get("recommendations"), list)
        else []
    )

    return {
        "structure_quality": {
            "score": score,
            "verdict": _verdict(score),
            "score_band": _score_band(score),
            "score_rationale": str(quality.get("score_rationale", "")),
            "reason": str(quality.get("reason", "")),
            "criteria": criteria,
            "criteria_rationales": criteria_rationales,
            "scope_fit": {
                "ideal_step_count": _normalize_int(raw_scope_fit.get("ideal_step_count")),
                "actual_step_count": _normalize_int(raw_scope_fit.get("actual_step_count")),
                "difference": _normalize_int(raw_scope_fit.get("difference")),
                "reason": scope_fit_reason,
            },
            "strengths": strengths,
            "issues": issues,
            "issue_count": len(issues),
            "recommendations": recommendations,
            "normalization_notes": consistency_notes,
        }
    }


class StructureQualityJudge:
    """Evaluate semantic structure quality with an LLM judge."""

    def __init__(self, llm=None):
        self.llm = llm or get_judge_llm(temperature=0.0, max_tokens=2048)

    def evaluate(
        self,
        roadmap: dict,
        question: str = "",
        category: str = "",
    ) -> dict[str, Any]:
        prompt = _PROMPT_TEMPLATE.format(
            question=question or "(not provided)",
            category=category or "(not provided)",
            roadmap=json.dumps(roadmap, ensure_ascii=False, indent=2)[:6000],
        )
        try:
            parsed, trace = invoke_json_with_retry(
                self.llm,
                prompt,
                operation="Structure Quality",
                parser=_parse_structure_response,
                repair_instruction=(
                    "Return the complete structure_quality JSON contract using only "
                    "structural vocabulary. Do not assess logical order, prerequisites, "
                    "or causal dependencies. Do not use the words logical or logically "
                    "anywhere in the output."
                ),
            )
            boundary_terms = _structure_boundary_terms(parsed)
            if boundary_terms:
                corrected, correction_trace = invoke_json_with_retry(
                    self.llm,
                    _LANGUAGE_CORRECTION_PROMPT.format(
                        assessment=json.dumps(parsed, ensure_ascii=False, indent=2)
                    ),
                    operation="Structure Quality language correction",
                    parser=_parse_structure_boundary_response,
                    repair_instruction=(
                        "Return the complete JSON with identical scores and findings, "
                        "but remove Logical Order terminology from every textual field."
                    ),
                )
                original_quality = parsed["structure_quality"]
                corrected_quality = corrected["structure_quality"]
                corrected_quality["score"] = original_quality.get("score")
                corrected_quality["criteria"] = original_quality.get("criteria")
                original_scope = original_quality.get("scope_fit") or {}
                corrected_scope = corrected_quality.get("scope_fit") or {}
                for field in ("ideal_step_count", "actual_step_count", "difference"):
                    corrected_scope[field] = original_scope.get(field)
                corrected_quality["scope_fit"] = corrected_scope
                parsed = corrected
                trace["language_correction"] = correction_trace
            normalized = _normalize_result(parsed)
            normalized["structure_quality"]["judge_execution"] = trace
            return normalized
        except Exception as exc:
            return _fallback_result(exc)
