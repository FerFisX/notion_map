"""Dedicated claim-level semantic judge for roadmap Grounding.

Grounding evaluates whether verifiable claims in existing roadmap steps are
supported by the retrieved context. It does not evaluate correctness against
external knowledge, coverage, order, actionability, distinctness, or structure.
"""

from __future__ import annotations

import json
from typing import Any

from evaluation.metrics.prompt_payloads import compact_contexts, compact_roadmap, prompt_json
from evaluation.structured_output import invoke_json_with_retry
from src.llm_provider import get_judge_llm


_SCORE_CORRECTION_PROMPT = """\
You are assigning the final Grounding score to claim diagnostics that have
already been validated. Return ONLY valid JSON with no markdown or extra text.

VALIDATED CLAIM DIAGNOSTICS
{diagnostics}

The claim statuses, importance levels, counts, and evidence are final. Do not
add, remove, or reclassify claims. Choose a precise decimal score inside the
mandatory {score_band} band and make the reason and recommendations describe
those diagnostics. Do not score completeness, correctness against external
knowledge, actionability, order, overlap, or structure.

Return exactly:
{{
  "grounding_score_correction": {{
    "score": <number in {score_band}>,
    "reason": "<short explanation>",
    "strengths": ["<grounding strength>"],
    "recommendations": ["<grounding recommendation>"]
  }}
}}
"""


_CLAIM_STATUSES = {
    "supported",
    "unsupported",
    "insufficient_evidence",
    "contradicted",
}
_CLAIM_IMPORTANCE = {"core", "supporting"}
_STEP_EVALUABILITY = {"evaluable", "not_applicable"}
_STEP_BATCH_SIZE = 3


_BATCH_PROMPT_TEMPLATE = """\
You are an expert evaluator of retrieval-grounded technical roadmaps.
Return ONLY valid JSON. Do not use markdown or extra text.

USER QUESTION
{question}

RETRIEVED CONTEXTS
{contexts}

ROADMAP STEPS IN THIS BATCH
{roadmap}

Evaluate only Grounding: whether atomic, materially verifiable claims in each
step are supported by the retrieved contexts. Use only those contexts as
evidence; do not use external knowledge or general plausibility.

Evaluate every supplied step exactly once and preserve its exact ID. Read its
label, description, and key points together. Split independent claims when
different evidence could support them. Return at most three claims per step,
prioritizing the main technical instruction and only material supporting
details. Do not invent claims or fragment minor wording into separate claims.

Evaluability:
- evaluable: the step makes a factual, technical, or procedural assertion.
- not_applicable: the step contains only a subjective goal, preference, or
  organizational decision. An unfamiliar product or procedure is still
  evaluable and unsupported when no context supports it.

Importance:
- core: defines the step's main technical instruction or justification.
- supporting: adds a secondary detail.

Status:
- supported: context directly or reasonably entails the complete claim.
- unsupported: no relevant context supports it.
- insufficient_evidence: related context supports only part of it.
- contradicted: context explicitly conflicts with it.

Supported and contradicted claims must cite valid context IDs and a concise
evidence excerpt. Unsupported claims must use empty context_ids and evidence.
Do not penalize missing coverage, order, actionability, overlap, structure, or
schema validity.

Return exactly:
{{
  "grounding": {{
    "step_claims": [
      {{
        "step_id": "<exact supplied step ID>",
        "evaluability": "evaluable|not_applicable",
        "reason": "<brief case-specific reason>",
        "claims": [
          {{
            "claim_id": "<step_id>_claim_<number>",
            "claim": "<atomic verifiable claim>",
            "importance": "core|supporting",
            "status": "supported|unsupported|insufficient_evidence|contradicted",
            "context_ids": ["context_1"],
            "evidence": "<concise excerpt or empty string>",
            "explanation": "<support assessment>",
            "recommendation": "<correction or evidence needed, or empty string>"
          }}
        ]
      }}
    ]
  }}
}}
"""


_SUMMARY_PROMPT_TEMPLATE = """\
You assign the final Grounding score to claim diagnostics that have already
been validated. Return ONLY valid JSON with no markdown or extra text.

USER QUESTION
{question}

VALIDATED CLAIM DIAGNOSTICS
{diagnostics}

Do not add, remove, or reclassify claims. Score only whether existing claims
are supported by retrieved context. Do not score completeness, external
correctness, actionability, order, overlap, structure, or schema validity.

Score bands:
- 8.0-10.0: all core claims supported; no unsupported or contradicted claims;
  at most one minor supporting insufficient-evidence claim.
- 5.0-7.9: support is mixed, one supporting claim is unsupported, or a
  localized core claim lacks sufficient evidence while most technical basis
  remains supported.
- 0.0-4.9: a core claim is contradicted, several core claims are unsupported,
  context is irrelevant, or most evaluable steps lack support.

The score is not a raw percentage. Select the band from material impact, then
choose a natural decimal within it.

Return exactly:
{{
  "grounding_summary": {{
    "score": <0-10>,
    "reason": "<short explanation supporting the score band>",
    "strengths": ["<grounding strength>"],
    "recommendations": ["<grounding recommendation>"]
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
        raise ValueError("Grounding response must be a JSON object")
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


def _parse_score_correction(
    raw: str,
    minimum: float,
    maximum: float,
) -> dict[str, Any]:
    parsed = _parse_json_response(raw)
    correction = parsed.get("grounding_score_correction")
    if not isinstance(correction, dict):
        raise ValueError("grounding_score_correction must be an object")
    score = _normalize_score(correction.get("score"))
    if not minimum <= score <= maximum:
        raise ValueError(
            f"Grounding score {score} must be inside {minimum}-{maximum}"
        )
    if not str(correction.get("reason", "")).strip():
        raise ValueError("Grounding score correction requires a reason")
    if not isinstance(correction.get("strengths", []), list):
        raise ValueError("Grounding strengths must be a list")
    if not isinstance(correction.get("recommendations", []), list):
        raise ValueError("Grounding recommendations must be a list")
    return parsed


def _parse_grounding_batch(
    raw: str,
    expected_step_ids: list[str],
) -> dict[str, Any]:
    parsed = _parse_json_response(raw)
    grounding = parsed.get("grounding")
    if not isinstance(grounding, dict):
        raise ValueError("grounding must be an object")
    step_claims = grounding.get("step_claims")
    if not isinstance(step_claims, list):
        raise ValueError("grounding.step_claims must be a list")
    observed_ids = []
    for item in step_claims:
        if not isinstance(item, dict):
            raise ValueError("Every step_claims entry must be an object")
        step_id = str(item.get("step_id", "")).strip()
        observed_ids.append(step_id)
        if not isinstance(item.get("claims", []), list):
            raise ValueError(f"Claims for {step_id or '(missing ID)'} must be a list")
        if len(item.get("claims", [])) > 3:
            raise ValueError(f"Grounding batch returned more than 3 claims for {step_id}")
    if observed_ids != expected_step_ids:
        raise ValueError(
            "Grounding batch must assess each supplied step exactly once in order; "
            f"expected {expected_step_ids}, received {observed_ids}"
        )
    return parsed


def _parse_grounding_summary(raw: str) -> dict[str, Any]:
    parsed = _parse_json_response(raw)
    summary = parsed.get("grounding_summary")
    if not isinstance(summary, dict):
        raise ValueError("grounding_summary must be an object")
    try:
        score = float(summary.get("score"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Grounding summary score must be numeric") from exc
    if not 0.0 <= score <= 10.0:
        raise ValueError("Grounding summary score must be between 0 and 10")
    if not str(summary.get("reason", "")).strip():
        raise ValueError("Grounding summary requires a reason")
    if not isinstance(summary.get("strengths", []), list):
        raise ValueError("Grounding summary strengths must be a list")
    if not isinstance(summary.get("recommendations", []), list):
        raise ValueError("Grounding summary recommendations must be a list")
    return parsed


def _roadmap_steps(roadmap: dict[str, Any]) -> list[dict[str, Any]]:
    raw = roadmap.get("steps", [])
    if not isinstance(raw, list):
        return []
    return [step for step in raw if isinstance(step, dict)]


def _contexts(values: list[Any] | None) -> list[dict[str, str]]:
    return compact_contexts(values)


def _fallback_result(roadmap: dict[str, Any], error: Exception | None = None) -> dict[str, Any]:
    reason = "Grounding judge failed; manual review is required."
    if error:
        reason = f"{reason} Error: {type(error).__name__}"
    assessments = [
        {
            "step_id": str(step.get("id", "")).strip(),
            "evaluability": "evaluable",
            "status": "ungrounded",
            "reason": reason,
            "claim_ids": [],
            "supported_claim_count": 0,
            "unsupported_claim_count": 0,
        }
        for step in _roadmap_steps(roadmap)
    ]
    return {
        "grounding": {
            "support_score": 5.0,
            "verdict": "NEEDS_REVIEW",
            "score_band": "5.0-7.9",
            "reason": reason,
            "claim_support_pct": 0.0,
            "step_grounded_pct": 0.0,
            "evaluable_claim_count": 0,
            "supported_claim_count": 0,
            "claims": [],
            "unsupported_claims": [],
            "unsupported_claim_count": 0,
            "contradictions": [],
            "contradiction_count": 0,
            "step_assessments": assessments,
            "strengths": [],
            "recommendations": ["Review Grounding manually."],
            "normalization_notes": [],
            "manual_review_required": True,
        }
    }


def _normalize_result(
    result: dict[str, Any],
    roadmap: dict[str, Any],
    contexts: list[dict[str, str]],
) -> dict[str, Any]:
    raw = result.get("grounding") if isinstance(result.get("grounding"), dict) else {}
    steps = _roadmap_steps(roadmap)
    step_by_id = {
        str(step.get("id", "")).strip(): step
        for step in steps
        if str(step.get("id", "")).strip()
    }
    context_ids = {item["id"] for item in contexts}
    notes: list[str] = []
    manual_review = False
    raw_steps = raw.get("step_claims") if isinstance(raw.get("step_claims"), list) else []
    raw_by_id: dict[str, dict[str, Any]] = {}
    for item in raw_steps:
        if not isinstance(item, dict):
            manual_review = True
            notes.append("Ignored a non-object step claim assessment.")
            continue
        step_id = str(item.get("step_id", "")).strip()
        if step_id not in step_by_id or step_id in raw_by_id:
            manual_review = True
            notes.append("Discarded an unknown or duplicate step claim assessment.")
            continue
        raw_by_id[step_id] = item

    claims = []
    assessments = []
    used_claim_ids: set[str] = set()
    for step_id in step_by_id:
        item = raw_by_id.get(step_id)
        if item is None:
            manual_review = True
            notes.append(f"Added fallback assessment for missing step {step_id}.")
            assessments.append({
                "step_id": step_id,
                "evaluability": "evaluable",
                "status": "ungrounded",
                "reason": "The judge did not assess this step.",
                "claim_ids": [],
                "supported_claim_count": 0,
                "unsupported_claim_count": 0,
            })
            continue
        evaluability = str(item.get("evaluability", "evaluable")).lower().strip()
        if evaluability not in _STEP_EVALUABILITY:
            evaluability = "evaluable"
            manual_review = True
            notes.append(f"Normalized evaluability for step {step_id}.")
        raw_claims = item.get("claims") if isinstance(item.get("claims"), list) else []
        if evaluability == "not_applicable" and raw_claims:
            raw_claims = []
            manual_review = True
            notes.append(f"Removed claims from not-applicable step {step_id}.")
        if evaluability == "evaluable" and not raw_claims:
            manual_review = True
            notes.append(f"Evaluable step {step_id} has no claims.")
        step_claims = []
        for index, claim in enumerate(raw_claims, 1):
            if not isinstance(claim, dict):
                manual_review = True
                continue
            claim_id = str(claim.get("claim_id", "")).strip() or f"{step_id}_claim_{index}"
            if claim_id in used_claim_ids:
                claim_id = f"{step_id}_{claim_id}_{index}"
                manual_review = True
                notes.append(f"Normalized duplicate claim ID in step {step_id}.")
            used_claim_ids.add(claim_id)
            status = str(claim.get("status", "unsupported")).lower().strip()
            if status not in _CLAIM_STATUSES:
                status = "unsupported"
                manual_review = True
            importance = str(claim.get("importance", "core")).lower().strip()
            if importance not in _CLAIM_IMPORTANCE:
                importance = "core"
                manual_review = True
            claim_text = str(claim.get("claim", "")).strip()
            if not claim_text:
                manual_review = True
                notes.append(f"Claim {claim_id} has no description.")
            evidence_ids = []
            raw_context_ids = claim.get("context_ids", [])
            if not isinstance(raw_context_ids, list):
                raw_context_ids = []
                manual_review = True
                notes.append(f"Normalized invalid context IDs for claim {claim_id}.")
            for value in raw_context_ids:
                context_id = str(value).strip()
                if context_id in context_ids and context_id not in evidence_ids:
                    evidence_ids.append(context_id)
                elif context_id:
                    manual_review = True
                    notes.append(f"Discarded unknown context ID from claim {claim_id}.")
            evidence = str(claim.get("evidence", "")).strip()
            if status == "unsupported":
                evidence_ids = []
                evidence = ""
            elif status in {"supported", "contradicted"} and (not evidence_ids or not evidence):
                status = "insufficient_evidence"
                manual_review = True
                notes.append(f"Downgraded claim {claim_id} without traceable evidence.")
            normalized = {
                "claim_id": claim_id,
                "step_id": step_id,
                "claim": claim_text,
                "importance": importance,
                "status": status,
                "context_ids": evidence_ids,
                "evidence": evidence,
                "explanation": str(claim.get("explanation", "")).strip(),
                "recommendation": str(claim.get("recommendation", "")).strip(),
            }
            claims.append(normalized)
            step_claims.append(normalized)
        supported = [claim for claim in step_claims if claim["status"] == "supported"]
        unsupported = [claim for claim in step_claims if claim["status"] != "supported"]
        if evaluability == "not_applicable":
            status = "not_applicable"
        elif supported and not unsupported:
            status = "grounded"
        elif supported:
            status = "partially_grounded"
        else:
            status = "ungrounded"
        assessments.append({
            "step_id": step_id,
            "evaluability": evaluability,
            "status": status,
            "reason": str(item.get("reason", "")).strip(),
            "claim_ids": [claim["claim_id"] for claim in step_claims],
            "supported_claim_count": len(supported),
            "unsupported_claim_count": len(unsupported),
        })

    supported_claims = [claim for claim in claims if claim["status"] == "supported"]
    unsupported_claims = [claim for claim in claims if claim["status"] != "supported"]
    contradictions = [claim for claim in claims if claim["status"] == "contradicted"]
    evaluable_steps = [item for item in assessments if item["evaluability"] == "evaluable"]
    grounded_steps = [item for item in evaluable_steps if item["status"] == "grounded"]
    claim_support_pct = round(len(supported_claims) / len(claims) * 100, 2) if claims else 0.0
    step_grounded_pct = (
        round(len(grounded_steps) / len(evaluable_steps) * 100, 2)
        if evaluable_steps else 0.0
    )
    score = _normalize_score(raw.get("score", 0))
    core_contradictions = [
        claim for claim in contradictions if claim["importance"] == "core"
    ]
    core_gaps = [
        claim for claim in unsupported_claims if claim["importance"] == "core"
    ]
    hard_gaps = [
        claim for claim in unsupported_claims
        if claim["status"] in {"unsupported", "contradicted"}
    ]
    if not unsupported_claims and claims and score < 8.0:
        notes.append("The score is below 8.0 despite full claim support.")
        manual_review = True
    if core_contradictions and score >= 5.0:
        notes.append("The score is 5.0 or higher despite a core contradiction.")
        manual_review = True
    if claims and not supported_claims and score >= 5.0:
        notes.append("The score is 5.0 or higher despite zero supported claims.")
        manual_review = True
    if score >= 8.0 and (core_gaps or hard_gaps):
        notes.append("The score is 8.0 or higher despite material grounding gaps.")
        manual_review = True
    reason = str(raw.get("reason", "")).strip()
    if not reason:
        reason = "The judge did not provide an overall Grounding reason."
        manual_review = True
    strengths = [str(value).strip() for value in raw.get("strengths", []) if str(value).strip()] if isinstance(raw.get("strengths"), list) else []
    recommendations = [str(value).strip() for value in raw.get("recommendations", []) if str(value).strip()] if isinstance(raw.get("recommendations"), list) else []
    return {
        "grounding": {
            "support_score": score,
            "verdict": _verdict(score),
            "score_band": _score_band(score),
            "reason": reason,
            "claim_support_pct": claim_support_pct,
            "step_grounded_pct": step_grounded_pct,
            "evaluable_claim_count": len(claims),
            "supported_claim_count": len(supported_claims),
            "claims": claims,
            "unsupported_claims": unsupported_claims,
            "unsupported_claim_count": len(unsupported_claims),
            "contradictions": contradictions,
            "contradiction_count": len(contradictions),
            "step_assessments": assessments,
            "strengths": strengths,
            "recommendations": recommendations,
            "normalization_notes": notes,
            "manual_review_required": manual_review,
        }
    }


class GroundingJudge:
    """Evaluate roadmap claims against retrieved contexts."""

    def __init__(self, llm=None):
        self.llm = llm or get_judge_llm(temperature=0.0, max_tokens=4096)

    def evaluate(
        self,
        roadmap: dict[str, Any],
        contexts: list[Any] | None,
        question: str = "",
    ) -> dict[str, Any]:
        normalized_contexts = _contexts(contexts)
        steps = _roadmap_steps(roadmap)
        batch_traces: list[dict[str, Any]] = []
        step_claims: list[dict[str, Any]] = []
        active_stage = "setup"
        try:
            if not steps:
                raise ValueError("Grounding requires at least one roadmap step")

            batches = [
                steps[index:index + _STEP_BATCH_SIZE]
                for index in range(0, len(steps), _STEP_BATCH_SIZE)
            ]
            for batch_index, batch_steps in enumerate(batches, 1):
                active_stage = f"claim_batch_{batch_index}_of_{len(batches)}"
                expected_ids = [
                    str(step.get("id", "")).strip() for step in batch_steps
                ]
                batch_roadmap = compact_roadmap({
                    "title": roadmap.get("title", ""),
                    "steps": batch_steps,
                })
                batch_prompt = _BATCH_PROMPT_TEMPLATE.format(
                    question=question or "(not provided)",
                    contexts=prompt_json(normalized_contexts),
                    roadmap=prompt_json(batch_roadmap),
                )
                batch_result, batch_trace = invoke_json_with_retry(
                    self.llm,
                    batch_prompt,
                    operation=f"Grounding claim batch {batch_index}/{len(batches)}",
                    parser=lambda raw, ids=expected_ids: _parse_grounding_batch(
                        raw, ids
                    ),
                    repair_instruction=(
                        "Return exactly one step_claims entry for each supplied "
                        f"step ID, in order: {expected_ids}. Return no overall score."
                    ),
                )
                for item in batch_result["grounding"]["step_claims"]:
                    step_id = str(item.get("step_id", "")).strip()
                    for claim_index, claim in enumerate(item.get("claims", []), 1):
                        claim["claim_id"] = f"{step_id}_claim_{claim_index}"
                    step_claims.append(item)
                batch_traces.append({
                    "batch": batch_index,
                    "step_ids": expected_ids,
                    **batch_trace,
                })

            diagnostic_result = _normalize_result(
                {
                    "grounding": {
                        "step_claims": step_claims,
                        "score": 0,
                        "reason": "Pending final Grounding score.",
                        "strengths": [],
                        "recommendations": [],
                    }
                },
                roadmap,
                normalized_contexts,
            )["grounding"]
            diagnostics = {
                "evaluable_claim_count": diagnostic_result["evaluable_claim_count"],
                "supported_claim_count": diagnostic_result["supported_claim_count"],
                "unsupported_claim_count": diagnostic_result[
                    "unsupported_claim_count"
                ],
                "contradiction_count": diagnostic_result["contradiction_count"],
                "claim_support_pct": diagnostic_result["claim_support_pct"],
                "step_grounded_pct": diagnostic_result["step_grounded_pct"],
                "claims": [
                    {
                        "claim_id": claim["claim_id"],
                        "step_id": claim["step_id"],
                        "claim": claim["claim"],
                        "importance": claim["importance"],
                        "status": claim["status"],
                    }
                    for claim in diagnostic_result["claims"]
                ],
            }
            active_stage = "summary"
            summary_result, summary_trace = invoke_json_with_retry(
                self.llm,
                _SUMMARY_PROMPT_TEMPLATE.format(
                    question=question or "(not provided)",
                    diagnostics=prompt_json(diagnostics),
                ),
                operation="Grounding summary",
                parser=_parse_grounding_summary,
                repair_instruction=(
                    "Return the complete grounding_summary object with a numeric "
                    "score, reason, strengths, and recommendations."
                ),
            )
            summary = summary_result["grounding_summary"]
            parsed = {
                "grounding": {
                    "step_claims": step_claims,
                    "score": summary["score"],
                    "reason": summary["reason"],
                    "strengths": summary.get("strengths", []),
                    "recommendations": summary.get("recommendations", []),
                }
            }
            normalized = _normalize_result(parsed, roadmap, normalized_contexts)
            consistency_prefixes = (
                "The score is below 8.0 despite full claim support.",
                "The score is 5.0 or higher despite a core contradiction.",
                "The score is 5.0 or higher despite zero supported claims.",
                "The score is 8.0 or higher despite material grounding gaps.",
            )
            consistency_notes = [
                str(note)
                for note in normalized["grounding"].get("normalization_notes", [])
                if str(note).startswith(consistency_prefixes)
            ]
            correction_trace = None
            if consistency_notes:
                active_stage = "score_correction"
                correction_reason = "; ".join(consistency_notes)
                print(
                    "      [LLM] Grounding score contradicts claim diagnostics; "
                    "starting one directed score correction.",
                    flush=True,
                )
                if any(
                    note.startswith(
                        (
                            "The score is 5.0 or higher despite a core contradiction.",
                            "The score is 5.0 or higher despite zero supported claims.",
                        )
                    )
                    for note in consistency_notes
                ):
                    minimum, maximum, mandatory_band = 0.0, 4.9, "0.0-4.9"
                elif any(
                    note.startswith(
                        "The score is below 8.0 despite full claim support."
                    )
                    for note in consistency_notes
                ):
                    minimum, maximum, mandatory_band = 8.0, 10.0, "8.0-10.0"
                else:
                    minimum, maximum, mandatory_band = 5.0, 7.9, "5.0-7.9"
                diagnostics = {
                    "evaluable_claim_count": normalized["grounding"][
                        "evaluable_claim_count"
                    ],
                    "supported_claim_count": normalized["grounding"][
                        "supported_claim_count"
                    ],
                    "unsupported_claims": normalized["grounding"][
                        "unsupported_claims"
                    ],
                    "contradictions": normalized["grounding"]["contradictions"],
                    "claim_support_pct": normalized["grounding"][
                        "claim_support_pct"
                    ],
                }
                correction_prompt = _SCORE_CORRECTION_PROMPT.format(
                    diagnostics=prompt_json(diagnostics),
                    score_band=mandatory_band,
                )
                correction_result, correction_trace = invoke_json_with_retry(
                    self.llm,
                    correction_prompt,
                    operation="Grounding score correction",
                    parser=lambda raw: _parse_score_correction(
                        raw, minimum, maximum
                    ),
                    repair_instruction=(
                        "Return the complete grounding_score_correction object and "
                        f"keep the score inside {mandatory_band}."
                    ),
                )
                correction = correction_result["grounding_score_correction"]
                parsed["grounding"]["score"] = correction["score"]
                parsed["grounding"]["reason"] = correction["reason"]
                parsed["grounding"]["strengths"] = correction.get("strengths", [])
                parsed["grounding"]["recommendations"] = correction.get(
                    "recommendations", []
                )
                normalized = _normalize_result(parsed, roadmap, normalized_contexts)
                normalized["grounding"]["normalization_notes"].append(
                    "Applied a directed score correction after: "
                    f"{correction_reason.rstrip('.')}."
                )
            all_traces = batch_traces + [summary_trace]
            attempt_count = sum(int(trace.get("attempt_count", 1)) for trace in all_traces)
            recovered = any(bool(trace.get("recovered")) for trace in all_traces)
            normalized["grounding"]["judge_execution"] = {
                "strategy": "step_batches_then_summary",
                "batch_size": _STEP_BATCH_SIZE,
                "step_batches": batch_traces,
                "summary": summary_trace,
                "attempt_count": attempt_count,
                "recovered": recovered and not normalized["grounding"].get(
                    "manual_review_required", False
                ),
            }
            if correction_trace is not None:
                normalized["grounding"]["judge_execution"]["attempt_count"] += int(
                    correction_trace.get("attempt_count", 1)
                )
                normalized["grounding"]["judge_execution"]["recovered"] = (
                    bool(normalized["grounding"]["judge_execution"]["recovered"])
                    or bool(correction_trace.get("recovered"))
                ) and not normalized["grounding"].get("manual_review_required", False)
                normalized["grounding"]["judge_execution"][
                    "score_correction"
                ] = correction_trace
            return normalized
        except Exception as exc:
            fallback = _fallback_result(roadmap, exc)
            fallback["grounding"]["judge_execution"] = {
                "strategy": "step_batches_then_summary",
                "batch_size": _STEP_BATCH_SIZE,
                "step_batches": batch_traces,
                "failed_stage": active_stage,
                "error_type": type(exc).__name__,
                "failure_trace": getattr(exc, "trace", None),
            }
            return fallback
