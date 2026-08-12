"""Dedicated claim-level semantic judge for roadmap Grounding.

Grounding evaluates whether verifiable claims in existing roadmap steps are
supported by the retrieved context. It does not evaluate correctness against
external knowledge, coverage, order, actionability, distinctness, or structure.
"""

from __future__ import annotations

import json
from typing import Any

from evaluation.structured_output import invoke_json_with_retry
from src.llm_provider import get_judge_llm, invoke_llm_text


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


_PROMPT_TEMPLATE = """\
You are an expert evaluator of retrieval-grounded technical roadmaps.
Return ONLY valid JSON. Do not use markdown or extra text.

USER QUESTION
{question}

RETRIEVED CONTEXTS
{contexts}

ROADMAP
{roadmap}

OBJECTIVE
Evaluate only Grounding: whether verifiable claims in each existing roadmap
step are supported by the retrieved contexts provided above.

EVIDENCE RULE
Use ONLY the retrieved contexts as evidence. Do not use external knowledge,
general plausibility, or assumptions about the technology. A technically
correct claim is still unsupported when the supplied context does not support
it.

CLAIM IDENTIFICATION
- Evaluate every roadmap step exactly once.
- Read each step label, description, and key points together.
- Extract atomic, materially verifiable technical or procedural claims.
- Split independent claims when different evidence could support them.
- Do not invent claims that the step does not make.
- Mark a step not_applicable only when it contains no factual, technical, or
  procedural assertion that can reasonably be checked against context.
- Planning preferences, organizational decisions, and purely subjective goals
  may be not_applicable.
- An instruction can contain a verifiable claim: for example, recommending a
  named function, configuration, dependency, or procedure asserts that it is
  appropriate for the stated task.
- An unfamiliar or apparently invented product, node, property, function, or
  procedure is still an evaluable claim. When the contexts provide no basis for
  it, classify the claim as unsupported; never use not_applicable merely
  because evidence is absent or the named capability is unknown.

CLAIM IMPORTANCE
- core: the claim materially defines the step's main technical instruction or
  justification.
- supporting: the claim adds a secondary detail without defining the step's
  main responsibility.

CLAIM STATUS
- supported: context directly or reasonably entails the full claim.
- unsupported: no relevant contextual evidence supports the claim.
- insufficient_evidence: context is related but does not establish the complete
  claim or required specificity.
- contradicted: context explicitly conflicts with the claim.

Distinguish unsupported from insufficient_evidence by the evidence relationship,
not by claim importance:
- A named feature, behavior, property, function, or procedure that has no basis
  anywhere in the contexts is unsupported, even when it is only a supporting
  detail. One such invented detail places the roadmap in needs review.
- Use insufficient_evidence only when a relevant context partially supports the
  claim but leaves part of it genuinely unresolved. A single minor supporting
  gap of this kind may remain in the grounded band.

Do not mark a claim supported merely because the context mentions the same
topic or vocabulary. Cite exact context IDs and a concise evidence excerpt.
Unsupported claims must have an empty context_ids list and empty evidence.
Contradicted claims must cite the context that contradicts them.

SCORE BANDS
- 8.0-10.0 / grounded:
  All core claims are supported. At most one supporting claim has insufficient
  evidence, and there are no unsupported or contradicted claims.
- 5.0-7.9 / needs review:
  Support is mixed, one supporting claim is unsupported, or a localized core
  claim lacks sufficient evidence, while most of the roadmap's technical basis
  remains supported.
- 0.0-4.9 / ungrounded:
  A central claim is contradicted, several core claims are unsupported, the
  context is irrelevant, most evaluable steps lack support, or none of the
  roadmap's evaluable claims are supported. Two or more unsupported core claims
  normally belong in this band when they represent most of the roadmap.

The score is NOT a raw arithmetic percentage. Select its band from the
importance and practical impact of unsupported claims, then choose a natural
value inside that band.

Calibration check: if your claim list contains any unsupported claim, do not
assign 8.0 or higher. If it contains only one minor supporting
insufficient_evidence claim and every core claim is supported, 8.0-10.0 remains
available. Never describe an unsupported claim as insufficient evidence in the
overall reason.

MECE BOUNDARIES
Do NOT lower Grounding because:
- requested content is missing; that belongs to Completeness;
- steps are in the wrong order; that belongs to Logical Order;
- steps are vague or hard to execute; that belongs to Actionability;
- steps duplicate work; that belongs to Step Distinctness and Step Overlap;
- framing, closure, granularity, navigation, or length is weak; that belongs
  to Structure Quality;
- the roadmap schema is invalid; that belongs to Schema Validity.

Do not reward extra coverage. An incomplete roadmap can be fully grounded when
every claim it does contain is supported.

CONSISTENCY REQUIREMENTS
- Return exactly one step_claims entry for every roadmap step.
- Preserve exact roadmap step IDs.
- Use only context IDs provided in RETRIEVED CONTEXTS.
- Every evaluable step must contain at least one claim.
- Every not_applicable step must contain no claims.
- Supported and contradicted claims must cite context IDs and evidence.
- Unsupported claims must not cite evidence.
- If a core claim is contradicted, the score MUST be in 0.0-4.9.
- A score of 8.0 or higher requires all core claims supported and no
  unsupported or contradicted claims. At most one supporting
  insufficient_evidence claim may remain.
- The score, reason, claim statuses, strengths, and recommendations must
  describe the same assessment.

Before returning JSON:
1. List all contradicted, unsupported, and insufficient claims.
2. Inspect whether each is core or supporting.
3. Select the score band from their material impact.
4. Verify that every roadmap step and evidence context ID is valid.

Expected JSON schema:
{{
  "grounding": {{
    "step_claims": [
      {{
        "step_id": "<exact roadmap step ID>",
        "evaluability": "evaluable|not_applicable",
        "reason": "<why this step is evaluable or not>",
        "claims": [
          {{
            "claim_id": "<unique concise ID>",
            "claim": "<atomic verifiable claim>",
            "importance": "core|supporting",
            "status": "supported|unsupported|insufficient_evidence|contradicted",
            "context_ids": ["context_1", "..."],
            "evidence": "<concise excerpt or empty string>",
            "explanation": "<case-specific support assessment>",
            "recommendation": "<correction or evidence needed, or empty string>"
          }}
        ]
      }}
    ],
    "score": <0-10>,
    "reason": "<short explanation supporting the score band>",
    "strengths": ["<grounding strength>", "..."],
    "recommendations": ["<grounding recommendation>", "..."]
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


def _roadmap_steps(roadmap: dict[str, Any]) -> list[dict[str, Any]]:
    raw = roadmap.get("steps", [])
    if not isinstance(raw, list):
        return []
    return [step for step in raw if isinstance(step, dict)]


def _contexts(values: list[Any] | None) -> list[dict[str, str]]:
    normalized = []
    for index, value in enumerate(values or [], 1):
        if isinstance(value, dict):
            context_id = str(value.get("id", "")).strip() or f"context_{index}"
            text = str(value.get("text", value.get("content", ""))).strip()
        else:
            context_id = f"context_{index}"
            text = str(value).strip()
        if text:
            normalized.append({"id": context_id, "text": text})
    return normalized


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

    def _invoke(self, prompt: str, attempt: int = 1) -> str:
        return invoke_llm_text(
            self.llm,
            prompt,
            operation="Grounding",
            attempt=attempt,
        )

    def evaluate(
        self,
        roadmap: dict[str, Any],
        contexts: list[Any] | None,
        question: str = "",
    ) -> dict[str, Any]:
        normalized_contexts = _contexts(contexts)
        prompt = _PROMPT_TEMPLATE.format(
            question=question or "(not provided)",
            contexts=json.dumps(normalized_contexts, ensure_ascii=False, indent=2),
            roadmap=json.dumps(roadmap, ensure_ascii=False, indent=2),
        )
        try:
            raw = self._invoke(prompt, attempt=1)
            retried = False
            retry_reason = ""
            try:
                parsed = _parse_json_response(raw)
            except (json.JSONDecodeError, ValueError):
                retry_reason = "the previous response was not valid JSON"
                retry_prompt = (
                    f"{prompt}\n\nRETRY REQUIREMENT\n"
                    "The previous response was not valid JSON. Return the complete JSON object only."
                )
                parsed = _parse_json_response(self._invoke(retry_prompt, attempt=2))
                retried = True
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
            if consistency_notes and not retried:
                retry_reason = "; ".join(consistency_notes)
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
                    diagnostics=json.dumps(
                        diagnostics, ensure_ascii=False, indent=2
                    )[:7000],
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
                retried = True
            if consistency_notes and retried:
                normalized["grounding"]["normalization_notes"].append(
                    "Applied a directed score correction after: "
                    f"{str(retry_reason).rstrip('.')}."
                )
            elif retried:
                normalized["grounding"]["normalization_notes"].append(
                    f"Retried once after an invalid response: {str(retry_reason).rstrip('.')}."
                )
            normalized["grounding"]["judge_execution"] = {
                "attempt_count": 2 if retried else 1,
                "recovered": retried and not normalized["grounding"].get(
                    "manual_review_required", False
                ),
                "retry_reason": retry_reason or None,
            }
            if consistency_notes and retried and "correction_trace" in locals():
                normalized["grounding"]["judge_execution"][
                    "score_correction"
                ] = correction_trace
            return normalized
        except Exception as exc:
            return _fallback_result(roadmap, exc)
