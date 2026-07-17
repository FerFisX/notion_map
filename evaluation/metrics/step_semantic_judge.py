"""LLM-based semantic judge for roadmap step metrics.

This replaces the previous token-overlap interpretation of step distinctness.
The judge evaluates whether steps contribute unique objectives, actions,
outputs, and necessary value. Step overlap is a conditional diagnostic that
explains weak steps when they exist.
"""

from __future__ import annotations

import json
from typing import Any

from src.llm_provider import get_llm


_PROMPT_TEMPLATE = """\
You are an expert evaluator of technical learning roadmaps.
Return ONLY valid JSON. Do not use markdown or extra text.

USER QUESTION
{question}

REFINED QUESTION
{refined_question}

CATEGORY
{category}

RETRIEVED CONTEXT
{context}

ROADMAP JSON
{roadmap}

TASK
Evaluate step-level quality semantically. Do NOT penalize repeated domain
vocabulary, repeated technology names, or necessary technical terms. Penalize
only weak contribution, unclear differentiation, or duplicated learning work.

STEP DISTINCTNESS
Score whether each step contributes a unique and necessary learning objective,
action, output, or understanding.

Evaluate exactly these four criteria:
1. Unique objective: each step teaches/accomplishes something different.
2. Unique action: each step asks the learner to do a distinct action.
3. Unique output: each step produces a different artifact, decision,
   validation, or understanding.
4. Necessary contribution: removing the step would lose important value.

Do NOT include "non-duplication" as a Step Distinctness criterion. If steps are
weak because they overlap, identify them as weak_steps and explain overlap in
the Step Overlap section.

Use this scoring scale explicitly:
- 8.0-10.0: strong roadmap; steps are unique, necessary, and clearly separated.
  Minor wording similarity or repeated domain terms are acceptable.
- 5.0-7.9: review needed; some steps have noticeable overlap, weak
  differentiation, or similar objectives, actions, outputs, or responsibilities.
- 0.0-4.9: unusable or structurally weak step design; several steps repeat the
  same task, output, learning objective, or responsibility, making the roadmap
  hard to use without restructuring.

Important distinction:
- Use 5.0-7.9 when the roadmap has isolated or moderate overlap but can still be
  approved after clarification.
- Use 0.0-4.9 when overlap affects multiple steps or core parts of the roadmap, so
  the roadmap should not be approved without restructuring. This does NOT mean
  every step is useless; it means the step design is not usable as-is.
- If the roadmap contains multiple duplicate clusters, repeated tasks across
  several steps, or 2+ meaningful overlapping pairs, prefer the 0.0-4.9 band.

When choosing the score, first identify the closest band from the scale, then
choose the numeric value inside that band. Explain the score using this scale.
The score is the final verdict. Do not give a high score if Step Overlap finds
meaningful redundancy.

Score consistency guidance:
- The score must be understandable on its own in a first reading.
- Use Step Overlap diagnostics to inform the score, not as a separate final
  verdict the reader must combine manually.
- A roadmap with no meaningful overlap should receive a high score.
- A roadmap with noticeable overlap should receive a middle score.
- A roadmap with strong repeated work should receive a low score.
- Do not assign a high score only because some steps remain useful if the
  roadmap has clear duplicated objectives, actions, outputs, or responsibilities.
- Use the full 0-10 scale naturally. Avoid clustering all imperfect roadmaps
  around the same value.

STEP OVERLAP
Evaluate overlap when Step Distinctness has weak_steps, score < 7, or score is
between 7 and 8.5 inclusive. The 7-8.5 band is important because medium-overlap
cases often hide behind otherwise useful steps.

If there are no weak steps and score > 8.5, set:
  evaluated=false, trigger="no_weak_steps", overall_severity="none",
  overlapping_pairs=[]

If evaluated, explain whether weak steps overlap due to:
- same_task
- same_output
- same_learning_objective
- same_responsibility
- weak_differentiation

Use this overlap severity scale:
- none: no meaningful overlap; repeated words are only necessary domain terms.
- low: slight similarity, but steps still have clearly different outputs or
  responsibilities.
- medium: noticeable overlap; one or more steps should be clarified, merged, or
  differentiated.
- high: strong duplication; steps repeat the same task, output, learning
  objective, or responsibility.

Use high severity when duplication affects multiple steps, repeated clusters,
or core roadmap actions, even if some other steps remain useful.

Expected JSON schema:
{{
    "step_distinctness": {{
    "score": <0-10>,
    "verdict": "PASS|NEEDS_REVIEW|FAIL",
    "score_band": "8.0-10.0|5.0-7.9|0.0-4.9",
    "score_rationale": "<why this score belongs to that band>",
    "reason": "<short explanation>",
    "strengths": ["<strength>", ...],
    "weak_steps": [
      {{
        "step": <step number>,
        "label": "<step label>",
        "reason": "<why this step is weak>"
      }}
    ]
  }},
  "step_overlap": {{
    "evaluated": <true|false>,
    "trigger": "no_weak_steps|weak_steps_detected|medium_band_review|low_distinctness_score",
    "overall_severity": "none|low|medium|high",
    "overlapping_pairs": [
      {{
        "steps": [<step number>, <step number>],
        "severity": "none|low|medium|high",
        "overlap_type": "same_task|same_output|same_learning_objective|same_responsibility|weak_differentiation",
        "explanation": "<why these steps overlap or are weakly differentiated>",
        "recommendation": "<how to clarify, split, or merge them>"
      }}
    ],
    "non_overlap_notes": [
      {{
        "steps": [<step number>, <step number>],
        "explanation": "<why repeated words do not mean overlap>"
      }}
    ]
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


def _verdict(score: float) -> str:
    if score >= 7:
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


def _fallback_result(error: Exception | None = None) -> dict:
    reason = "Step semantic judge failed; manual review is required."
    if error:
        reason = f"{reason} Error: {type(error).__name__}"
    return {
        "step_distinctness": {
            "score": 5.0,
            "verdict": "NEEDS_REVIEW",
            "score_band": "5.0-7.9",
            "score_rationale": reason,
            "reason": reason,
            "strengths": [],
            "weak_steps": [],
        },
        "step_overlap": {
            "evaluated": False,
            "trigger": "judge_error",
            "overall_severity": "none",
            "overlapping_pairs": [],
            "non_overlap_notes": [],
        },
    }


def _normalize_result(result: dict[str, Any]) -> dict:
    distinctness = result.get("step_distinctness") or {}
    overlap = result.get("step_overlap") or {}

    score = float(distinctness.get("score", 0))
    score = round(max(0.0, min(10.0, score)), 2)
    weak_steps = distinctness.get("weak_steps") or []
    if not isinstance(weak_steps, list):
        weak_steps = []

    evaluated = bool(overlap.get("evaluated"))
    pairs = overlap.get("overlapping_pairs") or []
    if not isinstance(pairs, list):
        pairs = []
    non_overlap_notes = overlap.get("non_overlap_notes") or []
    if not isinstance(non_overlap_notes, list):
        non_overlap_notes = []

    if not evaluated and (weak_steps or score <= 8.5):
        evaluated = True
    overall_severity = str(overlap.get("overall_severity") or "").lower().strip()
    if overall_severity not in {"none", "low", "medium", "high"}:
        if pairs:
            severity_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
            observed = [
                str(pair.get("severity", "")).lower().strip()
                for pair in pairs
                if str(pair.get("severity", "")).lower().strip() in severity_rank
            ]
            overall_severity = max(observed, key=lambda value: severity_rank[value]) if observed else "medium"
        else:
            overall_severity = "none"

    score_rationale = str(distinctness.get("score_rationale", ""))

    normalized_distinctness = {
        "score": score,
        "verdict": distinctness.get("verdict") or _verdict(score),
        "score_band": distinctness.get("score_band") or _score_band(score),
        "score_rationale": score_rationale,
        "reason": str(distinctness.get("reason", "")),
        "strengths": distinctness.get("strengths") if isinstance(distinctness.get("strengths"), list) else [],
        "weak_steps": weak_steps,
        "weak_step_count": len(weak_steps),
    }

    normalized_overlap = {
        "evaluated": evaluated,
        "trigger": overlap.get("trigger") or (
            "weak_steps_detected" if weak_steps else "medium_band_review" if score <= 8.5 else "no_weak_steps"
        ),
        "overall_severity": overall_severity,
        "overlapping_pairs": pairs,
        "issue_count": len(pairs),
        "non_overlap_notes": non_overlap_notes,
    }

    return {
        "step_distinctness": normalized_distinctness,
        "step_overlap": normalized_overlap,
    }


class StepSemanticJudge:
    """Evaluate step distinctness and conditional overlap with an LLM judge."""

    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.0, max_tokens=2048)

    def evaluate(
        self,
        roadmap: dict,
        question: str = "",
        refined_question: str = "",
        contexts: list[str] | None = None,
        category: str = "",
    ) -> dict:
        context_text = "\n---\n".join(contexts or [])
        prompt = _PROMPT_TEMPLATE.format(
            question=question or "(not provided)",
            refined_question=refined_question or "(not provided)",
            category=category or "(not provided)",
            context=context_text[:2500] or "(no context provided)",
            roadmap=json.dumps(roadmap, ensure_ascii=False, indent=2)[:5000],
        )
        try:
            raw = self.llm.invoke(prompt).content
            parsed = json.loads(_strip_markdown_json(raw))
            return _normalize_result(parsed)
        except Exception as exc:  # keep evaluation runs resilient
            return _fallback_result(exc)
