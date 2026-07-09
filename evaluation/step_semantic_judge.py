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

STEP OVERLAP
Only evaluate overlap when Step Distinctness has weak_steps or score < 7.
If there are no weak steps and score >= 7, set:
  evaluated=false, trigger="no_weak_steps", overlapping_pairs=[]

If evaluated, explain whether weak steps overlap due to:
- same_task
- same_output
- same_learning_objective
- same_responsibility
- weak_differentiation

Expected JSON schema:
{{
  "step_distinctness": {{
    "score": <0-10>,
    "verdict": "PASS|NEEDS_REVIEW|FAIL",
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
    "trigger": "no_weak_steps|weak_steps_detected|low_distinctness_score",
    "overlapping_pairs": [
      {{
        "steps": [<step number>, <step number>],
        "severity": "low|medium|high",
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


def _fallback_result(error: Exception | None = None) -> dict:
    reason = "Step semantic judge failed; manual review is required."
    if error:
        reason = f"{reason} Error: {type(error).__name__}"
    return {
        "step_distinctness": {
            "score": 5.0,
            "verdict": "NEEDS_REVIEW",
            "reason": reason,
            "strengths": [],
            "weak_steps": [],
        },
        "step_overlap": {
            "evaluated": False,
            "trigger": "judge_error",
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

    normalized_distinctness = {
        "score": score,
        "verdict": distinctness.get("verdict") or _verdict(score),
        "reason": str(distinctness.get("reason", "")),
        "strengths": distinctness.get("strengths") if isinstance(distinctness.get("strengths"), list) else [],
        "weak_steps": weak_steps,
        "weak_step_count": len(weak_steps),
    }

    evaluated = bool(overlap.get("evaluated"))
    pairs = overlap.get("overlapping_pairs") or []
    if not isinstance(pairs, list):
        pairs = []
    non_overlap_notes = overlap.get("non_overlap_notes") or []
    if not isinstance(non_overlap_notes, list):
        non_overlap_notes = []

    if not evaluated and (weak_steps or score < 7):
        evaluated = True

    normalized_overlap = {
        "evaluated": evaluated,
        "trigger": overlap.get("trigger") or ("weak_steps_detected" if weak_steps else "no_weak_steps"),
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

