"""Offline contract checks for LLM-facing prompts.

This module validates prompt language and compatibility requirements without making
any network or model call. User queries and legacy JSON keys are intentionally not
treated as Spanish prompt regressions.
"""

from __future__ import annotations

import json
import inspect
from types import SimpleNamespace

from evaluation.corpus_judge import _CHUNK_EVAL_PROMPT, _COVERAGE_PROMPT
from evaluation.llm_judge import _PROMPT_TEMPLATE
from src.rag_engine import RagEngine


class _FakeLLM:
    """Captures prompts and supplies deterministic responses for offline checks."""

    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content=next(self.responses))


_SPANISH_INSTRUCTION_MARKERS = (
    "Eres un ",
    "Responde SOLO",
    "Evalúa SOLO",
    "Consulta original:",
    "INTENCIÓN CLASIFICADA",
    "CRITERIOS CLÁSICOS",
    "CONTEXTO RECUPERADO",
)


def _assert_english_instructions(name: str, prompt: str, required: tuple[str, ...]) -> None:
    for phrase in required:
        assert phrase in prompt, f"{name}: missing expected instruction: {phrase!r}"
    for marker in _SPANISH_INSTRUCTION_MARKERS:
        assert marker not in prompt, f"{name}: Spanish instruction marker found: {marker!r}"


def validate_prompt_language() -> None:
    """Validate English instructions, output contracts, and output-language controls."""
    fake_llm = _FakeLLM(
        [
            json.dumps({"intent": "implementation", "confidence": 0.9}),
            json.dumps({"intent": "implementation", "refined_query": "Configure DAX measures", "confidence": 0.9}),
            "Configure DAX measures",
        ]
    )
    engine = object.__new__(RagEngine)
    engine.preprocessing_llm = fake_llm

    engine.classify_query_intent("¿Cómo configuro medidas DAX?")
    engine.analyze_and_rewrite_query("¿Cómo configuro medidas DAX?")
    engine.rewrite_query(
        "¿Cómo configuro medidas DAX?",
        {
            "intent": "implementation",
            "roadmap_goal": "Configure DAX measures",
            "retrieval_focus": "DAX measures",
            "generation_guidance": "Build an ordered roadmap.",
        },
    )

    classify_prompt, merged_prompt, rewrite_prompt = fake_llm.prompts
    _assert_english_instructions(
        "intent classification",
        classify_prompt,
        ("You are an intent analyst", "Return ONLY valid JSON", "User query:"),
    )
    _assert_english_instructions(
        "merged query analysis",
        merged_prompt,
        ("You are a query analyst", "Preserve the original language", "Return ONLY valid JSON"),
    )
    _assert_english_instructions(
        "query rewriting",
        rewrite_prompt,
        ("You are an expert in technical RAG systems", "Preserve the original language", "Improved query:"),
    )

    assert "No specific context was retrieved." in inspect.getsource(RagEngine.build_roadmap)

    _assert_english_instructions(
        "corpus chunk evaluation",
        _CHUNK_EVAL_PROMPT,
        ("You are an expert in RAG systems", "Return ONLY valid JSON", "Keep the JSON keys exactly as shown"),
    )
    _assert_english_instructions(
        "corpus coverage evaluation",
        _COVERAGE_PROMPT,
        ("Evaluate topical coverage", "Return ONLY valid JSON", "Keep the JSON keys exactly as shown"),
    )
    _assert_english_instructions(
        "legacy roadmap judge",
        _PROMPT_TEMPLATE,
        ("You are an expert evaluator", "Return ONLY valid JSON", "INSTRUCTIONS"),
    )

    # These legacy Spanish keys are an API contract for corpus aggregation/reporting.
    for key in ("coherencia", "densidad_tecnica", "utilidad_rag", "problema_detectado"):
        assert f'"{key}"' in _CHUNK_EVAL_PROMPT, f"Corpus contract key missing: {key}"


def main() -> None:
    validate_prompt_language()
    print("Prompt language validation: PASS")
    print("No LLM calls were made.")


if __name__ == "__main__":
    main()
