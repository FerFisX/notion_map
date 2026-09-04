"""Focused LLM judge for ambiguous source-intent boundaries."""

from __future__ import annotations

import json
import os

from src.llm_provider import get_query_preprocessing_llm, invoke_llm_text


JUDGE_MAX_TOKENS = int(os.getenv("INTENT_JUDGE_MAX_TOKENS", "160"))
JUDGE_MIN_CONFIDENCE = float(os.getenv("INTENT_JUDGE_MIN_CONFIDENCE", "0.50"))

INTENT_NAMES = {
    1: "KB_ONLY: use only the internal knowledge base.",
    2: "KB_PLUS_EXTERNAL: use the internal KB as the base and web as support.",
    3: "EXTERNAL_PLUS_KB: use current external sources as the base and KB as support.",
    4: "AMBIGUOUS: the source preference is not sufficiently clear.",
}


class SourceIntentJudge:
    """Choose between exactly two candidates on a known intent frontier."""

    def __init__(self, *, llm=None, invoke_text=None):
        self.llm = llm
        self.invoke_text = invoke_text

    @staticmethod
    def _parse(content: str) -> dict:
        raw = content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
        if not raw.strip().startswith("{"):
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                raw = raw[start:end + 1]
        return json.loads(raw.strip())

    @staticmethod
    def _prompt(query: str, pair: list[int]) -> str:
        options = "\n".join(
            f"- {intent}: {INTENT_NAMES[intent]}" for intent in pair
        )
        return f"""You are a source-preference boundary judge for a RAG system.
Choose exactly one of the two candidate intents using only the preference
expressed in the user's text. Do not inspect corpus coverage, do not answer the
question, and do not use topic knowledge.

USER QUERY:
{query}

CANDIDATES:
{options}

Return only valid JSON:
{{"intent": <candidate integer>, "confidence": <0.0-1.0>, "rationale": "<one sentence>"}}
"""

    def judge(self, query: str, candidates: list[int]) -> dict:
        pair = list(dict.fromkeys(
            int(value) for value in candidates if value in (1, 2, 3, 4)
        ))[:2]
        fallback = {
            "intent": None,
            "confidence": 0.0,
            "rationale": "boundary judge could not make a safe decision",
            "source": "fallback",
            "considered": pair,
        }
        if len(pair) != 2:
            return fallback
        try:
            prompt = self._prompt(query, pair)
            if self.invoke_text is not None:
                raw = self.invoke_text(prompt, "Source Intent Boundary Judge")
            else:
                if self.llm is None:
                    self.llm = get_query_preprocessing_llm(
                        temperature=0.0,
                        max_tokens=JUDGE_MAX_TOKENS,
                    )
                raw = invoke_llm_text(
                    self.llm,
                    prompt,
                    operation="Source Intent Boundary Judge",
                )
            data = self._parse(str(raw))
            intent = int(data.get("intent"))
            confidence = float(data.get("confidence") or 0.0)
            if intent not in pair or confidence < JUDGE_MIN_CONFIDENCE:
                return fallback
            return {
                "intent": intent,
                "confidence": round(confidence, 4),
                "rationale": str(data.get("rationale") or "")[:300],
                "source": "boundary_judge",
                "considered": pair,
            }
        except Exception as exc:
            return {
                **fallback,
                "error_type": type(exc).__name__,
                "error": str(exc)[:200],
            }

