"""
llm_judge.py — Sprint 6: LLM judge for the ambiguous Top-K frontier.

The deterministic + semantic + rejector layers commit intents 1/2/3 and defer
the genuinely ambiguous residual. For the samples that land on an ambiguous
boundary (the CRITICAL_FRONTIERS 2<->3, 2<->4, 3<->4 surfaced by recall@K but
not decided by the embedding top-1), a small LLM judge DISCRIMINATES between
the candidate intents instead of doing an open 4-way classification.

Design:
  * The judge is a focused bisection: given the query and exactly TWO candidate
    intents (a frontier pair), it picks the more faithful one. Bisection is
    easier for a small LLM than open classification, and it is cheap to bound.
  * It reuses the project's LLM client (get_llm) so config (provider, model,
    base URL) stays in .env — no new plumbing.
  * Fallback rate: if the LLM output is unparsable / low-confidence, the judge
    returns None (fallback to the semantic order / clarify).

Runtime note: the Ollama llama3.2:3b calls here are SLOW (28-60s each). The
Sprint 6 evaluation is therefore run on a SMALL targeted set (the frontier
minimal pairs + a few near-threshold items), never the full 100+ sample run.
"""

import json
import os

from src.llm_provider import get_llm

JUDGE_MAX_TOKENS = int(os.getenv("INTENT_JUDGE_MAX_TOKENS", "120"))

# label of each intent for the judge prompt (kept stable / readable)
INTENT_NAMES = {
    1: "KB_ONLY: use only the internal knowledge base; no external info.",
    2: "KB_PLUS_EXTERNAL: internal KB as the base, complemented by external/updated info.",
    3: "EXTERNAL_PLUS_KB: external/updated info as the primary basis, using the KB as a complement (or distrusting it).",
    4: "AMBIGUOUS: no clear source commitment; clarify before proceeding.",
}


class LLMJudge:
    """Bisects a (query, {candidate intents}) decision with the LLM."""

    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.0, max_tokens=JUDGE_MAX_TOKENS)

    def _prompt(self, query: str, candidates: list, expected_hint: str = "") -> str:
        opts = "\n".join(f"  - {c}: {INTENT_NAMES.get(c, str(c))}" for c in candidates)
        return (
            "You are a precise INTENT JUDGE for a RAG system. Read the user's query "
            "and pick the SINGLE best-fitting intent from the candidates below, based "
            "strictly on the TEXT's expressed trust in the internal knowledge base "
            "(do not judge the topic or consider any corpus contents).\n\n"
            f"QUERY: {query}\n\n"
            f"CANDIDATE INTENTS:\n{opts}\n\n"
            "Return ONLY a JSON object of the form {\"intent\": <int>, \"confidence\": 0.0-1.0, "
            "\"rationale\": \"<1 sentence>\"}. Pick exactly one of the candidate intents."
        )

    @staticmethod
    def _parse(content: str):
        raw = content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        if not raw.startswith("{"):
            s, e = raw.find("{"), raw.rfind("}")
            if s >= 0 and e > s:
                raw = raw[s:e + 1]
        return json.loads(raw)

    def judge(self, query: str, candidates: list):
        """Returns dict {intent, confidence, rationale, source} or a fallback
        dict indicating failure ({intent: None, source: 'fallback'}).
        `candidates` collapsed to a pair if more than two (bisection on the
        first two — callers pass an ordered frontier pair)."""
        pair = candidates[:2]
        fallback = {"intent": None, "confidence": 0.0,
                    "rationale": "judge fallback (unparsable/low-conf)",
                    "source": "fallback"}
        try:
            content = self.llm.invoke(self._prompt(query, pair)).content
            if isinstance(content, (list, dict)):
                content = str(content)
            data = self._parse(content)
        except Exception as e:
            print(f"  [LLM Judge] fallback ({e})")
            return fallback

        intent = data.get("intent")
        try:
            intent = int(intent)
        except (TypeError, ValueError):
            intent = None
        conf = float(data.get("confidence") or 0.0)
        if intent not in pair or conf < 0.4:
            return fallback
        return {
            "intent": intent,
            "confidence": round(conf, 3),
            "rationale": str(data.get("rationale") or "")[:200],
            "source": "llm_judge",
            "considered": pair,
        }
