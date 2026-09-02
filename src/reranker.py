"""
reranker.py — Sprint 4: contrastive reranking of candidate intents.

Given the Top-K candidate intents produced by the embedding retrieval
(semantic_trace.candidate_intents), a reranker scores each candidate against
its structured definition (IntentSpec description + positive examples) so the
final pick is NOT just max cosine similarity over clusters.

Approach: a cross-encoder (query, intent-candidate) that scores how well the
query matches each candidate's description. The reranker reorders the
candidates by this score and can veto an embedding-high but semantically-wrong
candidate (e.g. a hard negative that the bi-encoder scores highly).

Runtime: lazy-loads the cross-encoder on first use. If the model is
unavailable (no network / not installed), degrades to the embedding order
(passthrough) so nothing crashes.

Backend is configurable via env so the team can benchmark cross-encoder vs a
fine-tuned bi-encoder vs a small LLM judge later (Sprint 4 "options to
benchmark").
"""

import os

RERANKER_MODEL = os.getenv("INTENT_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_ENABLED = os.getenv("INTENT_RERANKER_ENABLED", "false").lower().strip() in ("true", "1", "yes")


class IntentReranker:
    """Scores (query, candidate-intent) pairs with a cross-encoder and returns
    the re-ranked candidate list.

    For each candidate we build a short passage from its IntentSpec
    (description + a couple of representative positive examples) and cross-encode
    the (query, passage) pair. The candidate with the highest score is the
    reranker's pick.
    """

    def __init__(self):
        self._model = None
        self._load_error = None

    def _load(self):
        if self._model is not None or self._load_error is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(RERANKER_MODEL)
        except Exception as e:  # network / install failure
            self._load_error = str(e)
            print(f"  [Intent Reranker] cross-encoder unavailable ({e}); "
                  f"falling back to embedding order.")

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    @staticmethod
    def _passage_for(spec) -> str:
        """Builds a compact retrieval passage for a candidate IntentSpec."""
        parts = [spec.description]
        if spec.positive_examples:
            parts.append(" ".join(spec.positive_examples[:2]))
        return " ".join(p for p in parts if p)

    def rerank(self, query: str, candidates: list, intent_specs: dict) -> dict:
        """candidates: list of intent ids (from semantic_trace.candidate_intents).
        intent_specs: {intent_id: IntentSpec}. Returns a dict with the ranked
        order + per-candidate scores, or an embedding-ordered passthrough."""
        if not self.available or not candidates:
            return {"ranked": list(candidates), "scores": {},
                    "source": "embedding" if self.available is False else "passthrough"}

        passages = {c: self._passage_for(intent_specs[c]) for c in candidates}
        pairs = [(query.lower(), passages[c]) for c in candidates]
        raw = self._model.predict(pairs)  # list of similarity scores
        scored = sorted(zip(candidates, raw), key=lambda x: x[1], reverse=True)
        ranked = [c for c, _ in scored]
        scores = {c: round(float(s), 4) for c, s in scored}
        return {"ranked": ranked, "scores": scores,
                "source": "cross_encoder", "model": RERANKER_MODEL}


# Convenience singleton (lazy).
_reranker = None


def get_reranker() -> IntentReranker:
    global _reranker
    if _reranker is None:
        _reranker = IntentReranker()
    return _reranker
