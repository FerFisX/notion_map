"""Offline contract checks for source-intent/source-mode integration."""

from unittest.mock import patch

from src import rag_engine as rag_module
from src.rag_engine import RagEngine
from src.intent_classifier import IntentClassifier
from src.source_intent_judge import SourceIntentJudge
from src.source_intent_integration import build_source_intent_plan


def _classification(intent, label, decision="proceed"):
    return {
        "intent": intent,
        "label": label,
        "decision": decision,
        "confidence": 0.95,
    }


def _retrieval():
    source = {
        "id": "corpus_1",
        "source_type": "corpus",
        "title": "Internal guide",
        "text": "Supported internal evidence.",
        "vector_score": 0.9,
    }
    return {"selected": [source], "pool": [source]}


def _web():
    return [{
        "id": "web_1",
        "source_type": "web",
        "title": "Official guide",
        "url": "https://example.test/guide",
        "text": "Supported external evidence.",
    }]


def validate_plan_matrix():
    explicit = build_source_intent_plan(
        "corpus", _classification(3, "external_plus_kb")
    )
    assert explicit.strategy == "corpus"
    assert not explicit.classifier_applied

    kb_only = build_source_intent_plan(
        "auto", _classification(1, "kb_only")
    )
    assert (kb_only.strategy, kb_only.preference) == ("corpus", "corpus")

    kb_first = build_source_intent_plan(
        "auto", _classification(2, "kb_plus_external")
    )
    assert (kb_first.strategy, kb_first.preference) == ("hybrid", "corpus")

    web_first = build_source_intent_plan(
        "auto", _classification(3, "external_plus_kb")
    )
    assert (web_first.strategy, web_first.preference) == ("hybrid", "web")

    ambiguous = build_source_intent_plan(
        "auto", _classification(4, "ambiguous", "clarify")
    )
    assert ambiguous.strategy == "route"

    no_roadmap = build_source_intent_plan(
        "auto", {"label": "no_retrieval", "decision": "no_retrieval"}
    )
    assert no_roadmap.no_retrieval

    # Rejector output is observational and must not override a valid source
    # contract until its operating threshold is separately validated.
    observed_reject = _classification(2, "kb_plus_external")
    observed_reject.update({
        "reject": True,
        "reject_score": 0.9,
        "reject_reason": "weak_evidence",
    })
    unchanged = build_source_intent_plan("auto", observed_reject)
    assert (unchanged.strategy, unchanged.preference) == ("hybrid", "corpus")


def validate_hybrid_priority():
    engine = RagEngine.__new__(RagEngine)
    engine.retrieve_contexts_scored = lambda _query: _retrieval()
    with patch.object(rag_module, "search_web_sources", return_value=_web()):
        corpus_first = engine.get_contexts_with_sources(
            "query",
            source_mode="auto",
            source_plan=build_source_intent_plan(
                "auto", _classification(2, "kb_plus_external")
            ),
        )
        web_first = engine.get_contexts_with_sources(
            "query",
            source_mode="auto",
            source_plan=build_source_intent_plan(
                "auto", _classification(3, "external_plus_kb")
            ),
        )
    assert [item["id"] for item in corpus_first["evidence_sources"]] == [
        "corpus_1", "web_1"
    ]
    assert [item["id"] for item in web_first["evidence_sources"]] == [
        "web_1", "corpus_1"
    ]


def validate_shared_embedding_adapter():
    class FakeSharedEmbedder:
        def __init__(self):
            self.document_calls = 0
            self.query_calls = 0

        def embed_documents(self, texts):
            self.document_calls += 1
            return [[float(len(text)), 1.0] for text in texts]

        def embed_query(self, text):
            self.query_calls += 1
            return [float(len(text)), 1.0]

    shared = FakeSharedEmbedder()
    classifier = IntentClassifier(semantic_embedder=shared)
    references = classifier._get_reference_embeddings()
    classifier._embed_query("controlled query")
    assert references
    assert shared.document_calls == len(references)
    assert shared.query_calls == 1
    first_vector = next(iter(references.values()))[0]
    assert abs(sum(value * value for value in first_vector) - 1.0) < 1e-9


def validate_focused_boundary_judge():
    calls = []

    def fake_invoke(prompt, operation):
        calls.append((prompt, operation))
        return '{"intent": 2, "confidence": 0.88, "rationale": "KB remains the base."}'

    judge = SourceIntentJudge(invoke_text=fake_invoke)
    judged = judge.judge("Use our notes as the base and add web context.", [2, 3])
    assert judged["intent"] == 2
    assert judged["source"] == "boundary_judge"
    assert calls[0][1] == "Source Intent Boundary Judge"

    classifier = IntentClassifier(boundary_judge=judge)
    trace = {
        "top_score": 0.60,
        "threshold": 0.55,
        "margin": 0.01,
        "winner": "internal_and_external",
        "candidate_intents": [2, 3],
    }
    assert classifier._boundary_pair(trace) == [2, 3]
    resolved = classifier._judge_semantic_boundary("controlled", trace)
    assert resolved["parsed"]["intent"] == 2
    assert resolved["parsed"]["decision_source"] == "boundary_judge"

    assert classifier._boundary_pair({**trace, "top_score": 0.40}) == []
    assert classifier._boundary_pair({**trace, "margin": 0.20}) == []
    assert classifier._boundary_pair({**trace, "candidate_intents": [1, 2]}) == []


def main():
    validate_plan_matrix()
    validate_hybrid_priority()
    validate_shared_embedding_adapter()
    validate_focused_boundary_judge()
    print("Source-intent integration validation: PASS")


if __name__ == "__main__":
    main()
