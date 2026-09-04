"""Offline contract checks for source-intent/source-mode integration."""

from unittest.mock import patch

from src import rag_engine as rag_module
from src.rag_engine import RagEngine
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


def main():
    validate_plan_matrix()
    validate_hybrid_priority()
    print("Source-intent integration validation: PASS")


if __name__ == "__main__":
    main()
