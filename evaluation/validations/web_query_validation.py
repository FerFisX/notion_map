"""Deterministic checks for provider-independent web query preparation."""

from unittest.mock import patch

import src.rag_engine as rag_module
from src.rag_engine import RagEngine
from src.web_search import prepare_web_query, rank_web_results


def run() -> None:
    cases = [
        (
            "  ¿Cómo comparar Amazon Bedrock y la API de Anthropic?  ",
            "¿Cómo comparar Amazon Bedrock y la API de Anthropic?",
        ),
        (
            "How do I compare GPT-5, Claude 4, and Gemini 2.5 pricing?",
            "How do I compare GPT-5, Claude 4, and Gemini 2.5 pricing?",
        ),
        (
            "Compare   DATEADD\nwith SAMEPERIODLASTYEAR in Power BI",
            "Compare DATEADD with SAMEPERIODLASTYEAR in Power BI",
        ),
    ]

    for raw, expected in cases:
        actual = prepare_web_query(raw)
        assert actual == expected, (raw, actual, expected)

    long_query = "technical-token " * 40
    compact = prepare_web_query(long_query, max_chars=80)
    assert len(compact) <= 80
    assert not compact.endswith(" ")
    assert "technical-token" in compact

    assert prepare_web_query("") == ""

    ranked = rank_web_results(
        "Amazon Bedrock Anthropic current official documentation pricing",
        [
            {
                "title": "Anthropic API vs AWS Bedrock: Which to use",
                "href": "https://example.ai/articles/anthropic-vs-bedrock",
                "body": "A third-party comparison.",
            },
            {
                "title": "Anthropic - Amazon Bedrock",
                "href": (
                    "https://docs.aws.amazon.com/bedrock/latest/"
                    "userguide/model-cards-anthropic.html"
                ),
                "body": "Official model documentation.",
            },
            {
                "title": "Claude on Amazon Bedrock legacy",
                "href": "https://docs.vendor.example/docs/bedrock-legacy",
                "body": "Legacy integration documentation.",
            },
        ],
    )
    assert "docs.aws.amazon.com" in ranked[0]["href"]
    assert "legacy" not in ranked[0]["href"]

    engine = object.__new__(RagEngine)

    def retrieve_with_score(score: float):
        return {
            "selected": [{"text": "corpus evidence"}],
            "pool": [{"vector_score": score}],
        }

    with (
        patch.object(rag_module, "WEB_FALLBACK", True),
        patch.object(rag_module, "WEB_FALLBACK_MIN", 0.50),
        patch.object(rag_module, "WEB_HYBRID_MIN", 0.65),
        patch("src.web_search.search_web", return_value=["web evidence"]),
    ):
        engine.retrieve_contexts_scored = lambda _: retrieve_with_score(0.40)
        web = engine.get_contexts_with_sources("refined", web_query="original")
        assert web["mode"] == "web"
        assert web["contexts"] == ["web evidence"]
        assert web["corpus_contexts"] == []
        assert web["corpus_candidate_count"] == 1

        engine.retrieve_contexts_scored = lambda _: retrieve_with_score(0.55)
        hybrid = engine.get_contexts_with_sources("refined", web_query="original")
        assert hybrid["mode"] == "hybrid"
        assert hybrid["contexts"] == ["web evidence", "corpus evidence"]

        engine.retrieve_contexts_scored = lambda _: retrieve_with_score(0.70)
        corpus = engine.get_contexts_with_sources("refined", web_query="original")
        assert corpus["mode"] == "corpus"
        assert corpus["contexts"] == ["corpus evidence"]
    print("Web query validation: PASS")


if __name__ == "__main__":
    run()
