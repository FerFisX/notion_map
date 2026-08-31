"""Deterministic checks for provider-independent web query preparation."""

from unittest.mock import patch

import src.rag_engine as rag_module
from src.rag_engine import RagEngine
from src.web_search import (
    build_web_query_candidates,
    compact_web_query,
    prepare_web_query,
    rank_web_results,
    requires_current_web_evidence,
)


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

    assert requires_current_web_evidence(
        "Compara los precios vigentes de Power BI"
    )
    assert requires_current_web_evidence(
        "Review the latest update before production adoption"
    )
    assert not requires_current_web_evidence(
        "Diseña retries cuando cambia un requisito del sistema"
    )

    compact = compact_web_query(
        "¿Cómo puedo construir y validar medidas YTD, QTD y MTD en DAX "
        "sin obtener resultados incorrectos cuando faltan fechas?"
    )
    assert compact == (
        "construir validar medidas YTD QTD MTD DAX obtener resultados "
        "incorrectos faltan fechas"
    )
    candidates = build_web_query_candidates(
        "¿Cómo puedo usar el modelo C4 para documentar un sistema?",
        "Documentar arquitectura de software con contexto contenedores componentes",
    )
    assert len(candidates) == 3
    assert candidates[0].startswith("¿Cómo puedo usar")
    assert candidates[1] == "modelo C4 documentar sistema"

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
            "selected": [{
                "id": "corpus_1",
                "source_type": "corpus",
                "text": "corpus evidence",
                "title": "Corpus source",
                "url": "",
                "vector_score": score,
            }],
            "pool": [{"vector_score": score}],
        }

    web_sources = [{
        "id": "web_1",
        "source_type": "web",
        "title": "Web source",
        "url": "https://example.test/source",
        "text": "web evidence",
        "content_origin": "page",
    }]

    with (
        patch.object(rag_module, "AUTO_WEB_THRESHOLD", 0.40),
        patch.object(rag_module, "AUTO_CORPUS_THRESHOLD", 0.55),
        patch.object(rag_module, "search_web_sources", return_value=web_sources),
    ):
        engine.retrieve_contexts_scored = lambda _: retrieve_with_score(0.30)
        web = engine.get_contexts_with_sources(
            "refined", web_query="original", source_mode="auto"
        )
        assert web["mode"] == "web"
        assert web["contexts"] == ["web evidence"]
        assert web["corpus_contexts"] == []
        assert web["corpus_candidate_count"] == 1

        engine.retrieve_contexts_scored = lambda _: retrieve_with_score(0.45)
        hybrid = engine.get_contexts_with_sources(
            "refined", web_query="original", source_mode="auto"
        )
        assert hybrid["mode"] == "hybrid"
        assert hybrid["contexts"] == ["web evidence", "corpus evidence"]

        engine.retrieve_contexts_scored = lambda _: retrieve_with_score(0.60)
        corpus = engine.get_contexts_with_sources(
            "refined", web_query="original", source_mode="auto"
        )
        assert corpus["mode"] == "corpus"
        assert corpus["contexts"] == ["corpus evidence"]

    with (
        patch.object(rag_module, "AUTO_WEB_THRESHOLD", 0.40),
        patch.object(rag_module, "AUTO_CORPUS_THRESHOLD", 0.55),
        patch.object(rag_module, "WEB_SEARCH_RETRY_DELAY", 0.0),
        patch.object(
            rag_module,
            "search_web_sources",
            side_effect=[[], web_sources],
        ),
    ):
        engine.retrieve_contexts_scored = lambda _: retrieve_with_score(0.30)
        recovered = engine.get_contexts_with_sources(
            "Compare current Anthropic models, regions, pricing, and limits",
            web_query=(
                "¿Cómo puedo comparar los modelos Anthropic disponibles en "
                "Amazon Bedrock considerando regiones, precios y límites?"
            ),
            source_mode="auto",
        )
        assert recovered["mode"] == "web"
        assert recovered["web_search_query"] != recovered["web_search_attempts"][0]["query"]
        assert len(recovered["web_search_attempts"]) == 2
        assert recovered["web_search_attempts"][0]["result_count"] == 0
        assert recovered["web_search_attempts"][1]["result_count"] == 1

    with (
        patch.object(rag_module, "WEB_SEARCH_RETRY_DELAY", 0.0),
        patch.object(
            rag_module,
            "search_web_sources",
            side_effect=[[], web_sources],
        ),
    ):
        recovered_web = engine.get_contexts_with_sources(
            "current n8n AI agent nodes and capabilities",
            web_query=(
                "¿Cómo puedo verificar los nodos y capacidades de agentes de IA "
                "disponibles en la versión actual de n8n?"
            ),
            source_mode="web",
        )
        assert recovered_web["mode"] == "web"
        assert recovered_web["corpus_sources"] == []
        assert len(recovered_web["web_search_attempts"]) == 2
        assert recovered_web["web_search_attempts"][1]["result_count"] == 1

    # Refinement may improve retrieval, but source routing must remain tied to
    # the stable original question. A temporary web failure in the hybrid band
    # degrades safely to corpus for a non-current request.
    def retrieve_by_query(query: str):
        return retrieve_with_score(0.90 if query == "expanded" else 0.45)

    with (
        patch.object(rag_module, "AUTO_WEB_THRESHOLD", 0.40),
        patch.object(rag_module, "AUTO_CORPUS_THRESHOLD", 0.55),
        patch.object(rag_module, "WEB_SEARCH_RETRY_DELAY", 0.0),
        patch.object(rag_module, "search_web_sources", return_value=[]),
    ):
        engine.retrieve_contexts_scored = retrieve_by_query
        stable = engine.get_contexts_with_sources(
            "expanded", web_query="original", source_mode="auto"
        )
        assert stable["best_score"] == 0.45
        assert stable["original_corpus_score"] == 0.45
        assert stable["refined_corpus_score"] == 0.90
        assert stable["corpus_selection_query"] == "refined"
        assert stable["mode"] == "corpus"
        assert stable["automatic_decision"] == "hybrid_corpus_fallback"
    print("Web query validation: PASS")


if __name__ == "__main__":
    run()
