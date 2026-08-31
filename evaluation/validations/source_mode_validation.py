"""Offline regression checks for strict roadmap source modes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

import api.main as api_module
from api.main import QueryRequest
from src import rag_engine as rag_module
from src.rag_engine import (
    RagEngine,
    _citation_contract_issues,
    _parse_grounding_gate_output,
)
from src.source_modes import (
    InsufficientEvidenceError,
    SourceMode,
    normalize_source_mode,
    timeout_for_mode,
)
from src import source_modes as source_mode_module


def _corpus_retrieval(score: float = 0.9) -> dict:
    source = {
        "id": "corpus_1",
        "source_type": "corpus",
        "text": "A supported internal procedure.",
        "title": "Internal guide",
        "url": "https://notion.example/internal",
        "vector_score": score,
    }
    return {"selected": [source], "pool": [source]}


def _web_sources() -> list[dict[str, str]]:
    return [{
        "id": "web_1",
        "source_type": "web",
        "title": "Official documentation",
        "url": "https://docs.example.test/guide",
        "text": "A supported web procedure.",
        "content_origin": "page",
    }]


def _engine_without_init(score: float = 0.9) -> RagEngine:
    engine = RagEngine.__new__(RagEngine)
    engine.retrieve_contexts_scored = lambda _query: _corpus_retrieval(score)
    return engine


def _validate_mode_contract() -> None:
    assert normalize_source_mode(None) is SourceMode.AUTO
    assert normalize_source_mode("CORPUS") is SourceMode.CORPUS
    controlled_timeouts = {
        SourceMode.CORPUS: 120.0,
        SourceMode.WEB: 300.0,
        SourceMode.AUTO: 210.0,
    }
    with patch.dict(
        source_mode_module.SOURCE_MODE_TIMEOUTS,
        controlled_timeouts,
        clear=True,
    ):
        assert timeout_for_mode("corpus") == 120.0
        assert timeout_for_mode("web") == 300.0
        assert timeout_for_mode("auto") == 210.0
    try:
        QueryRequest(question="x", source_mode="invalid")
    except ValidationError:
        pass
    else:
        raise AssertionError("API accepted an invalid source mode")


def _validate_corpus_isolation() -> None:
    engine = _engine_without_init()
    with patch.object(
        rag_module,
        "search_web_sources",
        side_effect=AssertionError("Corpus mode called web search"),
    ):
        result = engine.get_contexts_with_sources(
            "controlled query", source_mode="corpus"
        )
    assert result["requested_mode"] == "corpus"
    assert result["mode"] == "corpus"
    assert len(result["corpus_sources"]) == 1
    assert result["web_sources"] == []


def _validate_corpus_fails_closed() -> None:
    engine = _engine_without_init(score=-0.1)
    try:
        engine.get_contexts_with_sources("absent query", source_mode="corpus")
    except InsufficientEvidenceError:
        pass
    else:
        raise AssertionError("Corpus mode accepted irrelevant evidence")


def _validate_web_isolation() -> None:
    engine = _engine_without_init()
    engine.retrieve_contexts_scored = lambda _query: (_ for _ in ()).throw(
        AssertionError("Web mode queried the corpus")
    )
    with patch.object(rag_module, "search_web_sources", return_value=_web_sources()):
        result = engine.get_contexts_with_sources(
            "current controlled query", source_mode="web"
        )
    assert result["requested_mode"] == "web"
    assert result["mode"] == "web"
    assert result["corpus_sources"] == []
    assert len(result["web_sources"]) == 1


def _validate_automatic_selection() -> None:
    high_engine = _engine_without_init(score=0.99)
    with patch.object(
        rag_module,
        "search_web_sources",
        side_effect=AssertionError("High-coverage auto mode called web search"),
    ):
        high = high_engine.get_contexts_with_sources("covered", source_mode="auto")
    assert high["mode"] == "corpus"

    hybrid_score = (
        rag_module.AUTO_WEB_THRESHOLD + rag_module.AUTO_CORPUS_THRESHOLD
    ) / 2
    hybrid_engine = _engine_without_init(score=hybrid_score)
    with patch.object(rag_module, "search_web_sources", return_value=_web_sources()):
        hybrid = hybrid_engine.get_contexts_with_sources(
            "partially covered", source_mode="auto"
        )
    assert hybrid["mode"] == "hybrid"
    assert {source["source_type"] for source in hybrid["evidence_sources"]} == {
        "corpus",
        "web",
    }

    current_engine = _engine_without_init(score=0.99)
    with (
        patch.object(rag_module, "WEB_SEARCH_RETRY_DELAY", 0.0),
        patch.object(rag_module, "search_web_sources", return_value=[]),
    ):
        try:
            current_engine.get_contexts_with_sources(
                "Compare current version pricing", source_mode="auto"
            )
        except InsufficientEvidenceError:
            pass
        else:
            raise AssertionError(
                "Current-information auto request fell back to stale corpus"
            )


def _validate_citations_and_gate_contract() -> None:
    roadmap = {
        "steps": [{
            "id": "step_1",
            "label": "Apply the procedure",
            "description": "Follow the supported internal procedure.",
            "type": "inicio",
            "key_points": [],
            "evidence_ids": ["corpus_1"],
        }]
    }
    sources = _corpus_retrieval()["selected"]
    assert _citation_contract_issues(roadmap, sources, SourceMode.CORPUS) == []
    assert _citation_contract_issues(roadmap, sources, SourceMode.WEB)

    parsed = _parse_grounding_gate_output(
        '{"steps":[{"step_id":"step_1","supported":true,'
        '"reason":"The cited procedure directly supports the step",'
        '"unsupported_claims":[]}]}',
        ["step_1"],
    )
    assert parsed["supported"] is True


def _validate_source_bound_prompt() -> None:
    response = {
        "title": "Controlled roadmap",
        "steps": [{
            "id": "step_1",
            "label": "Apply the internal procedure",
            "description": "Follow the supported procedure.",
            "type": "inicio",
            "key_points": ["Use the documented sequence."],
            "evidence_ids": ["corpus_1"],
        }],
    }

    class FakeLLM:
        def __init__(self):
            self.prompt = ""

        def invoke(self, prompt: str):
            import json

            self.prompt = prompt
            return SimpleNamespace(
                content=json.dumps(response),
                response_metadata={"done_reason": "stop"},
            )

        async def ainvoke(self, prompt: str):
            return self.invoke(prompt)

    engine = RagEngine.__new__(RagEngine)
    engine.llm = FakeLLM()
    engine.parser = rag_module.JsonOutputParser(pydantic_object=rag_module.Roadmap)
    engine._last_build_diagnostics = {}
    result = engine.build_roadmap(
        "How do I apply the procedure?",
        "Apply the documented procedure",
        ["A supported internal procedure."],
        query_intent={"intent": "implementation"},
        evidence_sources=_corpus_retrieval()["selected"],
        source_mode="corpus",
        timeout_seconds=10,
    )
    assert result["steps"][0]["evidence_ids"] == ["corpus_1"]
    assert "Model training knowledge" in engine.llm.prompt
    assert "non-empty `evidence_ids`" in engine.llm.prompt
    assert "[inferido]" not in engine.llm.prompt


def _validate_end_to_end_status_contract() -> None:
    engine = RagEngine.__new__(RagEngine)
    engine._active_pipeline_deadline = None
    engine._active_source_mode = SourceMode.AUTO
    engine._last_build_diagnostics = {}
    engine.rewrite_query = lambda query, query_intent=None: query
    engine.get_contexts_with_sources = lambda *args, **kwargs: {
        "contexts": ["A supported internal procedure."],
        "evidence_sources": _corpus_retrieval()["selected"],
        "corpus_contexts": ["A supported internal procedure."],
        "web_contexts": [],
        "corpus_sources": _corpus_retrieval()["selected"],
        "web_sources": [],
        "best_score": 0.9,
        "original_corpus_score": 0.9,
        "refined_corpus_score": 0.9,
        "corpus_selection_query": "original",
        "requested_mode": "corpus",
        "mode": "corpus",
        "automatic_decision": "explicit_mode",
        "current_web_required": False,
        "web_search_query": "controlled",
        "corpus_candidate_count": 1,
        "timings": {"retrieval_s": 0.0, "web_search_s": 0.0},
    }

    def build(*args, **kwargs):
        engine._last_build_diagnostics = {"generation_attempt_count": 1}
        return {
            "title": "Controlled roadmap",
            "steps": [{
                "id": "step_1",
                "label": "Apply the procedure",
                "description": "Follow the documented internal procedure.",
                "type": "inicio",
                "key_points": [],
                "evidence_ids": ["corpus_1"],
            }],
        }

    engine.build_roadmap = build
    engine._run_grounding_gate = lambda roadmap, sources: {
        "supported": True,
        "steps": [{
            "step_id": "step_1",
            "supported": True,
            "reason": "The cited source supports the complete step.",
            "unsupported_claims": [],
        }],
        "trace": {"elapsed_s": 0.0},
    }
    with patch.object(rag_module, "QUERY_INTENT_ENABLED", False):
        result = engine.generate_roadmap_with_trace(
            "controlled", source_mode="corpus"
        )
    assert result["status"] == "ACCEPTED"
    assert result["roadmap"]["status"] == "ACCEPTED"
    assert result["roadmap"]["sources"]["requested_mode"] == "corpus"
    assert engine._active_pipeline_deadline is None


def _validate_user_api_contract() -> None:
    calls: list[tuple[str, SourceMode]] = []

    class FakeEngine:
        def generate_roadmap(self, question: str, source_mode: SourceMode):
            calls.append((question, source_mode))
            return {"title": "Controlled", "steps": [], "status": "ACCEPTED"}

    previous = api_module.engine
    api_module.engine = FakeEngine()
    try:
        result = asyncio.run(
            api_module.generate_roadmap_endpoint(
                QueryRequest(question="Use only internal evidence", source_mode="corpus")
            )
        )
    finally:
        api_module.engine = previous
    assert result["status"] == "ACCEPTED"
    assert calls == [("Use only internal evidence", SourceMode.CORPUS)]


def main() -> None:
    _validate_mode_contract()
    _validate_corpus_isolation()
    _validate_corpus_fails_closed()
    _validate_web_isolation()
    _validate_automatic_selection()
    _validate_citations_and_gate_contract()
    _validate_source_bound_prompt()
    _validate_end_to_end_status_contract()
    _validate_user_api_contract()
    print("Source mode validation: PASS")


if __name__ == "__main__":
    main()
