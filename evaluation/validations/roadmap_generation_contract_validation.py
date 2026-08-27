"""Offline checks for strict roadmap generation output handling."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from pydantic import ValidationError
from langchain_core.output_parsers import JsonOutputParser

from src.llm_provider import LLMInvocationTimeout, invoke_llm_with_trace
from src.rag_engine import RagEngine, Roadmap, _parse_roadmap_output
from evaluation.benchmarks.generation_reporting import (
    aggregate_generation_results,
    schema_not_evaluated,
)


_VALID_ROADMAP = {
    "title": "Controlled roadmap",
    "steps": [
        {
            "id": "step_1",
            "label": "Define the objective",
            "description": "Record the target outcome and its constraints.",
            "type": "inicio",
            "key_points": ["Write one measurable objective."],
        },
        {
            "id": "step_2",
            "label": "Validate the outcome",
            "description": "Check the completed work against the objective.",
            "type": "fin",
            "key_points": ["Document the validation result."],
        },
    ],
}


class _FakeLLM:
    def __init__(self, *contents: str):
        self.contents = list(contents) or [json.dumps(_VALID_ROADMAP)]
        self.call_count = 0
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        index = min(self.call_count, len(self.contents) - 1)
        content = self.contents[index]
        self.call_count += 1
        return SimpleNamespace(
            content=content,
            response_metadata={
                "done_reason": "stop",
                "prompt_eval_count": 120,
                "eval_count": 80,
            },
            usage_metadata={},
        )

    async def ainvoke(self, prompt: str) -> SimpleNamespace:
        return self.invoke(prompt)


class _SlowFakeLLM(_FakeLLM):
    async def ainvoke(self, prompt: str) -> SimpleNamespace:
        await asyncio.sleep(0.05)
        return self.invoke(prompt)


class _AdvancingClock:
    """Deterministic clock for exercising timeout-bounded retries offline."""

    def __init__(self, step: float = 7.0):
        self.value = -step
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


def validate_roadmap_generation_contract() -> None:
    parsed = _parse_roadmap_output(json.dumps(_VALID_ROADMAP))
    assert parsed["steps"][-1]["type"] == "fin"

    fenced = _parse_roadmap_output(
        f"```json\n{json.dumps(_VALID_ROADMAP)}\n```"
    )
    assert fenced == parsed

    truncated = '{"title":"Broken","steps":[{"id":"step_1"'
    try:
        _parse_roadmap_output(truncated)
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("Truncated JSON was accepted")

    incomplete = {
        "title": "Incomplete",
        "steps": [{"id": "step_1"}],
    }
    try:
        _parse_roadmap_output(json.dumps(incomplete))
    except ValidationError:
        pass
    else:
        raise AssertionError("Incomplete roadmap fields were accepted")

    text, trace = invoke_llm_with_trace(
        _FakeLLM(),
        "controlled prompt",
        operation="Controlled Roadmap",
    )
    assert text.startswith("{")
    assert trace["finish_reason"] == "stop"
    assert trace["input_tokens"] == 120
    assert trace["output_tokens"] == 80
    assert trace["response_characters"] == len(text)

    try:
        invoke_llm_with_trace(
            _SlowFakeLLM(),
            "controlled timeout",
            operation="Controlled Timeout",
            timeout_seconds=0.01,
        )
    except LLMInvocationTimeout:
        pass
    else:
        raise AssertionError("The hard LLM deadline was not enforced")

    engine = object.__new__(RagEngine)
    engine.llm = _FakeLLM()
    engine.parser = JsonOutputParser(pydantic_object=Roadmap)
    engine._last_build_diagnostics = {}
    generated = engine.build_roadmap(
        "Build a controlled roadmap",
        "Build a controlled roadmap",
        ["Controlled evidence"],
        {"intent": "disabled"},
    )
    assert generated["title"] == _VALID_ROADMAP["title"]
    assert engine._last_build_diagnostics["generation_contract_valid"] is True
    assert engine._last_build_diagnostics["generation_attempt_count"] == 1
    assert engine._last_build_diagnostics["generation_recovered"] is False
    assert "Return exactly one complete JSON object" in engine.llm.prompts[0]
    assert "internally verify" in engine.llm.prompts[0]

    recovering_llm = _FakeLLM(truncated, json.dumps(_VALID_ROADMAP))
    engine.llm = recovering_llm
    recovered = engine.build_roadmap(
        "Build a controlled roadmap",
        "Build a controlled roadmap",
        ["Controlled evidence"],
        {"intent": "disabled"},
    )
    assert recovered["title"] == _VALID_ROADMAP["title"]
    assert recovering_llm.call_count == 2
    assert engine._last_build_diagnostics["generation_contract_valid"] is True
    assert engine._last_build_diagnostics["generation_attempt_count"] == 2
    assert engine._last_build_diagnostics["generation_recovered"] is True
    assert engine._last_build_diagnostics["generation_attempts"][0]["contract_valid"] is False
    assert engine._last_build_diagnostics["generation_attempts"][1]["contract_valid"] is True

    third_attempt_llm = _FakeLLM(
        truncated,
        truncated,
        json.dumps(_VALID_ROADMAP),
    )
    engine.llm = third_attempt_llm
    third_attempt_recovery = engine.build_roadmap(
        "Build a controlled roadmap",
        "Build a controlled roadmap",
        ["Controlled evidence"],
        {"intent": "disabled"},
    )
    assert third_attempt_recovery["title"] == _VALID_ROADMAP["title"]
    assert third_attempt_llm.call_count == 3
    assert engine._last_build_diagnostics["generation_attempt_count"] == 3
    assert engine._last_build_diagnostics["generation_recovered"] is True

    malformed = json.dumps(_VALID_ROADMAP, indent=2).replace(
        '"label": "Validate the outcome"',
        '" "label": "Validate the outcome"',
    )
    syntax_recovery_llm = _FakeLLM(malformed, json.dumps(_VALID_ROADMAP))
    engine.llm = syntax_recovery_llm
    syntax_recovered = engine.build_roadmap(
        "Build a controlled roadmap",
        "Build a controlled roadmap",
        ["Controlled evidence"],
        {"intent": "disabled"},
    )
    assert syntax_recovered["title"] == _VALID_ROADMAP["title"]
    assert '" "label"' in syntax_recovery_llm.prompts[1]
    assert "untrusted excerpt" in syntax_recovery_llm.prompts[1]

    failing_llm = _FakeLLM(truncated, truncated)
    engine.llm = failing_llm
    engine._roadmap_generation_timeout_seconds = 20.0
    engine._roadmap_generation_clock = _AdvancingClock()
    failed = engine.build_roadmap(
        "Build a controlled roadmap",
        "Build a controlled roadmap",
        ["Controlled evidence"],
        {"intent": "disabled"},
    )
    assert failed["title"] == "Error"
    assert failing_llm.call_count == 2
    assert engine._last_build_diagnostics["generation_contract_valid"] is False
    assert engine._last_build_diagnostics["generation_attempt_count"] == 2
    assert engine._last_build_diagnostics["generation_recovered"] is False
    assert engine._last_build_diagnostics["generation_invalid_raw_response"] == truncated
    assert engine._last_build_diagnostics["generation_contract_elapsed_s"] >= 20.0
    assert engine._last_build_diagnostics["generation_contract_stop_reason"] == "deadline_reached"
    del engine._roadmap_generation_timeout_seconds
    del engine._roadmap_generation_clock

    not_evaluated = schema_not_evaluated("Generation failed")
    assert not_evaluated["score"] is None
    assert not_evaluated["verdict"] == "NOT_EVALUATED"
    summary = aggregate_generation_results(
        [{
            "generation_succeeded": False,
            "observed_total_s": 2.0,
            "within_target": True,
            "schema_validity": not_evaluated,
            "web_required": False,
            "web_requirement_passed": True,
            "generation_trace": {
                "timings": {"total_generation_s": 2.0},
                "generation_attempts": [
                    {"contract_valid": False},
                    {"contract_valid": False},
                ],
                "generation_recovered": False,
            },
        }],
        initialization_s=0.1,
    )
    assert summary["schema_validity_evaluated_count"] == 0
    assert summary["schema_validity_not_evaluated_count"] == 1
    assert summary["generation_initial_contract_failure_count"] == 1
    assert summary["generation_recovered_count"] == 0
    assert summary["generation_first_attempt_valid_count"] == 0
    assert summary["generation_retry_activation_rate"] == 1.0
    assert summary["generation_recovery_failure_rate"] == 1.0


def main() -> None:
    validate_roadmap_generation_contract()
    print("Roadmap generation contract validation: PASS")
    print("No LLM calls were made.")


if __name__ == "__main__":
    main()
