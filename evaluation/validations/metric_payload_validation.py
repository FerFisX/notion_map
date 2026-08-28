"""Offline regression checks for complete metric prompt payloads and retries."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Callable

from evaluation.metrics.grounding_judge import GroundingJudge
from evaluation.metrics.logical_order_judge import LogicalOrderJudge
from evaluation.metrics.prompt_payloads import compact_roadmap, prompt_json
from evaluation.metrics.step_semantic_judge import StepSemanticJudge
from evaluation.metrics.structure_quality_judge import StructureQualityJudge


def _roadmap(step_count: int = 20) -> dict[str, Any]:
    return {
        "title": "Long controlled roadmap",
        "steps": [
            {
                "id": f"step_{index}",
                "type": "inicio" if index == 1 else "fin" if index == step_count else "proceso",
                "label": f"Controlled step {index}",
                "description": (
                    f"Produce the distinct controlled output {index} using a concrete "
                    "method and retain evidence for the next roadmap phase."
                ),
                "key_points": [
                    f"Apply controlled method {index}.",
                    f"Validate controlled output {index}.",
                ],
            }
            for index in range(1, step_count + 1)
        ],
    }


class _FakeLLM:
    def __init__(self, responder: Callable[[str, int], str]):
        self.responder = responder
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.responder(prompt, len(self.prompts)))


def _extract_between(prompt: str, start: str, end: str) -> str:
    return prompt.split(start, 1)[1].split(end, 1)[0].strip()


def _validate_compact_payload() -> None:
    roadmap = _roadmap()
    serialized = prompt_json(compact_roadmap(roadmap))
    parsed = json.loads(serialized)
    assert parsed["step_count"] == 20
    assert len(parsed["steps"]) == 20
    assert parsed["steps"][-1]["id"] == "step_20"


def _validate_structure_payload() -> None:
    def respond(prompt: str, _: int) -> str:
        assert '"id":"step_20"' in prompt
        return json.dumps({
            "structure_quality": {
                "score": 9.0,
                "verdict": "PASS",
                "score_band": "8.0-10.0",
                "score_rationale": "Every structural phase is visible and complete.",
                "reason": "The roadmap has a clear structural arc and closure.",
                "criteria": {
                    "goal_framing": 9,
                    "closure_quality": 9,
                    "granularity": 9,
                    "flow_coherence": 9,
                    "structural_usefulness": 9,
                    "scope_fit": 9,
                },
                "criteria_rationales": {
                    "goal_framing": "The first phase defines the goal.",
                    "closure_quality": "The final phase validates the result.",
                    "granularity": "Phases have comparable scope.",
                    "flow_coherence": "Transitions form a cohesive arc.",
                    "structural_usefulness": "Every phase is navigable.",
                },
                "scope_fit": {
                    "ideal_step_count": 20,
                    "actual_step_count": 20,
                    "difference": 0,
                    "reason": "The length fits this controlled complex question.",
                },
                "strengths": ["Complete beginning and ending."],
                "issues": [],
                "recommendations": [],
            }
        })

    value = StructureQualityJudge(llm=_FakeLLM(respond)).evaluate(
        _roadmap(), question="How do I complete a complex controlled process?"
    )["structure_quality"]
    assert value["score"] == 9.0
    assert value["scope_fit"]["actual_step_count"] == 20


def _validate_step_retry_and_threshold() -> None:
    valid = {
        "step_distinctness": {
            "score": 7.5,
            "verdict": "PASS",
            "score_band": "8.0-10.0",
            "score_rationale": "One pair requires clearer boundaries.",
            "reason": "The roadmap is usable after clarification.",
            "strengths": [],
            "weak_steps": [
                {"step": 19, "label": "Controlled step 19", "reason": "Similar output."},
                {"step": 20, "label": "Controlled step 20", "reason": "Similar output."},
            ],
        },
        "step_overlap": {
            "evaluated": True,
            "trigger": "weak_steps_detected",
            "overall_severity": "medium",
            "overlapping_pairs": [{
                "steps": [19, 20],
                "severity": "medium",
                "overlap_type": "same_output",
                "explanation": "Both steps produce the same final artifact.",
                "recommendation": "Give each step a distinct deliverable.",
            }],
            "non_overlap_notes": [],
        },
    }

    def respond(prompt: str, attempt: int) -> str:
        assert '"id":"step_20"' in prompt
        return "not-json" if attempt == 1 else json.dumps(valid)

    value = StepSemanticJudge(llm=_FakeLLM(respond)).evaluate(
        _roadmap(), question="How do I complete a complex controlled process?"
    )["step_distinctness"]
    assert value["score"] == 7.5
    assert value["verdict"] == "NEEDS_REVIEW"
    assert value["score_band"] == "5.0-7.9"
    assert value["judge_execution"]["attempt_count"] == 2
    assert value["judge_execution"]["recovered"] is True


def _validate_grounding_batches() -> None:
    roadmap = _roadmap(8)

    def respond(prompt: str, _: int) -> str:
        if "ROADMAP STEPS IN THIS BATCH" in prompt:
            payload = json.loads(_extract_between(
                prompt,
                "ROADMAP STEPS IN THIS BATCH\n",
                "\n\nEvaluate only Grounding",
            ))
            entries = []
            for step in payload["steps"]:
                step_id = step["id"]
                entries.append({
                    "step_id": step_id,
                    "evaluability": "evaluable",
                    "reason": "The step contains a verifiable controlled claim.",
                    "claims": [{
                        "claim_id": "claim_1",
                        "claim": f"{step_id} produces its controlled output.",
                        "importance": "core",
                        "status": "supported",
                        "context_ids": ["context_1"],
                        "evidence": "Each controlled step produces its named output.",
                        "explanation": "The context directly supports the claim.",
                        "recommendation": "",
                    }],
                })
            return json.dumps({"grounding": {"step_claims": entries}})
        if "VALIDATED CLAIM DIAGNOSTICS" in prompt:
            return json.dumps({
                "grounding_summary": {
                    "score": 9.3,
                    "reason": "Every core claim is supported by the supplied context.",
                    "strengths": ["All steps have traceable support."],
                    "recommendations": [],
                }
            })
        raise AssertionError("Unexpected Grounding prompt")

    llm = _FakeLLM(respond)
    result = GroundingJudge(llm=llm).evaluate(
        roadmap,
        [{
            "id": "context_1",
            "text": "Each controlled step produces its named output.",
        }],
        question="How do I complete the controlled process?",
    )["grounding"]
    assert result["support_score"] == 9.3
    assert result["evaluable_claim_count"] == 8
    assert result["supported_claim_count"] == 8
    assert result["unsupported_claim_count"] == 0
    assert len({claim["claim_id"] for claim in result["claims"]}) == 8
    assert result["claims"][-1]["claim_id"] == "step_8_claim_1"
    assert result["manual_review_required"] is False
    assert len(result["judge_execution"]["step_batches"]) == 3
    assert result["judge_execution"]["strategy"] == "step_batches_then_summary"


def _validate_grounding_failure_trace() -> None:
    result = GroundingJudge(
        llm=_FakeLLM(lambda _prompt, _attempt: "invalid-json")
    ).evaluate(
        _roadmap(2),
        [{"id": "context_1", "text": "Controlled context."}],
        question="How do I complete the controlled process?",
    )["grounding"]
    execution = result["judge_execution"]
    assert result["manual_review_required"] is True
    assert execution["failed_stage"] == "claim_batch_1_of_1"
    assert execution["error_type"] == "StructuredOutputError"
    assert execution["failure_trace"]["attempt_count"] == 2


def _validate_logical_order_payload() -> None:
    def respond(prompt: str, _: int) -> str:
        if "CURRENT ROADMAP ORDER" in prompt:
            steps = json.loads(_extract_between(
                prompt,
                "CURRENT ROADMAP ORDER\nLower positions execute before higher positions.\n",
                "\n\nTASK",
            ))
            assert len(steps) == 20
            assert steps[-1]["id"] == "step_20"
            return json.dumps({
                "inverted_dependency_audit": [
                    {"step_id": step["id"], "later_dependencies": []}
                    for step in steps
                ]
            })
        if "VALIDATED DETERMINISTIC ORDER EVIDENCE" in prompt:
            assert '"id":"step_20"' in prompt
            return json.dumps({
                "logical_order_assessment": {
                    "criteria": {
                        "prerequisite_order": 9,
                        "causal_dependency_order": 9,
                        "learning_progression": 9,
                        "validation_timing": 9,
                    },
                    "criteria_rationales": {
                        "prerequisite_order": "No strict inversion was found.",
                        "causal_dependency_order": "Outputs precede their use.",
                        "learning_progression": "The progression remains usable.",
                        "validation_timing": "Validation follows production.",
                    },
                    "score": 9.0,
                    "reason": "No strict dependency inversion was found.",
                    "recommendations": [],
                }
            })
        raise AssertionError("Unexpected Logical Order prompt")

    result = LogicalOrderJudge(llm=_FakeLLM(respond)).evaluate(
        _roadmap(),
        question="How do I complete a complex controlled process?",
        contexts=["Controlled phases produce their outputs before validation."],
    )["logical_order"]
    assert result["score"] == 9.0
    assert result["manual_review_required"] is False
    assert result["judge_execution"]["dependency_extraction"]["attempt_count"] == 1


def main() -> None:
    _validate_compact_payload()
    _validate_structure_payload()
    _validate_step_retry_and_threshold()
    _validate_grounding_batches()
    _validate_grounding_failure_trace()
    _validate_logical_order_payload()
    print("Metric prompt payload validation: PASS")


if __name__ == "__main__":
    main()
