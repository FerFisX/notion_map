"""Controlled metric validation cases for the redesigned evaluation metrics.

This module intentionally avoids LLM calls. It validates deterministic behavior
around metric aliases, roadmap readiness gating, MLflow flattening, and report
rendering with small synthetic payloads.
"""

from __future__ import annotations

from pathlib import Path

from evaluation.config import config
from evaluation.llm_judge import LLMJudgeEvaluator
from evaluation.ragas_alignment import align_judge_with_ragas
from evaluation.reporter import save_html, save_human_review_csv
from evaluation.metrics.step_semantic_judge import StepSemanticJudge
from evaluation.tracking import _flatten_metrics


CASES = [
    {
        "name": "ready_case",
        "mese": {
            "mapping": 8,
            "exhaustiveness": 8,
            "sequence": 8,
            "experience": 8,
            "composite": 8.0,
        },
        "structure": {"score": 8, "verdict": "PASS"},
        "expected_readiness": "READY",
        "expected_reason_contains": "All gating dimensions",
    },
    {
        "name": "low_grounding_case",
        "mese": {
            "mapping": 4,
            "exhaustiveness": 9,
            "sequence": 9,
            "experience": 9,
            "composite": 7.75,
        },
        "structure": {"score": 9, "verdict": "PASS"},
        "expected_readiness": "FAIL",
        "expected_reason_contains": "Grounding below critical threshold",
    },
    {
        "name": "incomplete_case",
        "mese": {
            "mapping": 8,
            "exhaustiveness": 6,
            "sequence": 8,
            "experience": 8,
            "composite": 7.6,
        },
        "structure": {"score": 8, "verdict": "PASS"},
        "expected_readiness": "NEEDS_REVIEW",
        "expected_reason_contains": "Completeness needs review",
    },
    {
        "name": "bad_order_case",
        "mese": {
            "mapping": 8,
            "exhaustiveness": 8,
            "sequence": 4,
            "experience": 8,
            "composite": 6.6,
        },
        "structure": {"score": 8, "verdict": "PASS"},
        "expected_readiness": "FAIL",
        "expected_reason_contains": "Logical order below critical threshold",
    },
    {
        "name": "low_actionability_case",
        "mese": {
            "mapping": 8,
            "exhaustiveness": 8,
            "sequence": 8,
            "experience": 6,
            "composite": 7.6,
        },
        "structure": {"score": 8, "verdict": "PASS"},
        "expected_readiness": "NEEDS_REVIEW",
        "expected_reason_contains": "Actionability needs review",
    },
    {
        "name": "bad_structure_case",
        "mese": {
            "mapping": 8,
            "exhaustiveness": 8,
            "sequence": 8,
            "experience": 8,
            "composite": 8.0,
        },
        "structure": {"score": 4, "verdict": "FAIL"},
        "expected_readiness": "FAIL",
        "expected_reason_contains": "Structure below critical threshold",
    },
    {
        "name": "misleading_summary_score_case",
        "mese": {
            "mapping": 9,
            "exhaustiveness": 9,
            "sequence": 4,
            "experience": 9,
            "composite": 8.25,
        },
        "structure": {"score": 9, "verdict": "PASS"},
        "expected_readiness": "FAIL",
        "expected_reason_contains": "Logical order below critical threshold",
    },
]


def _sample_result(case: dict, readiness: dict) -> dict:
    name = case["name"]
    mese = case["mese"]
    structure = {
        "score": case["structure"]["score"],
        "verdict": case["structure"].get("verdict", "PASS"),
        "passed": 8,
        "total_checks": 10,
        "violations": [] if case["structure"].get("verdict") == "PASS" else ["Synthetic structure violation"],
    }
    step_distinctness = case.get("step_distinctness", {
        "score": 10.0,
        "verdict": "PASS",
        "reason": "Synthetic distinct steps.",
        "strengths": [],
        "weak_steps": [],
        "weak_step_count": 0,
    })
    step_overlap = case.get("step_overlap", {
        "evaluated": False,
        "trigger": "no_weak_steps",
        "overlapping_pairs": [],
        "issue_count": 0,
        "non_overlap_notes": [],
    })
    return {
        "question": f"Synthetic validation question: {name}",
        "category": "metric_validation",
        "query_intent": {"intent": "validation", "roadmap_goal": "Validate redesigned metrics"},
        "refined_question": f"Validate metric behavior for {name}",
        "rewrite_strategy": "synthetic",
        "ground_truth": "Synthetic ground truth",
        "answer": "Synthetic roadmap answer",
        "contexts": ["Synthetic context"],
        "retrieval": {},
        "judge_context_strategy": "synthetic",
        "roadmap": {"title": name, "steps": []},
        "steps": ["Inspect metric inputs", "Validate expected readiness"],
        "expected_steps": ["Inspect metric inputs", "Validate expected readiness"],
        "classic": {
            "relevance": {"score": 8, "justification": "Synthetic"},
            "completeness": {"score": 8, "justification": "Synthetic"},
            "coherence": {"score": 8, "justification": "Synthetic"},
            "technical_accuracy": {"score": 8, "justification": "Synthetic"},
            "context_fidelity": {"score": 8, "justification": "Synthetic"},
            "summary": "Synthetic validation sample.",
        },
        "sequence_eval": {
            "score": mese["sequence"],
            "is_valid": mese["sequence"] >= 7,
            "out_of_order_steps": [] if mese["sequence"] >= 7 else ["Validate expected readiness"],
            "suggested_fix": "Orden correcto" if mese["sequence"] >= 7 else "Move dependent steps later",
            "explanation": "Synthetic sequence validation.",
        },
        "mese": mese,
        "step_distinctness": step_distinctness,
        "step_overlap": step_overlap,
        "roadmap_readiness": readiness,
        "structure": structure,
        "similarity": {"semantic": 0.0, "tfidf": 0.0},
        "response_time": 0.0,
        "overall_score": 8.0,
        "verdict": "PASS",
        "mese_verdict": "PASS" if mese["composite"] >= config.mese_pass_threshold else "FAIL",
    }


def _validate_readiness_cases() -> list[dict]:
    samples = []
    for case in CASES:
        readiness = LLMJudgeEvaluator._roadmap_readiness(
            case["mese"],
            case["structure"],
            case.get("step_distinctness"),
        )
        assert readiness["status"] == case["expected_readiness"], (
            case["name"],
            readiness["status"],
            case["expected_readiness"],
        )
        assert case["expected_reason_contains"] in readiness["reason"], (
            case["name"],
            readiness["reason"],
        )
        samples.append(_sample_result(case, readiness))
    return samples


def _validate_step_distinctness() -> None:
    class _FakeResponse:
        content = """{
          "step_distinctness": {
            "score": 6.5,
            "verdict": "NEEDS_REVIEW",
            "reason": "Two steps have weak unique outputs.",
            "strengths": ["The roadmap separates validation from application."],
            "weak_steps": [
              {"step": 1, "label": "Create date table", "reason": "Weak unique output"},
              {"step": 2, "label": "Build calendar table", "reason": "Similar responsibility"}
            ]
          },
          "step_overlap": {
            "evaluated": true,
            "trigger": "weak_steps_detected",
            "overlapping_pairs": [
              {
                "steps": [1, 2],
                "severity": "medium",
                "overlap_type": "same_output",
                "explanation": "Both steps create a date/calendar table.",
                "recommendation": "Merge or clarify separate outputs."
              }
            ],
            "non_overlap_notes": []
          }
        }"""

    class _FakeLLM:
        def invoke(self, _prompt):
            return _FakeResponse()

    result = StepSemanticJudge(llm=_FakeLLM()).evaluate({"steps": []})
    distinctness = result["step_distinctness"]
    overlap = result["step_overlap"]
    assert distinctness["score"] == 6.5
    assert distinctness["weak_step_count"] == 2
    assert overlap["evaluated"] is True
    assert overlap["issue_count"] == 1

    readiness = LLMJudgeEvaluator._roadmap_readiness(
        {"mapping": 8, "exhaustiveness": 8, "sequence": 8, "experience": 8},
        {"score": 8, "verdict": "PASS"},
        distinctness,
    )
    assert readiness["status"] == "NEEDS_REVIEW"
    assert "Step distinctness" in readiness["reason"]


def _validate_aliases_and_flatten(samples: list[dict]) -> dict:
    aggregated = LLMJudgeEvaluator._aggregate(
        samples,
        ["relevance", "completeness", "coherence", "technical_accuracy", "context_fidelity"],
        ["mapping", "exhaustiveness", "sequence", "experience"],
    )
    assert "roadmap" in aggregated
    assert "grounding" in aggregated
    assert "readiness" in aggregated

    judge_results = {
        "per_sample": samples,
        "aggregated": aggregated,
        "pass_rate": 1.0,
        "mese_pass_rate": 1.0,
        "readiness_ready_rate": aggregated["readiness"]["ready_rate"],
        "readiness_fail_rate": aggregated["readiness"]["fail_rate"],
    }
    metrics = _flatten_metrics(judge_results, None, None)

    required_metrics = [
        "roadmap.completeness",
        "roadmap.logical_order",
        "roadmap.actionability",
        "roadmap.step_distinctness",
        "step.distinctness.score",
        "step.distinctness.weak_step_count",
        "step.overlap.issue_count",
        "step.overlap.evaluated_count",
        "roadmap.summary_score",
        "grounding.support_score",
        "roadmap.readiness_code",
        "roadmap.ready_rate",
        "roadmap.needs_review_rate",
        "roadmap.fail_rate",
    ]
    missing = [metric for metric in required_metrics if metric not in metrics]
    assert not missing, missing
    return judge_results


def _validate_ragas_alignment() -> None:
    case = {
        "name": "ragas_grounding_override_case",
        "mese": {
            "mapping": 9,
            "exhaustiveness": 9,
            "sequence": 9,
            "experience": 9,
            "composite": 9.0,
        },
        "structure": {"score": 9, "verdict": "PASS"},
    }
    sample = _sample_result(
        case,
        LLMJudgeEvaluator._roadmap_readiness(case["mese"], case["structure"]),
    )
    judge_results = {
        "per_sample": [sample],
        "aggregated": LLMJudgeEvaluator._aggregate(
            [sample],
            ["relevance", "completeness", "coherence", "technical_accuracy", "context_fidelity"],
            ["mapping", "exhaustiveness", "sequence", "experience"],
        ),
        "pass_rate": 1.0,
        "mese_pass_rate": 1.0,
    }
    ragas_results = {
        "per_sample": [{
            "question": sample["question"],
            "category": sample["category"],
            "scores": {
                "faithfulness": 0.4,
                "answer_relevancy": 0.8,
                "context_precision": 0.7,
                "context_recall": 0.6,
            },
        }],
        "aggregated": {
            "faithfulness": {"mean": 0.4},
            "answer_relevancy": {"mean": 0.8},
            "context_precision": {"mean": 0.7},
            "context_recall": {"mean": 0.6},
        },
    }

    aligned = align_judge_with_ragas(judge_results, ragas_results)
    aligned_sample = aligned["per_sample"][0]
    assert aligned_sample["mese"]["mapping"] == 4.0
    assert aligned_sample["legacy_mese"]["mapping"] == 9
    assert aligned_sample["roadmap_readiness"]["status"] == "FAIL"
    assert aligned["aggregated"]["grounding"]["support_score"] == 4.0
    assert aligned["aggregated"]["retrieval"]["context_precision"] == 0.7
    assert aligned["aggregated"]["retrieval"]["context_recall"] == 0.6
    assert aligned["aggregated"]["answer"]["relevancy"] == 0.8


def _validate_report_rendering(judge_results: dict) -> None:
    report_dir = Path(config.reports_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / "_metric_validation_smoke.html"
    csv_path = report_dir / "_metric_validation_smoke.csv"

    save_html(None, judge_results, str(html_path))
    save_human_review_csv(judge_results, str(csv_path))

    html_text = html_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8-sig")

    for expected in [
        "Batch Overview",
        "Case Details",
        "Readiness",
        "Grounding",
        "Completeness",
        "Logical Order",
        "Actionability",
    ]:
        assert expected in html_text, expected

    for expected in ["Readiness", "Readiness reason", "Grounding", "Logical Order"]:
        assert expected in csv_text, expected

    html_path.unlink(missing_ok=True)
    csv_path.unlink(missing_ok=True)


def run() -> None:
    samples = _validate_readiness_cases()
    _validate_step_distinctness()
    judge_results = _validate_aliases_and_flatten(samples)
    _validate_ragas_alignment()
    _validate_report_rendering(judge_results)

    print("Controlled metric validation passed")
    print(f"  cases: {len(CASES)}")
    print("  validated: readiness gating, step distinctness, RAGAS alignment, aliases, MLflow flattening, report rendering")


if __name__ == "__main__":
    run()
