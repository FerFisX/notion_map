"""Align redesigned metrics with RAGAS when RAGAS results are available."""

from __future__ import annotations

from copy import deepcopy

from evaluation.config import config
from evaluation.llm_judge import LLMJudgeEvaluator


def _score_0_10(value: float | int | None) -> float:
    """Convert a RAGAS 0-1 score to the project 0-10 score scale."""
    if value is None:
        return 0.0
    return round(max(0.0, min(1.0, float(value))) * 10, 2)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _mese_composite(mese: dict) -> float:
    weights = config.mese_weights
    return round(
        mese["mapping"] * weights["mapping"]
        + mese["exhaustiveness"] * weights["exhaustiveness"]
        + mese["sequence"] * weights["sequence"]
        + mese["experience"] * weights["experience"],
        2,
    )


def align_judge_with_ragas(judge_results: dict | None, ragas_results: dict | None) -> dict | None:
    """Use RAGAS as the primary source for RAG/retrieval metrics.

    The alignment intentionally avoids exposing duplicate first-class metrics:
    - faithfulness becomes the official grounding support score;
    - context precision/recall become retrieval metrics;
    - answer relevancy becomes answer relevance.

    Legacy judge grounding is preserved only under ``legacy_mese`` for raw/debug
    traceability.
    """
    if not judge_results or not ragas_results:
        return judge_results

    judge_samples = judge_results.get("per_sample", [])
    ragas_samples = ragas_results.get("per_sample", [])
    if not judge_samples or len(judge_samples) != len(ragas_samples):
        return judge_results

    aligned_samples = []
    for judge_sample, ragas_sample in zip(judge_samples, ragas_samples):
        sample = deepcopy(judge_sample)
        ragas_scores = ragas_sample.get("scores", {})

        faithfulness = ragas_scores.get("faithfulness")
        answer_relevancy = ragas_scores.get("answer_relevancy")
        context_precision = ragas_scores.get("context_precision")
        context_recall = ragas_scores.get("context_recall")

        sample["legacy_mese"] = deepcopy(sample.get("mese", {}))
        sample["ragas_alignment"] = {
            "grounding_source": "ragas.faithfulness",
            "retrieval_precision_source": "ragas.context_precision",
            "retrieval_recall_source": "ragas.context_recall",
            "answer_relevancy_source": "ragas.answer_relevancy",
        }

        if faithfulness is not None:
            sample["mese"]["mapping"] = _score_0_10(faithfulness)
            sample["mese"]["mapping_justification"] = (
                "Grounding score aligned from RAGAS faithfulness."
            )

        sample["mese"]["composite"] = _mese_composite(sample["mese"])
        sample["mese_verdict"] = (
            "PASS" if sample["mese"]["composite"] >= config.mese_pass_threshold else "FAIL"
        )
        sample["roadmap_readiness"] = LLMJudgeEvaluator._roadmap_readiness(
            sample["mese"],
            sample.get("structure", {}),
        )
        sample["grounding"] = {
            "support_score": sample["mese"]["mapping"],
            "source": "ragas.faithfulness" if faithfulness is not None else "legacy_fallback",
        }
        sample["retrieval_metrics"] = {
            "context_precision": round(float(context_precision or 0), 4),
            "context_recall": round(float(context_recall or 0), 4),
        }
        sample["answer_metrics"] = {
            "relevancy": round(float(answer_relevancy or 0), 4),
        }
        aligned_samples.append(sample)

    aggregated = LLMJudgeEvaluator._aggregate(
        aligned_samples,
        ["relevance", "completeness", "coherence", "technical_accuracy", "context_fidelity"],
        ["mapping", "exhaustiveness", "sequence", "experience"],
    )

    retrieval_precision = [
        s["retrieval_metrics"]["context_precision"] for s in aligned_samples
    ]
    retrieval_recall = [
        s["retrieval_metrics"]["context_recall"] for s in aligned_samples
    ]
    answer_relevancy = [
        s["answer_metrics"]["relevancy"] for s in aligned_samples
    ]
    aggregated["retrieval"] = {
        "context_precision": _mean(retrieval_precision),
        "context_recall": _mean(retrieval_recall),
    }
    aggregated["answer"] = {
        "relevancy": _mean(answer_relevancy),
    }
    aggregated["grounding"]["source"] = "ragas.faithfulness"

    pass_rate = sum(1 for s in aligned_samples if s["verdict"] == "PASS") / len(aligned_samples)
    mese_pass_rate = sum(1 for s in aligned_samples if s["mese_verdict"] == "PASS") / len(aligned_samples)
    ready_rate = sum(1 for s in aligned_samples if s["roadmap_readiness"]["status"] == "READY") / len(aligned_samples)
    fail_rate = sum(1 for s in aligned_samples if s["roadmap_readiness"]["status"] == "FAIL") / len(aligned_samples)

    judge_results["per_sample"] = aligned_samples
    judge_results["aggregated"] = aggregated
    judge_results["pass_rate"] = round(pass_rate, 4)
    judge_results["mese_pass_rate"] = round(mese_pass_rate, 4)
    judge_results["readiness_ready_rate"] = round(ready_rate, 4)
    judge_results["readiness_fail_rate"] = round(fail_rate, 4)
    judge_results["metric_alignment"] = {
        "grounding": "ragas.faithfulness",
        "retrieval.context_precision": "ragas.context_precision",
        "retrieval.context_recall": "ragas.context_recall",
        "answer.relevancy": "ragas.answer_relevancy",
    }
    return judge_results
