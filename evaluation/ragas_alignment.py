"""Attach explicitly requested RAGAS diagnostics without overriding judges."""

from __future__ import annotations

from copy import deepcopy


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def align_judge_with_ragas(
    judge_results: dict | None,
    ragas_results: dict | None,
) -> dict | None:
    """Attach RAGAS as secondary diagnostics to canonical judge results.

    This function never changes Grounding, readiness, legacy aliases, or any
    canonical metric score. It remains available for explicit comparison tools;
    the default runner modes execute Judge and RAGAS independently.
    """
    if not judge_results or not ragas_results:
        return judge_results

    judge_samples = judge_results.get("per_sample", [])
    ragas_samples = ragas_results.get("per_sample", [])
    if not judge_samples or len(judge_samples) != len(ragas_samples):
        return judge_results

    aligned_samples = []
    for judge_sample, ragas_sample in zip(judge_samples, ragas_samples):
        if judge_sample.get("question") != ragas_sample.get("question"):
            return judge_results
        sample = deepcopy(judge_sample)
        scores = deepcopy(ragas_sample.get("scores", {}))
        sample["ragas_diagnostics"] = scores
        sample["retrieval_metrics"] = {
            "context_precision": round(float(scores.get("context_precision") or 0), 4),
            "context_recall": round(float(scores.get("context_recall") or 0), 4),
        }
        sample["answer_metrics"] = {
            "relevancy": round(float(scores.get("answer_relevancy") or 0), 4),
        }
        aligned_samples.append(sample)

    aligned = deepcopy(judge_results)
    aligned["per_sample"] = aligned_samples
    aggregated = deepcopy(judge_results.get("aggregated", {}))
    aggregated["ragas"] = deepcopy(ragas_results.get("aggregated", {}))
    aggregated["retrieval"] = {
        "context_precision": _mean([
            sample["retrieval_metrics"]["context_precision"]
            for sample in aligned_samples
        ]),
        "context_recall": _mean([
            sample["retrieval_metrics"]["context_recall"]
            for sample in aligned_samples
        ]),
    }
    aggregated["answer"] = {
        "relevancy": _mean([
            sample["answer_metrics"]["relevancy"]
            for sample in aligned_samples
        ]),
    }
    aligned["aggregated"] = aggregated
    aligned["metric_alignment"] = {
        "grounding": "canonical.grounding.support_score",
        "ragas.faithfulness": "secondary_diagnostic",
        "retrieval.context_precision": "ragas.context_precision",
        "retrieval.context_recall": "ragas.context_recall",
        "answer.relevancy": "ragas.answer_relevancy",
    }
    return aligned
