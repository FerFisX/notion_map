"""
ragas_evaluator.py — Evaluación automática con el framework RAGAS.
Métricas: faithfulness, answer_relevancy, context_precision, context_recall.
"""

import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

from evaluation.config import config
from evaluation.dataset import EvalSample
from evaluation.rag_adapter import RagAdapter
from src.llm_provider import get_llm


def run_ragas(adapter: RagAdapter, samples: List[EvalSample]) -> dict:
    """
    Corre las 4 métricas RAGAS sobre todas las muestras.

    Returns:
        {
            "per_sample": [ { "question", "scores": {...} } ],
            "aggregated": { metric: { mean, std, min, max } },
        }
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.dataset_schema import SingleTurnSample
        from ragas import EvaluationDataset
    except ImportError:
        raise ImportError("Instala RAGAS: pip install ragas>=0.2.0")

    llm        = LangchainLLMWrapper(get_llm(temperature=config.judge_temperature, max_tokens=2048))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm = llm
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings

    ragas_samples = []
    raw_results   = []

    print(f"\n  Ejecutando RAGAS sobre {len(samples)} muestras...")
    for i, sample in enumerate(samples, 1):
        print(f"  [{i}/{len(samples)}] {sample.question[:60]}...")
        result = adapter.query(sample.question)

        ragas_samples.append(SingleTurnSample(
            user_input=sample.question,
            response=result["answer"],
            retrieved_contexts=result["contexts"],
            reference=sample.ground_truth,
        ))
        raw_results.append(result)

    dataset = EvaluationDataset(samples=ragas_samples)
    scores  = evaluate(dataset=dataset, metrics=metrics)
    df      = scores.to_pandas()

    per_sample = []
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    for i, sample in enumerate(samples):
        row = df.iloc[i]
        per_sample.append({
            "question": sample.question,
            "category": sample.category,
            "scores": {m: round(float(row.get(m, 0)), 4) for m in metric_names},
            "answer":   raw_results[i]["answer"],
            "contexts": raw_results[i]["contexts"],
        })

    # Estadísticas agregadas
    aggregated = {}
    for m in metric_names:
        vals = [s["scores"][m] for s in per_sample if m in s["scores"]]
        if vals:
            aggregated[m] = {
                "mean": round(sum(vals) / len(vals), 4),
                "min":  round(min(vals), 4),
                "max":  round(max(vals), 4),
            }

    print("  RAGAS completado.")
    return {"per_sample": per_sample, "aggregated": aggregated}
