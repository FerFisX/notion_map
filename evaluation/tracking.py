"""
Capa opcional de MLflow para registrar cada corrida de evaluación.
Si MLflow no está instalado, todo se omite silenciosamente.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH      = os.path.join(BASE_DIR, "mlflow.db")
ARTIFACT_DIR = os.path.join(BASE_DIR, "mlartifacts")
EXPERIMENT   = "NotionMap-RAG-Eval"

# MLflow 3.x: el file store ('./mlruns') está deprecado; usamos backend SQLite.
TRACKING_URI = "sqlite:///" + DB_PATH.replace("\\", "/")


def mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


# aplana los resultados a un dict de métricas numéricas
def _flatten_metrics(judge: dict, ragas: dict, corpus: dict) -> dict:
    m: dict[str, float] = {}

    if judge:
        agg = judge.get("aggregated", {})
        metric_results = agg.get("metrics", {})
        readiness = agg.get("readiness", {})
        diagnostics = agg.get("diagnostics", {})
        schema = agg.get("schema_validity", {})
        rt = agg.get("response_time", {})

        m["evaluation.sample_count"] = judge.get("sample_count", 0)
        m["evaluation.failure_count"] = judge.get("failure_count", 0)
        for name, values in metric_results.items():
            prefix = f"roadmap.{name}"
            m[f"{prefix}.mean_score"] = values.get("mean_score", 0)
            m[f"{prefix}.pass_rate"] = values.get("pass_rate", 0)
            m[f"{prefix}.needs_review_rate"] = values.get("needs_review_rate", 0)
            m[f"{prefix}.fail_rate"] = values.get("fail_rate", 0)

        if schema:
            m["schema_validity.mean_score"] = schema.get("mean_score", 0)
            m["schema_validity.pass_rate"] = schema.get("pass_rate", 0)
        for name, value in diagnostics.items():
            m[f"diagnostics.{name}"] = value

        m["roadmap.ready_rate"] = readiness.get("ready_rate", 0)
        m["roadmap.needs_review_rate"] = readiness.get("needs_review_rate", 0)
        m["roadmap.fail_rate"] = readiness.get("fail_rate", 0)
        m["roadmap.not_evaluated_rate"] = readiness.get("not_evaluated_rate", 0)
        m["response_time.generation_mean_s"] = rt.get("mean_s", 0)
        m["response_time.generation_max_s"] = rt.get("max_s", 0)
        m["response_time.total_wall_s"] = agg.get("total_wall_time_s", 0)
        for name, values in agg.get("generation_stage_timings", {}).items():
            stage = name.removesuffix("_s")
            m[f"generation.{stage}.mean_s"] = values.get("mean_s", 0)
            m[f"generation.{stage}.max_s"] = values.get("max_s", 0)
        for name, values in agg.get("metric_timings", {}).items():
            m[f"response_time.{name}.mean_s"] = values.get("mean_s", 0)
            m[f"response_time.{name}.max_s"] = values.get("max_s", 0)

    if ragas:
        for name, vals in ragas.get("aggregated", {}).items():
            m[f"ragas.{name}"] = vals.get("mean", 0)

    if corpus and "overall_corpus_score" in corpus:
        ca = corpus.get("aggregated", {})
        sd = corpus.get("semantic_diversity", {}) or {}
        m["corpus.overall_score"]      = corpus.get("overall_corpus_score", 0)
        m["corpus.avg_quality"]        = ca.get("avg_quality", 0)
        m["corpus.avg_coherencia"]     = ca.get("avg_coherencia", 0)
        m["corpus.avg_densidad"]       = ca.get("avg_densidad_tecnica", 0)
        m["corpus.avg_utilidad_rag"]   = ca.get("avg_utilidad_rag", 0)
        if sd.get("diversity_score") is not None:
            m["corpus.diversity"]      = sd.get("diversity_score", 0)
        m["corpus.redundant_pairs"]    = sd.get("redundant_pairs", 0)

    return m


# registra una corrida completa
def log_evaluation(
    run_name:       str,
    params:         dict,
    judge_results:  dict = None,
    ragas_results:  dict = None,
    corpus_results: dict = None,
    artifacts:      list[str] = None,
    enabled:        bool = True,
) -> None:
    """Crea un run de MLflow con params, métricas y artefactos."""
    if not enabled:
        return
    if not mlflow_available():
        print("  [MLflow] no instalado — tracking omitido. (pip install mlflow)")
        return

    import mlflow

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)

    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=Path(ARTIFACT_DIR).as_uri())
    mlflow.set_experiment(EXPERIMENT)

    metrics = _flatten_metrics(judge_results, ragas_results, corpus_results)

    with mlflow.start_run(run_name=run_name):
        # params a str para evitar problemas de tipos
        mlflow.log_params({k: str(v) for k, v in params.items()})
        if metrics:
            mlflow.log_metrics(metrics)
        for path in (artifacts or []):
            if path and os.path.exists(path):
                mlflow.log_artifact(path)

    print(f"  [MLflow] Run '{run_name}' registrado ({len(metrics)} métricas).")
    print(f"  [MLflow] Dashboard: python -m mlflow ui --backend-store-uri \"{TRACKING_URI}\"")


def log_generation_benchmark(
    run_name: str,
    params: dict,
    metrics: dict[str, float],
    artifacts: list[str] | None = None,
    enabled: bool = True,
) -> None:
    """Persist a generation-only benchmark without fabricating judge results."""
    if not enabled:
        return
    if not mlflow_available():
        print("  [MLflow] no instalado — benchmark no registrado.")
        return

    import mlflow

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(
            EXPERIMENT, artifact_location=Path(ARTIFACT_DIR).as_uri()
        )
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("run_family", "roadmap_generation_benchmark")
        mlflow.log_params({key: str(value) for key, value in params.items()})
        if metrics:
            mlflow.log_metrics(metrics)
        for path in artifacts or []:
            if path and os.path.exists(path):
                mlflow.log_artifact(path)

    print(f"  [MLflow] Benchmark '{run_name}' registrado ({len(metrics)} métricas).")
