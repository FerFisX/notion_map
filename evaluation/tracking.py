"""
tracking.py — Capa opcional de MLflow para registrar cada corrida de evaluación.

NO reemplaza el sistema de evaluación; se pone encima.
Registra por cada run:
  - params:    modelo, método de re-ranking, k, umbrales, nº de muestras...
  - metrics:   MESE, estructura, similitud, tiempos, RAGAS, corpus...
  - artifacts: eval_report.html / .json / human_review.csv

Si MLflow no está instalado, todo se omite silenciosamente (no rompe la evaluación).

Ver el dashboard:
  python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
  -> http://localhost:5000
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


# ── Aplana los resultados a un dict plano de métricas numéricas ───────────────

def _flatten_metrics(judge: dict, ragas: dict, corpus: dict) -> dict:
    m: dict[str, float] = {}

    if judge:
        agg    = judge.get("aggregated", {})
        mese   = agg.get("mese", {})
        seq    = agg.get("sequence", {})
        struct = agg.get("structure", {})
        sim    = agg.get("similarity", {})
        rt     = agg.get("response_time", {})

        m["judge.overall_score"]   = agg.get("overall_score", 0)
        m["judge.pass_rate"]       = judge.get("pass_rate", 0)
        m["judge.mese_pass_rate"]  = judge.get("mese_pass_rate", 0)
        m["mese.composite"]        = mese.get("composite", 0)
        m["mese.mapping"]          = mese.get("mapping", 0)
        m["mese.exhaustiveness"]   = mese.get("exhaustiveness", 0)
        m["mese.sequence"]         = mese.get("sequence", 0)
        m["mese.experience"]       = mese.get("experience", 0)
        m["sequence.mean_score"]   = seq.get("mean_score", 0)
        m["sequence.valid_pct"]    = seq.get("valid_pct", 0)
        m["structure.mean_score"]  = struct.get("mean_score", 0)
        m["structure.pass_rate"]   = struct.get("pass_rate", 0)
        m["similarity.semantic"]   = sim.get("mean_semantic", 0)
        m["similarity.tfidf"]      = sim.get("mean_tfidf", 0)
        m["response_time.mean_s"]  = rt.get("mean_s", 0)
        m["response_time.max_s"]   = rt.get("max_s", 0)

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


# ── Registra una corrida completa ─────────────────────────────────────────────

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

    # Crear el experimento con ubicación de artefactos local si no existe
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=Path(ARTIFACT_DIR).as_uri())
    mlflow.set_experiment(EXPERIMENT)

    metrics = _flatten_metrics(judge_results, ragas_results, corpus_results)

    with mlflow.start_run(run_name=run_name):
        # params (todo a str para evitar problemas de tipos)
        mlflow.log_params({k: str(v) for k, v in params.items()})
        if metrics:
            mlflow.log_metrics(metrics)
        for path in (artifacts or []):
            if path and os.path.exists(path):
                mlflow.log_artifact(path)

    print(f"  [MLflow] Run '{run_name}' registrado ({len(metrics)} métricas).")
    print(f"  [MLflow] Dashboard: python -m mlflow ui --backend-store-uri \"{TRACKING_URI}\"")
