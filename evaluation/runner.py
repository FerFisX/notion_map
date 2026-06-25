"""
runner.py — CLI para correr las evaluaciones.

Modos disponibles:
  python -m evaluation.runner                   # todo: RAGAS + Judge (con estructura) + Corpus
  python -m evaluation.runner --mode judge      # solo LLM Judge (incluye estructura y tiempo)
  python -m evaluation.runner --mode ragas      # solo RAGAS
  python -m evaluation.runner --mode corpus     # solo juez del knowledge base
  python -m evaluation.runner --mode structure  # solo validación de estructura (sin LLM)
  python -m evaluation.runner --samples 3       # evaluar 3 muestras
  python -m evaluation.runner --verbose         # mostrar justificaciones en consola

MLflow (tracking de experimentos):
  python -m evaluation.runner --run-name "mmr_k10"   # nombra la corrida
  python -m evaluation.runner --no-mlflow            # no registrar en MLflow
  python -m mlflow ui --backend-store-uri file:./mlruns   # ver dashboard (http://localhost:5000)
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from evaluation.config  import config
from evaluation.dataset import EVAL_SAMPLES
from evaluation.rag_adapter import RagAdapter
from evaluation.reporter import save_json, save_html, save_human_review_csv
from evaluation.tracking import log_evaluation


def print_summary(judge_results: dict, ragas_results: dict, corpus_results: dict = None):
    print("\n" + "=" * 60)
    print("  RESUMEN DE EVALUACIÓN")
    print("=" * 60)

    if judge_results:
        agg    = judge_results["aggregated"]
        mese   = agg.get("mese", {})
        seq    = agg.get("sequence", {})
        struct = agg.get("structure", {})
        rt     = agg.get("response_time", {})

        print(f"\n  LLM Judge")
        print(f"  ├─ Score General:       {agg.get('overall_score', 0):.2f}/10")
        print(f"  ├─ Pass Rate:           {judge_results['pass_rate']:.0%}")
        print(f"  ├─ MESE Compuesto:      {mese.get('composite', 0):.2f}/10")
        print(f"  │    Mapping:           {mese.get('mapping', 0):.2f}")
        print(f"  │    Exhaustividad:     {mese.get('exhaustiveness', 0):.2f}")
        print(f"  │    Secuencia:         {mese.get('sequence', 0):.2f}  <- peso 35%")
        print(f"  │    Experiencia:       {mese.get('experience', 0):.2f}")
        print(f"  ├─ MESE Pass Rate:      {judge_results['mese_pass_rate']:.0%}")
        print(f"  ├─ Secuencias OK:       {seq.get('valid_pct', 0):.0%}")
        print(f"  ├─ Estructura:          {struct.get('mean_score', 0):.2f}/10  (pass: {struct.get('pass_rate', 0):.0%})")
        print(f"  └─ Tiempo respuesta:    {rt.get('mean_s', 0):.1f}s promedio (max: {rt.get('max_s', 0):.1f}s)")

    if ragas_results:
        agg = ragas_results["aggregated"]
        print(f"\n  RAGAS")
        for metric, vals in agg.items():
            print(f"  ├─ {metric:<25} {vals.get('mean', 0):.4f}")

    if corpus_results and "overall_corpus_score" in corpus_results:
        ca = corpus_results.get("aggregated", {})
        sd = corpus_results.get("semantic_diversity", {})
        cv = corpus_results.get("coverage", {})
        print(f"\n  Corpus Judge")
        print(f"  ├─ Score Corpus:        {corpus_results['overall_corpus_score']:.2f}/10  [{corpus_results['verdict']}]")
        print(f"  ├─ Chunks evaluados:    {corpus_results['n_chunks_evaluated']}/{corpus_results['n_chunks_total']}")
        print(f"  ├─ Calidad media:       {ca.get('avg_quality', 0):.2f}/10")
        print(f"  │    Coherencia:        {ca.get('avg_coherencia', 0):.2f}")
        print(f"  │    Densidad técnica:  {ca.get('avg_densidad_tecnica', 0):.2f}")
        print(f"  │    Utilidad RAG:      {ca.get('avg_utilidad_rag', 0):.2f}")
        print(f"  ├─ Diversidad semántica: {sd.get('diversity_score', 'N/A')}")
        print(f"  │    Pares redundantes: {sd.get('redundant_pairs', 0)}")
        print(f"  └─ Chunks con issues:  {ca.get('chunks_with_issues', 0)}")
        if cv.get("observacion"):
            print(f"\n  Obs. corpus: {cv['observacion']}")

    print()
    print("=" * 60)


def run(mode: str = "all", n_samples: int = None, html: bool = True,
        verbose: bool = False, max_corpus_chunks: int = 20,
        run_name: str = None, mlflow_enabled: bool = True):

    if not os.getenv("AWS_ACCESS_KEY_ID"):
        print("[!] AWS_ACCESS_KEY_ID no encontrada en .env")
        sys.exit(1)

    samples = EVAL_SAMPLES[:n_samples] if n_samples else EVAL_SAMPLES

    if mode not in ("corpus",):
        print(f"\n  Evaluando {len(samples)} muestras | Modo: {mode}")

    adapter        = RagAdapter()
    ragas_results  = None
    judge_results  = None
    corpus_results = None

    # ── RAGAS ────────────────────────────────────────────────────────────────
    if mode in ("ragas", "all"):
        from evaluation.ragas_evaluator import run_ragas
        ragas_results = run_ragas(adapter, samples)

    # ── LLM Judge (incluye estructura + tiempo de respuesta) ─────────────────
    if mode in ("judge", "all"):
        from evaluation.llm_judge import run_judge
        judge_results = run_judge(adapter, samples, verbose=verbose)

    # ── Solo validación de estructura (sin LLM) ───────────────────────────────
    if mode == "structure":
        from evaluation.structure_validator import StructureValidator
        validator = StructureValidator()
        struct_results = []
        for sample in samples:
            result = adapter.query(sample.question)
            sv = validator.validate(result["roadmap"])
            struct_results.append({
                "question": sample.question,
                "structure": sv,
            })
            status = "[PASS]" if sv["verdict"] == "PASS" else "[FAIL]"
            print(f"  {status} {sample.question[:60]}")
            print(f"         score={sv['score']}/10  checks={sv['passed']}/{sv['total_checks']}")
            for v in sv["violations"]:
                print(f"         ! {v}")
        print(f"\n  Estructura evaluada para {len(struct_results)} muestras.")
        return

    # ── Corpus Judge ──────────────────────────────────────────────────────────
    if mode in ("corpus", "all"):
        from evaluation.corpus_judge import run_corpus_judge
        corpus_results = run_corpus_judge(max_chunks=max_corpus_chunks)

    # ── Guardar resultados ────────────────────────────────────────────────────
    os.makedirs(config.reports_dir, exist_ok=True)

    combined = {
        "ragas":  ragas_results,
        "judge":  judge_results,
        "corpus": corpus_results,
        "config": {
            "model":        config.bedrock_model_id,
            "n_samples":    len(samples),
            "mese_weights": config.mese_weights,
            "thresholds": {
                "pass":     config.pass_threshold,
                "mese":     config.mese_pass_threshold,
                "sequence": config.sequence_min_score,
            },
        },
    }

    json_path = os.path.join(config.reports_dir, "eval_report.json")
    save_json(combined, json_path)

    if html:
        html_path = os.path.join(config.reports_dir, "eval_report.html")
        save_html(ragas_results, judge_results, html_path, corpus_results=corpus_results)

    csv_path = None
    if judge_results:
        csv_path = os.path.join(config.reports_dir, "human_review.csv")
        save_human_review_csv(judge_results, csv_path)

    # ── MLflow tracking (registra esta corrida para comparar experimentos) ────
    rerank_method = os.getenv("RERANK_METHOD", "mmr")
    params = {
        "mode":          mode,
        "model":         config.bedrock_model_id,
        "llm_provider":  os.getenv("LLM_PROVIDER", "bedrock"),
        "n_samples":     len(samples),
        "rerank_method": rerank_method,
        "pool_size":     os.getenv("RETRIEVAL_POOL_SIZE", "10"),
        "top_n":         os.getenv("RETRIEVAL_TOP_N", "5"),
        "judge_temp":    config.judge_temperature,
        "pass_threshold":      config.pass_threshold,
        "mese_pass_threshold": config.mese_pass_threshold,
    }
    artifacts = [json_path]
    if html:
        artifacts.append(os.path.join(config.reports_dir, "eval_report.html"))
    if csv_path:
        artifacts.append(csv_path)

    log_evaluation(
        run_name=run_name or f"{mode}-{rerank_method}",
        params=params,
        judge_results=judge_results,
        ragas_results=ragas_results,
        corpus_results=corpus_results,
        artifacts=artifacts,
        enabled=mlflow_enabled,
    )

    print_summary(judge_results, ragas_results, corpus_results)

    print(f"  Archivos generados en: {config.reports_dir}")
    print(f"  ├─ eval_report.json")
    if html:
        print(f"  ├─ eval_report.html  <- abrir en navegador")
    if judge_results:
        print(f"  └─ human_review.csv  <- tabla para revisión humana")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NotionMap Evaluation Runner")
    parser.add_argument("--mode", choices=["all", "ragas", "judge", "corpus", "structure"],
                        default="all")
    parser.add_argument("--samples",      type=int, default=None,
                        help="Número de muestras (default: todas)")
    parser.add_argument("--no-html",      action="store_true", help="No generar HTML")
    parser.add_argument("--verbose",      action="store_true",
                        help="Mostrar scores y justificaciones en consola")
    parser.add_argument("--max-chunks",   type=int, default=20,
                        help="Máximo chunks a evaluar en corpus judge (default: 20)")
    parser.add_argument("--run-name",     type=str, default=None,
                        help="Nombre del experimento en MLflow (default: modo-rerank)")
    parser.add_argument("--no-mlflow",    action="store_true",
                        help="No registrar la corrida en MLflow")
    args = parser.parse_args()

    run(
        mode=args.mode,
        n_samples=args.samples,
        html=not args.no_html,
        verbose=args.verbose,
        max_corpus_chunks=args.max_chunks,
        run_name=args.run_name,
        mlflow_enabled=not args.no_mlflow,
    )
