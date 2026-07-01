"""Run normalized single-query experiments and log them to MLflow.

This runner is intentionally small and focused on comparing query-refinement
variants for the same question. Each variant runs in its own Python process so
environment flags are read cleanly when RagEngine is imported.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


VARIANTS = {
    "01_baseline_current": {
        "run_name": "normalized-01-baseline-current-time-intelligence-dax",
        "query_intent_enabled": "false",
        "context_strategy": "legacy_original_query",
        "description": "Generic query rewrite; judge retrieves its own context from the original question.",
    },
    "02_shared_context": {
        "run_name": "normalized-02-shared-context-time-intelligence-dax",
        "query_intent_enabled": "false",
        "context_strategy": "generator_context",
        "description": "Generic query rewrite; judge uses the same context as the generator.",
    },
    "03_intent_aware_query": {
        "run_name": "normalized-03-intent-aware-query-time-intelligence-dax",
        "query_intent_enabled": "true",
        "context_strategy": "generator_context",
        "description": "Intent-aware query rewrite; judge uses the same context as the generator.",
    },
}


def _git_value(args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()
    except Exception:
        return default


def _run_all(question: str, comparison_group: str, batch_name: str) -> None:
    for variant in VARIANTS:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        cmd = [
            sys.executable,
            "-m",
            "evaluation.run_query_experiments",
            "--variant",
            variant,
            "--question",
            question,
            "--comparison-group",
            comparison_group,
            "--batch-name",
            batch_name,
        ]
        print(f"\n=== Running {variant} ===")
        subprocess.run(cmd, check=True, env=env)


def _aggregate_single(sample_result: dict) -> dict:
    classic_keys = ["relevance", "completeness", "coherence", "technical_accuracy", "context_fidelity"]
    mese_keys = ["mapping", "exhaustiveness", "sequence", "experience", "composite"]
    return {
        "overall_score": sample_result["overall_score"],
        "classic": {
            key: sample_result["classic"][key]["score"]
            for key in classic_keys
        },
        "mese": {
            key: sample_result["mese"][key]
            for key in mese_keys
        },
        "sequence": {
            "mean_score": sample_result["sequence_eval"]["score"],
            "valid_pct": 1.0 if sample_result["sequence_eval"].get("is_valid") else 0.0,
        },
        "structure": {
            "mean_score": sample_result["structure"]["score"],
            "pass_rate": 1.0 if sample_result["structure"]["verdict"] == "PASS" else 0.0,
        },
        "similarity": {
            "mean_semantic": sample_result["similarity"].get("semantic", 0),
            "mean_tfidf": sample_result["similarity"].get("tfidf", 0),
        },
        "response_time": {
            "mean_s": sample_result["response_time"],
            "max_s": sample_result["response_time"],
            "min_s": sample_result["response_time"],
        },
    }


def _write_markdown(path: Path, payload: dict) -> None:
    sample = payload["judge"]["per_sample"][0]
    intent = sample.get("query_intent", {}) or {}
    retrieval = sample.get("retrieval", {}) or {}
    lines = [
        f"# {payload['run_name']}",
        "",
        f"- Variante: `{payload['variant']}`",
        f"- Pregunta original: `{sample['question']}`",
        f"- Intención: `{intent.get('intent', '')}`",
        f"- Objetivo: `{intent.get('roadmap_goal', '')}`",
        f"- Consulta refinada: `{sample.get('refined_question', '')}`",
        f"- Estrategia de contexto judge: `{sample.get('judge_context_strategy', '')}`",
        f"- Modo retrieval: `{retrieval.get('mode', '')}`",
        f"- Score general: `{sample['overall_score']}/10`",
        f"- MESE compuesto: `{sample['mese']['composite']}/10`",
        f"- Estructura: `{sample['structure']['score']}/10 ({sample['structure']['verdict']})`",
        f"- Similitud semántica: `{sample['similarity'].get('semantic', 0)}`",
        "",
        "## Roadmap generado",
        "",
        sample["answer"],
        "",
        "## Judge summary",
        "",
        sample["classic"].get("summary", ""),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_one(variant: str, question: str, comparison_group: str, batch_name: str) -> None:
    spec = VARIANTS[variant]
    os.environ["QUERY_INTENT_ENABLED"] = spec["query_intent_enabled"]
    os.environ["EVAL_CONTEXT_STRATEGY"] = spec["context_strategy"]

    import mlflow

    from evaluation.config import config
    from evaluation.llm_judge import LLMJudgeEvaluator
    from evaluation.rag_adapter import RagAdapter
    from evaluation.reporter import save_human_review_csv, save_html, save_json
    from evaluation.similarity_metrics import query_answer_relevance
    from evaluation.structure_validator import StructureValidator
    from evaluation.tracking import ARTIFACT_DIR, EXPERIMENT, TRACKING_URI
    from src.llm_provider import active_model_name

    started = time.time()
    adapter = RagAdapter()
    result = adapter.query(question)
    response_time = round(time.time() - started, 2)

    roadmap = result["roadmap"]
    answer = result["answer"]
    contexts = result.get("contexts", [])
    steps = adapter.steps_as_list(roadmap)
    steps_str = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(steps))

    judge = LLMJudgeEvaluator()
    raw_judge = judge._judge_single(
        question=question,
        roadmap_text=answer,
        contexts=contexts,
        ground_truth="(No disponible - evaluación ad-hoc normalizada)",
        steps_list=steps_str,
        refined_question=result.get("refined_question", ""),
    )
    raw_judge["mese"]["composite"] = judge._mese_composite(raw_judge["mese"])

    classic_keys = ["relevance", "completeness", "coherence", "technical_accuracy", "context_fidelity"]
    overall = round(sum(raw_judge["classic"][k]["score"] for k in classic_keys) / len(classic_keys), 2)
    verdict = "PASS" if overall >= config.pass_threshold else "FAIL"
    mese_verdict = "PASS" if raw_judge["mese"]["composite"] >= config.mese_pass_threshold else "FAIL"
    structure = StructureValidator().validate(roadmap)
    similarity = query_answer_relevance(question, answer)

    query_intent = result.get("query_intent", {}) or {}
    retrieval = result.get("retrieval", {}) or {}
    sources = roadmap.get("sources", {}) or {}
    branch = _git_value(["branch", "--show-current"])
    commit = _git_value(["rev-parse", "--short", "HEAD"])

    sample_result = {
        "question": question,
        "category": "time_intelligence_dax",
        "query_intent": query_intent,
        "intent": query_intent.get("intent", ""),
        "rewrite_strategy": "intent_aware" if spec["query_intent_enabled"] == "true" else "generic",
        "refined_question": result.get("refined_question", ""),
        "ground_truth": "(No disponible - evaluación ad-hoc normalizada)",
        "answer": answer,
        "contexts": contexts,
        "retrieval": retrieval,
        "judge_context_strategy": result.get("judge_context_strategy", ""),
        "roadmap": roadmap,
        "steps": steps,
        "expected_steps": [],
        "classic": raw_judge["classic"],
        "sequence_eval": raw_judge["sequence_eval"],
        "mese": raw_judge["mese"],
        "structure": structure,
        "similarity": similarity,
        "response_time": response_time,
        "overall_score": overall,
        "verdict": verdict,
        "mese_verdict": mese_verdict,
    }

    judge_results = {
        "per_sample": [sample_result],
        "aggregated": _aggregate_single(sample_result),
        "pass_rate": 1.0 if verdict == "PASS" else 0.0,
        "mese_pass_rate": 1.0 if mese_verdict == "PASS" else 0.0,
        "seq_valid_pct": 1.0 if raw_judge["sequence_eval"].get("is_valid") else 0.0,
    }

    payload = {
        "run_name": spec["run_name"],
        "variant": variant,
        "variant_description": spec["description"],
        "question": question,
        "judge": judge_results,
        "config": {
            "query_intent_enabled": spec["query_intent_enabled"],
            "context_strategy": spec["context_strategy"],
            "llm_provider": os.getenv("LLM_PROVIDER", "bedrock"),
            "model": active_model_name(),
            "rerank_method": os.getenv("RERANK_METHOD", "mmr"),
            "retrieval_top_n": os.getenv("RETRIEVAL_TOP_N", "5"),
            "retrieval_pool_size": os.getenv("RETRIEVAL_POOL_SIZE", "10"),
            "branch": branch,
            "commit": commit,
        },
    }

    report_dir = Path(config.reports_dir) / "normalized_time_intelligence"
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = variant
    json_path = report_dir / f"{stem}.json"
    html_path = report_dir / f"{stem}.html"
    md_path = report_dir / f"{stem}.md"
    csv_path = report_dir / f"{stem}_human_review.csv"

    save_json(payload, str(json_path))
    save_html(None, judge_results, str(html_path))
    save_human_review_csv(judge_results, str(csv_path))
    _write_markdown(md_path, payload)

    metrics = {
        "judge.overall_score": overall,
        "judge.classic.relevance": raw_judge["classic"]["relevance"]["score"],
        "judge.classic.completeness": raw_judge["classic"]["completeness"]["score"],
        "judge.classic.coherence": raw_judge["classic"]["coherence"]["score"],
        "judge.classic.technical_accuracy": raw_judge["classic"]["technical_accuracy"]["score"],
        "judge.classic.context_fidelity": raw_judge["classic"]["context_fidelity"]["score"],
        "judge.sequence.score": raw_judge["sequence_eval"]["score"],
        "judge.sequence.valid": 1 if raw_judge["sequence_eval"].get("is_valid") else 0,
        "mese.mapping": raw_judge["mese"]["mapping"],
        "mese.exhaustiveness": raw_judge["mese"]["exhaustiveness"],
        "mese.sequence": raw_judge["mese"]["sequence"],
        "mese.experience": raw_judge["mese"]["experience"],
        "mese.composite": raw_judge["mese"]["composite"],
        "structure.score": structure.get("score", 0),
        "structure.pass": 1 if structure.get("verdict") == "PASS" else 0,
        "similarity.tfidf": similarity.get("tfidf", 0),
        "similarity.semantic": similarity.get("semantic", 0),
        "roadmap.step_count": len(steps),
        "retrieval.best_score": retrieval.get("best_score", sources.get("best_score", 0)),
        "retrieval.n_contexts": retrieval.get("n_contexts", len(contexts)),
        "source.corpus_pct": sources.get("corpus_pct", 0),
        "source.web_pct": sources.get("web_pct", 0),
        "total_elapsed_s": response_time,
        "intent.confidence": float(query_intent.get("confidence") or 0),
        "comparison.overall_score": overall,
        "comparison.mese_composite": raw_judge["mese"]["composite"],
        "comparison.structure_score": structure.get("score", 0),
        "comparison.similarity_semantic": similarity.get("semantic", 0),
        "comparison.similarity_tfidf": similarity.get("tfidf", 0),
        "comparison.roadmap_step_count": len(steps),
        "comparison.retrieval_best_score": retrieval.get("best_score", sources.get("best_score", 0)),
        "comparison.source_corpus_pct": sources.get("corpus_pct", 0),
        "comparison.source_web_pct": sources.get("web_pct", 0),
    }
    params = {
        "comparison_group": comparison_group,
        "comparison_slot": variant,
        "comparison_batch": batch_name,
        "query_domain": "time_intelligence_dax",
        "question": question,
        "query_intent_enabled": spec["query_intent_enabled"],
        "query_intent": query_intent.get("intent", ""),
        "intent_goal": query_intent.get("roadmap_goal", ""),
        "refined_question": result.get("refined_question", ""),
        "judge_context_strategy": result.get("judge_context_strategy", ""),
        "llm_provider": os.getenv("LLM_PROVIDER", "bedrock"),
        "model": active_model_name(),
        "branch": branch,
        "commit": commit,
        "retrieval_mode": retrieval.get("mode", ""),
        "rerank_method": os.getenv("RERANK_METHOD", "mmr"),
        "retrieval_top_n": os.getenv("RETRIEVAL_TOP_N", "5"),
        "retrieval_pool_size": os.getenv("RETRIEVAL_POOL_SIZE", "10"),
    }

    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=Path(ARTIFACT_DIR).resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=spec["run_name"]) as run:
        mlflow.set_tags({
            "comparison_group": comparison_group,
            "comparison_slot": variant,
            "comparison_batch": batch_name,
            "query_domain": "time_intelligence_dax",
            "variant": variant,
        })
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics(metrics)
        for artifact in (json_path, html_path, md_path, csv_path):
            mlflow.log_artifact(str(artifact), artifact_path="eval")
        run_id = run.info.run_id

    print(json.dumps({
        "variant": variant,
        "run_name": spec["run_name"],
        "run_id": run_id,
        "html_path": str(html_path),
        "overall_score": overall,
        "mese_composite": raw_judge["mese"]["composite"],
        "structure_score": structure.get("score", 0),
        "query_intent": query_intent.get("intent", ""),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run normalized Time Intelligence query experiments.")
    parser.add_argument("--all", action="store_true", help="Run all normalized variants.")
    parser.add_argument("--variant", choices=sorted(VARIANTS), help="Run a single variant.")
    parser.add_argument("--question", default="¿Que es time intelligence en dax?")
    parser.add_argument("--comparison-group", default="time-intelligence-dax-normalized")
    parser.add_argument("--batch-name", default=time.strftime("normalized-%Y%m%d-%H%M%S"))
    args = parser.parse_args()

    if args.all:
        _run_all(args.question, args.comparison_group, args.batch_name)
        return
    if not args.variant:
        parser.error("Use --all or --variant.")
    _run_one(args.variant, args.question, args.comparison_group, args.batch_name)


if __name__ == "__main__":
    main()
