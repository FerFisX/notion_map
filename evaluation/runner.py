"""CLI for canonical roadmap, optional RAGAS, and corpus evaluations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import urllib.request
from typing import Iterable

from evaluation.config import config
from evaluation.dataset import EVAL_SAMPLES, EvalSample
from evaluation.rag_adapter import RagAdapter
from evaluation.reporter import save_html, save_json
from evaluation.run_session import (
    RunCheckpoint,
    RunProgress,
    create_run_dir,
    selected_metric_count,
)
from evaluation.tracking import log_evaluation
from src.llm_provider import active_model_name


CLI_METRICS = (
    "grounding",
    "completeness",
    "actionability",
    "logical_order",
    "structure_quality",
    "step_distinctness",
    "schema_validity",
)

_ORCHESTRATOR_NAMES = {
    "grounding": "grounding",
    "completeness": "completeness",
    "actionability": "actionability",
    "logical_order": "logical_order",
    "structure_quality": "structure_quality",
    "step_distinctness": "step_semantics",
    "schema_validity": "schema_validity",
}


def _check_ollama(model: str) -> tuple[bool, str]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = {
            str(item.get("name", ""))
            for item in payload.get("models", [])
            if isinstance(item, dict)
        }
        available = model in names or any(name.split(":")[0] == model for name in names)
        return available, f"Ollama model {'available' if available else 'not found'}: {model}"
    except Exception as exc:
        return False, f"Ollama unavailable at {base_url}: {type(exc).__name__}"


def _preflight(
    mode: str,
    samples: list[EvalSample],
    mlflow_enabled: bool,
    requires_generation: bool,
) -> None:
    checks: list[tuple[bool, str, bool]] = []
    if mode != "corpus":
        checks.append((bool(samples), f"Samples selected: {len(samples)}", True))

    provider = os.getenv("LLM_PROVIDER", "bedrock").lower().strip()
    model = active_model_name()
    checks.append((provider in {"bedrock", "anthropic", "openai", "gemini", "ollama"}, f"Provider: {provider} | model: {model}", True))
    credential_vars = {
        "bedrock": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        "anthropic": ("ANTHROPIC_API_KEY",),
        "openai": ("OPENAI_API_KEY",),
        "gemini": ("GEMINI_API_KEY",),
    }
    if provider == "ollama":
        ok, message = _check_ollama(model)
        checks.append((ok, message, True))
    elif provider in credential_vars:
        missing = [name for name in credential_vars[provider] if not os.getenv(name)]
        checks.append((not missing, "Provider credentials configured" if not missing else f"Missing credentials: {', '.join(missing)}", True))

    if requires_generation or mode == "corpus":
        checks.append((os.path.isdir(config.vectorstore_dir), f"Vectorstore: {config.vectorstore_dir}", True))
    if mode == "ragas":
        checks.append((importlib.util.find_spec("ragas") is not None, "RAGAS dependency installed", True))
    if mlflow_enabled:
        checks.append((importlib.util.find_spec("mlflow") is not None, "MLflow dependency installed", False))

    print("\nPREFLIGHT")
    blocking_failures = []
    for passed, message, blocking in checks:
        label = "OK" if passed else "FAIL" if blocking else "WARN"
        print(f"  [{label}] {message}")
        if not passed and blocking:
            blocking_failures.append(message)
    if blocking_failures:
        raise ValueError("Preflight failed: " + "; ".join(blocking_failures))


def _print_execution_plan(
    mode: str,
    samples: list[EvalSample],
    selected_metrics: list[str] | None,
    requires_generation: bool,
    mlflow_enabled: bool,
    resume: bool,
) -> None:
    metrics = selected_metrics or list(_ORCHESTRATOR_NAMES.values())
    semantic_calls = sum(name != "schema_validity" for name in metrics)
    print("\nEXECUTION PLAN")
    print(f"  Mode: {mode}")
    if mode == "judge":
        print(f"  Samples: {len(samples)}")
        print(f"  Metrics: {', '.join(metrics)}")
        print(f"  Roadmap generations: {len(samples) if requires_generation else 0}")
        print(f"  Judge calls: {len(samples) * semantic_calls}")
        print(f"  Deterministic schema checks: {len(samples) if 'schema_validity' in metrics else 0}")
        print(f"  Resume: {'yes' if resume else 'no'}")
    print(f"  MLflow: {'enabled' if mlflow_enabled else 'disabled'}")
    print("  No LLM calls were made by this dry run.")


def parse_metrics(value: str | Iterable[str] | None) -> list[str] | None:
    """Normalize public metric names into the internal orchestrator selection."""
    if value is None:
        return None
    raw = value.split(",") if isinstance(value, str) else list(value)
    names = []
    for item in raw:
        name = str(item).strip().lower()
        if name and name not in names:
            names.append(name)
    unknown = set(names) - set(CLI_METRICS)
    if unknown:
        raise ValueError(
            f"Unknown metrics: {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(CLI_METRICS)}"
        )
    return [_ORCHESTRATOR_NAMES[name] for name in names]


def parse_sample_indices(value: str | None) -> list[int] | None:
    """Parse one-based sample positions while preserving the requested order."""
    if value is None:
        return None
    tokens = [item.strip() for item in value.split(",") if item.strip()]
    if not tokens:
        raise ValueError("--sample-indices requires at least one index.")
    try:
        indices = [int(item) for item in tokens]
    except ValueError as exc:
        raise ValueError("--sample-indices must contain comma-separated integers.") from exc
    if any(index < 1 for index in indices):
        raise ValueError("--sample-indices uses one-based positive indices.")
    if len(indices) != len(set(indices)):
        raise ValueError("--sample-indices cannot contain duplicate indices.")
    return indices


def _print_judge_summary(results: dict) -> None:
    aggregate = results.get("aggregated", {})
    metrics = aggregate.get("metrics", {})
    readiness = aggregate.get("readiness", {})
    schema = aggregate.get("schema_validity", {})
    print("\n" + "=" * 60)
    print("  ROADMAP EVALUATION SUMMARY")
    print("=" * 60)
    for name, values in metrics.items():
        label = name.replace("_", " ").title()
        print(f"  {label:<24} {values.get('mean_score', 0):5.2f}/10")
    if schema:
        print(f"  {'Schema Validity':<24} {schema.get('mean_score', 0):5.2f}/10")
    counts = readiness.get("counts", {})
    print("\n  Readiness")
    print(f"    Ready:        {counts.get('READY', 0)}")
    print(f"    Needs review: {counts.get('NEEDS_REVIEW', 0)}")
    print(f"    Failed:       {counts.get('FAIL', 0)}")
    print(f"    Not evaluated:{counts.get('NOT_EVALUATED', 0):2d}")
    response = aggregate.get("response_time", {})
    print(f"\n  Generation time: {response.get('mean_s', 0):.1f}s mean")


def _print_ragas_summary(results: dict) -> None:
    print("\nRAGAS ADDITIONAL DIAGNOSTICS")
    for name, values in results.get("aggregated", {}).items():
        print(f"  {name:<24} {values.get('mean', 0):.4f}")


def _print_corpus_summary(results: dict) -> None:
    aggregate = results.get("aggregated", {})
    print("\nCORPUS DIAGNOSTICS")
    print(f"  Overall score:       {results.get('overall_corpus_score', 0):.2f}/10")
    print(f"  Average quality:     {aggregate.get('avg_quality', 0):.2f}/10")
    print(f"  Chunks evaluated:    {results.get('n_chunks_evaluated', 0)}")


def run(
    mode: str = "judge",
    n_samples: int | None = None,
    html: bool = True,
    verbose: bool = False,
    max_corpus_chunks: int = 20,
    run_name: str | None = None,
    mlflow_enabled: bool = True,
    metrics: str | Iterable[str] | None = None,
    resume: str | None = None,
    dry_run: bool = False,
    sample_indices: str | None = None,
) -> None:
    if mode not in {"judge", "ragas", "corpus"}:
        raise ValueError("Mode must be judge, ragas, or corpus.")
    if mode != "judge" and metrics is not None:
        raise ValueError("--metrics is available only with --mode judge.")
    if resume and mode != "judge":
        raise ValueError("--resume is available only with --mode judge.")
    if resume and metrics is not None:
        raise ValueError("A resumed run uses the metric selection stored in its checkpoint.")
    if n_samples is not None and sample_indices is not None:
        raise ValueError("Use either --samples or --sample-indices, not both.")
    if resume and (n_samples is not None or sample_indices is not None):
        raise ValueError("A resumed run uses the samples stored in its checkpoint.")

    rerank_method = os.getenv("RERANK_METHOD", "mmr")
    checkpoint = None
    if resume:
        checkpoint = RunCheckpoint.load(resume)
        if checkpoint.state.get("mode") != "judge":
            raise ValueError("Only judge checkpoints can be resumed.")
        selected_metrics = list(checkpoint.state.get("selected_metrics", []))
        samples = [
            EvalSample(**entry["sample"])
            for entry in checkpoint.state.get("samples", [])
        ]
        effective_run_name = str(checkpoint.state.get("run_name", "resumed-judge"))
        run_dir = checkpoint.run_dir
    else:
        selected_metrics = parse_metrics(metrics)
        requested_indices = parse_sample_indices(sample_indices)
        if requested_indices:
            invalid = [index for index in requested_indices if index > len(EVAL_SAMPLES)]
            if invalid:
                raise ValueError(
                    "--sample-indices out of range: "
                    f"{', '.join(map(str, invalid))}; dataset has {len(EVAL_SAMPLES)} samples."
                )
            samples = [EVAL_SAMPLES[index - 1] for index in requested_indices]
        else:
            samples = EVAL_SAMPLES[:n_samples] if n_samples else EVAL_SAMPLES
        effective_run_name = run_name or f"{mode}-{rerank_method}"
        run_dir = None
    requires_generation = (
        any(not entry.get("generated") for entry in checkpoint.state.get("samples", []))
        if checkpoint
        else mode in {"judge", "ragas"}
    )
    _preflight(mode, samples, mlflow_enabled, requires_generation)
    if dry_run:
        _print_execution_plan(
            mode,
            samples,
            selected_metrics,
            requires_generation,
            mlflow_enabled,
            bool(resume),
        )
        return
    if checkpoint:
        checkpoint.mark("RUNNING")
    else:
        run_dir = create_run_dir(config.reports_dir, effective_run_name)
        if mode == "judge":
            checkpoint = RunCheckpoint.create(
                run_dir,
                effective_run_name,
                mode,
                selected_metrics or list(_ORCHESTRATOR_NAMES.values()),
                samples,
                metadata={
                    "model": active_model_name(),
                    "llm_provider": os.getenv("LLM_PROVIDER", "bedrock"),
                    "ollama_seed": os.getenv("OLLAMA_SEED", ""),
                },
            )
    total_units = (
        checkpoint.remaining_units()
        if checkpoint
        else len(samples) * (1 + selected_metric_count(selected_metrics))
        if mode == "judge"
        else 1
    )
    progress = RunProgress(run_dir, total_units)
    ragas_results = None
    judge_results = None
    corpus_results = None

    if mode == "judge":
        from evaluation.metric_orchestrator import CanonicalMetricEvaluator, run_canonical_metrics

        print(f"\nEvaluating {len(samples)} samples | mode: judge")
        try:
            needs_generation = any(
                not entry.get("generated")
                for entry in checkpoint.state.get("samples", [])
            )
            adapter = RagAdapter() if needs_generation else None
            judge_results = run_canonical_metrics(
                adapter,
                samples,
                verbose=verbose,
                enabled_metrics=selected_metrics,
                progress=progress,
                checkpoint=checkpoint,
            )
        except KeyboardInterrupt:
            checkpoint.mark("INTERRUPTED")
            partial_samples = checkpoint.completed_results()
            partial_results = {
                "per_sample": partial_samples,
                "aggregated": (
                    CanonicalMetricEvaluator._aggregate(partial_samples)
                    if partial_samples else {}
                ),
                "sample_count": len(partial_samples),
                "failure_count": 0,
                "failures": [],
                "metric_contract": "canonical_v1_partial",
            }
            partial_path = str(run_dir / "eval_report.partial.json")
            save_json({
                "mode": "judge",
                "status": "INTERRUPTED",
                "judge": partial_results,
                "checkpoint": str(checkpoint.path),
            }, partial_path)
            if html and partial_samples:
                save_html(
                    None,
                    partial_results,
                    str(run_dir / "eval_report.partial.html"),
                )
            progress.finish("INTERRUPTED")
            print(f"\nPartial artifacts saved in: {run_dir}")
            print(f"Resume with: python -m evaluation.runner --resume \"{run_dir}\"")
            return
        except Exception as exc:
            checkpoint.mark("FAILED")
            progress.stage_failed("Judge evaluation", exc)
            progress.finish("FAILED")
            raise
        judge_results["aggregated"]["total_wall_time_s"] = progress.elapsed_seconds()
        _print_judge_summary(judge_results)
    elif mode == "ragas":
        from evaluation.ragas_evaluator import run_ragas

        print(f"\nEvaluating {len(samples)} samples | mode: ragas")
        progress.stage_started("RAGAS evaluation")
        adapter = RagAdapter()
        started = progress.elapsed_seconds()
        ragas_results = run_ragas(adapter, samples)
        progress.stage_completed(
            "RAGAS evaluation",
            progress.elapsed_seconds() - started,
        )
        _print_ragas_summary(ragas_results)
    else:
        from evaluation.corpus_judge import run_corpus_judge

        progress.stage_started("Corpus evaluation")
        started = progress.elapsed_seconds()
        corpus_results = run_corpus_judge(max_chunks=max_corpus_chunks)
        progress.stage_completed(
            "Corpus evaluation",
            progress.elapsed_seconds() - started,
        )
        _print_corpus_summary(corpus_results)

    combined = {
        "mode": mode,
        "ragas": ragas_results,
        "judge": judge_results,
        "corpus": corpus_results,
        "config": {
            "model": active_model_name(),
            "n_samples": len(samples) if mode != "corpus" else None,
            "metrics": selected_metrics if mode == "judge" else None,
            "metric_contract": "canonical_v1" if mode == "judge" else None,
            "run_directory": str(run_dir),
        },
    }
    json_path = str(run_dir / "eval_report.json")
    save_json(combined, json_path)

    artifacts = [json_path]
    if html:
        html_path = str(run_dir / "eval_report.html")
        save_html(ragas_results, judge_results, html_path, corpus_results=corpus_results)
        artifacts.append(html_path)

    params = {
        "mode": mode,
        "model": active_model_name(),
        "llm_provider": os.getenv("LLM_PROVIDER", "bedrock"),
        "n_samples": len(samples) if mode != "corpus" else 0,
        "metrics": (
            ",".join(selected_metrics or list(_ORCHESTRATOR_NAMES.values()))
            if mode == "judge"
            else "not_applicable"
        ),
        "metric_contract": "canonical_v1" if mode == "judge" else "not_applicable",
        "rerank_method": rerank_method,
        "pool_size": os.getenv("RETRIEVAL_POOL_SIZE", "10"),
        "top_n": os.getenv("RETRIEVAL_TOP_N", "5"),
        "judge_temp": config.judge_temperature,
    }
    artifacts.append(str(progress.log_path))
    if checkpoint:
        artifacts.append(str(checkpoint.path))
    log_evaluation(
        run_name=effective_run_name,
        params=params,
        judge_results=judge_results,
        ragas_results=ragas_results,
        corpus_results=corpus_results,
        artifacts=artifacts,
        enabled=mlflow_enabled,
    )

    if checkpoint:
        checkpoint.mark("COMPLETED")
    progress.finish("COMPLETED")
    print(f"\nFiles generated in: {run_dir}")
    print("  - eval_report.json")
    if html:
        print("  - eval_report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NotionMap Evaluation Runner")
    parser.add_argument(
        "--mode",
        choices=("judge", "ragas", "corpus"),
        default="judge",
        help="Evaluation family to run (default: judge).",
    )
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help="Comma-separated one-based dataset indices, for example: 15,16,17.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help=(
            "Comma-separated canonical metrics for judge mode: "
            + ",".join(CLI_METRICS)
        ),
    )
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-chunks", type=int, default=20)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume an interrupted judge run from its directory or checkpoint.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print the execution plan without LLM calls.",
    )
    parser.add_argument("--list-metrics", action="store_true")
    parser.add_argument("--list-samples", action="store_true")
    args = parser.parse_args()
    if args.list_metrics:
        for name in CLI_METRICS:
            print(name)
        raise SystemExit(0)
    if args.list_samples:
        for index, sample in enumerate(EVAL_SAMPLES, 1):
            print(f"{index}: [{sample.category}] {sample.question}")
        raise SystemExit(0)
    try:
        run(
            mode=args.mode,
            n_samples=args.samples,
            sample_indices=args.sample_indices,
            html=not args.no_html,
            verbose=args.verbose,
            max_corpus_chunks=args.max_chunks,
            run_name=args.run_name,
            mlflow_enabled=not args.no_mlflow,
            metrics=args.metrics,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        parser.error(str(exc))
