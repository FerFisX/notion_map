"""Benchmark roadmap generation without running evaluation judges."""

from __future__ import annotations

import argparse
import html
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.config import config
from evaluation.benchmarks.generation_reporting import (
    aggregate_generation_results as _aggregate,
    recovery_status as _recovery_status,
    schema_display as _schema_display,
    schema_not_evaluated as _schema_not_evaluated,
)
from evaluation.dataset import EVAL_SAMPLES
from evaluation.metrics.structure_validator import StructureValidator
from evaluation.rag_adapter import RagAdapter
from evaluation.tracking import log_generation_benchmark
from src.llm_provider import active_model_name
from src.source_modes import timeout_for_mode
from src.rag_engine import (
    GENERATION_MAX_OUTPUT_TOKENS,
    QUERY_PREPROCESSING_REASONING_MODE,
)


DEFAULT_SAMPLE_INDICES = tuple(range(15, 45))
def _parse_indices(value: str) -> list[int]:
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("Sample indices must be comma-separated integers.") from exc
    if not indices or any(index < 1 or index > len(EVAL_SAMPLES) for index in indices):
        raise ValueError(f"Sample indices must be between 1 and {len(EVAL_SAMPLES)}.")
    if len(indices) != len(set(indices)):
        raise ValueError("Sample indices cannot contain duplicates.")
    return indices


def _run_directory(base: str | Path, run_name: str) -> Path:
    slug = "".join(
        character.lower() if character.isalnum() else "-"
        for character in run_name
    ).strip("-") or "generation-benchmark"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(base) / "generation_benchmarks" / f"{slug}-{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "roadmaps").mkdir()
    return path


def _mlflow_metrics(summary: dict[str, Any]) -> dict[str, float]:
    total = summary["total_generation"]
    metrics = {
        "generation.initialization_s": summary["initialization_s"],
        "generation.sample_count": summary["sample_count"],
        "generation.failure_count": summary["generation_failure_count"],
        "generation.success_rate": summary["generation_success_rate"],
        "generation.web_requirement_pass_rate": summary["web_requirement_pass_rate"],
        "generation.within_target_rate": summary["within_target_rate"],
        "generation.total.mean_s": total["mean_s"],
        "generation.total.min_s": total["min_s"],
        "generation.total.max_s": total["max_s"],
        "generation.initial_contract_failure_count": summary[
            "generation_initial_contract_failure_count"
        ],
        "generation.first_attempt_valid_rate": summary[
            "generation_first_attempt_valid_rate"
        ],
        "generation.retry_activation_rate": summary[
            "generation_retry_activation_rate"
        ],
        "generation.recovered_count": summary["generation_recovered_count"],
    }
    if summary["schema_validity_pass_rate"] is not None:
        metrics["generation.schema_validity_pass_rate"] = summary[
            "schema_validity_pass_rate"
        ]
    if summary["generation_recovery_rate"] is not None:
        metrics["generation.recovery_rate"] = summary["generation_recovery_rate"]
        metrics["generation.recovery_failure_rate"] = summary[
            "generation_recovery_failure_rate"
        ]
    for name, values in summary["stage_timings"].items():
        stage = name.removesuffix("_s")
        metrics[f"generation.{stage}.mean_s"] = values["mean_s"]
        metrics[f"generation.{stage}.max_s"] = values["max_s"]
    return metrics


def _save_html(payload: dict[str, Any], path: Path) -> None:
    rows = []
    for result in payload["results"]:
        trace = result.get("generation_trace", {})
        timings = trace.get("timings", {})
        rows.append(
            "<tr>"
            f"<td>{result['sample_index']}</td>"
            f"<td>{html.escape(result['category'])}</td>"
            f"<td>{html.escape(result['question'])}</td>"
            f"<td>{result['observed_total_s']:.2f}s</td>"
            f"<td>{timings.get('query_preprocessing_s', 0):.2f}s</td>"
            f"<td>{timings.get('intent_classification_s', 0):.2f}s</td>"
            f"<td>{timings.get('query_refinement_s', 0):.2f}s</td>"
            f"<td>{timings.get('retrieval_s', 0):.2f}s</td>"
            f"<td>{timings.get('web_search_s', 0):.2f}s</td>"
            f"<td>{timings.get('roadmap_generation_s', 0):.2f}s</td>"
            f"<td>{timings.get('source_attribution_s', 0):.2f}s</td>"
            f"<td>{trace.get('roadmap_step_count', 0)}</td>"
            f"<td>{'PASS' if result['generation_succeeded'] else 'FAIL'}</td>"
            f"<td>{trace.get('generation_attempt_count', 0)}</td>"
            f"<td>{html.escape(_recovery_status(trace))}</td>"
            f"<td>{html.escape(_schema_display(result['schema_validity']))}</td>"
            f"<td>{'YES' if result['web_required'] else 'NO'}</td>"
            f"<td>{trace.get('web_context_count', 0)}</td>"
            f"<td>{html.escape(result.get('retrieval', {}).get('mode', ''))}</td>"
            f"<td>{'PASS' if result['web_requirement_passed'] else 'FAIL'}</td>"
            f"<td>{'PASS' if result['within_target'] else 'ABOVE TARGET'}</td>"
            "</tr>"
        )
    summary = payload["summary"]
    recovery_summary = (
        f"{summary['generation_recovered_count']}/"
        f"{summary['generation_initial_contract_failure_count']} initial failures"
        if summary["generation_initial_contract_failure_count"]
        else "N/A — no initial failures"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Roadmap Generation Benchmark</title>
<style>body{{font-family:Arial,sans-serif;margin:28px;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d8deea;padding:8px;text-align:left}}th{{background:#eef2f8}}.cards{{display:flex;gap:16px;margin:20px 0}}.card{{padding:16px;border:1px solid #d8deea;border-radius:8px}}</style>
</head><body><h1>Roadmap Generation Benchmark</h1>
<p>Generated: {html.escape(payload['generated_at'])} | Provider: {html.escape(payload['configuration']['provider'])} | Model: {html.escape(payload['configuration']['model'])}</p>
<div class="cards"><div class="card"><b>Mean</b><br>&le; target: {summary['target_seconds']:.0f}s<br>observed: {summary['total_generation']['mean_s']:.2f}s</div><div class="card"><b>Within target</b><br>{summary['within_target_count']}/{summary['sample_count']}</div><div class="card"><b>First-pass valid</b><br>{summary['generation_first_attempt_valid_count']}/{summary['sample_count']}</div><div class="card"><b>Recovery</b><br>{recovery_summary}</div><div class="card"><b>Initialization</b><br>{summary['initialization_s']:.2f}s</div></div>
<table><thead><tr><th>#</th><th>Category</th><th>Question</th><th>Total</th><th>Merged preprocessing</th><th>Intent</th><th>Refinement</th><th>Retrieval</th><th>Web time</th><th>Roadmap</th><th>Attribution</th><th>Steps</th><th>Generation</th><th>Attempts</th><th>Recovery</th><th>Schema validity</th><th>Web required</th><th>Web contexts</th><th>Retrieval mode</th><th>Web check</th><th>Target</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def run(
    sample_indices: list[int],
    run_name: str,
    output_dir: str | Path,
    mlflow_enabled: bool = True,
    source_mode: str = "auto",
) -> Path:
    target_seconds = timeout_for_mode(source_mode)
    run_dir = _run_directory(output_dir, run_name)
    print(f"Run directory: {run_dir}")
    initialization_started = time.perf_counter()
    adapter = RagAdapter()
    initialization_s = time.perf_counter() - initialization_started
    schema_validator = StructureValidator()
    print(f"Initialization completed in {initialization_s:.2f}s")

    results = []
    roadmap_artifacts = []
    for position, sample_index in enumerate(sample_indices, 1):
        sample = EVAL_SAMPLES[sample_index - 1]
        print(f"\n[{position}/{len(sample_indices)}] {sample.question}")
        started = time.perf_counter()
        generated = adapter.query(sample.question, source_mode=source_mode)
        observed_total_s = round(time.perf_counter() - started, 4)
        roadmap = generated.get("roadmap", {})
        succeeded = roadmap.get("title") != "Error" and bool(roadmap.get("steps"))
        schema_validity = (
            schema_validator.validate(roadmap)
            if succeeded
            else _schema_not_evaluated("Roadmap generation failed before schema validation.")
        )
        step_count = len(roadmap.get("steps", []))
        web_required = bool(sample.requires_web)
        web_context_count = int(
            generated.get("generation_trace", {}).get("web_context_count", 0)
        )
        web_requirement_passed = not web_required or web_context_count > 0
        result = {
            "sample_index": sample_index,
            "question": sample.question,
            "category": sample.category,
            "observed_total_s": observed_total_s,
            "within_target": observed_total_s <= target_seconds,
            "generation_succeeded": succeeded,
            "web_required": web_required,
            "web_requirement_passed": web_requirement_passed,
            "schema_validity": schema_validity,
            "refined_question": generated.get("refined_question", ""),
            "web_search_query": generated.get("web_search_query", ""),
            "retrieval": generated.get("retrieval", {}),
            "generation_trace": generated.get("generation_trace", {}),
            "roadmap_artifact": f"roadmaps/sample_{sample_index}.json",
        }
        results.append(result)
        roadmap_path = run_dir / result["roadmap_artifact"]
        roadmap_path.write_text(
            json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        roadmap_artifacts.append(str(roadmap_path))
        print(
            f"Completed in {observed_total_s:.2f}s | "
            f"steps={step_count} | generation="
            f"{'PASS' if succeeded else 'FAIL'} | "
            f"schema={_schema_display(schema_validity)} | "
            f"web={'PASS' if web_requirement_passed else 'FAIL'} | "
            f"target={'PASS' if result['within_target'] else 'ABOVE'}"
        )

    summary = _aggregate(results, initialization_s, target_seconds)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark": "roadmap_generation_v1",
        "configuration": {
            "provider": os.getenv("LLM_PROVIDER", "ollama").lower().strip(),
            "model": active_model_name(),
            "generation_reasoning_mode": os.getenv(
                "LLM_GENERATION_REASONING_MODE", "provider_default"
            ).lower().strip(),
            "query_preprocessing_reasoning_mode": (
                QUERY_PREPROCESSING_REASONING_MODE
            ),
            "query_preprocessing_mode": os.getenv(
                "QUERY_PREPROCESSING_MODE", "sequential"
            ).lower().strip(),
            "generation_prompt_version": next(
                (
                    result.get("generation_trace", {}).get(
                        "generation_prompt_version"
                    )
                    for result in results
                    if result.get("generation_trace", {}).get(
                        "generation_prompt_version"
                    )
                ),
                "unknown",
            ),
            "generation_timeout_seconds": next(
                (
                    result.get("generation_trace", {}).get(
                        "generation_timeout_seconds"
                    )
                    for result in results
                    if result.get("generation_trace", {}).get(
                        "generation_timeout_seconds"
                    ) is not None
                ),
                None,
            ),
            "generation_max_output_tokens": next(
                (
                    result.get("generation_trace", {}).get(
                        "generation_max_output_tokens"
                    )
                    for result in results
                    if result.get("generation_trace", {}).get(
                        "generation_max_output_tokens"
                    ) is not None
                ),
                None,
            ),
            "generation_retry_policy": next(
                (
                    result.get("generation_trace", {}).get(
                        "generation_retry_policy"
                    )
                    for result in results
                    if result.get("generation_trace", {}).get(
                        "generation_retry_policy"
                    )
                ),
                "unknown",
            ),
            "sample_indices": sample_indices,
            "source_mode": source_mode,
            "target_seconds": target_seconds,
        },
        "summary": summary,
        "results": results,
    }
    json_path = run_dir / "benchmark_results.json"
    html_path = run_dir / "benchmark_report.html"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_html(payload, html_path)
    log_generation_benchmark(
        run_name=run_name,
        params={**payload["configuration"], "benchmark": payload["benchmark"]},
        metrics=_mlflow_metrics(summary),
        artifacts=[str(json_path), str(html_path), *roadmap_artifacts],
        enabled=mlflow_enabled,
    )
    print("\nGENERATION BENCHMARK SUMMARY")
    print(f"  Mean:          {summary['total_generation']['mean_s']:.2f}s")
    print(f"  Range:         {summary['total_generation']['min_s']:.2f}s - {summary['total_generation']['max_s']:.2f}s")
    print(f"  Within target: {summary['within_target_count']}/{summary['sample_count']}")
    print(
        "  First-pass:    "
        f"{summary['generation_first_attempt_valid_count']}/"
        f"{summary['sample_count']} valid"
    )
    print(
        "  Valid schema:   "
        f"{summary['schema_validity_pass_count']}/"
        f"{summary['schema_validity_evaluated_count']} evaluated"
    )
    recovery_summary = (
        f"{summary['generation_recovered_count']}/"
        f"{summary['generation_initial_contract_failure_count']} initial failures"
        if summary["generation_initial_contract_failure_count"]
        else "N/A (no initial failures)"
    )
    print(f"  Recovered:      {recovery_summary}")
    print(
        "  Required web:  "
        f"{summary['web_requirement_pass_count']}/{summary['web_required_count']}"
    )
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark roadmap generation without evaluation judges."
    )
    parser.add_argument(
        "--sample-indices",
        default=",".join(map(str, DEFAULT_SAMPLE_INDICES)),
        help="One-based dataset indices (default: 15-44, 30 generation cases).",
    )
    parser.add_argument("--run-name", default="roadmap-generation-baseline-v1")
    parser.add_argument("--output-dir", default=config.reports_dir)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--source-mode",
        choices=("corpus", "web", "auto"),
        default="auto",
        help="Evidence policy used for every generated roadmap.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        indices = _parse_indices(args.sample_indices)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print("Roadmap generation benchmark")
        print(f"  Provider: {os.getenv('LLM_PROVIDER', 'ollama')}")
        print(f"  Model: {active_model_name()}")
        print(f"  Target: <= {timeout_for_mode(args.source_mode):.0f}s per roadmap")
        print(f"  Source mode: {args.source_mode}")
        print(f"  Max output tokens: {GENERATION_MAX_OUTPUT_TOKENS}")
        print(
            "  Generation reasoning: "
            f"{os.getenv('LLM_GENERATION_REASONING_MODE', 'provider_default')}"
        )
        print(
            "  Query preprocessing reasoning: "
            f"{QUERY_PREPROCESSING_REASONING_MODE}"
        )
        print(
            "  Query preprocessing: "
            f"{os.getenv('QUERY_PREPROCESSING_MODE', 'sequential')}"
        )
        for index in indices:
            sample = EVAL_SAMPLES[index - 1]
            web = " | web required" if sample.requires_web else ""
            print(f"  {index}: [{sample.category}] {sample.question}{web}")
        print("  Judge calls: 0")
        return
    run(
        indices,
        args.run_name,
        args.output_dir,
        not args.no_mlflow,
        args.source_mode,
    )


if __name__ == "__main__":
    main()
