"""Orchestrate the six canonical roadmap metrics over one generated roadmap."""

from __future__ import annotations

import time
import traceback
from copy import deepcopy
from typing import Any, Iterable

from evaluation.config import EvalConfig, config
from evaluation.metric_contract import (
    canonical_scores,
    validate_canonical_metrics,
)
from evaluation.metrics.actionability_judge import ActionabilityJudge
from evaluation.metrics.completeness_judge import CompletenessJudge
from evaluation.metrics.grounding_judge import GroundingJudge
from evaluation.metrics.logical_order_judge import LogicalOrderJudge
from evaluation.metrics.step_semantic_judge import StepSemanticJudge
from evaluation.metrics.structure_quality_judge import StructureQualityJudge
from evaluation.metrics.structure_validator import StructureValidator
from evaluation.readiness import roadmap_readiness
from evaluation.run_session import metric_result_summary
from src.llm_provider import active_model_name, get_judge_llm


METRIC_ORDER = (
    "grounding",
    "completeness",
    "actionability",
    "logical_order",
    "structure_quality",
    "step_semantics",
    "schema_validity",
)


def _context_texts(contexts: list[Any] | None) -> list[str]:
    values = []
    for context in contexts or []:
        if isinstance(context, dict):
            text = str(context.get("text", context.get("content", ""))).strip()
        else:
            text = str(context).strip()
        if text:
            values.append(text)
    return values


class CanonicalMetricEvaluator:
    """Evaluate dedicated metrics sequentially and expose canonical results."""

    def __init__(
        self,
        cfg: EvalConfig = config,
        judges: dict[str, Any] | None = None,
        structure_validator: StructureValidator | None = None,
    ):
        self.cfg = cfg
        if judges is None:
            judges = {
                "grounding": GroundingJudge(get_judge_llm(
                    temperature=cfg.judge_temperature, max_tokens=4096
                )),
                "completeness": CompletenessJudge(get_judge_llm(
                    temperature=cfg.judge_temperature, max_tokens=4096
                )),
                "actionability": ActionabilityJudge(get_judge_llm(
                    temperature=cfg.judge_temperature, max_tokens=4096
                )),
                "logical_order": LogicalOrderJudge(get_judge_llm(
                    temperature=cfg.judge_temperature, max_tokens=2048
                )),
                "structure_quality": StructureQualityJudge(get_judge_llm(
                    temperature=cfg.judge_temperature, max_tokens=2048
                )),
                "step_semantics": StepSemanticJudge(get_judge_llm(
                    temperature=cfg.judge_temperature, max_tokens=2048
                )),
            }
        self.judges = judges
        self.structure_validator = structure_validator or StructureValidator()

    @staticmethod
    def _timed(call) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        result = call()
        return result, round(time.perf_counter() - started, 2)

    def evaluate_generated(
        self,
        sample: Any,
        generated: dict[str, Any],
        enabled_metrics: Iterable[str] | None = None,
        progress: Any | None = None,
        existing_result: dict[str, Any] | None = None,
        on_metric_completed: Any | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Evaluate a pre-generated roadmap without repeating retrieval/generation."""
        enabled = set(enabled_metrics or METRIC_ORDER)
        unknown = enabled - set(METRIC_ORDER)
        if unknown:
            raise ValueError(f"Unknown canonical metrics: {', '.join(sorted(unknown))}")

        roadmap = generated["roadmap"]
        question = str(getattr(sample, "question", generated.get("question", "")))
        category = str(getattr(sample, "category", ""))
        ground_truth = str(getattr(sample, "ground_truth", ""))
        refined_question = str(generated.get("refined_question", ""))
        contexts = generated.get("contexts", [])
        context_texts = _context_texts(contexts)
        result_keys = (
            "grounding",
            "completeness",
            "actionability",
            "logical_order",
            "structure_quality",
            "step_distinctness",
            "step_overlap",
            "schema_validity",
        )
        previous = existing_result or {}
        metrics: dict[str, dict[str, Any]] = {
            name: deepcopy(previous[name]) for name in result_keys if name in previous
        }
        timings: dict[str, float] = deepcopy(previous.get("metric_timings", {}))

        def persist(metric_name: str) -> None:
            if on_metric_completed:
                on_metric_completed(
                    metric_name,
                    self._build_sample_result(sample, generated, metrics, timings),
                )

        if "grounding" in enabled and "grounding" not in metrics:
            if progress:
                progress.stage_started("Grounding")
            elif verbose:
                print("    [Metric] Grounding...", flush=True)
            value, timings["grounding"] = self._timed(
                lambda: self.judges["grounding"].evaluate(
                    roadmap, contexts, question=question
                )
            )
            metrics["grounding"] = value["grounding"]
            self._report_metric_progress(progress, "grounding", value, timings["grounding"], verbose)
            persist("grounding")

        if "completeness" in enabled and "completeness" not in metrics:
            if progress:
                progress.stage_started("Completeness")
            elif verbose:
                print("    [Metric] Completeness...", flush=True)
            value, timings["completeness"] = self._timed(
                lambda: self.judges["completeness"].evaluate(
                    roadmap,
                    question=question,
                    refined_question=refined_question,
                    ground_truth=ground_truth,
                    expected_elements=list(
                        getattr(sample, "expected_elements", []) or []
                    ),
                )
            )
            metrics["completeness"] = value["completeness"]
            self._report_metric_progress(progress, "completeness", value, timings["completeness"], verbose)
            persist("completeness")

        if "actionability" in enabled and "actionability" not in metrics:
            if progress:
                progress.stage_started("Actionability")
            elif verbose:
                print("    [Metric] Actionability...", flush=True)
            value, timings["actionability"] = self._timed(
                lambda: self.judges["actionability"].evaluate(
                    roadmap, question=question, category=category
                )
            )
            metrics["actionability"] = value["actionability"]
            self._report_metric_progress(progress, "actionability", value, timings["actionability"], verbose)
            persist("actionability")

        if "logical_order" in enabled and "logical_order" not in metrics:
            if progress:
                progress.stage_started("Logical Order")
            elif verbose:
                print("    [Metric] Logical Order...", flush=True)
            value, timings["logical_order"] = self._timed(
                lambda: self.judges["logical_order"].evaluate(
                    roadmap,
                    question=question,
                    category=category,
                    contexts=context_texts,
                )
            )
            metrics["logical_order"] = value["logical_order"]
            self._report_metric_progress(progress, "logical_order", value, timings["logical_order"], verbose)
            persist("logical_order")

        if "structure_quality" in enabled and "structure_quality" not in metrics:
            if progress:
                progress.stage_started("Structure Quality")
            elif verbose:
                print("    [Metric] Structure Quality...", flush=True)
            value, timings["structure_quality"] = self._timed(
                lambda: self.judges["structure_quality"].evaluate(
                    roadmap, question=question, category=category
                )
            )
            metrics["structure_quality"] = value["structure_quality"]
            self._report_metric_progress(progress, "structure_quality", value, timings["structure_quality"], verbose)
            persist("structure_quality")

        if "step_semantics" in enabled and "step_distinctness" not in metrics:
            if progress:
                progress.stage_started("Step Distinctness + Overlap")
            elif verbose:
                print("    [Metric] Step Distinctness + Overlap...", flush=True)
            value, timings["step_semantics"] = self._timed(
                lambda: self.judges["step_semantics"].evaluate(
                    roadmap,
                    question=question,
                    refined_question=refined_question,
                    contexts=context_texts,
                    category=category,
                )
            )
            metrics["step_distinctness"] = value["step_distinctness"]
            metrics["step_overlap"] = value["step_overlap"]
            self._report_metric_progress(progress, "step_semantics", value, timings["step_semantics"], verbose)
            persist("step_semantics")

        if "schema_validity" in enabled and "schema_validity" not in metrics:
            if progress:
                progress.stage_started("Schema Validity")
            elif verbose:
                print("    [Metric] Schema Validity...", flush=True)
            value, timings["schema_validity"] = self._timed(
                lambda: self.structure_validator.validate(roadmap)
            )
            metrics["schema_validity"] = value
            self._report_metric_progress(
                progress,
                "schema_validity",
                {"schema_validity": value},
                timings["schema_validity"],
                verbose,
            )
            persist("schema_validity")

        return self._build_sample_result(sample, generated, metrics, timings)

    def _build_sample_result(
        self,
        sample: Any,
        generated: dict[str, Any],
        metrics: dict[str, dict[str, Any]],
        timings: dict[str, float],
    ) -> dict[str, Any]:
        roadmap = generated["roadmap"]
        question = str(getattr(sample, "question", generated.get("question", "")))
        category = str(getattr(sample, "category", ""))
        ground_truth = str(getattr(sample, "ground_truth", ""))
        refined_question = str(generated.get("refined_question", ""))
        contexts = generated.get("contexts", [])
        canonical_semantic = {
            name: metrics[name]
            for name in (
                "grounding",
                "completeness",
                "actionability",
                "logical_order",
                "structure_quality",
                "step_distinctness",
            )
            if name in metrics
        }
        contract_errors = (
            validate_canonical_metrics(canonical_semantic)
            if len(canonical_semantic) == 6
            else []
        )
        readiness = roadmap_readiness(metrics, contract_errors)

        return {
            "question": question,
            "query_intent": generated.get("query_intent", {}),
            "refined_question": refined_question,
            "category": category,
            "ground_truth": ground_truth,
            "answer": generated.get("answer", ""),
            "contexts": contexts,
            "retrieval": generated.get("retrieval", {}),
            "generation_trace": generated.get("generation_trace", {}),
            "judge_context_strategy": generated.get("judge_context_strategy", ""),
            "roadmap": roadmap,
            "steps": [
                str(step.get("label", ""))
                for step in roadmap.get("steps", [])
                if isinstance(step, dict)
            ],
            "expected_steps": list(getattr(sample, "expected_step_order", [])),
            **metrics,
            "metric_scores": canonical_scores(canonical_semantic),
            "metric_timings": timings,
            "metric_contract_errors": contract_errors,
            "roadmap_readiness": readiness,
        }

    @staticmethod
    def _report_metric_progress(
        progress: Any | None,
        name: str,
        result: dict[str, Any],
        elapsed: float,
        verbose: bool = False,
    ) -> None:
        score, verdict = metric_result_summary(name, result)
        label = {
            "grounding": "Grounding",
            "completeness": "Completeness",
            "actionability": "Actionability",
            "logical_order": "Logical Order",
            "structure_quality": "Structure Quality",
            "step_semantics": "Step Distinctness + Overlap",
            "schema_validity": "Schema Validity",
        }[name]
        if progress:
            progress.stage_completed(label, elapsed, score, verdict)
        elif verbose:
            print(f"    [Metric] {label} completed in {elapsed:.2f}s", flush=True)

    def evaluate(
        self,
        adapter: Any,
        samples: list[Any],
        verbose: bool = False,
        enabled_metrics: Iterable[str] | None = None,
        progress: Any | None = None,
        checkpoint: Any | None = None,
    ) -> dict[str, Any]:
        """Generate each roadmap once, evaluate it, and preserve failed samples."""
        per_sample = []
        failures = []
        if verbose or progress:
            print(f"\n[Canonical Metrics] {len(samples)} samples | model: {active_model_name()}")
        for index, sample in enumerate(samples, 1):
            if progress:
                progress.sample_started(index, len(samples), sample.question)
            elif verbose:
                print(f"  [{index}/{len(samples)}] {sample.question[:65]}...", flush=True)
            started = time.perf_counter()
            try:
                checkpoint_entry = checkpoint.entry(index - 1) if checkpoint else {}
                generated = checkpoint_entry.get("generated")
                if generated:
                    generation_seconds = float(checkpoint_entry.get("generation_seconds") or 0)
                    if progress:
                        progress.message("  SKIP Roadmap generation (restored from checkpoint)")
                else:
                    if progress:
                        progress.stage_started("Roadmap generation")
                    generated = adapter.query(sample.question)
                    generation_seconds = round(time.perf_counter() - started, 2)
                    if checkpoint:
                        checkpoint.record_generation(index - 1, generated, generation_seconds)
                    if progress:
                        progress.stage_completed("Roadmap generation", generation_seconds)
                result = self.evaluate_generated(
                    sample,
                    generated,
                    enabled_metrics,
                    progress=progress,
                    existing_result=checkpoint_entry.get("result"),
                    on_metric_completed=(
                        lambda metric, partial, sample_index=index - 1:
                        checkpoint.record_result(sample_index, metric, partial)
                    ) if checkpoint else None,
                    verbose=verbose,
                )
                result["response_time"] = generation_seconds
                if checkpoint:
                    checkpoint.entry(index - 1)["result"] = result
                    checkpoint.save()
                per_sample.append(result)
                if verbose:
                    scores = ", ".join(
                        f"{name}={score:.1f}"
                        for name, score in result["metric_scores"].items()
                    )
                    print(f"    {scores}")
            except Exception as exc:
                failures.append({
                    "question": str(getattr(sample, "question", "")),
                    "error_type": type(exc).__name__,
                })
                print(f"  Error in sample {index}:")
                traceback.print_exc()

        if not per_sample:
            raise RuntimeError("No valid canonical metric result was generated.")
        aggregated = self._aggregate(per_sample)
        return {
            "per_sample": per_sample,
            "aggregated": aggregated,
            "sample_count": len(per_sample),
            "failure_count": len(failures),
            "failures": failures,
            "metric_contract": "canonical_v1",
        }

    @staticmethod
    def _aggregate(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate canonical scores and diagnostics without a primary composite."""
        count = len(per_sample)
        metric_names = (
            "grounding",
            "completeness",
            "actionability",
            "logical_order",
            "structure_quality",
            "step_distinctness",
        )
        metric_aggregates = {}
        for name in metric_names:
            values = [sample[name] for sample in per_sample if name in sample]
            if not values:
                continue
            scores = [sample["metric_scores"][name] for sample in per_sample if name in sample]
            verdicts = [str(value.get("verdict", "")) for value in values]
            metric_aggregates[name] = {
                "mean_score": round(sum(scores) / len(scores), 2),
                "pass_rate": round(sum(value == "PASS" for value in verdicts) / len(values), 4),
                "needs_review_rate": round(
                    sum(value == "NEEDS_REVIEW" for value in verdicts) / len(values), 4
                ),
                "fail_rate": round(sum(value == "FAIL" for value in verdicts) / len(values), 4),
            }

        readiness_statuses = [
            sample.get("roadmap_readiness", {}).get("status", "NOT_EVALUATED")
            for sample in per_sample
        ]
        readiness_counts = {
            status: readiness_statuses.count(status)
            for status in ("READY", "NEEDS_REVIEW", "FAIL", "NOT_EVALUATED")
        }
        generation_times = [float(sample.get("response_time", 0)) for sample in per_sample]
        generation_stage_names = sorted({
            name
            for sample in per_sample
            for name in sample.get("generation_trace", {}).get("timings", {})
        })
        generation_stage_timings = {}
        for name in generation_stage_names:
            values = [
                float(sample["generation_trace"]["timings"][name])
                for sample in per_sample
                if name in sample.get("generation_trace", {}).get("timings", {})
            ]
            generation_stage_timings[name] = {
                "mean_s": round(sum(values) / len(values), 4),
                "max_s": round(max(values), 4),
                "min_s": round(min(values), 4),
            }
        metric_timing_names = sorted({
            name for sample in per_sample for name in sample.get("metric_timings", {})
        })
        metric_timings = {}
        for name in metric_timing_names:
            values = [
                float(sample["metric_timings"][name])
                for sample in per_sample
                if name in sample.get("metric_timings", {})
            ]
            metric_timings[name] = {
                "mean_s": round(sum(values) / len(values), 2),
                "max_s": round(max(values), 2),
            }

        diagnostics = {
            "unsupported_claim_count": sum(
                sample.get("grounding", {}).get("unsupported_claim_count", 0)
                for sample in per_sample
            ),
            "missing_element_count": sum(
                sample.get("completeness", {}).get("missing_element_count", 0)
                for sample in per_sample
            ),
            "weak_action_step_count": sum(
                sample.get("actionability", {}).get("weak_action_step_count", 0)
                for sample in per_sample
            ),
            "dependency_violation_count": sum(
                sample.get("logical_order", {}).get("dependency_violation_count", 0)
                for sample in per_sample
            ),
            "structure_issue_count": sum(
                sample.get("structure_quality", {}).get("issue_count", 0)
                for sample in per_sample
            ),
            "weak_step_count": sum(
                sample.get("step_distinctness", {}).get("weak_step_count", 0)
                for sample in per_sample
            ),
            "overlap_issue_count": sum(
                sample.get("step_overlap", {}).get("issue_count", 0)
                for sample in per_sample
            ),
        }
        schema_values = [
            sample["schema_validity"]
            for sample in per_sample
            if "schema_validity" in sample
        ]
        schema_aggregate = {}
        if schema_values:
            schema_aggregate = {
                "mean_score": round(
                    sum(float(value.get("score", 0)) for value in schema_values)
                    / len(schema_values),
                    2,
                ),
                "pass_rate": round(
                    sum(value.get("verdict") == "PASS" for value in schema_values)
                    / len(schema_values),
                    4,
                ),
            }
        return {
            "metrics": metric_aggregates,
            "schema_validity": schema_aggregate,
            "readiness": {
                "ready_rate": round(readiness_counts["READY"] / count, 4),
                "needs_review_rate": round(readiness_counts["NEEDS_REVIEW"] / count, 4),
                "fail_rate": round(readiness_counts["FAIL"] / count, 4),
                "not_evaluated_rate": round(readiness_counts["NOT_EVALUATED"] / count, 4),
                "counts": readiness_counts,
            },
            "diagnostics": diagnostics,
            "response_time": {
                "mean_s": round(sum(generation_times) / count, 2),
                "max_s": round(max(generation_times), 2),
                "min_s": round(min(generation_times), 2),
            },
            "generation_stage_timings": generation_stage_timings,
            "metric_timings": metric_timings,
        }


def run_canonical_metrics(
    adapter: Any,
    samples: list[Any],
    verbose: bool = False,
    enabled_metrics: Iterable[str] | None = None,
    progress: Any | None = None,
    checkpoint: Any | None = None,
) -> dict[str, Any]:
    """Functional entry point used during staged runner migration."""
    return CanonicalMetricEvaluator().evaluate(
        adapter,
        samples,
        verbose=verbose,
        enabled_metrics=enabled_metrics,
        progress=progress,
        checkpoint=checkpoint,
    )
