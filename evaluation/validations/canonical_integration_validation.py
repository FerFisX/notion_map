"""Controlled smoke validation for the canonical metric orchestrator.

The suite injects hand-written roadmaps through an in-memory adapter. It never
calls retrieval or roadmap generation, so failures and timings belong only to
the canonical evaluation pipeline.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.config import config
from evaluation.dataset import EvalSample
from evaluation.metric_contract import canonical_scores, validate_canonical_metrics
from evaluation.metric_orchestrator import CanonicalMetricEvaluator, run_canonical_metrics
from evaluation.readiness import roadmap_readiness
from evaluation.reporter import save_html, save_json
from evaluation.runner import parse_metrics
from src.llm_provider import active_model_name


REPORT_DIR = Path(config.reports_dir) / "canonical_integration_validation"
SEMANTIC_METRICS = (
    "grounding",
    "completeness",
    "actionability",
    "logical_order",
    "structure_quality",
    "step_distinctness",
)


def _step(
    index: int,
    label: str,
    description: str,
    points: list[str],
    step_type: str = "proceso",
) -> dict[str, Any]:
    return {
        "id": f"step_{index}",
        "label": label,
        "description": description,
        "type": step_type,
        "key_points": points,
    }


def _context(index: int, text: str) -> dict[str, str]:
    return {"id": f"context_{index}", "text": text}


@dataclass(frozen=True)
class ControlledCase:
    name: str
    sample: EvalSample
    generated: dict[str, Any]
    score_ranges: dict[str, tuple[float, float]]
    expected_readiness: str


STRONG_ROADMAP = {
    "title": "Build and Validate Year-over-Year Sales in DAX",
    "steps": [
        _step(
            1,
            "Define the comparison goal",
            "Specify that the report must compare current sales with the same period in the prior year. Define a validated YoY percentage as the final output.",
            ["current versus prior year", "sales measure", "validated YoY output"],
            "inicio",
        ),
        _step(
            2,
            "Create the calendar table",
            "Create a continuous Date table, mark it as the model's date table, and relate Date[Date] to Sales[OrderDate]. This establishes the required time-intelligence foundation.",
            ["continuous dates", "mark as date table", "active date relationship"],
        ),
        _step(
            3,
            "Create the base sales measure",
            "Define [Total Sales] as SUM(Sales[SalesAmount]). Validate it against a known total before applying any time shift.",
            ["SUM", "Sales[SalesAmount]", "known-total check"],
        ),
        _step(
            4,
            "Calculate prior-year sales",
            "Define [Sales PY] with CALCULATE([Total Sales], SAMEPERIODLASTYEAR(Date[Date])). Check a period that has data in both years.",
            ["CALCULATE", "SAMEPERIODLASTYEAR", "comparable period"],
        ),
        _step(
            5,
            "Calculate the YoY percentage",
            "Define [Sales YoY %] with DIVIDE([Total Sales] - [Sales PY], [Sales PY]). Format the measure as a percentage and preserve safe blank handling.",
            ["DIVIDE", "percentage format", "blank-safe result"],
        ),
        _step(
            6,
            "Validate and reuse the pattern",
            "Place current sales, prior-year sales, and YoY percentage in a visual by month. Compare known periods, document the checks, and reuse the pattern only after the results are correct.",
            ["monthly visual", "known-period validation", "documented reusable pattern"],
            "fin",
        ),
    ],
}

STRONG_CONTEXTS = [
    _context(1, "DAX year-over-year analysis compares a current business measure with the same period in the previous year."),
    _context(2, "Time intelligence requires a continuous calendar table marked as the date table and related to the fact date column."),
    _context(3, "Total Sales can be defined as SUM(Sales[SalesAmount]) and should be validated before derived measures are created."),
    _context(4, "Prior-year sales can be calculated with CALCULATE([Total Sales], SAMEPERIODLASTYEAR(Date[Date]))."),
    _context(5, "YoY percentage can use DIVIDE([Total Sales] - [Sales PY], [Sales PY]); DIVIDE safely handles a zero denominator."),
    _context(6, "Validate time-intelligence measures in a visual across periods with known results before reusing the pattern."),
]

POOR_ROADMAP = {
    "title": "Automatic DAX Time Intelligence",
    "steps": [
        _step(
            1,
            "Define a rough reporting goal",
            "State that the report should show a year-over-year sales result, but leave the business measure, comparison period, and acceptance criteria undecided.",
            ["rough YoY goal", "undecided measure", "no acceptance criteria"],
            "inicio",
        ),
        _step(
            2,
            "Publish the report immediately",
            "Publish the final report before creating the date model or any sales measures. Assume the cloud service will automatically create and repair every missing DAX calculation.",
            ["publish before model", "automatic measure creation", "automatic repair"],
        ),
        _step(
            3,
            "Validate the missing YoY output",
            "Validate the final YoY result before the date model and measures exist. Treat a visual that renders without an error as proof that the calculation is correct.",
            ["validate before output exists", "visual rendering is sufficient"],
        ),
        _step(
            4,
            "Configure automatic YoY",
            "Use the nonexistent AUTOYOY function with default options instead of defining a Total Sales base measure, a prior-year measure, or an explicit YoY percentage measure.",
            ["AUTOYOY", "skip base measure", "skip explicit YoY measures"],
        ),
        _step(
            5,
            "Repeat the automatic calculation",
            "Configure the same nonexistent AUTOYOY function again with the same defaults, without adding a different output, validation, or responsibility.",
            ["duplicate AUTOYOY setup", "same defaults", "same output"],
        ),
        _step(
            6,
            "Add placeholder date data last",
            "After publishing, validating, and configuring YoY, add an incomplete date column. Do not create a continuous marked Date table and do not define its relationship to the sales date.",
            ["date data added last", "no marked date table", "no relationship"],
        ),
        _step(
            7,
            "Close with an unresolved visual check",
            "Render the report once, record that the required measure, comparison period, date relationship, and acceptance criteria remain unresolved, and stop without correcting them.",
            ["single visual check", "document unresolved gaps", "stop without correction"],
            "fin",
        ),
    ],
}

POOR_CONTEXTS = [
    _context(1, "Build and validate the data model before publishing a Power BI report. Publishing does not repair missing DAX calculations."),
    _context(2, "DAX has no AUTOYOY function. Prior-year analysis requires an explicit base measure and a valid date table."),
    _context(3, "Create a continuous date table and relationship before applying SAMEPERIODLASTYEAR to a validated base measure."),
    _context(4, "A visual rendering without an error is not sufficient validation; compare the measure with known period results."),
]


CASES = {
    "strong": ControlledCase(
        name="strong_supported_dax",
        sample=EvalSample(
            question="How do I build and validate year-over-year sales in DAX?",
            ground_truth=(
                "Define the comparison goal, create a continuous marked date table and relationship, "
                "validate a Total Sales base measure, calculate prior-year sales, calculate YoY percentage, "
                "and validate known periods before reuse."
            ),
            expected_keywords=["date table", "Total Sales", "SAMEPERIODLASTYEAR", "DIVIDE", "validate"],
            category="dax_power_bi",
            expected_step_order=[
                "Define the comparison goal",
                "Create the calendar table",
                "Create the base sales measure",
                "Calculate prior-year sales",
                "Calculate the YoY percentage",
                "Validate and reuse the pattern",
            ],
        ),
        generated={
            "question": "How do I build and validate year-over-year sales in DAX?",
            "refined_question": "How can I implement and verify a reusable year-over-year sales pattern in DAX?",
            "query_intent": {"intent": "implementation"},
            "contexts": STRONG_CONTEXTS,
            "corpus_contexts": STRONG_CONTEXTS,
            "web_contexts": [],
            "retrieval": {"context_strategy": "controlled", "n_contexts": len(STRONG_CONTEXTS)},
            "judge_context_strategy": "controlled_same_context",
            "roadmap": STRONG_ROADMAP,
            "answer": "Controlled strong DAX roadmap.",
        },
        score_ranges={name: (8.0, 10.0) for name in SEMANTIC_METRICS} | {"schema_validity": (7.0, 10.0)},
        expected_readiness="READY",
    ),
    "poor": ControlledCase(
        name="poor_cross_metric_dax",
        sample=EvalSample(
            question="How do I build and validate year-over-year sales in DAX?",
            ground_truth=(
                "A correct solution needs a date table and relationship, a validated base measure, explicit prior-year "
                "and YoY measures, known-period validation, and publication only after validation."
            ),
            expected_keywords=["date table", "base measure", "prior year", "YoY", "validation"],
            category="dax_power_bi",
            expected_step_order=[
                "Define the comparison goal",
                "Prepare the date model",
                "Create the base measure",
                "Create prior-year and YoY measures",
                "Validate known periods",
                "Publish the report",
            ],
        ),
        generated={
            "question": "How do I build and validate year-over-year sales in DAX?",
            "refined_question": "How can I implement and verify a reusable year-over-year sales pattern in DAX?",
            "query_intent": {"intent": "implementation"},
            "contexts": POOR_CONTEXTS,
            "corpus_contexts": POOR_CONTEXTS,
            "web_contexts": [],
            "retrieval": {"context_strategy": "controlled", "n_contexts": len(POOR_CONTEXTS)},
            "judge_context_strategy": "controlled_same_context",
            "roadmap": POOR_ROADMAP,
            "answer": "Controlled poor DAX roadmap.",
        },
        score_ranges={
            "grounding": (0.0, 4.9),
            "completeness": (0.0, 4.9),
            "actionability": (5.0, 7.9),
            "logical_order": (0.0, 4.9),
            "structure_quality": (5.0, 7.9),
            "step_distinctness": (5.0, 7.9),
            "schema_validity": (7.0, 10.0),
        },
        expected_readiness="FAIL",
    ),
}


class ControlledAdapter:
    """Return exact fixtures and prove that no generation engine is required."""

    def __init__(self, cases: list[ControlledCase]):
        self._generated = {case.sample.question + case.name: case.generated for case in cases}
        self._case_names = [case.name for case in cases]
        self.query_count = 0

    def query(self, question: str) -> dict[str, Any]:
        case_name = self._case_names[self.query_count]
        self.query_count += 1
        return deepcopy(self._generated[question + case_name])


def _attach_expectations(results: dict[str, Any], cases: list[ControlledCase]) -> dict[str, Any]:
    matched = 0
    total = 0
    for sample_result, case in zip(results["per_sample"], cases):
        checks = {}
        for name, (minimum, maximum) in case.score_ranges.items():
            if name not in sample_result:
                continue
            score = (
                float(sample_result[name].get("support_score", 0))
                if name == "grounding"
                else float(sample_result[name].get("score", 0))
            )
            ok = minimum <= score <= maximum
            checks[name] = {
                "score": score,
                "expected_range": [minimum, maximum],
                "matched": ok,
            }
            matched += int(ok)
            total += 1
        actual_readiness = sample_result["roadmap_readiness"]["status"]
        if actual_readiness != "NOT_EVALUATED":
            readiness_ok = actual_readiness == case.expected_readiness
            checks["readiness"] = {
                "actual": actual_readiness,
                "expected": case.expected_readiness,
                "matched": readiness_ok,
            }
            matched += int(readiness_ok)
            total += 1
        sample_result["controlled_case"] = case.name
        sample_result["expected_check"] = checks

    results["controlled_validation"] = {
        "matched_checks": matched,
        "total_checks": total,
        "match_rate": round(matched / total, 4) if total else 0.0,
        "interpretation": "Observed outputs are preserved even when they differ from expected ranges.",
    }
    return results


def run(case_selection: str, metrics: str | None = None) -> dict[str, Any]:
    selected = list(CASES.values()) if case_selection == "all" else [CASES[case_selection]]
    enabled_metrics = parse_metrics(metrics)
    adapter = ControlledAdapter(selected)
    results = run_canonical_metrics(
        adapter,
        [case.sample for case in selected],
        verbose=True,
        enabled_metrics=enabled_metrics,
    )
    if adapter.query_count != len(selected):
        raise AssertionError("The orchestrator did not consume each controlled roadmap exactly once.")
    results = _attach_expectations(results, selected)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metric_suffix = "full" if not metrics else metrics.replace(",", "-")
    stem = f"canonical_integration_{case_selection}_{metric_suffix}"
    json_path = REPORT_DIR / f"{stem}.json"
    html_path = REPORT_DIR / f"{stem}.html"
    payload = {
        "suite": "canonical_controlled_integration",
        "case_selection": case_selection,
        "metrics": metrics or "canonical_default",
        "model": active_model_name(),
        "generator_calls": 0,
        "controlled_adapter_calls": adapter.query_count,
        "judge": results,
    }
    save_json(payload, str(json_path))
    save_html(None, results, str(html_path))

    validation = results["controlled_validation"]
    print("\nCONTROLLED INTEGRATION RESULT")
    print(f"  Generator calls: 0")
    print(f"  Controlled roadmaps: {adapter.query_count}")
    print(f"  Expected checks: {validation['matched_checks']}/{validation['total_checks']}")
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")
    return payload


def _load_sample(filename: str) -> dict[str, Any]:
    path = REPORT_DIR / filename
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload["judge"]["per_sample"][0]


def _merge_samples(base: dict[str, Any], additions: list[dict[str, Any]]) -> dict[str, Any]:
    merged = deepcopy(base)
    result_keys = (*SEMANTIC_METRICS, "step_overlap", "schema_validity")
    for addition in additions:
        for name in result_keys:
            if name in addition:
                merged[name] = deepcopy(addition[name])
        merged.setdefault("metric_timings", {}).update(addition.get("metric_timings", {}))

    canonical = {name: merged[name] for name in SEMANTIC_METRICS}
    merged["metric_scores"] = canonical_scores(canonical)
    merged["metric_contract_errors"] = validate_canonical_metrics(canonical)
    readiness_inputs = {name: merged[name] for name in result_keys if name in merged}
    merged["roadmap_readiness"] = roadmap_readiness(
        readiness_inputs,
        merged["metric_contract_errors"],
    )
    merged["response_time"] = 0.0
    return merged


def combine_existing() -> dict[str, Any]:
    """Combine completed partial runs without invoking any LLM or generator."""
    strong = _merge_samples(
        _load_sample("canonical_integration_strong.json"),
        [_load_sample("canonical_integration_strong_actionability.json")],
    )
    poor = _merge_samples(
        _load_sample("canonical_integration_poor_grounding.json"),
        [
            _load_sample("canonical_integration_poor_completeness.json"),
            _load_sample("canonical_integration_poor_actionability.json"),
            _load_sample(
                "canonical_integration_poor_logical_order-structure_quality-"
                "step_distinctness-schema_validity.json"
            ),
        ],
    )
    results = {
        "per_sample": [strong, poor],
        "sample_count": 2,
        "failure_count": 0,
        "failures": [],
        "metric_contract": "canonical_v1",
    }
    results["aggregated"] = CanonicalMetricEvaluator._aggregate(results["per_sample"])
    results = _attach_expectations(results, [CASES["strong"], CASES["poor"]])

    json_path = REPORT_DIR / "canonical_integration_controlled_final.json"
    html_path = REPORT_DIR / "canonical_integration_controlled_final.html"
    payload = {
        "suite": "canonical_controlled_integration",
        "case_selection": "strong+poor",
        "metrics": "canonical_default",
        "model": active_model_name(),
        "generator_calls": 0,
        "source": "completed controlled partial runs",
        "judge": results,
    }
    save_json(payload, str(json_path))
    save_html(None, results, str(html_path))
    validation = results["controlled_validation"]
    print("\nCONSOLIDATED CONTROLLED RESULT")
    print("  Generator calls: 0")
    print(f"  Cases: {results['sample_count']}")
    print(f"  Expected checks: {validation['matched_checks']}/{validation['total_checks']}")
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonical controlled integration validation")
    parser.add_argument("--case", choices=("strong", "poor", "all"), default="strong")
    parser.add_argument(
        "--metrics",
        default=None,
        help="Optional comma-separated canonical metric selection.",
    )
    parser.add_argument(
        "--combine-existing",
        action="store_true",
        help="Combine completed partial controlled runs without calling the LLM.",
    )
    args = parser.parse_args()
    if args.combine_existing:
        combine_existing()
    else:
        run(args.case, args.metrics)
