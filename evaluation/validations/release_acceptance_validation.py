"""Fast release acceptance for the canonical orchestrator and reports.

This suite does not call an LLM. It validates deterministic behavior, wiring,
report integrity, checkpoints, tracking, and available controlled evidence.
Subjective semantic and visual checks are reported as MANUAL_REQUIRED.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import time
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from evaluation.config import config
from evaluation.dataset import EvalSample
from evaluation.metric_contract import SEMANTIC_METRICS, validate_canonical_metrics
from evaluation.metric_orchestrator import CanonicalMetricEvaluator, METRIC_ORDER
from evaluation.ragas_alignment import align_judge_with_ragas
from evaluation.readiness import roadmap_readiness
from evaluation.reporter import save_html, save_json
from evaluation.run_session import RunCheckpoint, RunProgress, create_run_dir
from evaluation.runner import CLI_METRICS, parse_metrics
from evaluation.tracking import _flatten_metrics
from evaluation.validations.canonical_integration_validation import (
    REPORT_DIR as CANONICAL_REPORT_DIR,
    run as run_live_orchestrator,
)


STATUS_COLORS = {
    "PASS": "#237804",
    "FAIL": "#cf1322",
    "SKIP": "#8c8c8c",
    "MANUAL_REQUIRED": "#d46b08",
}


@dataclass
class CheckResult:
    id: str
    area: str
    description: str
    severity: str
    validation_type: str
    status: str
    evidence: str
    elapsed_ms: int = 0


class AcceptanceSuite:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def auto(
        self,
        check_id: str,
        area: str,
        description: str,
        severity: str,
        validator: Callable[[], str | None],
    ) -> None:
        started = time.perf_counter()
        try:
            evidence = validator() or "Validated."
            status = "PASS"
        except Exception as exc:
            status = "FAIL"
            evidence = f"{type(exc).__name__}: {exc}"
        self.results.append(CheckResult(
            check_id,
            area,
            description,
            severity,
            "AUTO",
            status,
            evidence,
            round((time.perf_counter() - started) * 1000),
        ))

    def evidence(
        self,
        check_id: str,
        area: str,
        description: str,
        severity: str,
        validator: Callable[[], str | None] | None,
        missing_reason: str,
    ) -> None:
        if validator is None:
            self.results.append(CheckResult(
                check_id, area, description, severity, "EVIDENCE", "SKIP", missing_reason
            ))
            return
        started = time.perf_counter()
        try:
            detail = validator() or "Controlled evidence validated."
            status = "PASS"
        except Exception as exc:
            status = "FAIL"
            detail = f"{type(exc).__name__}: {exc}"
        self.results.append(CheckResult(
            check_id,
            area,
            description,
            severity,
            "EVIDENCE",
            status,
            detail,
            round((time.perf_counter() - started) * 1000),
        ))

    def manual(
        self,
        check_id: str,
        area: str,
        description: str,
        severity: str,
        evidence: str,
    ) -> None:
        self.results.append(CheckResult(
            check_id,
            area,
            description,
            severity,
            "MANUAL",
            "MANUAL_REQUIRED",
            evidence,
        ))


def _assert(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _verdict(score: float) -> str:
    return "PASS" if score >= 8 else "NEEDS_REVIEW" if score >= 5 else "FAIL"


def _step(index: int, label: str) -> dict[str, Any]:
    return {
        "id": f"step_{index}",
        "label": label,
        "description": f"Execute {label.lower()} with a named input and verifiable output.",
        "type": "inicio" if index == 1 else "fin" if index == 2 else "proceso",
        "key_points": ["input", "method", "expected output"],
    }


ROADMAP = {
    "title": "Controlled Roadmap",
    "steps": [_step(1, "Prepare the model"), _step(2, "Validate the result")],
}


def _metric_payloads(score: float = 9.0) -> dict[str, dict[str, Any]]:
    verdict = _verdict(score)
    reason = f"Controlled {verdict.lower()} rationale."
    return {
        "grounding": {
            "support_score": score,
            "verdict": verdict,
            "reason": reason,
            "claim_support_pct": 100 if score >= 8 else 20,
            "step_grounded_pct": 100 if score >= 8 else 0,
            "unsupported_claim_count": 0 if score >= 8 else 1,
            "unsupported_claims": [] if score >= 8 else [{
                "step_id": "step_1",
                "claim": "Unsupported controlled claim",
                "status": "unsupported",
                "explanation": "No supporting evidence.",
            }],
            "manual_review_required": False,
        },
        "completeness": {
            "score": score,
            "verdict": verdict,
            "reason": reason,
            "coverage_pct": 100 if score >= 8 else 40,
            "missing_element_count": 0 if score >= 8 else 1,
            "missing_elements": [] if score >= 8 else [{
                "name": "Validation",
                "importance": "critical",
                "coverage_status": "missing",
                "recommendation": "Add validation.",
            }],
            "manual_review_required": False,
        },
        "actionability": {
            "score": score,
            "verdict": verdict,
            "reason": reason,
            "actionability_density": 1.0 if score >= 8 else 0.5,
            "actionable_step_count": 2 if score >= 8 else 1,
            "weak_action_step_count": 0 if score >= 8 else 1,
            "weak_action_steps": [] if score >= 8 else [{
                "step_id": "step_1", "score": score, "reason": reason
            }],
            "step_actionability": [
                {"step_id": "step_1", "score": score},
                {"step_id": "step_2", "score": score},
            ],
            "manual_review_required": False,
        },
        "logical_order": {
            "score": score,
            "verdict": verdict,
            "reason": reason,
            "has_valid_sequence": score >= 8,
            "dependency_violation_count": 0 if score >= 8 else 1,
            "dependency_violations": [] if score >= 8 else [{
                "dependent_step_id": "step_1",
                "required_predecessor_step_id": "step_2",
                "dependency_type": "prerequisite",
                "explanation": reason,
            }],
            "manual_review_required": False,
        },
        "structure_quality": {
            "score": score,
            "verdict": verdict,
            "reason": reason,
            "criteria": {
                "goal_framing": score,
                "closure_quality": score,
                "granularity": score,
                "flow_coherence": score,
                "structural_usefulness": score,
                "scope_fit": score,
            },
            "criteria_rationales": {
                name: reason for name in (
                    "goal_framing", "closure_quality", "granularity",
                    "flow_coherence", "structural_usefulness", "scope_fit",
                )
            },
            "issue_count": 0 if score >= 8 else 1,
        },
        "step_distinctness": {
            "score": score,
            "verdict": verdict,
            "reason": reason,
            "weak_step_count": 0 if score >= 8 else 1,
            "weak_steps": [] if score >= 8 else [{
                "step_id": "step_1", "label": "Prepare", "reason": reason
            }],
        },
        "step_overlap": {
            "overall_severity": "none" if score >= 8 else "medium",
            "issue_count": 0 if score >= 8 else 1,
            "overlapping_pairs": [] if score >= 8 else [{
                "steps": [1, 2], "explanation": reason
            }],
        },
        "schema_validity": {
            "score": 10.0,
            "verdict": "PASS",
            "passed": 8,
            "total_checks": 8,
            "violations": [],
        },
    }


class FakeJudge:
    def __init__(self, name: str, calls: list[str], payloads: dict[str, dict[str, Any]]):
        self.name = name
        self.calls = calls
        self.payloads = payloads
        self.roadmaps: list[dict[str, Any]] = []

    def evaluate(self, roadmap: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(self.name)
        self.roadmaps.append(deepcopy(roadmap))
        if self.name == "step_semantics":
            return {
                "step_distinctness": deepcopy(self.payloads["step_distinctness"]),
                "step_overlap": deepcopy(self.payloads["step_overlap"]),
            }
        return {self.name: deepcopy(self.payloads[self.name])}


class FakeSchema:
    def __init__(self, calls: list[str], payload: dict[str, Any]):
        self.calls = calls
        self.payload = payload
        self.roadmaps: list[dict[str, Any]] = []

    def validate(self, roadmap: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("schema_validity")
        self.roadmaps.append(deepcopy(roadmap))
        return deepcopy(self.payload)


class FakeAdapter:
    def __init__(self, generated: dict[str, Any]):
        self.generated = generated
        self.calls = 0

    def query(self, question: str) -> dict[str, Any]:
        self.calls += 1
        return deepcopy(self.generated)


def _sample() -> EvalSample:
    return EvalSample(
        question="Controlled question <script>alert(1)</script>",
        ground_truth="Prepare the model and validate the result.",
        expected_keywords=["prepare", "validate"],
        category="controlled",
        expected_step_order=["Prepare the model", "Validate the result"],
    )


def _generated() -> dict[str, Any]:
    return {
        "question": _sample().question,
        "refined_question": "Controlled refined prompt",
        "query_intent": {"intent": "implementation"},
        "contexts": [{"id": "c1", "text": "Controlled context"}],
        "retrieval": {"context_strategy": "controlled"},
        "judge_context_strategy": "controlled",
        "roadmap": deepcopy(ROADMAP),
        "answer": "Controlled answer",
    }


def _fake_evaluation(
    score: float = 9.0,
    enabled_metrics: list[str] | None = None,
) -> tuple[dict[str, Any], FakeAdapter, list[str], list[FakeJudge], FakeSchema]:
    calls: list[str] = []
    payloads = _metric_payloads(score)
    judges = {
        name: FakeJudge(name, calls, payloads)
        for name in (
            "grounding", "completeness", "actionability", "logical_order",
            "structure_quality", "step_semantics",
        )
    }
    schema = FakeSchema(calls, payloads["schema_validity"])
    evaluator = CanonicalMetricEvaluator(judges=judges, structure_validator=schema)
    adapter = FakeAdapter(_generated())
    results = evaluator.evaluate(
        adapter,
        [_sample()],
        enabled_metrics=enabled_metrics,
    )
    return results, adapter, calls, list(judges.values()), schema


def _sample_result(score: float, question: str) -> dict[str, Any]:
    payloads = _metric_payloads(score)
    metrics = {
        name: payloads[name]
        for name in (*SEMANTIC_METRICS, "step_overlap", "schema_validity")
    }
    readiness = roadmap_readiness(metrics, validate_canonical_metrics({
        name: metrics[name] for name in SEMANTIC_METRICS
    }))
    return {
        "question": question,
        "refined_question": "Controlled refined prompt",
        "category": "controlled",
        "ground_truth": "Controlled truth",
        "answer": "Controlled answer",
        "contexts": [],
        "roadmap": deepcopy(ROADMAP),
        "steps": [step["label"] for step in ROADMAP["steps"]],
        "expected_steps": ["Prepare the model", "Validate the result"],
        **metrics,
        "metric_scores": {
            name: payloads[name].get("support_score", payloads[name].get("score", 0))
            for name in SEMANTIC_METRICS
        },
        "metric_timings": {name: 0.1 for name in METRIC_ORDER},
        "metric_contract_errors": [],
        "roadmap_readiness": readiness,
        "response_time": 0.1,
    }


def _judge_report_results() -> dict[str, Any]:
    samples = [
        _sample_result(9.0, "Strong <script>alert(1)</script> roadmap"),
        _sample_result(6.0, "Roadmap requiring review"),
        _sample_result(3.0, "Deliberately weak roadmap"),
    ]
    return {
        "per_sample": samples,
        "aggregated": CanonicalMetricEvaluator._aggregate(samples),
        "sample_count": 2,
        "failure_count": 0,
        "failures": [],
        "metric_contract": "canonical_v1",
    }


def _load_evidence(path: Path | None) -> tuple[Path | None, dict[str, Any] | None]:
    if path and path.exists():
        with path.open("r", encoding="utf-8") as file:
            return path, json.load(file)
    default = Path(config.reports_dir) / "canonical_integration_validation" / "canonical_integration_controlled_final.json"
    if default.exists():
        with default.open("r", encoding="utf-8") as file:
            return default, json.load(file)
    return None, None


def _find_evidence_cases(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    samples = payload["judge"]["per_sample"]
    strong = next(item for item in samples if item.get("controlled_case") == "strong_supported_dax")
    poor = next(item for item in samples if item.get("controlled_case") == "poor_cross_metric_dax")
    return strong, poor


def _write_summary(
    output_dir: Path,
    results: list[CheckResult],
    evidence_path: Path | None,
) -> tuple[Path, Path, dict[str, Any]]:
    counts = {
        status: sum(item.status == status for item in results)
        for status in STATUS_COLORS
    }
    release_status = (
        "FAILED" if counts["FAIL"] else
        "REVIEW_REQUIRED" if counts["SKIP"] or counts["MANUAL_REQUIRED"] else
        "APPROVED"
    )
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "release_status": release_status,
        "counts": counts,
        "evidence_path": str(evidence_path) if evidence_path else None,
        "checks": [asdict(item) for item in results],
    }
    json_path = output_dir / "release_acceptance_results.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    cards = "".join(
        f'<div class="card"><strong style="color:{STATUS_COLORS[status]}">{count}</strong><span>{status.replace("_", " ")}</span></div>'
        for status, count in counts.items()
    )
    rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(item.id)}</code></td>"
        f"<td>{html.escape(item.area)}</td>"
        f"<td>{html.escape(item.description)}</td>"
        f"<td>{html.escape(item.severity)}</td>"
        f"<td>{html.escape(item.validation_type)}</td>"
        f'<td><strong style="color:{STATUS_COLORS[item.status]}">{html.escape(item.status)}</strong></td>'
        f"<td>{html.escape(item.evidence)}</td>"
        "</tr>"
        for item in results
    )
    html_path = output_dir / "release_acceptance_results.html"
    html_path.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Release Acceptance</title>
<style>body{{font-family:Segoe UI,sans-serif;margin:24px;color:#262626}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{padding:16px;min-width:150px;border:1px solid #ddd;border-radius:8px}}.card strong,.card span{{display:block}}.card strong{{font-size:28px}}table{{border-collapse:collapse;width:100%;margin-top:20px}}th,td{{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}}th{{background:#fafafa;position:sticky;top:0}}code{{background:#f5f5f5;padding:2px 4px}}.meta{{color:#666}}</style></head>
<body><h1>Canonical Evaluation — Release Acceptance</h1>
<p class="meta">Status: <strong>{release_status}</strong> | Generated: {payload['generated_at']} | Evidence: {html.escape(str(evidence_path or 'not available'))}</p>
<div class="cards">{cards}</div>
<h2>Checklist Results</h2><table><thead><tr><th>ID</th><th>Area</th><th>Check</th><th>Severity</th><th>Type</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>""", encoding="utf-8")
    return json_path, html_path, payload


def _executive_results(results: list[CheckResult]) -> list[CheckResult]:
    """Collapse implementation checks into the release decisions humans need."""
    by_id = {item.id: item for item in results}
    groups = (
        ("ENV-01", "Environment", "Real environment preflight", "BLOCKER", ("ENV-05",)),
        ("ORCH-01", "Orchestrator", "Consistent end-to-end orchestration", "BLOCKER",
         ("ORCH-01", "ORCH-02", "ORCH-03", "ORCH-04", "ORCH-05", "ORCH-06", "ORCH-07", "ORCH-17")),
        ("ORCH-02", "Orchestrator", "Developer execution visibility", "IMPORTANT",
         ("ORCH-09", "ORCH-10")),
        ("ORCH-03", "Orchestrator", "Checkpoint and resume integrity", "BLOCKER",
         ("ORCH-11", "ORCH-12", "ORCH-13", "ORCH-14")),
        ("ORCH-04", "Orchestrator", "Real interruption workflow", "BLOCKER",
         ("ORCH-15", "ORCH-16")),
        ("MET-01", "Metrics", "Score contract consistency", "BLOCKER",
         ("MET-01", "MET-02", "MET-10")),
        ("MET-02", "Metrics", "Metric-specific diagnostics", "BLOCKER",
         ("MET-03", "MET-04", "MET-05", "MET-06", "MET-07", "MET-08", "MET-09")),
        ("MET-03", "Metrics", "Explanation usefulness", "IMPORTANT",
         ("MET-11", "MET-12")),
        ("MET-04", "Metrics", "MECE metric ownership", "BLOCKER",
         tuple(f"MECE-{index:02d}" for index in range(1, 9))),
        ("REP-01", "Report", "JSON-to-HTML integrity", "BLOCKER",
         tuple(f"REP-{index:02d}" for index in range(1, 9))),
        ("REP-02", "Report", "Partial and legacy-safe rendering", "BLOCKER",
         ("REP-09", "REP-10", "REP-11")),
        ("REP-03", "Report", "Safe dynamic content", "BLOCKER", ("REP-12",)),
        ("REP-04", "Report", "Human report usability", "IMPORTANT",
         ("REP-13", "REP-14", "REP-15", "REP-16")),
        ("AUX-01", "Supporting outputs", "RAGAS, Corpus, and MLflow separation", "BLOCKER",
         ("AUX-01", "AUX-02", "AUX-03", "AUX-04", "AUX-05")),
        ("AUX-02", "Supporting outputs", "Real MLflow artifacts", "IMPORTANT", ("AUX-06",)),
    )
    priority = {"PASS": 0, "MANUAL_REQUIRED": 1, "SKIP": 2, "FAIL": 3}
    executive: list[CheckResult] = []
    for check_id, area, description, severity, member_ids in groups:
        members = [by_id[member_id] for member_id in member_ids]
        status = max((item.status for item in members), key=priority.__getitem__)
        types = sorted({item.validation_type for item in members})
        if status == "PASS":
            evidence = f"{len(members)} internal checks passed."
        else:
            affected = [item.description for item in members if item.status == status]
            evidence = "; ".join(affected)
        executive.append(CheckResult(
            check_id,
            area,
            description,
            severity,
            " + ".join(types),
            status,
            evidence,
            sum(item.elapsed_ms for item in members),
        ))
    return executive


def _print_acceptance_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 64)
    print("PHASE 3 — RELEASE ACCEPTANCE SUMMARY")
    print("=" * 64)
    print(f"  Status: {summary['release_status']}")
    for status, count in summary["counts"].items():
        print(f"  {status:<16} {count}")
    print(f"  JSON: {summary['json_path']}")
    print(f"  HTML: {summary['html_path']}")


def run(
    evidence_json: Path | None = None,
    *,
    emit_summary: bool = True,
) -> dict[str, Any]:
    suite = AcceptanceSuite()
    output_dir = Path(config.reports_dir) / "release_acceptance" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    runner_source = (Path(config.reports_dir).parent / "runner.py").read_text(encoding="utf-8")
    reporter_source = (Path(config.reports_dir).parent / "reporter.py").read_text(encoding="utf-8")
    full_results, adapter, calls, judges, schema = _fake_evaluation()
    full_sample = full_results["per_sample"][0]
    partial_results, partial_adapter, partial_calls, _, _ = _fake_evaluation(
        enabled_metrics=["grounding", "schema_validity"]
    )

    suite.auto("ENV-01", "Environment", "Evaluation modules compile", "BLOCKER", lambda: (
        compile(runner_source, "runner.py", "exec") and "Runner source compiled."
    ))
    suite.auto("ENV-02", "Environment", "Canonical CLI metrics are exact", "BLOCKER", lambda: (
        _assert(tuple(CLI_METRICS) == (*SEMANTIC_METRICS, "schema_validity"), str(CLI_METRICS)) or
        "Seven canonical CLI names found."
    ))
    suite.auto("ENV-03", "Environment", "CLI modes exclude all and structure", "IMPORTANT", lambda: (
        _assert('choices=("judge", "ragas", "corpus")' in runner_source, "Canonical mode choices not found") or
        _assert('choices=["all"' not in runner_source, "Legacy all mode found") or
        "Only judge, ragas, and corpus are exposed."
    ))
    suite.auto("ENV-04", "Environment", "QoL CLI options are exposed", "IMPORTANT", lambda: (
        _assert(all(flag in runner_source for flag in ("--dry-run", "--list-metrics", "--list-samples", "--resume")), "Missing QoL flag") or
        "Discovery, dry-run, and resume flags found."
    ))
    suite.manual("ENV-05", "Environment", "Real team environment passes preflight", "BLOCKER", "Run the documented dry-run on the target machine.")

    suite.auto("ORCH-01", "Orchestrator", "One roadmap generation per sample", "BLOCKER", lambda: (
        _assert(adapter.calls == 1, f"adapter calls={adapter.calls}") or "Adapter called once."
    ))
    suite.auto("ORCH-02", "Orchestrator", "All judges receive the same roadmap", "BLOCKER", lambda: (
        _assert(all(judge.roadmaps == [ROADMAP] for judge in judges) and schema.roadmaps == [ROADMAP], "Roadmap changed between judges") or
        "Identical roadmap reached every judge."
    ))
    suite.auto("ORCH-03", "Orchestrator", "Default executes complete contract", "BLOCKER", lambda: (
        _assert(calls == list(METRIC_ORDER), str(calls)) or "Canonical call order completed."
    ))
    suite.auto("ORCH-04", "Orchestrator", "Selective metrics execute alone", "IMPORTANT", lambda: (
        _assert(partial_calls == ["grounding", "schema_validity"], str(partial_calls)) or
        _assert(set(partial_results["per_sample"][0]["metric_scores"]) == {"grounding"}, "Unexpected semantic score") or
        "Only Grounding and Schema executed."
    ))
    suite.auto("ORCH-05", "Orchestrator", "Canonical execution order", "IMPORTANT", lambda: (
        _assert(calls == list(METRIC_ORDER), str(calls)) or "Order matches METRIC_ORDER."
    ))
    suite.auto("ORCH-06", "Orchestrator", "Metric payloads are preserved", "BLOCKER", lambda: (
        _assert(full_sample["grounding"]["reason"] == "Controlled pass rationale.", "Reason changed") or
        _assert(full_sample["step_overlap"]["overall_severity"] == "none", "Diagnostic changed") or
        "Canonical reasons and diagnostics preserved."
    ))
    suite.auto("ORCH-07", "Orchestrator", "Readiness uses canonical verdicts", "BLOCKER", lambda: (
        _assert(full_sample["roadmap_readiness"]["status"] == "READY", "Full PASS is not READY") or
        _assert(_sample_result(3, "poor")["roadmap_readiness"]["status"] == "FAIL", "Semantic FAIL is not FAIL") or
        _assert(partial_results["per_sample"][0]["roadmap_readiness"]["status"] == "NOT_EVALUATED", "Partial contract not marked") or
        "READY, FAIL, and NOT_EVALUATED paths validated."
    ))
    suite.auto("ORCH-08", "Orchestrator", "Legacy fields are absent", "BLOCKER", lambda: (
        _assert(not ({"classic", "mese", "sequence_eval"} & set(full_sample)), "Legacy field found") or
        "No legacy field in canonical sample."
    ))

    progress_dir = create_run_dir(str(output_dir), "progress-check")
    progress = RunProgress(progress_dir, 2)
    progress.stage_started("Grounding")
    progress.stage_completed("Grounding", 1.0, 9.0, "PASS")
    progress.stage_started("Completeness")
    progress.stage_completed("Completeness", 1.0, 9.0, "PASS")
    progress.finish()
    log_text = progress.log_path.read_text(encoding="utf-8")
    suite.auto("ORCH-09", "Orchestrator", "Progress includes active/completed metric", "IMPORTANT", lambda: (
        _assert("-> Grounding" in log_text and "OK Grounding" in log_text and "9.0/10" in log_text, "Progress detail missing") or
        "Metric start, completion, score, and verdict logged."
    ))
    suite.auto("ORCH-10", "Orchestrator", "ETA and total time are available", "IMPORTANT", lambda: (
        _assert("estimated remaining" in log_text and "total time" in log_text, "Timing detail missing") or
        "ETA and total time logged."
    ))
    second_progress_dir = create_run_dir(str(output_dir), "progress-check")
    suite.auto("ORCH-11", "Orchestrator", "Run directories are unique", "IMPORTANT", lambda: (
        _assert(progress_dir != second_progress_dir and progress_dir.exists() and second_progress_dir.exists(), "Directory collision") or
        "Repeated names created distinct directories."
    ))

    checkpoint_dir = create_run_dir(str(output_dir), "checkpoint-check")
    checkpoint = RunCheckpoint.create(
        checkpoint_dir, "checkpoint-check", "judge", ["grounding", "completeness"], [_sample()]
    )
    checkpoint.record_generation(0, _generated(), 1.0)
    partial = deepcopy(full_sample)
    for key in ("completeness", "actionability", "logical_order", "structure_quality", "step_distinctness", "step_overlap", "schema_validity"):
        partial.pop(key, None)
    partial["metric_scores"] = {"grounding": 9.0}
    checkpoint.record_result(0, "grounding", partial)
    reloaded = RunCheckpoint.load(checkpoint.path)
    suite.auto("ORCH-12", "Orchestrator", "Checkpoint persists generated and completed work", "BLOCKER", lambda: (
        _assert(reloaded.entry(0)["generated"]["roadmap"] == ROADMAP, "Roadmap not restored") or
        _assert(reloaded.entry(0)["completed_metrics"] == ["grounding"], "Metric state not restored") or
        "Generation and Grounding survived reload."
    ))

    resume_calls: list[str] = []
    resume_payloads = _metric_payloads()
    resume_judges = {
        "grounding": SimpleNamespace(evaluate=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Grounding repeated"))),
        "completeness": FakeJudge("completeness", resume_calls, resume_payloads),
    }
    no_generation = SimpleNamespace(query=lambda question: (_ for _ in ()).throw(AssertionError("Generation repeated")))
    resumed = CanonicalMetricEvaluator(judges=resume_judges).evaluate(
        no_generation,
        [_sample()],
        enabled_metrics=["grounding", "completeness"],
        checkpoint=reloaded,
    )
    suite.auto("ORCH-13", "Orchestrator", "Resume skips completed work", "BLOCKER", lambda: (
        _assert(resume_calls == ["completeness"], str(resume_calls)) or
        _assert(set(resumed["per_sample"][0]["metric_scores"]) == {"grounding", "completeness"}, "Merged result incomplete") or
        "Only pending Completeness executed."
    ))
    interrupted = RunCheckpoint.load(checkpoint.path)
    interrupted.mark("INTERRUPTED")
    suite.auto("ORCH-14", "Orchestrator", "Interrupted checkpoint remains recoverable", "BLOCKER", lambda: (
        _assert(interrupted.state["status"] == "INTERRUPTED", "Status not persisted") or
        _assert(interrupted.remaining_units() >= 0, "Invalid remaining units") or
        "Interrupted status and remaining work are readable."
    ))
    suite.manual("ORCH-15", "Orchestrator", "Real Ctrl+C creates partial artifacts", "BLOCKER", "Interrupt one real Judge run after a completed metric.")
    suite.manual("ORCH-16", "Orchestrator", "Resume preserves original run selection", "IMPORTANT", "Confirm on a real interrupted run using the documented command.")

    suite.auto("MET-01", "Metrics", "Canonical scores stay in 0–10", "BLOCKER", lambda: (
        _assert(all(0 <= value <= 10 for value in full_sample["metric_scores"].values()), str(full_sample["metric_scores"])) or
        "All synthetic canonical scores are in range."
    ))
    suite.auto("MET-02", "Metrics", "Score and verdict bands agree", "BLOCKER", lambda: (
        _assert(all(_verdict(score) == full_sample[name]["verdict"] for name, score in full_sample["metric_scores"].items()), "Band mismatch") or
        "All semantic verdicts match score bands."
    ))

    evidence_path, evidence_payload = _load_evidence(evidence_json)
    evidence_cases = _find_evidence_cases(evidence_payload) if evidence_payload else None
    missing = "Controlled final evidence JSON was not found; run the documented controlled validation."

    suite.evidence("ORCH-17", "Orchestrator", "Real judges complete the controlled orchestration", "BLOCKER", (
        (lambda: (
            _assert(evidence_payload.get("generator_calls") == 0, "Roadmap generator was called") or
            _assert(evidence_payload.get("controlled_adapter_calls") == 2, "Both controlled roadmaps were not consumed") or
            _assert(evidence_payload["judge"].get("sample_count") == 2, "Expected two controlled samples") or
            _assert(evidence_payload["judge"].get("failure_count") == 0, "A judge execution failed") or
            _assert(all(
                all(name in sample for name in (*SEMANTIC_METRICS, "schema_validity"))
                for sample in evidence_payload["judge"]["per_sample"]
            ), "A canonical metric result is missing") or
            "Two controlled roadmaps completed every real judge without generator calls."
        )) if evidence_payload and "controlled_adapter_calls" in evidence_payload else None
    ), missing)

    suite.evidence("MET-03", "Metrics", "Grounding support diagnostics are coherent", "BLOCKER", (
        (lambda: (
            _assert(evidence_cases[0]["grounding"]["support_score"] >= 8, "Strong Grounding too low") or
            _assert(evidence_cases[1]["grounding"]["unsupported_claim_count"] > 0, "Poor claims not detected") or
            "Strong support and poor unsupported claims validated."
        )) if evidence_cases else None
    ), missing)
    suite.evidence("MET-04", "Metrics", "Completeness coverage diagnostics are coherent", "BLOCKER", (
        (lambda: (
            _assert(evidence_cases[0]["completeness"]["missing_element_count"] == 0, "Strong case has gaps") or
            _assert(evidence_cases[1]["completeness"]["missing_element_count"] == len(evidence_cases[1]["completeness"]["missing_elements"]), "Missing count mismatch") or
            "Coverage and missing elements validated."
        )) if evidence_cases else None
    ), missing)
    suite.evidence("MET-05", "Metrics", "Actionability per-step diagnostics are coherent", "BLOCKER", (
        (lambda: (
            _assert(all(
                item["step_id"] in {
                    str(step.get("id", ""))
                    for step in evidence_cases[1]["roadmap"].get("steps", [])
                    if isinstance(step, dict)
                }
                for item in evidence_cases[1]["actionability"]["step_actionability"]
            ), "Unknown step ID") or
            _assert(evidence_cases[1]["actionability"]["weak_action_step_count"] == len(evidence_cases[1]["actionability"]["weak_action_steps"]), "Weak count mismatch") or
            "Step IDs and weak-step count validated."
        )) if evidence_cases else None
    ), missing)
    suite.evidence("MET-06", "Metrics", "Logical Order dependency diagnostics are coherent", "BLOCKER", (
        (lambda: (
            _assert(evidence_cases[1]["logical_order"]["dependency_violation_count"] == len(evidence_cases[1]["logical_order"]["dependency_violations"]), "Violation count mismatch") or
            "Dependency count matches violation list."
        )) if evidence_cases else None
    ), missing)
    suite.evidence("MET-07", "Metrics", "Structure exposes six dimensions", "IMPORTANT", (
        (lambda: (
            _assert(set(evidence_cases[0]["structure_quality"]["criteria"]) == {"goal_framing", "closure_quality", "granularity", "flow_coherence", "structural_usefulness", "scope_fit"}, "Dimension set mismatch") or
            "Six Structure dimensions found."
        )) if evidence_cases else None
    ), missing)
    suite.evidence("MET-08", "Metrics", "Schema stays independent from semantic quality", "BLOCKER", (
        (lambda: (
            _assert(evidence_cases[1]["schema_validity"]["score"] == 10, "Poor schema did not pass") or
            _assert(evidence_cases[1]["roadmap_readiness"]["status"] == "FAIL", "Poor semantic roadmap did not fail") or
            "Schema 10/10 coexists with semantic FAIL."
        )) if evidence_cases else None
    ), missing)
    suite.evidence("MET-09", "Metrics", "Step Distinctness owns score; overlap is diagnostic", "BLOCKER", (
        (lambda: (
            _assert("score" in evidence_cases[1]["step_distinctness"], "Distinctness score missing") or
            _assert("score" not in evidence_cases[1]["step_overlap"], "Overlap duplicate score found") or
            "Distinctness score and overlap diagnostic are separate."
        )) if evidence_cases else None
    ), missing)
    suite.evidence("MET-10", "Metrics", "Controlled contract has no errors", "BLOCKER", (
        (lambda: (
            _assert(all(not sample["metric_contract_errors"] for sample in evidence_payload["judge"]["per_sample"]), "Contract errors found") or
            "No canonical metric contract errors were found."
        )) if evidence_payload else None
    ), missing)
    suite.manual("MET-11", "Metrics", "Reasons explain observed score bands", "IMPORTANT", "Read reasons for strong and poor controlled cases.")
    suite.manual("MET-12", "Metrics", "Recommendations are actionable and case-specific", "IMPORTANT", "Review recommendations in Case Details.")

    for check_id, description in (
        ("MECE-01", "Complete but unsupported ownership"),
        ("MECE-02", "Grounded but incomplete ownership"),
        ("MECE-03", "Complete but vague ownership"),
        ("MECE-04", "Dependency-inverted ownership"),
        ("MECE-05", "Distinct but poorly framed ownership"),
        ("MECE-06", "Redundant roadmap ownership"),
        ("MECE-07", "Strong content with invalid schema"),
        ("MECE-08", "Shared terminology without duplicated responsibility"),
    ):
        suite.manual(check_id, "MECE", description, "BLOCKER" if check_id not in {"MECE-03", "MECE-05"} else "IMPORTANT", "Run or inspect the corresponding orthogonal controlled case; semantic ownership requires human review.")

    judge_results = _judge_report_results()
    judge_json = output_dir / "synthetic_judge_report.json"
    judge_html = output_dir / "synthetic_judge_report.html"
    save_json({"judge": judge_results}, str(judge_json))
    save_html(None, judge_results, str(judge_html))
    report_text = judge_html.read_text(encoding="utf-8")
    report_json = json.loads(judge_json.read_text(encoding="utf-8"))

    suite.auto("REP-01", "Report", "Judge report is standalone UTF-8 HTML", "BLOCKER", lambda: (
        _assert(report_text.startswith("<!doctype html>") and '<meta charset="utf-8">' in report_text and "</html>" in report_text, "Invalid HTML shell") or
        "Standalone UTF-8 HTML generated."
    ))
    suite.auto("REP-02", "Report", "General Summary shows canonical cards", "BLOCKER", lambda: (
        _assert(all(label in report_text for label in ("Batch Status", "Evaluated Roadmaps", "Grounding", "Completeness", "Actionability", "Logical Order", "Structure Quality", "Step Distinctness", "Schema Validity")), "Summary card missing") or
        "Canonical summary cards found."
    ))
    suite.auto("REP-03", "Report", "Batch row count and data align", "BLOCKER", lambda: (
        _assert(report_text.count('class="case" id="case-') == len(report_json["judge"]["per_sample"]), "Case count mismatch") or
        _assert("READY" in report_text and "NEEDS REVIEW" in report_text and "FAIL" in report_text, "Statuses missing") or
        "Three JSON samples and three HTML cases align."
    ))
    suite.auto("REP-04", "Report", "Case Details preserve roadmap inputs", "BLOCKER", lambda: (
        _assert(all(value in report_text for value in ("Roadmap Output", "Expected Sequence", "Refined prompt", "Prepare the model", "Validate the result")), "Case input missing") or
        "Roadmap, expected sequence, and refined prompt found."
    ))
    suite.auto("REP-05", "Report", "Every metric has detailed diagnostics", "BLOCKER", lambda: (
        _assert(all(value in report_text for value in ("Grounding evidence", "Completeness gaps", "Actionability by step", "Logical Order dependencies", "Structure dimensions", "Step Distinctness and Overlap")), "Diagnostic section missing") or
        "All metric diagnostics found."
    ))
    suite.auto("REP-06", "Report", "HTML Readiness matches JSON", "BLOCKER", lambda: (
        _assert(all(sample["roadmap_readiness"]["status"].replace("_", " ") in report_text for sample in report_json["judge"]["per_sample"]), "Readiness mismatch") or
        "JSON readiness statuses appear in HTML."
    ))
    suite.auto("REP-07", "Report", "Score colors use canonical bands", "IMPORTANT", lambda: (
        _assert(all(color in report_text for color in ("#52c41a", "#fa8c16", "#f5222d")), "Expected score colors missing") or
        "PASS, NEEDS_REVIEW, and FAIL colors rendered."
    ))
    suite.auto("REP-08", "Report", "Quality Signals expose relevant issues", "IMPORTANT", lambda: (
        _assert("Quality Signals" in report_text and "Grounding:" in report_text and "Completeness:" in report_text, "Issue list incomplete") or
        "Multiple quality issues are visible."
    ))

    partial_sample = deepcopy(judge_results["per_sample"][0])
    for key in (*SEMANTIC_METRICS, "step_overlap", "schema_validity"):
        if key != "structure_quality":
            partial_sample.pop(key, None)
    partial_sample["metric_scores"] = {"structure_quality": 9.0}
    partial_sample["roadmap_readiness"] = {"status": "NOT_EVALUATED", "issues": [], "reasons": ["Partial"]}
    partial_report = {
        "per_sample": [partial_sample],
        "aggregated": CanonicalMetricEvaluator._aggregate([partial_sample]),
        "sample_count": 1,
    }
    partial_html = output_dir / "synthetic_partial_report.html"
    save_html(None, partial_report, str(partial_html))
    partial_text = partial_html.read_text(encoding="utf-8")
    suite.auto("REP-09", "Report", "Partial report omits absent metrics", "BLOCKER", lambda: (
        _assert("Schema Validity" not in partial_text and "Grounding evidence" not in partial_text, "Absent metric rendered") or
        "Unrequested Schema and Grounding are omitted."
    ))
    suite.auto("REP-10", "Report", "Legacy report content is absent", "BLOCKER", lambda: (
        _assert(not any(value in report_text for value in ("MESE", "Score General", "Revisión Humana", "Export Human", "Classic Metrics")), "Legacy content found") or
        "No legacy score or human-review UI found."
    ))
    suite.auto("REP-11", "Report", "Judge report excludes RAGAS and Corpus", "IMPORTANT", lambda: (
        _assert("RAGAS — Additional Diagnostics" not in report_text and "Corpus Diagnostics" not in report_text, "Auxiliary report mixed in") or
        "Judge report contains no RAGAS/Corpus sections."
    ))
    suite.auto("REP-12", "Report", "Dynamic content is HTML-escaped", "BLOCKER", lambda: (
        _assert("<script>alert(1)</script>" not in report_text and "&lt;script&gt;alert(1)&lt;/script&gt;" in report_text, "HTML injection not escaped") or
        "Injected script text rendered safely as text."
    ))
    suite.manual("REP-13", "Report", "Multi-case layout remains readable", "IMPORTANT", "Open the generated synthetic Judge HTML and review batch/case navigation.")
    suite.manual("REP-14", "Report", "Report language is consistently English", "IMPORTANT", "Automated labels are English; visually inspect free-form model text separately.")
    suite.manual("REP-15", "Report", "Responsive layout is usable", "IMPORTANT", "Review desktop and narrow viewport rendering.")
    suite.manual("REP-16", "Report", "Stakeholder interpretation is immediate", "IMPORTANT", "Confirm status and drivers are understandable without opening diagnostics.")

    ragas_results = {
        "per_sample": [{"question": "Q", "category": "test", "scores": {
            "faithfulness": 0.9, "answer_relevancy": 0.8,
            "context_precision": 0.7, "context_recall": 0.6,
        }}],
        "aggregated": {},
    }
    ragas_html = output_dir / "synthetic_ragas_report.html"
    save_html(ragas_results, None, str(ragas_html))
    ragas_text = ragas_html.read_text(encoding="utf-8")
    suite.auto("AUX-01", "Auxiliary", "RAGAS report remains independent", "IMPORTANT", lambda: (
        _assert("RAGAS" in ragas_text and "Batch Status" not in ragas_text and "Corpus Diagnostics" not in ragas_text, "RAGAS report mixed") or
        "Independent RAGAS report generated."
    ))
    corpus_results = {
        "overall_corpus_score": 8.0,
        "n_chunks_evaluated": 2,
        "n_chunks_total": 2,
        "aggregated": {"avg_quality": 8, "avg_coherencia": 8, "avg_densidad_tecnica": 8, "avg_utilidad_rag": 8},
        "semantic_diversity": {"diversity_score": 0.8, "redundant_pairs": 0},
    }
    corpus_html = output_dir / "synthetic_corpus_report.html"
    save_html(None, None, str(corpus_html), corpus_results=corpus_results)
    corpus_text = corpus_html.read_text(encoding="utf-8")
    suite.auto("AUX-02", "Auxiliary", "Corpus report remains independent", "IMPORTANT", lambda: (
        _assert("Corpus Diagnostics" in corpus_text and "Batch Status" not in corpus_text and "RAGAS" not in corpus_text, "Corpus report mixed") or
        "Independent Corpus report generated."
    ))
    aligned = align_judge_with_ragas(judge_results, {
        "per_sample": [
            {"question": sample["question"], "scores": {"faithfulness": 0.1, "context_precision": 0.2, "context_recall": 0.3, "answer_relevancy": 0.4}}
            for sample in judge_results["per_sample"]
        ],
        "aggregated": {},
    })
    suite.auto("AUX-03", "Auxiliary", "RAGAS cannot override canonical Grounding", "BLOCKER", lambda: (
        _assert(aligned["per_sample"][0]["grounding"]["support_score"] == judge_results["per_sample"][0]["grounding"]["support_score"], "Grounding overridden") or
        _assert(aligned["per_sample"][0]["roadmap_readiness"] == judge_results["per_sample"][0]["roadmap_readiness"], "Readiness overridden") or
        "Grounding and Readiness preserved."
    ))
    flattened = _flatten_metrics(judge_results, None, None)
    suite.auto("AUX-04", "Tracking", "MLflow flattening uses canonical names", "BLOCKER", lambda: (
        _assert("roadmap.grounding.mean_score" in flattened and "roadmap.step_distinctness.mean_score" in flattened and "roadmap.ready_rate" in flattened, "Canonical tracking key missing") or
        _assert(not any("mese" in key or "classic" in key for key in flattened), "Legacy tracking key found") or
        "Canonical metric and readiness names found."
    ))
    partial_flattened = _flatten_metrics(partial_report, None, None)
    suite.auto("AUX-05", "Tracking", "Unrequested Schema is omitted from MLflow", "IMPORTANT", lambda: (
        _assert("schema_validity.mean_score" not in partial_flattened, "False Schema metric logged") or
        "Partial run contains no fake Schema score."
    ))
    suite.manual("AUX-06", "Tracking", "MLflow artifacts open correctly", "IMPORTANT", "Complete one tracked run and open JSON, HTML, checkpoint, and run.log.")

    executive_results = _executive_results(suite.results)
    json_path, html_path, summary = _write_summary(output_dir, executive_results, evidence_path)
    summary["json_path"] = str(json_path)
    summary["html_path"] = str(html_path)
    if emit_summary:
        _print_acceptance_summary(summary)
    if summary["counts"]["FAIL"] and emit_summary:
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast canonical evaluation release acceptance")
    parser.add_argument("--evidence-json", type=Path, default=None)
    parser.add_argument(
        "--live-orchestrator",
        action="store_true",
        help="Run strong and poor controlled roadmaps through every real judge before acceptance checks.",
    )
    args = parser.parse_args()
    evidence_json = args.evidence_json
    if args.live_orchestrator:
        print("\n" + "=" * 64)
        print("PHASE 1 — LIVE CONTROLLED ORCHESTRATOR")
        print("Real judges enabled; roadmap generation disabled.")
        print("=" * 64)
        run_live_orchestrator("all")
        evidence_json = CANONICAL_REPORT_DIR / "canonical_integration_all_full.json"
    print("\n" + "=" * 64)
    print("PHASE 2 — DETERMINISTIC REGRESSION CHECKS")
    print("No LLM calls are made in this phase.")
    print("=" * 64)
    with redirect_stdout(io.StringIO()):
        summary = run(evidence_json, emit_summary=False)
    print("Deterministic regression checks completed.")
    _print_acceptance_summary(summary)
    if summary["counts"]["FAIL"]:
        raise SystemExit(1)
