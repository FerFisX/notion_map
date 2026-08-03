"""Controlled validation for the dedicated Completeness metric."""

from __future__ import annotations

import argparse
import html
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.config import config
from evaluation.metrics.completeness_judge import CompletenessJudge
from src.llm_provider import active_model_name


def _step(index: int, label: str, description: str) -> dict[str, Any]:
    return {
        "id": f"step_{index}",
        "label": label,
        "description": description,
        "type": "inicio" if index == 1 else "proceso",
        "key_points": [],
    }


def _expected(element_id: str, name: str, importance: str) -> dict[str, str]:
    return {"id": element_id, "name": name, "importance": importance}


@dataclass(frozen=True)
class CompletenessCase:
    name: str
    question: str
    category: str
    roadmap: dict[str, Any]
    expected_elements: list[dict[str, str]]
    score_range: tuple[float, float]
    coverage_range: tuple[float, float]
    expected_gap_ids: frozenset[str]


BASE_EXPECTED = [
    _expected("calendar", "Create and configure a calendar table", "critical"),
    _expected("base_measure", "Create the base sales measure", "critical"),
    _expected("prior_period", "Create a prior-period comparison measure", "critical"),
    _expected("change_measure", "Calculate period-over-period change", "important"),
    _expected("validation", "Validate results against known periods", "important"),
]


CASES = [
    CompletenessCase(
        "good_complete_dax",
        "How do I build and validate year-over-year sales analysis in DAX?",
        "dax_power_bi",
        {"title": "Complete YoY Analysis", "steps": [
            _step(1, "Create the calendar", "Create and mark a continuous Date table."),
            _step(2, "Create total sales", "Define the Total Sales measure."),
            _step(3, "Create prior-year sales", "Define the prior-year comparison."),
            _step(4, "Calculate YoY change", "Calculate absolute and percentage change."),
            _step(5, "Validate results", "Compare known periods with approved values."),
        ]},
        BASE_EXPECTED, (8, 10), (100, 100), frozenset(),
    ),
    CompletenessCase(
        "medium_missing_validation",
        "How do I build and validate year-over-year sales analysis in DAX?",
        "dax_power_bi",
        {"title": "YoY Without Validation", "steps": [
            _step(1, "Create the calendar", "Create and mark a continuous Date table."),
            _step(2, "Create total sales", "Define the Total Sales measure."),
            _step(3, "Create prior-year sales", "Define the prior-year comparison."),
            _step(4, "Calculate YoY change", "Calculate percentage change."),
        ]},
        BASE_EXPECTED, (5, 7.9), (75, 90), frozenset({"validation"}),
    ),
    CompletenessCase(
        "poor_missing_prior_period",
        "How do I build and validate year-over-year sales analysis in DAX?",
        "dax_power_bi",
        {"title": "Sales Without Comparison", "steps": [
            _step(1, "Create the calendar", "Create and mark a continuous Date table."),
            _step(2, "Create total sales", "Define the Total Sales measure."),
            _step(3, "Format the visual", "Display total sales in a chart."),
            _step(4, "Validate totals", "Compare total sales with an approved source."),
        ]},
        BASE_EXPECTED, (0, 4.9), (45, 75),
        frozenset({"prior_period", "change_measure", "validation"}),
    ),
    CompletenessCase(
        "poor_shell_without_core",
        "How do I automate API records into Google Sheets with n8n?",
        "n8n_automation",
        {"title": "Automation Shell", "steps": [
            _step(1, "Define the objective", "Describe the desired automation."),
            _step(2, "Review the workflow", "Review the proposed workflow."),
            _step(3, "Close the project", "Document that the project is complete."),
        ]},
        [
            _expected("retrieve", "Retrieve records from the API", "critical"),
            _expected("transform", "Map source fields to destination fields", "critical"),
            _expected("write", "Write records to Google Sheets", "critical"),
            _expected("test", "Test the end-to-end workflow", "important"),
        ],
        (0, 4.9), (0, 20),
        frozenset({"retrieve", "transform", "write", "test"}),
    ),
    CompletenessCase(
        "orthogonal_complete_misordered",
        "How do I build and validate year-over-year sales analysis in DAX?",
        "dax_power_bi",
        {"title": "Complete but Misordered", "steps": [
            _step(1, "Validate results", "Compare known periods with approved values."),
            _step(2, "Calculate YoY change", "Calculate percentage change."),
            _step(3, "Create prior-year sales", "Define the prior-year comparison."),
            _step(4, "Create total sales", "Define the Total Sales measure."),
            _step(5, "Create the calendar", "Create and mark a continuous Date table."),
        ]},
        BASE_EXPECTED, (8, 10), (100, 100), frozenset(),
    ),
    CompletenessCase(
        "orthogonal_complete_vague",
        "How do I build and validate year-over-year sales analysis in DAX?",
        "dax_power_bi",
        {"title": "Complete but Vague", "steps": [
            _step(1, "Calendar", "Handle the date model."),
            _step(2, "Base measure", "Create the main sales calculation."),
            _step(3, "Previous period", "Add the prior-year calculation."),
            _step(4, "Change", "Add the year-over-year comparison."),
            _step(5, "Validation", "Check the expected periods."),
        ]},
        BASE_EXPECTED, (8, 10), (100, 100), frozenset(),
    ),
    CompletenessCase(
        "orthogonal_grounded_but_incomplete",
        "How do I build and validate year-over-year sales analysis in DAX?",
        "dax_power_bi",
        {"title": "Supported but Incomplete", "steps": [
            _step(1, "Create the calendar", "Create and mark a continuous Date table."),
            _step(2, "Create total sales", "Define the Total Sales measure."),
            _step(3, "Create prior-year sales", "Define the prior-year comparison."),
        ]},
        BASE_EXPECTED, (5, 7.9), (50, 75),
        frozenset({"change_measure", "validation"}),
    ),
    CompletenessCase(
        "good_narrow_scope",
        "How do I mark an existing Date table as the date table in Power BI?",
        "power_bi",
        {"title": "Mark a Date Table", "steps": [
            _step(1, "Select the Date table", "Select the existing Date table."),
            _step(2, "Mark the table", "Use Mark as date table and select Date[Date]."),
            _step(3, "Confirm the setting", "Confirm the table is marked successfully."),
        ]},
        [
            _expected("select", "Select the existing date table", "critical"),
            _expected("mark", "Mark it using its unique date column", "critical"),
            _expected("confirm", "Confirm the date-table setting", "important"),
        ],
        (8, 10), (100, 100), frozenset(),
    ),
]

SMOKE_NAMES = {"good_complete_dax", "medium_missing_validation", "poor_shell_without_core"}


def _validate(case: CompletenessCase, result: dict[str, Any]) -> dict[str, Any]:
    value = result["completeness"]
    score = float(value["score"])
    coverage = float(value["coverage_pct"])
    gap_ids = {
        str(item["expected_element_id"])
        for item in value["missing_elements"]
    }
    score_ok = case.score_range[0] <= score <= case.score_range[1]
    coverage_ok = case.coverage_range[0] <= coverage <= case.coverage_range[1]
    gaps_ok = gap_ids == set(case.expected_gap_ids)
    return {
        "passed": score_ok and coverage_ok and gaps_ok,
        "score_ok": score_ok,
        "coverage_ok": coverage_ok,
        "gaps_ok": gaps_ok,
        "expected_score_range": list(case.score_range),
        "expected_coverage_range": list(case.coverage_range),
        "expected_gap_ids": sorted(case.expected_gap_ids),
        "observed_gap_ids": sorted(gap_ids),
    }


def _run(cases: list[CompletenessCase], suite: str) -> dict[str, Any]:
    judge = CompletenessJudge()
    started = time.perf_counter()
    results = []
    for index, case in enumerate(cases, 1):
        print(f"  [{index}/{len(cases)}] {case.name}", flush=True)
        case_started = time.perf_counter()
        evaluated = judge.evaluate(
            case.roadmap,
            question=case.question,
            expected_elements=case.expected_elements,
        )
        results.append({
            "name": case.name,
            "question": case.question,
            "category": case.category,
            "roadmap": case.roadmap,
            "expected_reference": case.expected_elements,
            "completeness": evaluated["completeness"],
            "validation": _validate(case, evaluated),
            "duration_seconds": round(time.perf_counter() - case_started, 2),
        })
    passed = sum(item["validation"]["passed"] for item in results)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": active_model_name(),
        "suite": suite,
        "ollama_seed": os.getenv("OLLAMA_SEED", "unset"),
        "case_count": len(results),
        "passed_count": passed,
        "failed_count": len(results) - passed,
        "pass_rate": round(passed / len(results), 4),
        "duration_seconds": round(time.perf_counter() - started, 2),
        "results": results,
    }


def _score_color(score: float) -> str:
    if score >= 8:
        return "#52c41a"
    if score >= 5:
        return "#fa8c16"
    return "#f5222d"


def _badge(label: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{html.escape(label)}</span>'


def _verdict_badge(verdict: str) -> str:
    colors = {"PASS": "#52c41a", "NEEDS_REVIEW": "#fa8c16", "FAIL": "#f5222d"}
    label = verdict.replace("_", " ")
    return _badge(label, colors.get(verdict, "#8c8c8c"))


def _status_badge(status: str) -> str:
    colors = {"covered": "#52c41a", "partial": "#fa8c16", "missing": "#f5222d"}
    return _badge(status.upper(), colors.get(status, "#8c8c8c"))


def _display_name(value: str) -> str:
    acronyms = {"dax": "DAX", "bi": "BI", "n8n": "n8n"}
    return " ".join(
        acronyms.get(word.lower(), word.capitalize())
        for word in value.replace("_", " ").split()
    )


def _case_detail(item: dict[str, Any], index: int) -> str:
    value = item["completeness"]
    validation = item["validation"]
    score = float(value["score"])
    steps = "".join(
        f"<li><strong>{html.escape(str(step.get('label', '')))}</strong>: "
        f"{html.escape(str(step.get('description', '')))}</li>"
        for step in item["roadmap"].get("steps", [])
    )
    element_rows = "".join(
        f"<tr><td><strong>{html.escape(str(element['name']))}</strong><br>"
        f"<code>{html.escape(str(element['id']))}</code></td>"
        f"<td>{html.escape(str(element['importance']).title())}</td>"
        f"<td>{_status_badge(str(element['coverage_status']))}</td>"
        f"<td>{html.escape(', '.join(element['evidence_step_ids']) or '—')}</td>"
        f"<td>{html.escape(str(element['reason']))}</td></tr>"
        for element in value["expected_elements"]
    )
    gap_rows = "".join(
        f"<tr><td>{html.escape(str(gap['name']))}</td>"
        f"<td>{_badge(str(gap['severity']).upper(), {'high':'#f5222d','medium':'#fa8c16','low':'#1677ff'}.get(str(gap['severity']), '#8c8c8c'))}</td>"
        f"<td>{html.escape(str(gap['coverage_status']).title())}</td>"
        f"<td>{html.escape(str(gap['explanation']))}</td>"
        f"<td>{html.escape(str(gap['expected_phase']) or '—')}</td>"
        f"<td>{html.escape(str(gap['recommendation']) or '—')}</td></tr>"
        for gap in value["missing_elements"]
    ) or "<tr><td colspan='6' class='empty-state'>No coverage gaps reported.</td></tr>"
    strengths = "".join(
        f"<li>{html.escape(str(text))}</li>" for text in value["strengths"]
    ) or "<li class='empty-state'>No strengths reported.</li>"
    recommendations = "".join(
        f"<li>{html.escape(str(text))}</li>" for text in value["recommendations"]
    ) or "<li class='empty-state'>No recommendations reported.</li>"
    notes = ""
    if value["normalization_notes"]:
        items = "".join(
            f"<li>{html.escape(str(note))}</li>"
            for note in value["normalization_notes"]
        )
        notes = f"<details class='notes'><summary>Processing notes</summary><ul>{items}</ul></details>"
    expected_score = validation["expected_score_range"]
    expected_coverage = validation["expected_coverage_range"]
    return f"""
    <details class="case-detail" id="case-{index}">
      <summary>{index}. {html.escape(_display_name(item["name"]))} — completeness {score:.1f}/10</summary>
      <div class="detail-grid">
        <div>
          <h3>Roadmap</h3>
          <p class="question"><strong>Question:</strong> {html.escape(item["question"])}</p>
          <ol>{steps}</ol>
        </div>
        <div>
          <div class="metric-heading">
            <div><h3>Completeness</h3><span class="muted">Band: {html.escape(value["score_band"])}</span></div>
            <strong class="metric-score" style="color:{_score_color(score)}">{score:.1f}<small>/10</small></strong>
          </div>
          <section class="assessment"><h4>Overall Assessment</h4><p>{html.escape(value["reason"])}</p></section>
          <div class="stats">
            <span><strong>Coverage:</strong> {value["coverage_pct"]:.0f}%</span>
            <span><strong>Covered:</strong> {value["covered_element_count"]}</span>
            <span><strong>Partial:</strong> {value["partial_element_count"]}</span>
            <span><strong>Gaps:</strong> {value["missing_element_count"]}</span>
            <span><strong>Expected score:</strong> {expected_score[0]:.1f}-{expected_score[1]:.1f}</span>
            <span><strong>Expected coverage:</strong> {expected_coverage[0]:.0f}-{expected_coverage[1]:.0f}%</span>
          </div>
          <div class="findings-grid">
            <section class="finding"><h4>Strengths</h4><ul>{strengths}</ul></section>
            <section class="finding"><h4>Recommendations</h4><ul>{recommendations}</ul></section>
          </div>
          {notes}
        </div>
      </div>
      <section class="diagnostics">
        <h3>Expected Coverage Assessment</h3>
        <div class="table-scroll"><table class="nested">
          <thead><tr><th>Expected Element</th><th>Importance</th><th>Status</th><th>Evidence</th><th>Evaluation</th></tr></thead>
          <tbody>{element_rows}</tbody>
        </table></div>
        <h3>Coverage Gaps</h3>
        <div class="table-scroll"><table class="nested">
          <thead><tr><th>Element</th><th>Severity</th><th>Status</th><th>Explanation</th><th>Expected Phase</th><th>Recommendation</th></tr></thead>
          <tbody>{gap_rows}</tbody>
        </table></div>
      </section>
    </details>
    """


def _save(payload: dict[str, Any], output_dir: str, stem: str) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{stem}.json"
    html_path = directory / f"{stem}.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = "".join(
        f"<tr><td>{index}</td><td><a href='#case-{index}'><strong>{html.escape(_display_name(item['name']))}</strong></a><br>"
        f"<span class='muted'>{html.escape(_display_name(item['category']))}</span></td>"
        f"<td style='color:{_score_color(float(item['completeness']['score']))}'><strong>{item['completeness']['score']:.1f}/10</strong></td>"
        f"<td>{_verdict_badge(item['completeness']['verdict'])}</td>"
        f"<td>{item['completeness']['coverage_pct']:.0f}%</td>"
        f"<td>{item['completeness']['missing_element_count']}</td>"
        f"<td>{item['duration_seconds']:.1f}s</td>"
        f"<td>{_badge('EXPECTED RESULT' if item['validation']['passed'] else 'NEEDS REVIEW', '#52c41a' if item['validation']['passed'] else '#f5222d')}</td></tr>"
        for index, item in enumerate(payload["results"], 1)
    )
    details = "".join(
        _case_detail(item, index)
        for index, item in enumerate(payload["results"], 1)
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Completeness Controlled Validation</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f5f5f5;color:#222;padding:24px}}
h1{{margin-bottom:4px}}h3,h4{{margin-top:0}}.meta,.muted{{color:#666}}.meta{{margin-bottom:20px}}
.cards{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px}}.card{{background:#fff;border-radius:8px;padding:16px 20px;box-shadow:0 2px 8px #00000014;min-width:150px}}.card strong{{display:block;font-size:26px;margin-bottom:4px}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px #00000014;margin-bottom:24px}}th,td{{padding:10px 12px;border-bottom:1px solid #eee;text-align:left;vertical-align:top;font-size:13px}}th{{background:#fafafa}}
.badge{{color:#fff;padding:2px 8px;border-radius:12px;font-size:12px;white-space:nowrap}}a{{color:#222;text-decoration:none}}a:hover{{color:#1677ff}}
.case-detail{{background:#fff;margin-bottom:12px;padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px #0000000f}}.case-detail>summary{{cursor:pointer;font-weight:600}}
.detail-grid{{display:grid;grid-template-columns:minmax(300px,.85fr) minmax(560px,1.5fr);gap:28px;margin-top:18px}}li{{margin-bottom:7px}}.question{{color:#555}}
.metric-heading{{display:flex;justify-content:space-between;gap:16px;padding-bottom:14px;border-bottom:1px solid #eee}}.metric-heading h3{{margin-bottom:4px}}.metric-score{{font-size:30px;white-space:nowrap}}.metric-score small{{font-size:14px;color:#777}}
.assessment{{background:#f7f9fc;border-left:4px solid #1677ff;padding:12px 14px;margin:16px 0}}.assessment h4{{margin-bottom:6px}}.assessment p{{margin:0}}
.stats{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}}.stats span{{background:#f0f2f5;border-radius:12px;padding:4px 9px;font-size:12px}}
.findings-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.finding{{border:1px solid #eee;border-radius:7px;padding:14px}}.finding ul{{margin:0;padding-left:20px}}
.notes{{margin-top:14px;border:1px solid #eee;border-radius:7px;padding:10px 12px}}.diagnostics{{margin-top:24px;padding-top:20px;border-top:1px solid #eee}}.nested{{box-shadow:none;border:1px solid #eee}}.table-scroll{{overflow-x:auto}}.empty-state{{color:#777;font-style:italic}}code{{background:#f0f2f5;padding:1px 4px}}
@media(max-width:1100px){{.detail-grid{{grid-template-columns:1fr}}}}@media(max-width:700px){{body{{padding:12px}}.findings-grid{{grid-template-columns:1fr}}.overview{{display:block;overflow-x:auto}}}}
</style></head><body>
<h1>Completeness Controlled Validation</h1>
<p class="meta">Generated: {html.escape(payload["generated_at"])} | Model: {html.escape(payload["model"])} | Suite: {html.escape(payload["suite"])} | Ollama seed: {html.escape(payload["ollama_seed"])}</p>
<div class="cards"><div class="card"><strong>{payload["case_count"]}</strong>Total Cases</div><div class="card"><strong style="color:#52c41a">{payload["passed_count"]}</strong>Expected Results</div><div class="card"><strong style="color:#f5222d">{payload["failed_count"]}</strong>Needs Review</div><div class="card"><strong>{payload["pass_rate"]:.0%}</strong>Expected Match Rate</div><div class="card"><strong>{payload["duration_seconds"]/60:.1f}</strong>Total Minutes</div></div>
<h2>Completeness Overview</h2><table class="overview"><thead><tr><th>#</th><th>Case</th><th>Score</th><th>Metric Result</th><th>Coverage</th><th>Gaps</th><th>Duration</th><th>Score Check</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Case Details</h2>{details}</body></html>"""
    html_path.write_text(page, encoding="utf-8")
    return json_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Completeness with controlled roadmaps.")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--output-dir",
        default=os.path.join(config.reports_dir, "completeness_metric_validation"),
    )
    args = parser.parse_args()
    selected = CASES
    if args.case:
        names = set(args.case)
        selected = [case for case in CASES if case.name in names]
        unknown = names - {case.name for case in CASES}
        if unknown:
            raise SystemExit(f"Unknown Completeness cases: {', '.join(sorted(unknown))}")
    elif args.suite == "smoke":
        selected = [case for case in CASES if case.name in SMOKE_NAMES]
    payload = _run(selected, args.suite)
    stem = f"completeness_metric_validation_{args.suite}"
    json_path, html_path = _save(payload, args.output_dir, stem)
    print(f"Completeness validation complete: {payload['passed_count']}/{payload['case_count']}")
    print(f"  json: {json_path.resolve()}")
    print(f"  html: {html_path.resolve()}")


if __name__ == "__main__":
    main()
