"""Controlled validation for semantic step metrics.

This script bypasses the roadmap generator and evaluates handcrafted roadmaps
with known step-overlap quality. It is intended to validate whether
StepSemanticJudge reacts correctly to:

- well-separated steps,
- medium semantic overlap,
- strong semantic overlap.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.config import config
from evaluation.step_semantic_judge import StepSemanticJudge
from src.llm_provider import active_model_name


@dataclass(frozen=True)
class StepMetricCase:
    name: str
    expected_level: str
    question: str
    category: str
    roadmap: dict[str, Any]
    expected_score_min: float
    expected_score_max: float
    expected_overlap_min: int
    expected_overlap_max: int | None = None
    expected_severities: tuple[str, ...] = ()
    expected_overall_severities: tuple[str, ...] = ()


def _step(idx: int, label: str, description: str, points: list[str] | None = None) -> dict[str, Any]:
    step_type = "inicio" if idx == 1 else "fin" if label.lower().startswith(("validar", "publicar", "documentar")) else "proceso"
    return {
        "id": f"step_{idx}",
        "label": label,
        "description": description,
        "type": step_type,
        "key_points": points or [],
    }


CASES: list[StepMetricCase] = [
    StepMetricCase(
        name="good_dax_time_intelligence",
        expected_level="good_no_overlap",
        question="How do I learn Time Intelligence in DAX to compare sales across periods?",
        category="dax_power_bi",
        expected_score_min=8.0,
        expected_score_max=10.0,
        expected_overlap_min=0,
        expected_overlap_max=0,
        expected_overall_severities=("none",),
        roadmap={
            "title": "Learn DAX Time Intelligence for Period Comparisons",
            "steps": [
                _step(1, "Create a calendar table", "Build a complete date table with continuous dates and required date attributes.", ["date column", "year", "month", "quarter"]),
                _step(2, "Mark the date table", "Configure the table as the official date table in Power BI so DAX time functions use it correctly.", ["mark as date table"]),
                _step(3, "Create base sales measures", "Define reusable measures such as total sales and transaction count before adding time logic.", ["SUM", "base measure"]),
                _step(4, "Add period comparison measures", "Use functions such as DATEADD and SAMEPERIODLASTYEAR to compare current and previous periods.", ["YoY", "MoM"]),
                _step(5, "Visualize comparisons", "Place the measures in charts and matrices to compare trends across months, quarters, and years.", ["line chart", "matrix"]),
                _step(6, "Validate against known totals", "Check the DAX outputs against trusted totals and manually calculated samples.", ["quality check"]),
            ],
        },
    ),
    StepMetricCase(
        name="good_n8n_api_to_sheets",
        expected_level="good_no_overlap",
        question="How do I automate loading API data into Google Sheets with n8n?",
        category="n8n_automation",
        expected_score_min=8.0,
        expected_score_max=10.0,
        expected_overlap_min=0,
        expected_overlap_max=0,
        expected_overall_severities=("none",),
        roadmap={
            "title": "Automate API Data Loading into Google Sheets with n8n",
            "steps": [
                _step(1, "Define the trigger", "Choose whether the workflow runs manually, on a schedule, or from a webhook.", ["schedule", "webhook"]),
                _step(2, "Configure the API request", "Set the HTTP method, endpoint, headers, query parameters, and authentication.", ["HTTP Request"]),
                _step(3, "Transform the response", "Map nested API fields into the tabular structure expected by Google Sheets.", ["field mapping"]),
                _step(4, "Handle missing or invalid values", "Normalize empty fields, dates, and numeric formats before writing rows.", ["data cleaning"]),
                _step(5, "Write rows to Google Sheets", "Connect the Google Sheets node and choose append or update behavior.", ["append", "update"]),
                _step(6, "Add error monitoring", "Capture failed executions and send a notification with the failing payload.", ["retry", "alert"]),
            ],
        },
    ),
    StepMetricCase(
        name="medium_power_bi_modeling_overlap",
        expected_level="medium_overlap",
        question="How do I design a star schema in Power BI?",
        category="power_bi_modeling",
        expected_score_min=5.0,
        expected_score_max=8.5,
        expected_overlap_min=1,
        expected_overlap_max=3,
        expected_severities=("low", "medium"),
        expected_overall_severities=("low", "medium"),
        roadmap={
            "title": "Design a Star Schema in Power BI",
            "steps": [
                _step(1, "Identify business metrics", "List the measures that the report must answer, such as sales, margin, and quantity.", ["business questions"]),
                _step(2, "Define the fact table", "Select the transactional table and confirm the grain of each row.", ["grain", "fact"]),
                _step(3, "Define dimension tables", "Create descriptive tables for date, product, customer, and geography.", ["dimensions"]),
                _step(4, "Clean product dimension attributes", "Remove duplicate product attributes and standardize product descriptions inside the dimension table.", ["dimension quality"]),
                _step(5, "Standardize product dimension fields", "Review the same product attributes again to standardize product descriptions and improve model usability.", ["cardinality", "descriptions"]),
                _step(6, "Validate relationships", "Check one-to-many relationships and confirm filters flow from dimensions to facts.", ["relationships"]),
            ],
        },
    ),
    StepMetricCase(
        name="medium_abstraction_levels_overlap",
        expected_level="medium_overlap",
        question="How do I explain a technical system using abstraction levels?",
        category="abstraction_levels",
        expected_score_min=5.0,
        expected_score_max=8.5,
        expected_overlap_min=1,
        expected_overlap_max=3,
        expected_severities=("low", "medium"),
        expected_overall_severities=("low", "medium"),
        roadmap={
            "title": "Explain a Technical System with Abstraction Levels",
            "steps": [
                _step(1, "Identify the audience", "Separate executive, functional, and technical readers and their decisions.", ["audience"]),
                _step(2, "Create the executive summary", "Summarize business impact, risks, and decisions for executives without implementation detail.", ["business impact"]),
                _step(3, "Create the functional view", "Describe user flows, responsibilities, and major process stages.", ["functional flow"]),
                _step(4, "Create the technical view", "Describe components, dependencies, interfaces, and operational constraints.", ["components"]),
                _step(5, "Prepare the business impact summary", "Create another executive-facing summary of the same business impact, risks, and decisions without implementation detail.", ["executive summary"]),
                _step(6, "Validate with stakeholders", "Ask each audience whether the level of detail supports their decisions.", ["feedback"]),
            ],
        },
    ),
    StepMetricCase(
        name="realistic_strong_dax_diagnostic_overlap",
        expected_level="medium_overlap",
        question="How do I optimize slow DAX measures?",
        category="dax_power_bi",
        expected_score_min=5.0,
        expected_score_max=8.5,
        expected_overlap_min=2,
        expected_severities=("medium", "high"),
        expected_overall_severities=("medium", "high"),
        roadmap={
            "title": "Optimize Slow DAX Measures with Diagnostic Tools",
            "steps": [
                _step(1, "Identify slow visuals", "Use Performance Analyzer to locate visuals with slow DAX query execution.", ["Performance Analyzer", "diagnosis"]),
                _step(2, "Identify slow measures", "Use the Performance Analyzer output to identify the DAX measures causing those slow visuals.", ["Performance Analyzer", "diagnosis"]),
                _step(3, "Profile slow measures in DAX Studio", "Open the same measures in DAX Studio and measure their execution time.", ["DAX Studio", "timing"]),
                _step(4, "Inspect server timings", "Use DAX Studio Server Timings to confirm which of the same measures are slow.", ["DAX Studio", "server timings"]),
                _step(5, "Review formula bottlenecks", "Review the slow measures again to identify expensive filters, iterators, or repeated calculations.", ["formula review"]),
                _step(6, "Rewrite expensive formulas", "Refactor the slow measures by reducing unnecessary filters, iterators, and repeated calculations.", ["optimization"]),
            ],
        },
    ),
    StepMetricCase(
        name="realistic_strong_n8n_extraction_mapping_overlap",
        expected_level="medium_overlap",
        question="How do I load API data into Google Sheets with n8n?",
        category="n8n_automation",
        expected_score_min=5.0,
        expected_score_max=8.5,
        expected_overlap_min=2,
        expected_severities=("medium", "high"),
        expected_overall_severities=("medium", "high"),
        roadmap={
            "title": "Load API Data into Google Sheets with n8n",
            "steps": [
                _step(1, "Configure the API request", "Set the endpoint, method, headers, and authentication in the HTTP Request node.", ["HTTP Request"]),
                _step(2, "Fetch API records", "Use the HTTP Request node to fetch records from the configured API endpoint.", ["HTTP Request"]),
                _step(3, "Retrieve updated records", "Call the same API endpoint again to retrieve the records that will be sent to Google Sheets.", ["HTTP Request"]),
                _step(4, "Normalize the API response", "Convert the API response fields into a consistent row-based structure.", ["mapping"]),
                _step(5, "Map fields for Google Sheets", "Map the same API response fields into the row structure required by Google Sheets.", ["mapping"]),
                _step(6, "Prepare rows for Sheets", "Format the mapped API response into rows ready to append to Google Sheets.", ["mapping"]),
            ],
        },
    ),
]


CONTEXT = [
    "Roadmap steps should be mutually exclusive when they teach or produce different objectives, actions, outputs, or decisions.",
    "Repeated domain vocabulary is acceptable when the actual learning work is different.",
    "Duplicate or weakly differentiated steps should be flagged as overlap only when they repeat the same task, output, learning objective, or responsibility.",
]


def _case_status(case: StepMetricCase, result: dict[str, Any]) -> dict[str, Any]:
    distinctness = result["step_distinctness"]
    overlap = result["step_overlap"]
    score = float(distinctness["score"])
    issue_count = int(overlap["issue_count"])
    severities = {
        str(pair.get("severity", "")).lower()
        for pair in overlap.get("overlapping_pairs", [])
        if pair.get("severity")
    }
    overall_severity = str(overlap.get("overall_severity", "none")).lower()

    score_ok = case.expected_score_min <= score <= case.expected_score_max
    min_ok = issue_count >= case.expected_overlap_min
    max_ok = True if case.expected_overlap_max is None else issue_count <= case.expected_overlap_max
    severity_ok = True
    if case.expected_severities and issue_count > 0:
        severity_ok = bool(severities & set(case.expected_severities))
    overall_severity_ok = True
    if case.expected_overall_severities:
        overall_severity_ok = overall_severity in set(case.expected_overall_severities)

    return {
        "score_ok": score_ok,
        "overlap_min_ok": min_ok,
        "overlap_max_ok": max_ok,
        "severity_ok": severity_ok,
        "overall_severity_ok": overall_severity_ok,
        "passed": score_ok and min_ok and max_ok and severity_ok and overall_severity_ok,
        "observed_score": score,
        "observed_issue_count": issue_count,
        "observed_severities": sorted(severities),
        "observed_overall_severity": overall_severity,
        "expected_score_range": [case.expected_score_min, case.expected_score_max],
        "expected_overlap_min": case.expected_overlap_min,
        "expected_overlap_max": case.expected_overlap_max,
        "expected_severities": list(case.expected_severities),
        "expected_overall_severities": list(case.expected_overall_severities),
    }


def _run_cases() -> dict[str, Any]:
    judge = StepSemanticJudge()
    results = []

    for case in CASES:
        result = judge.evaluate(
            case.roadmap,
            question=case.question,
            refined_question=case.question,
            contexts=CONTEXT,
            category=case.category,
        )
        validation = _case_status(case, result)
        results.append({
            "name": case.name,
            "expected_level": case.expected_level,
            "question": case.question,
            "category": case.category,
            "roadmap": case.roadmap,
            "result": result,
            "validation": validation,
        })

    passed = sum(1 for item in results if item["validation"]["passed"])
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": active_model_name(),
        "case_count": len(results),
        "passed_count": passed,
        "failed_count": len(results) - passed,
        "pass_rate": round(passed / len(results), 4),
        "context": CONTEXT,
        "results": results,
    }


def _score_color(score: float) -> str:
    if score >= 8:
        return "#52c41a"
    if score >= 5:
        return "#fa8c16"
    return "#f5222d"


def _status_badge(passed: bool) -> str:
    color = "#52c41a" if passed else "#f5222d"
    text = "EXPECTED RESULT" if passed else "NEEDS REVIEW"
    return f'<span class="badge" style="background:{color}">{text}</span>'


def _metric_verdict_label(verdict: str) -> str:
    labels = {
        "PASS": "Meets metric",
        "NEEDS_REVIEW": "Needs metric review",
        "FAIL": "Metric failed",
    }
    return labels.get(verdict, verdict or "N/A")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_html(payload: dict[str, Any], path: Path) -> None:
    rows = []
    detail_sections = []

    for idx, item in enumerate(payload["results"], 1):
        result = item["result"]
        validation = item["validation"]
        distinctness = result["step_distinctness"]
        overlap = result["step_overlap"]
        score = float(distinctness["score"])
        issue_count = int(overlap["issue_count"])
        rows.append(f"""
        <tr>
          <td>{idx}</td>
          <td><strong>{html.escape(item["name"])}</strong><br><span>{html.escape(item["expected_level"])}</span></td>
          <td>{html.escape(item["category"])}</td>
          <td style="color:{_score_color(score)}"><strong>{score:.1f}/10</strong></td>
          <td>{distinctness.get("weak_step_count", 0)}</td>
          <td>{'Yes' if overlap.get("evaluated") else 'No'}</td>
          <td>{html.escape(str(overlap.get("overall_severity", "none")))}</td>
          <td>{issue_count}</td>
          <td>{html.escape(', '.join(validation.get("observed_severities", [])) or '—')}</td>
          <td>{_status_badge(bool(validation["passed"]))}</td>
        </tr>
        """)

        steps = "\n".join(
            f"<li><strong>{html.escape(step.get('label', ''))}</strong>: {html.escape(step.get('description', ''))}</li>"
            for step in item["roadmap"].get("steps", [])
        )
        weak_steps = "\n".join(
            f"<li>Step {ws.get('step')}: <strong>{html.escape(str(ws.get('label', '')))}</strong> — {html.escape(str(ws.get('reason', '')))}</li>"
            for ws in distinctness.get("weak_steps", [])
        ) or "<li>No weak steps reported.</li>"
        pairs = "\n".join(
            f"<li>Steps {html.escape(', '.join(str(s) for s in pair.get('steps', [])))} "
            f"<strong>{html.escape(str(pair.get('severity', '')))}</strong> / "
            f"{html.escape(str(pair.get('overlap_type', '')))} — "
            f"{html.escape(str(pair.get('explanation', '')))} "
            f"<em>{html.escape(str(pair.get('recommendation', '')))}</em></li>"
            for pair in overlap.get("overlapping_pairs", [])
        ) or "<li>No overlapping pairs reported.</li>"

        detail_sections.append(f"""
        <details>
          <summary>{idx}. {html.escape(item["name"])} — {score:.1f}/10, {issue_count} overlap issues</summary>
          <div class="detail-grid">
            <div>
              <h3>Roadmap</h3>
              <ol>{steps}</ol>
            </div>
            <div>
              <h3>Step Distinctness</h3>
              <p><strong>Metric Verdict:</strong> {html.escape(_metric_verdict_label(str(distinctness.get("verdict", ""))))}</p>
              <p><strong>Reason:</strong> {html.escape(str(distinctness.get("reason", "")))}</p>
              <ul>{weak_steps}</ul>
              <h3>Step Overlap</h3>
              <p><strong>Evaluated:</strong> {'Yes' if overlap.get("evaluated") else 'No'} | <strong>Trigger:</strong> {html.escape(str(overlap.get("trigger", "")))} | <strong>Overall Severity:</strong> {html.escape(str(overlap.get("overall_severity", "none")))}</p>
              <ul>{pairs}</ul>
            </div>
          </div>
        </details>
        """)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Step Metrics Controlled Validation</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#f5f5f5; color:#222; padding:24px; }}
    h1 {{ margin-bottom:4px; }}
    .meta {{ color:#666; margin-bottom:20px; }}
    .cards {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }}
    .card {{ background:#fff; border-radius:8px; padding:16px 20px; box-shadow:0 2px 8px rgba(0,0,0,.08); min-width:150px; }}
    .card strong {{ display:block; font-size:26px; margin-bottom:4px; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,.08); margin-bottom:24px; }}
    th, td {{ padding:10px 12px; border-bottom:1px solid #eee; text-align:left; vertical-align:top; font-size:13px; }}
    th {{ background:#fafafa; }}
    .badge {{ color:white; padding:2px 8px; border-radius:12px; font-size:12px; }}
    details {{ background:#fff; margin-bottom:12px; padding:12px 16px; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,.06); }}
    summary {{ cursor:pointer; font-weight:600; }}
    .detail-grid {{ display:grid; grid-template-columns: minmax(280px, 1fr) minmax(280px, 1fr); gap:24px; margin-top:12px; }}
    li {{ margin-bottom:6px; }}
  </style>
</head>
<body>
  <h1>Step Metrics Controlled Validation</h1>
  <div class="meta">Generated: {html.escape(payload["generated_at"])} | Model: {html.escape(payload["model"])}</div>

  <div class="cards">
    <div class="card"><strong>{payload["case_count"]}</strong>Cases</div>
    <div class="card"><strong style="color:#52c41a">{payload["passed_count"]}</strong>Expected Results</div>
    <div class="card"><strong style="color:#f5222d">{payload["failed_count"]}</strong>Needs Review</div>
    <div class="card"><strong>{payload["pass_rate"]:.0%}</strong>Expected Match Rate</div>
  </div>

  <h2>Overview</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Case</th>
        <th>Category</th>
        <th>Distinctness</th>
        <th>Weak Steps</th>
        <th>Overlap Evaluated</th>
        <th>Overall Severity</th>
        <th>Overlap Issues</th>
        <th>Severities</th>
        <th>Expected Check</th>
      </tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>

  <h2>Case Details</h2>
  {''.join(detail_sections)}
</body>
</html>
"""
    path.write_text(html_doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate semantic step metrics with controlled roadmaps.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(config.reports_dir, "step_metric_validation"),
        help="Directory where JSON and HTML results will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _run_cases()
    json_path = output_dir / "step_metric_validation.json"
    html_path = output_dir / "step_metric_validation.html"
    _write_json(payload, json_path)
    _write_html(payload, html_path)

    print("Step metric controlled validation complete")
    print(f"  cases: {payload['case_count']}")
    print(f"  passed: {payload['passed_count']}")
    print(f"  failed/review: {payload['failed_count']}")
    print(f"  json: {json_path}")
    print(f"  html: {html_path}")


if __name__ == "__main__":
    main()
