"""Controlled validation for the dedicated Logical Order metric."""

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
from evaluation.metrics.logical_order_judge import LogicalOrderJudge
from src.llm_provider import active_model_name


@dataclass(frozen=True)
class LogicalOrderCase:
    name: str
    expected_level: str
    question: str
    category: str
    roadmap: dict[str, Any]
    expected_score_min: float
    expected_score_max: float


def _step(idx: int, label: str, description: str) -> dict[str, Any]:
    return {
        "id": f"step_{idx}",
        "label": label,
        "description": description,
        "type": "inicio" if idx == 1 else "proceso",
        "key_points": [],
    }


CASES: list[LogicalOrderCase] = [
    LogicalOrderCase(
        name="good_dax_dependency_order",
        expected_level="valid_sequence",
        question="How do I build and validate DAX time-intelligence measures?",
        category="dax_power_bi",
        expected_score_min=8.0,
        expected_score_max=10.0,
        roadmap={
            "title": "Build DAX Time Intelligence",
            "steps": [
                _step(1, "Define the comparison goal", "Identify the periods and business measures that must be compared."),
                _step(2, "Create the calendar table", "Create a continuous date table and connect it to the sales model."),
                _step(3, "Create base measures", "Define the base sales measures used by later calculations."),
                _step(4, "Add time-intelligence measures", "Build period comparison measures from the calendar and base measures."),
                _step(5, "Build comparison visuals", "Display the completed measures in visuals for inspection."),
                _step(6, "Validate known periods", "Compare visual results against trusted values after the measures exist."),
            ],
        },
    ),
    LogicalOrderCase(
        name="good_n8n_workflow_order",
        expected_level="valid_sequence",
        question="How do I load API data into Google Sheets with n8n?",
        category="n8n_automation",
        expected_score_min=8.0,
        expected_score_max=10.0,
        roadmap={
            "title": "Load API Data with n8n",
            "steps": [
                _step(1, "Define source and destination", "Identify the API endpoint and destination spreadsheet."),
                _step(2, "Configure credentials", "Configure API and Google Sheets credentials before making requests."),
                _step(3, "Retrieve API records", "Run the authenticated request and retrieve the source records."),
                _step(4, "Transform the response", "Map the retrieved payload into the destination column structure."),
                _step(5, "Write rows to Sheets", "Append the transformed records to the spreadsheet."),
                _step(6, "Test and monitor execution", "Validate the completed workflow and configure failure notifications."),
            ],
        },
    ),
    LogicalOrderCase(
        name="medium_validation_before_dashboard",
        expected_level="isolated_validation_violation",
        question="How do I build an executive Power BI dashboard?",
        category="power_bi_reporting",
        expected_score_min=5.0,
        expected_score_max=7.9,
        roadmap={
            "title": "Build an Executive Dashboard",
            "steps": [
                _step(1, "Define executive decisions", "Identify the decisions and KPIs the dashboard must support."),
                _step(2, "Validate the finished dashboard", "Test filters, totals, and stakeholder scenarios on the finished dashboard."),
                _step(3, "Prepare the data model", "Create the fact and dimension model required by the KPIs."),
                _step(4, "Create KPI measures", "Build the measures required by the executive views."),
                _step(5, "Design dashboard visuals", "Create the dashboard that will later be validated."),
                _step(6, "Publish the report", "Publish the validated report for the executive audience."),
            ],
        },
    ),
    LogicalOrderCase(
        name="medium_credentials_after_request",
        expected_level="isolated_prerequisite_violation",
        question="How do I connect n8n to a protected REST API?",
        category="n8n_automation",
        expected_score_min=5.0,
        expected_score_max=7.9,
        roadmap={
            "title": "Connect n8n to a Protected API",
            "steps": [
                _step(1, "Define the API operation", "Identify the endpoint, method, and expected response."),
                _step(2, "Send the authenticated request", "Call the protected endpoint using the required credential."),
                _step(3, "Configure the API credential", "Create the credential required by the previous request step."),
                _step(4, "Parse the response", "Transform the returned payload after the request succeeds."),
                _step(5, "Store the records", "Write the parsed records to the target system."),
                _step(6, "Validate the workflow", "Test the completed integration and its error path."),
            ],
        },
    ),
    LogicalOrderCase(
        name="poor_reversed_deployment_workflow",
        expected_level="multiple_core_violations",
        question="How do I build, test, and deploy a small data service?",
        category="data_engineering",
        expected_score_min=0.0,
        expected_score_max=4.9,
        roadmap={
            "title": "Deploy a Data Service Backwards",
            "steps": [
                _step(1, "Deploy the production service", "Release the completed and tested service to production."),
                _step(2, "Run integration tests", "Test the built service against its configured dependencies."),
                _step(3, "Implement the service", "Build the API and processing logic required by the tests."),
                _step(4, "Configure the environment", "Create the runtime configuration required by implementation and testing."),
                _step(5, "Define requirements", "Specify service inputs, outputs, and acceptance criteria."),
                _step(6, "Review deployment results", "Review the production deployment performed at the beginning."),
            ],
        },
    ),
    LogicalOrderCase(
        name="poor_practice_before_foundations",
        expected_level="multiple_learning_violations",
        question="How do I learn DAX context transition?",
        category="dax_power_bi",
        expected_score_min=0.0,
        expected_score_max=4.9,
        roadmap={
            "title": "Learn Context Transition in Reverse",
            "steps": [
                _step(1, "Optimize complex context-transition measures", "Refactor advanced measures that depend on context-transition knowledge."),
                _step(2, "Build an advanced calculation", "Apply context transition in a complex business calculation."),
                _step(3, "Practice a guided example", "Use context transition in a controlled example."),
                _step(4, "Learn row and filter context", "Establish the concepts required to understand context transition."),
                _step(5, "Introduce CALCULATE", "Explain how CALCULATE causes context transition."),
                _step(6, "Validate conceptual understanding", "Check understanding after the advanced work has already been attempted."),
            ],
        },
    ),
    LogicalOrderCase(
        name="orthogonal_incomplete_but_ordered",
        expected_level="valid_relative_order",
        question="How do I implement a production API ingestion workflow?",
        category="data_engineering",
        expected_score_min=8.0,
        expected_score_max=10.0,
        roadmap={
            "title": "Incomplete but Ordered API Workflow",
            "steps": [
                _step(1, "Define the endpoint", "Identify the API endpoint and required response."),
                _step(2, "Configure authentication", "Create authentication before calling the endpoint."),
                _step(3, "Fetch one response", "Call the endpoint after authentication is available."),
                _step(4, "Inspect the response", "Inspect the response after it has been retrieved."),
            ],
        },
    ),
    LogicalOrderCase(
        name="orthogonal_vague_but_ordered",
        expected_level="valid_relative_order",
        question="How do I create and test a small automation?",
        category="automation",
        expected_score_min=8.0,
        expected_score_max=10.0,
        roadmap={
            "title": "Vague but Ordered Automation",
            "steps": [
                _step(1, "Plan the automation", "Plan what is needed."),
                _step(2, "Set up the environment", "Set things up."),
                _step(3, "Build the automation", "Build the main thing."),
                _step(4, "Test the automation", "Test what was built."),
                _step(5, "Release the automation", "Release it after testing."),
            ],
        },
    ),
]

SMOKE_CASE_NAMES = {
    "good_dax_dependency_order",
    "medium_credentials_after_request",
    "poor_reversed_deployment_workflow",
    "orthogonal_vague_but_ordered",
}


def _case_status(case: LogicalOrderCase, result: dict[str, Any]) -> dict[str, Any]:
    logical_order = result["logical_order"]
    score = float(logical_order["score"])
    score_ok = case.expected_score_min <= score <= case.expected_score_max
    return {
        "score_ok": score_ok,
        "passed": score_ok,
        "observed_score": score,
        "expected_score_range": [case.expected_score_min, case.expected_score_max],
    }


def _run_cases(cases: list[LogicalOrderCase], suite: str) -> dict[str, Any]:
    judge = LogicalOrderJudge()
    results = []
    started_at = time.perf_counter()
    for idx, case in enumerate(cases, 1):
        print(f"  [{idx}/{len(cases)}] {case.name}", flush=True)
        case_started_at = time.perf_counter()
        result = judge.evaluate(
            case.roadmap,
            question=case.question,
            category=case.category,
        )
        duration_seconds = round(time.perf_counter() - case_started_at, 2)
        validation = _case_status(case, result)
        results.append({
            "name": case.name,
            "expected_level": case.expected_level,
            "question": case.question,
            "category": case.category,
            "roadmap": case.roadmap,
            "logical_order": result["logical_order"],
            "validation": validation,
            "duration_seconds": duration_seconds,
        })
    passed = sum(1 for item in results if item["validation"]["passed"])
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": active_model_name(),
        "suite": suite,
        "logical_order_reasoning": "model_default",
        "duration_seconds": round(time.perf_counter() - started_at, 2),
        "case_count": len(results),
        "passed_count": passed,
        "failed_count": len(results) - passed,
        "pass_rate": round(passed / len(results), 4),
        "results": results,
    }


def _score_color(score: float) -> str:
    if score >= 8.0:
        return "#52c41a"
    if score >= 5.0:
        return "#fa8c16"
    return "#f5222d"


def _score_status_badge(passed: bool) -> str:
    color = "#52c41a" if passed else "#f5222d"
    text = "EXPECTED SCORE" if passed else "UNEXPECTED SCORE"
    return f'<span class="badge" style="background:{color}">{text}</span>'


def _metric_result_badge(score: float) -> str:
    if score >= 8.0:
        color, text = "#52c41a", "PASS"
    elif score >= 5.0:
        color, text = "#fa8c16", "NEEDS REVIEW"
    else:
        color, text = "#f5222d", "FAIL"
    return f'<span class="badge" style="background:{color}">{text}</span>'


def _display_name(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_html(payload: dict[str, Any], path: Path) -> None:
    rows = []
    details = []
    for idx, item in enumerate(payload["results"], 1):
        result = item["logical_order"]
        validation = item["validation"]
        score = float(result["score"])
        expected_range = validation["expected_score_range"]
        rows.append(f"""
        <tr>
          <td>{idx}</td>
          <td><strong>{html.escape(item['name'])}</strong><br><span>{html.escape(item['expected_level'])}</span></td>
          <td>{html.escape(item['category'])}</td>
          <td style="color:{_score_color(score)}"><strong>{score:.1f}/10</strong></td>
          <td>{_metric_result_badge(score)}</td>
          <td>{expected_range[0]:.1f}-{expected_range[1]:.1f}</td>
          <td>{item['duration_seconds']:.1f}s</td>
          <td>{_score_status_badge(bool(validation['passed']))}</td>
        </tr>
        """)

        steps = "".join(
            f"<li><code>{html.escape(str(step.get('id', '')))}</code> "
            f"<strong>{html.escape(str(step.get('label', '')))}</strong>: "
            f"{html.escape(str(step.get('description', '')))}</li>"
            for step in item["roadmap"].get("steps", [])
        )
        rationales = result.get("criteria_rationales", {})
        criteria_rows = "".join(
            f"<tr><td><strong>{html.escape(_display_name(name))}</strong></td>"
            f"<td class='score-cell' style='color:{_score_color(float(value))}'><strong>{float(value):.1f}/10</strong></td>"
            f"<td>{html.escape(str(rationales.get(name, '')))}</td></tr>"
            for name, value in result.get("criteria", {}).items()
        )
        violations = "".join(
            f"<tr><td><code>{html.escape(str(v.get('dependent_step_id', '')))}</code></td>"
            f"<td><code>{html.escape(str(v.get('required_predecessor_step_id', '')))}</code></td>"
            f"<td>{html.escape(_display_name(str(v.get('dependency_type', ''))))}</td>"
            f"<td>{html.escape(str(v.get('severity', '')).upper())}</td>"
            f"<td>{html.escape(str(v.get('explanation', '')))}<br>"
            f"<em>{html.escape(str(v.get('suggested_fix', '')))}</em></td></tr>"
            for v in result.get("dependency_violations", [])
        ) or "<tr><td colspan='5' class='empty-state'>No dependency violations reported.</td></tr>"
        recommendations = "".join(
            f"<li>{html.escape(str(value))}</li>"
            for value in result.get("recommendations", [])
        ) or "<li class='empty-state'>No recommendations reported.</li>"
        suggested_order = " → ".join(
            f"<code>{html.escape(str(step_id))}</code>"
            for step_id in result.get("suggested_order", [])
        )
        details.append(f"""
        <details class="case-detail">
          <summary>{idx}. {html.escape(item['name'])} — logical order {score:.1f}/10</summary>
          <div class="detail-grid">
            <div>
              <h3>Roadmap</h3>
              <p class="question"><strong>Question:</strong> {html.escape(item['question'])}</p>
              <ol>{steps}</ol>
            </div>
            <div>
              <div class="metric-heading">
                <div><h3>Logical Order</h3><span class="score-band">Band: {result['score_band']}</span></div>
                <strong class="metric-score" style="color:{_score_color(score)}">{score:.1f}<small>/10</small></strong>
              </div>
              <section class="assessment-block">
                <h4>Overall Assessment</h4>
                <p>{html.escape(str(result.get('reason', '')))}</p>
              </section>
              <section>
                <h4>Criteria Assessment</h4>
                <table class="criteria-table">
                  <thead><tr><th>Criterion</th><th>Score</th><th>Evaluation</th></tr></thead>
                  <tbody>{criteria_rows}</tbody>
                </table>
              </section>
              <section>
                <h4>Dependency Violations <span class="count-badge">{result['dependency_violation_count']}</span></h4>
                <table class="violations-table">
                  <thead><tr><th>Dependent</th><th>Requires</th><th>Type</th><th>Severity</th><th>Explanation / Fix</th></tr></thead>
                  <tbody>{violations}</tbody>
                </table>
              </section>
              <section class="order-block">
                <h4>Suggested Order</h4>
                <p>{suggested_order}</p>
              </section>
              <section class="recommendations-block">
                <h4>Recommendations</h4>
                <ul>{recommendations}</ul>
              </section>
            </div>
          </div>
        </details>
        """)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Logical Order Controlled Validation</title>
  <style>
    body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f5f5f5; color:#222; padding:24px; }}
    h1 {{ margin-bottom:4px; }}
    h3, h4 {{ margin-top:0; }}
    .meta {{ color:#666; margin-bottom:20px; }}
    .cards {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }}
    .card {{ background:#fff; border-radius:8px; padding:16px 20px; box-shadow:0 2px 8px rgba(0,0,0,.08); min-width:150px; }}
    .card strong {{ display:block; font-size:26px; margin-bottom:4px; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,.08); margin-bottom:24px; }}
    th, td {{ padding:10px 12px; border-bottom:1px solid #eee; text-align:left; vertical-align:top; font-size:13px; }}
    th {{ background:#fafafa; }}
    .badge {{ color:#fff; padding:2px 8px; border-radius:12px; font-size:12px; }}
    details.case-detail {{ background:#fff; margin-bottom:12px; padding:12px 16px; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,.06); }}
    details.case-detail > summary {{ cursor:pointer; font-weight:600; }}
    .detail-grid {{ display:grid; grid-template-columns:minmax(300px,.85fr) minmax(560px,1.5fr); gap:28px; margin-top:18px; }}
    li {{ margin-bottom:7px; }}
    .question {{ color:#555; line-height:1.45; }}
    .metric-heading {{ display:flex; align-items:flex-start; justify-content:space-between; gap:16px; padding-bottom:14px; border-bottom:1px solid #eee; }}
    .metric-heading h3 {{ margin-bottom:4px; }}
    .score-band {{ color:#666; font-size:12px; }}
    .metric-score {{ font-size:30px; line-height:1; white-space:nowrap; }}
    .metric-score small {{ font-size:14px; color:#777; }}
    .assessment-block {{ background:#f7f9fc; border-left:4px solid #1677ff; border-radius:4px; padding:12px 14px; margin:16px 0 20px; }}
    .assessment-block h4 {{ margin-bottom:6px; }}
    .assessment-block p {{ margin:0; line-height:1.45; }}
    .criteria-table, .violations-table {{ box-shadow:none; border:1px solid #eee; margin:8px 0 20px; }}
    .criteria-table th:nth-child(1) {{ width:155px; }}
    .criteria-table th:nth-child(2) {{ width:70px; text-align:right; }}
    .score-cell {{ text-align:right; white-space:nowrap; }}
    .count-badge {{ background:#f0f2f5; border-radius:10px; padding:2px 7px; font-size:11px; color:#555; }}
    .empty-state {{ color:#777; font-style:italic; }}
    .order-block, .recommendations-block {{ border:1px solid #eee; border-radius:7px; padding:14px; margin-bottom:14px; }}
    .order-block p, .recommendations-block ul {{ margin-bottom:0; }}
    code {{ background:#f0f2f5; border-radius:3px; padding:1px 4px; }}
    @media (max-width:1100px) {{ .detail-grid {{ grid-template-columns:1fr; }} }}
    @media (max-width:700px) {{ body {{ padding:12px; }} .criteria-table,.violations-table {{ display:block; overflow-x:auto; }} }}
  </style>
</head>
<body>
  <h1>Logical Order Controlled Validation</h1>
  <div class="meta">Generated: {html.escape(payload['generated_at'])} | Model: {html.escape(payload['model'])} | Suite: {html.escape(payload['suite'])} | Logical Order reasoning: {html.escape(payload['logical_order_reasoning'])}</div>
  <div class="cards">
    <div class="card"><strong>{payload['case_count']}</strong>Total Cases</div>
    <div class="card"><strong style="color:#52c41a">{payload['passed_count']}</strong>Expected Scores</div>
    <div class="card"><strong style="color:#f5222d">{payload['failed_count']}</strong>Unexpected Scores</div>
    <div class="card"><strong>{payload['pass_rate']:.0%}</strong>Score Match Rate</div>
    <div class="card"><strong>{payload['duration_seconds'] / 60:.1f}</strong>Total Minutes</div>
  </div>
  <h2>Logical Order Overview</h2>
  <table>
    <thead><tr><th>#</th><th>Case</th><th>Category</th><th>Score</th><th>Metric Result</th><th>Expected Range</th><th>Duration</th><th>Score Check</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Case Details</h2>
  {''.join(details)}
</body>
</html>
"""
    path.write_text(html_doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Logical Order with controlled roadmaps.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(config.reports_dir, "logical_order_metric_validation"),
        help="Directory where JSON and HTML results will be written.",
    )
    parser.add_argument(
        "--suite",
        choices=("smoke", "full"),
        default="smoke",
        help="Run four representative cases by default, or all eight cases.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_names",
        help="Run only a named case. Repeat the option to select multiple cases.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.case_names:
        requested_names = set(args.case_names)
        unknown_names = requested_names - {case.name for case in CASES}
        if unknown_names:
            parser.error(f"Unknown case names: {', '.join(sorted(unknown_names))}")
        selected_cases = [case for case in CASES if case.name in requested_names]
        suite_name = "custom"
    elif args.suite == "full":
        selected_cases = CASES
        suite_name = "full"
    else:
        selected_cases = [case for case in CASES if case.name in SMOKE_CASE_NAMES]
        suite_name = "smoke"

    payload = _run_cases(selected_cases, suite_name)
    report_stem = f"logical_order_metric_validation_{suite_name}"
    json_path = output_dir / f"{report_stem}.json"
    html_path = output_dir / f"{report_stem}.html"
    _write_json(payload, json_path)
    _write_html(payload, html_path)

    print("Logical Order controlled validation complete")
    print(f"  cases: {payload['case_count']}")
    print(f"  expected scores: {payload['passed_count']}")
    print(f"  unexpected scores: {payload['failed_count']}")
    print(f"  duration: {payload['duration_seconds']:.1f}s")
    print(f"  json: {json_path}")
    print(f"  html: {html_path}")


if __name__ == "__main__":
    main()
