"""Controlled validation for the dedicated Actionability metric."""

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
from evaluation.metrics.actionability_judge import ActionabilityJudge
from src.llm_provider import active_model_name


@dataclass(frozen=True)
class ActionabilityCase:
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


CASES: list[ActionabilityCase] = [
    ActionabilityCase(
        name="good_dax_executable_steps",
        expected_level="actionable",
        question="How do I build a year-over-year sales measure in DAX?",
        category="dax_power_bi",
        expected_score_min=8.0,
        expected_score_max=10.0,
        roadmap={
            "title": "Build a Year-over-Year DAX Measure",
            "steps": [
                _step(1, "Define the comparison", "Create a comparison-requirements note with four fields: sales measure, current period, prior period, and percentage-change output required by the report."),
                _step(2, "Create the calendar table", "Create a continuous Date table covering the sales range, mark it as the model date table, and relate Date[Date] to Sales[OrderDate]."),
                _step(3, "Create the base measure", "Create [Total Sales] with SUM(Sales[SalesAmount]) and verify it returns the expected total for one known month."),
                _step(4, "Create prior-year sales", "Create [Sales PY] with CALCULATE([Total Sales], SAMEPERIODLASTYEAR(Date[Date])) so the measure returns the comparable prior-year value."),
                _step(5, "Create year-over-year change", "Create [Sales YoY %] with DIVIDE([Total Sales] - [Sales PY], [Sales PY]) and format the result as a percentage."),
                _step(6, "Validate the measures", "Place Date[Month], [Total Sales], [Sales PY], and [Sales YoY %] in a table and compare two known periods with trusted source values."),
            ],
        },
    ),
    ActionabilityCase(
        name="good_n8n_concise_workflow",
        expected_level="actionable",
        question="How do I copy API records into Google Sheets with n8n?",
        category="n8n_automation",
        expected_score_min=8.0,
        expected_score_max=10.0,
        roadmap={
            "title": "Copy API Records to Google Sheets",
            "steps": [
                _step(1, "Configure the API request", "Add an HTTP Request node, set Method to GET, enter the records endpoint, and attach the API credential."),
                _step(2, "Map the response fields", "Add an Edit Fields node and map each API field to the destination spreadsheet column name."),
                _step(3, "Append the rows", "Add a Google Sheets node, select Append Row, choose the target spreadsheet and sheet, and map the prepared fields."),
                _step(4, "Test one execution", "Run the workflow manually with a small response and confirm that the expected row values appear in the target sheet."),
            ],
        },
    ),
    ActionabilityCase(
        name="medium_concrete_labels_vague_details",
        expected_level="local_clarification",
        question="How do I create an n8n workflow for customer synchronization?",
        category="n8n_automation",
        expected_score_min=5.0,
        expected_score_max=7.9,
        roadmap={
            "title": "Synchronize Customers with n8n",
            "steps": [
                _step(1, "Choose the customer source", "Select the source that seems appropriate."),
                _step(2, "Configure API credentials", "Create an n8n credential with the customer API key and test the credential from the node settings."),
                _step(3, "Retrieve customer records", "Use an HTTP Request node with the configured credential to fetch one page of customer JSON."),
                _step(4, "Transform customer records", "Prepare the records for the next part of the workflow."),
                _step(5, "Write customers to the CRM", "Use the CRM node to map customer ID, name, and email, then upsert each prepared record."),
                _step(6, "Validate the synchronization", "Check that everything works correctly."),
            ],
        },
    ),
    ActionabilityCase(
        name="medium_missing_results",
        expected_level="local_clarification",
        question="How do I clean a sales table in Power Query?",
        category="power_bi",
        expected_score_min=5.0,
        expected_score_max=7.9,
        roadmap={
            "title": "Clean Sales Data in Power Query",
            "steps": [
                _step(1, "Load the sales query", "Open Power Query and select the Sales query from the Queries pane."),
                _step(2, "Correct column types", "Use the Data Type menu to assign Date, Whole Number, and Currency types to the relevant columns."),
                _step(3, "Handle missing values", "Use Replace Values and Remove Rows on columns with null values."),
                _step(4, "Standardize product names", "Apply Trim, Clean, and capitalization transformations to ProductName."),
                _step(5, "Review the transformed table", "Inspect the transformed data."),
                _step(6, "Finish the cleanup", "Complete the process when the table looks acceptable."),
            ],
        },
    ),
    ActionabilityCase(
        name="poor_abstract_power_bi_roadmap",
        expected_level="not_actionable",
        question="How do I improve a Power BI report?",
        category="power_bi",
        expected_score_min=0.0,
        expected_score_max=4.9,
        roadmap={
            "title": "Improve a Power BI Report",
            "steps": [
                _step(1, "Understand the report", "Learn what is important."),
                _step(2, "Explore the data", "Review the data carefully."),
                _step(3, "Improve the model", "Apply suitable improvements."),
                _step(4, "Enhance the visuals", "Use visualization best practices."),
                _step(5, "Optimize performance", "Make the report perform better."),
                _step(6, "Finalize the solution", "Complete the remaining work."),
            ],
        },
    ),
    ActionabilityCase(
        name="poor_non_executable_approval_outline",
        expected_level="not_actionable",
        question="How do I automate an approval process in n8n?",
        category="n8n_automation",
        expected_score_min=0.0,
        expected_score_max=4.9,
        roadmap={
            "title": "Approval Automation Outline",
            "steps": [
                _step(1, "Preparation", "Consider the relevant aspects of the approval process."),
                _step(2, "Request handling", "Work with incoming items as needed."),
                _step(3, "Evaluation", "Review each situation in an appropriate way."),
                _step(4, "Decision handling", "Continue according to the result."),
                _step(5, "Communication", "Keep the relevant people informed."),
                _step(6, "Monitoring", "Observe progress and improve things over time."),
            ],
        },
    ),
    ActionabilityCase(
        name="orthogonal_misordered_but_actionable",
        expected_level="actionable_despite_order",
        question="How do I build and publish a Power BI dashboard?",
        category="power_bi",
        expected_score_min=8.0,
        expected_score_max=10.0,
        roadmap={
            "title": "Misordered but Executable Dashboard Steps",
            "steps": [
                _step(1, "Publish the report", "In Power BI Desktop, select Publish, choose the Sales workspace, and confirm that the report appears there."),
                _step(2, "Validate dashboard totals", "Add a table with Sales[OrderID] and [Total Sales], then compare its totals with the approved monthly source file."),
                _step(3, "Create the sales measures", "Create [Total Sales] with SUM(Sales[Amount]) and [Order Count] with DISTINCTCOUNT(Sales[OrderID])."),
                _step(4, "Build the data model", "Relate Sales[ProductID] to Product[ProductID] and Sales[DateKey] to Date[DateKey], then confirm both relationships are active."),
                _step(5, "Load the source tables", "Use Get Data to load Sales, Product, and Date from the specified SQL connection into the model."),
            ],
        },
    ),
    ActionabilityCase(
        name="orthogonal_incomplete_inaccurate_but_actionable",
        expected_level="actionable_despite_other_quality",
        question="How do I create a complete DAX time-intelligence solution?",
        category="dax_power_bi",
        expected_score_min=8.0,
        expected_score_max=10.0,
        roadmap={
            "title": "Clear but Incomplete DAX Instructions",
            "steps": [
                _step(1, "Create the sales total", "Create [Total Sales] with AVERAGE(Sales[SalesAmount]) and format the result as Currency."),
                _step(2, "Create the prior-period measure", "Create [Sales PY] with CALCULATE([Total Sales], DATEADD(Date[Date], -1, YEAR)) and add it to a table visual."),
                _step(3, "Display the comparison", "Place Date[Year], [Total Sales], and [Sales PY] in a matrix and confirm that each year displays both values."),
            ],
        },
    ),
]


SMOKE_CASE_NAMES = {
    "good_dax_executable_steps",
    "medium_concrete_labels_vague_details",
    "poor_abstract_power_bi_roadmap",
    "orthogonal_misordered_but_actionable",
}


def _case_status(
    case: ActionabilityCase,
    result: dict[str, Any],
) -> dict[str, Any]:
    actionability = result["actionability"]
    score = float(actionability["score"])
    score_ok = case.expected_score_min <= score <= case.expected_score_max
    return {
        "score_ok": score_ok,
        "passed": score_ok,
        "observed_score": score,
        "expected_score_range": [
            case.expected_score_min,
            case.expected_score_max,
        ],
    }


def _run_cases(
    cases: list[ActionabilityCase],
    suite: str,
) -> dict[str, Any]:
    judge = ActionabilityJudge()
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
            "actionability": result["actionability"],
            "validation": validation,
            "duration_seconds": duration_seconds,
        })
    passed = sum(1 for item in results if item["validation"]["passed"])
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": active_model_name(),
        "suite": suite,
        "actionability_reasoning": "model_default",
        "ollama_seed": os.getenv("OLLAMA_SEED", "unset"),
        "duration_seconds": round(time.perf_counter() - started_at, 2),
        "case_count": len(results),
        "passed_count": passed,
        "failed_count": len(results) - passed,
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "results": results,
    }


def _score_class(score: float) -> str:
    if score >= 8.0:
        return "pass"
    if score >= 5.0:
        return "review"
    return "fail"


def _display_name(value: str) -> str:
    acronyms = {
        "bi": "BI",
        "dax": "DAX",
        "n8n": "n8n",
    }
    return " ".join(
        acronyms.get(word.lower(), word.capitalize())
        for word in value.replace("_", " ").split()
    )


def _metric_badge(verdict: str) -> str:
    normalized = verdict.strip().upper()
    labels = {
        "PASS": ("PASS", "#52c41a"),
        "NEEDS_REVIEW": ("NEEDS REVIEW", "#fa8c16"),
        "FAIL": ("FAIL", "#f5222d"),
        "ACTIONABLE": ("ACTIONABLE", "#52c41a"),
        "NEEDS_CLARIFICATION": ("NEEDS CLARIFICATION", "#fa8c16"),
        "NOT_ACTIONABLE": ("NOT ACTIONABLE", "#f5222d"),
    }
    label, color = labels.get(normalized, (normalized or "N/A", "#8c8c8c"))
    return f'<span class="badge" style="background:{color}">{html.escape(label)}</span>'


def _expected_badge(passed: bool) -> str:
    color = "#52c41a" if passed else "#f5222d"
    label = "EXPECTED RESULT" if passed else "NEEDS REVIEW"
    return f'<span class="badge" style="background:{color}">{label}</span>'


def _roadmap_steps_html(item: dict[str, Any]) -> str:
    return "".join(
        f"<li><strong>{html.escape(str(step.get('label', '')))}</strong>: "
        f"{html.escape(str(step.get('description', '')))}</li>"
        for step in item["roadmap"].get("steps", [])
    )


def _steps_html(item: dict[str, Any]) -> str:
    evaluations = {
        entry["step_id"]: entry
        for entry in item["actionability"]["step_actionability"]
    }
    rows = []
    for step in item["roadmap"]["steps"]:
        step_id = str(step.get("id", ""))
        evaluation = evaluations.get(step_id, {})
        score = float(evaluation.get("score", 0))
        missing = ", ".join(evaluation.get("missing_elements", [])) or "None"
        criteria = evaluation.get("criteria", {})
        optional = evaluation.get("optional_improvements", [])
        criteria_html = "".join(
            f"<li><strong>{html.escape(_display_name(name))}:</strong> "
            f"{html.escape(_display_name(str(value)))}</li>"
            for name, value in criteria.items()
        )
        details = f"""
        <details class="inline-detail">
          <summary>View criteria</summary>
          <ul>{criteria_html}</ul>
          <p><strong>Optional improvements:</strong> {html.escape(", ".join(optional) or "None")}</p>
          <p><strong>Recommendation:</strong> {html.escape(str(evaluation.get("recommendation", "")) or "None")}</p>
        </details>
        """
        rows.append(f"""
        <tr>
          <td><strong>{html.escape(str(step.get("label", "")))}</strong><br>
              <code>{html.escape(step_id)}</code><br>
              <span class="muted">{html.escape(str(step.get("description", "")))}</span></td>
          <td class="score-cell {_score_class(score)}">{score:.1f}/10</td>
          <td>{_metric_badge(str(evaluation.get("status", "")))}</td>
          <td>{html.escape(str(evaluation.get("reason", "")))}</td>
          <td><strong>Missing:</strong> {html.escape(missing)}{details}</td>
        </tr>
        """)
    return "".join(rows)


def _criteria_html(actionability: dict[str, Any]) -> str:
    rows = []
    criteria = actionability.get("criteria", {})
    rationales = actionability.get("criteria_rationales", {})
    for name, score in criteria.items():
        numeric = float(score)
        rows.append(f"""
        <tr>
          <td>{html.escape(_display_name(name))}</td>
          <td class="score-cell {_score_class(numeric)}">{numeric:.1f}/10</td>
          <td>{html.escape(str(rationales.get(name, "")))}</td>
        </tr>
        """)
    return "".join(rows)


def _case_details_html(item: dict[str, Any], idx: int) -> str:
    actionability = item["actionability"]
    validation = item["validation"]
    expected = validation["expected_score_range"]
    score = float(actionability["score"])
    notes = actionability.get("normalization_notes", [])
    notes_html = ""
    if notes:
        note_items = "".join(
            f"<li>{html.escape(str(note))}</li>"
            for note in notes
        )
        notes_html = f"""
        <details class="processing-notes">
          <summary>Processing notes</summary>
          <ul>{note_items}</ul>
        </details>
        """
    strengths = "".join(
        f"<li>{html.escape(str(value))}</li>"
        for value in actionability.get("strengths", [])
    ) or '<li class="empty-state">No strengths reported.</li>'
    recommendations = "".join(
        f"<li>{html.escape(str(value))}</li>"
        for value in actionability.get("recommendations", [])
    ) or '<li class="empty-state">No recommendations reported.</li>'
    manual_review = (
        '<span class="review">Required</span>'
        if actionability.get("manual_review_required")
        else "Not required"
    )
    return f"""
    <details class="case-detail" id="case-{idx}">
      <summary>{idx}. {html.escape(_display_name(item["name"]))} — actionability {score:.1f}/10</summary>
      <div class="detail-grid">
        <div class="roadmap-column">
          <h3>Roadmap</h3>
          <p class="question"><strong>Question:</strong> {html.escape(item["question"])}</p>
          <ol>{_roadmap_steps_html(item)}</ol>
        </div>
        <div class="metric-column">
          <div class="metric-heading">
            <div>
              <h3>Actionability</h3>
              <span class="score-band">Band: {html.escape(str(actionability.get("score_band", "")))}</span>
            </div>
            <strong class="metric-score {_score_class(score)}">{score:.1f}<small>/10</small></strong>
          </div>

          <section class="assessment-block">
            <h4>Overall Assessment</h4>
            <p>{html.escape(str(actionability.get("reason", "")))}</p>
          </section>

          <div class="stats">
            <span><strong>Expected:</strong> {expected[0]:.1f}-{expected[1]:.1f}</span>
            <span><strong>Density:</strong> {float(actionability.get("actionability_density", 0)):.0%}</span>
            <span><strong>Weak steps:</strong> {int(actionability.get("weak_action_step_count", 0))}</span>
            <span><strong>Manual review:</strong> {manual_review}</span>
            <span><strong>Score check:</strong> {_expected_badge(bool(validation["score_ok"]))}</span>
          </div>

          <section>
            <h4>Roadmap-level Criteria</h4>
            <table class="criteria-table">
              <thead><tr><th>Criterion</th><th>Score</th><th>Evaluation</th></tr></thead>
              <tbody>{_criteria_html(actionability)}</tbody>
            </table>
          </section>

          <div class="findings-grid">
            <section class="findings-block">
              <h4>Strengths</h4>
              <ul>{strengths}</ul>
            </section>
            <section class="findings-block">
              <h4>Recommendations</h4>
              <ul>{recommendations}</ul>
            </section>
          </div>
          {notes_html}
        </div>
      </div>

      <section class="step-diagnostics">
        <h3>Step Diagnostics</h3>
        <div class="table-scroll">
          <table class="step-table">
            <thead><tr><th>Step</th><th>Score</th><th>Metric Result</th><th>Evaluation</th><th>Gaps and Details</th></tr></thead>
            <tbody>{_steps_html(item)}</tbody>
          </table>
        </div>
      </section>
    </details>
    """


def _overview_rows(payload: dict[str, Any]) -> str:
    rows = []
    for idx, item in enumerate(payload["results"], 1):
        actionability = item["actionability"]
        validation = item["validation"]
        expected = validation["expected_score_range"]
        score = float(actionability["score"])
        rows.append(f"""
        <tr>
          <td>{idx}</td>
          <td><a class="case-link" href="#case-{idx}"><strong>{html.escape(_display_name(item["name"]))}</strong></a><br>
              <span class="muted">{html.escape(_display_name(item["expected_level"]))}</span></td>
          <td>{html.escape(_display_name(item["category"]))}</td>
          <td class="score-cell {_score_class(score)}">{score:.1f}/10</td>
          <td>{_metric_badge(str(actionability["verdict"]))}</td>
          <td>{expected[0]:.1f}-{expected[1]:.1f}</td>
          <td>{float(actionability.get("actionability_density", 0)):.0%}</td>
          <td>{int(actionability.get("weak_action_step_count", 0))}</td>
          <td>{item["duration_seconds"]:.1f}s</td>
          <td>{_expected_badge(bool(validation["score_ok"]))}</td>
        </tr>
        """)
    return "".join(rows)


def _save_reports(payload: dict[str, Any], output_dir: str, stem: str) -> tuple[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{stem}.json"
    html_path = directory / f"{stem}.html"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    details = [
        _case_details_html(item, idx)
        for idx, item in enumerate(payload["results"], 1)
    ]

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Actionability Controlled Validation</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f5f5f5; color:#222; padding:24px; }}
h1 {{ margin-bottom:4px; }}
h3, h4 {{ margin-top:0; }}
.meta, .muted {{ color:#666; }}
.meta {{ margin-bottom:20px; line-height:1.5; }}
.cards {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }}
.card {{ background:#fff; border-radius:8px; padding:16px 20px; box-shadow:0 2px 8px rgba(0,0,0,.08); min-width:150px; }}
.card strong {{ display:block; font-size:26px; margin-bottom:4px; }}
table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,.08); margin-bottom:24px; }}
th, td {{ padding:10px 12px; border-bottom:1px solid #eee; text-align:left; vertical-align:top; font-size:13px; }}
th {{ background:#fafafa; }}
.badge {{ color:#fff; padding:2px 8px; border-radius:12px; font-size:12px; white-space:nowrap; }}
.pass {{ color:#52c41a; font-weight:700; }}
.review {{ color:#fa8c16; font-weight:700; }}
.fail {{ color:#f5222d; font-weight:700; }}
.score-cell {{ white-space:nowrap; font-weight:700; }}
.case-link {{ color:#222; text-decoration:none; }}
.case-link:hover {{ color:#1677ff; text-decoration:underline; }}
details.case-detail {{ background:#fff; margin-bottom:12px; padding:12px 16px; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,.06); }}
details.case-detail > summary {{ cursor:pointer; font-weight:600; }}
.detail-grid {{ display:grid; grid-template-columns:minmax(300px,.85fr) minmax(560px,1.5fr); gap:28px; margin-top:18px; }}
.roadmap-column li {{ margin-bottom:9px; line-height:1.4; }}
.question {{ color:#555; line-height:1.45; }}
.metric-heading {{ display:flex; align-items:flex-start; justify-content:space-between; gap:16px; padding-bottom:14px; border-bottom:1px solid #eee; }}
.metric-heading h3 {{ margin-bottom:4px; }}
.score-band {{ color:#666; font-size:12px; }}
.metric-score {{ font-size:30px; line-height:1; white-space:nowrap; }}
.metric-score small {{ font-size:14px; color:#777; }}
.assessment-block {{ background:#f7f9fc; border-left:4px solid #1677ff; border-radius:4px; padding:12px 14px; margin:16px 0; }}
.assessment-block h4 {{ margin-bottom:6px; }}
.assessment-block p {{ margin:0; line-height:1.45; }}
.stats {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:20px; }}
.stats > span {{ background:#f0f2f5; border-radius:12px; padding:4px 9px; font-size:12px; }}
.stats .badge {{ font-size:10px; }}
.criteria-table, .step-table {{ box-shadow:none; border:1px solid #eee; margin:8px 0 20px; }}
.criteria-table th:nth-child(1) {{ width:155px; }}
.criteria-table th:nth-child(2) {{ width:78px; text-align:right; }}
.criteria-table .score-cell {{ text-align:right; }}
.findings-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.findings-block {{ border:1px solid #eee; border-radius:7px; padding:14px; }}
.findings-block h4 {{ margin-bottom:10px; }}
.findings-block ul {{ margin:0; padding-left:20px; }}
.findings-block li {{ margin-bottom:6px; }}
.empty-state {{ color:#777; font-style:italic; }}
.processing-notes {{ margin-top:14px; border:1px solid #eee; border-radius:7px; padding:10px 12px; }}
.processing-notes > summary, .inline-detail > summary {{ cursor:pointer; font-weight:600; }}
.step-diagnostics {{ margin-top:24px; padding-top:20px; border-top:1px solid #eee; }}
.step-table th:nth-child(1) {{ min-width:280px; }}
.step-table th:nth-child(2) {{ width:75px; }}
.step-table th:nth-child(3) {{ width:115px; }}
.step-table th:nth-child(4) {{ min-width:260px; }}
.step-table th:nth-child(5) {{ min-width:220px; }}
.inline-detail {{ margin-top:8px; }}
.inline-detail ul {{ padding-left:18px; }}
code {{ background:#f0f2f5; border-radius:3px; padding:1px 4px; color:#555; }}
.table-scroll {{ overflow-x:auto; }}
@media (max-width:1100px) {{
  .detail-grid {{ grid-template-columns:1fr; }}
}}
@media (max-width:700px) {{
  body {{ padding:12px; }}
  .findings-grid {{ grid-template-columns:1fr; }}
  .overview-table {{ display:block; overflow-x:auto; }}
  .card {{ flex:1 1 130px; min-width:0; }}
}}
</style>
</head>
<body>
<h1>Actionability Controlled Validation</h1>
<p class="meta">Generated: {html.escape(payload["generated_at"])} |
Model: {html.escape(payload["model"])} |
Suite: {html.escape(payload["suite"])} |
Reasoning: {html.escape(payload["actionability_reasoning"])} |
Ollama seed: {html.escape(payload["ollama_seed"])}</p>
<div class="cards">
  <div class="card"><strong>{payload["case_count"]}</strong>Total Cases</div>
  <div class="card"><strong class="pass">{payload["passed_count"]}</strong>Expected Results</div>
  <div class="card"><strong class="fail">{payload["failed_count"]}</strong>Needs Review</div>
  <div class="card"><strong>{payload["pass_rate"]:.0%}</strong>Expected Match Rate</div>
  <div class="card"><strong>{payload["duration_seconds"] / 60:.1f}</strong>Total Minutes</div>
</div>
<h2>Actionability Overview</h2>
<table class="overview-table">
  <thead><tr><th>#</th><th>Case</th><th>Category</th><th>Score</th><th>Metric Result</th><th>Expected Range</th><th>Density</th><th>Weak Steps</th><th>Duration</th><th>Score Check</th></tr></thead>
  <tbody>{_overview_rows(payload)}</tbody>
</table>
<h2>Case Details</h2>
{"".join(details)}
</body>
</html>
"""
    html_path.write_text(page, encoding="utf-8")
    return str(json_path.resolve()), str(html_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run controlled validation for the Actionability metric."
    )
    parser.add_argument(
        "--suite",
        choices=("smoke", "full"),
        default="smoke",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run one named case. Repeat to run multiple cases.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(config.reports_dir, "actionability_metric_validation"),
    )
    args = parser.parse_args()

    requested_names = set(args.case)
    if requested_names:
        unknown_names = requested_names - {case.name for case in CASES}
        if unknown_names:
            raise SystemExit(f"Unknown Actionability cases: {', '.join(sorted(unknown_names))}")
        selected_cases = [case for case in CASES if case.name in requested_names]
        suite_name = "custom"
    elif args.suite == "full":
        selected_cases = CASES
        suite_name = "full"
    else:
        selected_cases = [case for case in CASES if case.name in SMOKE_CASE_NAMES]
        suite_name = "smoke"

    payload = _run_cases(selected_cases, suite_name)
    stem = f"actionability_metric_validation_{suite_name}"
    json_path, html_path = _save_reports(payload, args.output_dir, stem)
    print("Actionability controlled validation complete")
    print(f"  cases: {payload['case_count']}")
    print(f"  expected scores: {payload['passed_count']}")
    print(f"  unexpected scores: {payload['failed_count']}")
    print(f"  duration: {payload['duration_seconds']:.1f}s")
    print(f"  json: {json_path}")
    print(f"  html: {html_path}")


if __name__ == "__main__":
    main()
