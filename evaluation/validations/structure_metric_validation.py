"""Controlled validation for schema validity and structure quality metrics."""

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
from evaluation.metrics.structure_quality_judge import StructureQualityJudge
from evaluation.metrics.structure_validator import StructureValidator
from src.llm_provider import active_model_name


@dataclass(frozen=True)
class StructureMetricCase:
    name: str
    expected_level: str
    question: str
    category: str
    roadmap: dict[str, Any]
    expected_quality_min: float
    expected_quality_max: float
    expected_schema_min: float = 7.0


@dataclass(frozen=True)
class SchemaValidityCase:
    name: str
    expected_level: str
    roadmap: dict[str, Any]
    expected_schema_min: float
    expected_schema_max: float


def _step(
    idx: int,
    label: str,
    description: str,
    points: list[str],
    step_type: str = "proceso",
) -> dict[str, Any]:
    return {
        "id": f"step_{idx}",
        "label": label,
        "description": description,
        "type": step_type,
        "key_points": points,
    }


CASES: list[StructureMetricCase] = [
    StructureMetricCase(
        name="good_dax_learning_structure",
        expected_level="solid_structure",
        question="How do I learn Time Intelligence in DAX?",
        category="dax_power_bi",
        expected_quality_min=8.0,
        expected_quality_max=10.0,
        roadmap={
            "title": "Learn Time Intelligence in DAX",
            "steps": [
                _step(1, "Frame the time analysis goal", "Define the business question and the time comparisons the learner needs. This creates a clear target before writing any DAX.", ["business question", "comparison goal", "expected output"], "inicio"),
                _step(2, "Create the calendar foundation", "Build and mark a calendar table with continuous dates. This gives the roadmap a stable structural base for time calculations.", ["date table", "mark as date", "relationships"]),
                _step(3, "Define baseline measures", "Create base sales measures before adding time-intelligence logic. This separates the foundation from the time comparison layer.", ["total sales", "base measure", "validation"]),
                _step(4, "Add time comparison measures", "Create YoY and MoM measures using appropriate DAX functions. This is the central technical layer of the roadmap.", ["YoY", "MoM", "DAX functions"]),
                _step(5, "Validate comparisons visually", "Place the measures into visuals and compare known periods. This checks whether the structure leads to interpretable results.", ["visual check", "known periods", "sanity test"]),
                _step(6, "Document and apply the pattern", "Summarize the reusable pattern and apply it to a second metric. This gives the roadmap clear closure and transfer value.", ["pattern summary", "second metric", "reuse"], "fin"),
            ],
        },
    ),
    StructureMetricCase(
        name="good_n8n_workflow_structure",
        expected_level="solid_structure",
        question="How do I automate API data loading with n8n?",
        category="n8n_automation",
        expected_quality_min=8.0,
        expected_quality_max=10.0,
        roadmap={
            "title": "Automate API Data Loading with n8n",
            "steps": [
                _step(1, "Frame the automation goal", "Define the source API, target sheet, schedule, and success criteria. This gives the workflow a clear operational purpose.", ["source", "target", "success criteria"], "inicio"),
                _step(2, "Configure secure access", "Set up credentials for the API and Google Sheets. This prepares the required connections before the workflow starts moving data.", ["credentials", "API access", "Sheets access"]),
                _step(3, "Extract API data", "Use an HTTP Request node to retrieve the source records. This step creates the raw input for the rest of the workflow.", ["HTTP Request", "records", "payload"]),
                _step(4, "Transform records", "Map and normalize fields into the structure expected by the destination sheet. This creates a clean handoff between extraction and loading.", ["mapping", "normalization", "rows"]),
                _step(5, "Load rows into Sheets", "Append or update the prepared rows in Google Sheets. This completes the main automation path.", ["append", "update", "destination"]),
                _step(6, "Validate and monitor the workflow", "Run a controlled test and add error notifications. This closes the roadmap with operational validation.", ["test run", "errors", "monitoring"], "fin"),
            ],
        },
    ),
    StructureMetricCase(
        name="medium_weak_closure",
        expected_level="needs_review",
        question="How do I build an executive Power BI dashboard?",
        category="power_bi_reporting",
        expected_quality_min=5.0,
        expected_quality_max=7.9,
        roadmap={
            "title": "Build an Executive Power BI Dashboard",
            "steps": [
                _step(1, "Frame executive decisions", "Identify the audience and the decisions the dashboard must support. This gives the report a useful starting point.", ["audience", "decisions", "KPIs"], "inicio"),
                _step(2, "Select financial KPIs", "Choose metrics such as revenue, margin, cost, and budget variance. This defines the dashboard's core signal.", ["revenue", "margin", "variance"]),
                _step(3, "Prepare the data model", "Create the fact and dimension tables needed by the KPIs. This gives the dashboard a coherent modeling base.", ["facts", "dimensions", "relationships"]),
                _step(4, "Design executive visuals", "Place KPI cards, trends, and variance visuals in a simple layout. This organizes the information for decision makers.", ["cards", "trends", "layout"]),
                _step(5, "Add filters and interactions", "Add slicers and interactions for period, region, and department. This gives users controlled exploration paths.", ["slicers", "period", "region"]),
                _step(6, "Finish the dashboard", "Review the dashboard and make final adjustments. This closes the roadmap, but the completion target is generic.", ["review", "adjust", "finish"], "fin"),
            ],
        },
    ),
    StructureMetricCase(
        name="medium_uneven_granularity",
        expected_level="needs_review",
        question="How do I document a data process at multiple abstraction levels?",
        category="abstraction_levels",
        expected_quality_min=5.0,
        expected_quality_max=7.9,
        roadmap={
            "title": "Document a Data Process with Abstraction Levels",
            "steps": [
                _step(1, "Frame the documentation purpose", "Define who will read the documentation and what decisions it should support. This gives the roadmap a useful starting frame.", ["audience", "purpose", "decisions"], "inicio"),
                _step(2, "Write the entire business context", "Document business objectives, process boundaries, stakeholders, data ownership, risks, governance rules, and success criteria in one large step. This step is useful but too broad compared with the others.", ["business", "ownership", "governance"]),
                _step(3, "List source systems", "Identify the source systems that feed the data process. This is much narrower than the previous step.", ["sources", "systems", "input"]),
                _step(4, "Draw the process flow", "Create a functional flow that shows movement from source to transformation to output. This provides the central structure.", ["flow", "transformation", "output"]),
                _step(5, "Describe operational checks", "Add refresh checks, ownership notes, and failure handling. This gives the document operational value.", ["refresh", "owner", "failure"]),
                _step(6, "Validate document levels", "Review whether each audience can read the process at the right level of detail. This gives the roadmap a meaningful closure.", ["review", "audience", "detail"], "fin"),
            ],
        },
    ),
    StructureMetricCase(
        name="poor_fragmented_checklist",
        expected_level="not_usable",
        question="How do I learn DAX context transition?",
        category="dax_power_bi",
        expected_quality_min=0.0,
        expected_quality_max=4.9,
        roadmap={
            "title": "DAX Context Transition Notes",
            "steps": [
                _step(1, "Open Power BI", "Open the Power BI Desktop application and locate the report file. This starts the activity but does not frame the learning goal.", ["Power BI", "file", "open"], "inicio"),
                _step(2, "Read about CALCULATE", "Read a short note about CALCULATE in DAX. This item is disconnected from a larger learning structure.", ["CALCULATE", "read", "note"]),
                _step(3, "Look at a table", "Look at a table that contains product and sales columns. This is a loose observation rather than a structured learning stage.", ["table", "columns", "sales"]),
                _step(4, "Try SUMX", "Try writing a SUMX expression somewhere in the model. This jumps into implementation without framing or scaffolding.", ["SUMX", "expression", "model"]),
                _step(5, "Check a visual", "Check whether a visual changes after the formula is added. This is useful but not connected to a clear validation plan.", ["visual", "change", "formula"]),
                _step(6, "Stop working", "Stop the exercise and save the file. This ends the roadmap without synthesis, validation, or transfer.", ["save", "stop", "file"], "fin"),
            ],
        },
    ),
    StructureMetricCase(
        name="poor_no_framing_no_closure",
        expected_level="not_usable",
        question="How do I connect n8n with a database and another system?",
        category="n8n_automation",
        expected_quality_min=0.0,
        expected_quality_max=4.9,
        roadmap={
            "title": "n8n Database Work",
            "steps": [
                _step(1, "Create a query", "Write a SQL query against the database without first defining the workflow objective. This starts in the middle of the process.", ["SQL", "query", "database"], "inicio"),
                _step(2, "Send data somewhere", "Send the query result to another system without specifying the target contract. This keeps the roadmap structurally vague.", ["send", "target", "system"]),
                _step(3, "Change some fields", "Modify fields after sending data instead of defining a transformation stage. This creates an unclear structural shape.", ["fields", "modify", "mapping"]),
                _step(4, "Add credentials later", "Add database credentials after the query and transfer steps. This makes the roadmap feel structurally incoherent.", ["credentials", "database", "access"]),
                _step(5, "Maybe add error handling", "Consider adding error handling if there is time. This makes an important structural component optional and vague.", ["errors", "optional", "handling"]),
                _step(6, "Continue improving", "Continue improving the workflow in the future. This does not provide meaningful closure or validation.", ["future", "improve", "continue"], "fin"),
            ],
        },
    ),
]


SCHEMA_CASES: list[SchemaValidityCase] = [
    SchemaValidityCase(
        name="schema_valid_contract",
        expected_level="valid_schema",
        expected_schema_min=9.5,
        expected_schema_max=10.0,
        roadmap={
            "title": "Valid Roadmap Contract",
            "steps": [
                _step(1, "Frame the objective", "Define the roadmap objective and expected result. This gives the structure a clear start.", ["objective", "scope", "result"], "inicio"),
                _step(2, "Create the foundation", "Build the initial foundation required by the process. This keeps the contract clear and complete.", ["foundation", "setup", "base"]),
                _step(3, "Validate the inputs", "Review the inputs before continuing with the workflow. This prevents structural gaps in later steps.", ["inputs", "review", "quality"]),
                _step(4, "Configure the process", "Set the main process configuration and required options. This creates a valid middle section.", ["configuration", "options", "process"]),
                _step(5, "Document the output", "Record the expected output and relevant notes. This gives reviewers enough structured information.", ["output", "notes", "review"]),
                _step(6, "Finalize the validation", "Confirm that the roadmap contract is complete and usable. This provides a clear final node.", ["confirm", "complete", "usable"], "fin"),
            ],
        },
    ),
    SchemaValidityCase(
        name="schema_missing_start_and_end_nodes",
        expected_level="invalid_start_end_contract",
        expected_schema_min=7.0,
        expected_schema_max=7.5,
        roadmap={
            "title": "Missing Start and End Nodes",
            "steps": [
                _step(1, "Define the objective", "Define the objective and expected result for the workflow. This gives useful context.", ["objective", "scope", "result"]),
                _step(2, "Create the foundation", "Create the foundation required by the process. This keeps the roadmap usable.", ["foundation", "setup", "base"]),
                _step(3, "Validate the inputs", "Validate the inputs before continuing to later stages. This reduces structural risk.", ["inputs", "review", "quality"]),
                _step(4, "Configure the process", "Configure the central process and its required options. This gives the roadmap a middle section.", ["configuration", "options", "process"]),
                _step(5, "Document the output", "Document the output and relevant implementation notes. This supports later review.", ["output", "notes", "review"]),
                _step(6, "Finalize the validation", "Finalize the roadmap by confirming the expected result. This gives semantic closure but not a fin node.", ["confirm", "complete", "usable"]),
            ],
        },
    ),
    SchemaValidityCase(
        name="schema_duplicate_ids_and_bad_types",
        expected_level="invalid_identity_and_type_contract",
        expected_schema_min=7.5,
        expected_schema_max=8.0,
        roadmap={
            "title": "Duplicate IDs and Invalid Types",
            "steps": [
                _step(1, "Frame the objective", "Define the objective and expected result for the workflow. This gives useful context.", ["objective", "scope", "result"], "inicio"),
                {**_step(2, "Create the foundation", "Create the foundation required by the process. This keeps the roadmap usable.", ["foundation", "setup", "base"]), "id": "step_1"},
                _step(3, "Validate the inputs", "Validate the inputs before continuing to later stages. This reduces structural risk.", ["inputs", "review", "quality"], "invalid_type"),
                _step(4, "Configure the process", "Configure the central process and its required options. This gives the roadmap a middle section.", ["configuration", "options", "process"]),
                _step(5, "Document the output", "Document the output and relevant implementation notes. This supports later review.", ["output", "notes", "review"]),
                _step(6, "Finalize the validation", "Finalize the roadmap by confirming the expected result. This gives the roadmap a valid final node.", ["confirm", "complete", "usable"], "fin"),
            ],
        },
    ),
    SchemaValidityCase(
        name="schema_invalid_key_points_contract",
        expected_level="invalid_key_points_contract",
        expected_schema_min=8.5,
        expected_schema_max=9.0,
        roadmap={
            "title": "Invalid Key Points Contract",
            "steps": [
                _step(1, "Frame objective", "Define the objective. This gives the roadmap a clear start.", ["objective"], "inicio"),
                {"id": "step_2", "label": "Create foundation", "description": "Create the foundation. This keeps the roadmap usable.", "type": "proceso", "key_points": "foundation"},
                {"id": "step_3", "label": "Validate inputs", "description": "Validate the inputs. This reduces structural risk.", "type": "proceso"},
                _step(4, "Configure process", "Configure the process. This gives the roadmap a middle section.", []),
                _step(5, "Document output", "Document the output. This supports later review.", ["output"]),
                _step(6, "Finalize validation", "Finalize the roadmap. This gives the roadmap a valid final node.", ["confirm"], "fin"),
            ],
        },
    ),
    SchemaValidityCase(
        name="schema_missing_title_and_empty_fields",
        expected_level="invalid_required_fields_contract",
        expected_schema_min=7.5,
        expected_schema_max=8.0,
        roadmap={
            "title": "",
            "steps": [
                _step(1, "Frame the objective", "Define the objective and expected result for the workflow. This gives useful context.", ["objective", "scope", "result"], "inicio"),
                {"id": "", "label": "Create the foundation", "description": "Create the foundation required by the process. This keeps the roadmap usable.", "type": "proceso", "key_points": ["foundation", "setup", "base"]},
                {"id": "step_3", "label": "", "description": "Validate the inputs before continuing to later stages. This reduces structural risk.", "type": "proceso", "key_points": ["inputs", "review", "quality"]},
                {"id": "step_4", "label": "Configure the process", "description": "", "type": "proceso", "key_points": ["configuration", "options", "process"]},
                _step(5, "Document the output", "Document the output and relevant implementation notes. This supports later review.", ["output", "notes", "review"]),
                _step(6, "Finalize the validation", "Finalize the roadmap by confirming the expected result. This gives the roadmap a valid final node.", ["confirm", "complete", "usable"], "fin"),
            ],
        },
    ),
]


def _case_status(case: StructureMetricCase, schema: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    quality_score = float(quality["structure_quality"]["score"])
    schema_score = float(schema.get("score", 0))
    quality_ok = case.expected_quality_min <= quality_score <= case.expected_quality_max
    schema_ok = schema_score >= case.expected_schema_min
    return {
        "passed": quality_ok and schema_ok,
        "quality_score_ok": quality_ok,
        "schema_score_ok": schema_ok,
        "observed_quality_score": quality_score,
        "observed_schema_score": schema_score,
        "expected_quality_range": [case.expected_quality_min, case.expected_quality_max],
        "expected_schema_min": case.expected_schema_min,
    }


def _schema_case_status(case: SchemaValidityCase, schema: dict[str, Any]) -> dict[str, Any]:
    schema_score = float(schema.get("score", 0))
    schema_ok = case.expected_schema_min <= schema_score <= case.expected_schema_max
    return {
        "passed": schema_ok,
        "schema_score_ok": schema_ok,
        "observed_schema_score": schema_score,
        "expected_schema_range": [case.expected_schema_min, case.expected_schema_max],
    }


def _run_cases() -> dict[str, Any]:
    schema_validator = StructureValidator()
    quality_judge = StructureQualityJudge()
    results = []

    for case in CASES:
        schema = schema_validator.validate(case.roadmap)
        quality = quality_judge.evaluate(
            case.roadmap,
            question=case.question,
            category=case.category,
        )
        validation = _case_status(case, schema, quality)
        results.append({
            "name": case.name,
            "expected_level": case.expected_level,
            "question": case.question,
            "category": case.category,
            "roadmap": case.roadmap,
            "schema_validity": schema,
            "structure_quality": quality["structure_quality"],
            "validation": validation,
        })

    schema_results = []
    for case in SCHEMA_CASES:
        schema = schema_validator.validate(case.roadmap)
        validation = _schema_case_status(case, schema)
        schema_results.append({
            "name": case.name,
            "expected_level": case.expected_level,
            "roadmap": case.roadmap,
            "schema_validity": schema,
            "validation": validation,
        })

    passed = sum(1 for item in results if item["validation"]["passed"])
    schema_passed = sum(1 for item in schema_results if item["validation"]["passed"])
    total_cases = len(results) + len(schema_results)
    total_passed = passed + schema_passed
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": active_model_name(),
        "case_count": total_cases,
        "quality_case_count": len(results),
        "schema_case_count": len(schema_results),
        "passed_count": total_passed,
        "quality_passed_count": passed,
        "schema_passed_count": schema_passed,
        "failed_count": total_cases - total_passed,
        "pass_rate": round(total_passed / total_cases, 4),
        "results": results,
        "schema_results": schema_results,
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


def _display_name(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _severity_badge(severity: str) -> str:
    normalized = severity.lower()
    colors = {"low": "#1677ff", "medium": "#fa8c16", "high": "#f5222d"}
    color = colors.get(normalized, "#8c8c8c")
    return f'<span class="severity-badge" style="background:{color}">{html.escape(normalized.upper())}</span>'


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_html(payload: dict[str, Any], path: Path) -> None:
    rows = []
    schema_rows = []
    details = []

    for idx, item in enumerate(payload["results"], 1):
        quality = item["structure_quality"]
        schema = item["schema_validity"]
        validation = item["validation"]
        score = float(quality["score"])
        rows.append(f"""
        <tr>
          <td>{idx}</td>
          <td><strong>{html.escape(item["name"])}</strong><br><span>{html.escape(item["expected_level"])}</span></td>
          <td>{html.escape(item["category"])}</td>
          <td style="color:{_score_color(score)}"><strong>{score:.1f}/10</strong></td>
          <td>{html.escape(str(quality.get("score_band", "")))}</td>
          <td>{quality.get("issue_count", 0)}</td>
          <td>{schema.get("score", 0):.1f}/10</td>
          <td>{_status_badge(bool(validation["passed"]))}</td>
        </tr>
        """)

        criteria_rationales = (
            quality.get("criteria_rationales")
            if isinstance(quality.get("criteria_rationales"), dict)
            else {}
        )
        scope_fit = quality.get("scope_fit", {}) if isinstance(quality.get("scope_fit"), dict) else {}
        criteria_rows = []
        for name, value in quality.get("criteria", {}).items():
            rationale = criteria_rationales.get(name) or "No case-specific evaluation available."
            scope_context = ""
            if name == "scope_fit" and scope_fit:
                rationale = scope_fit.get("reason") or rationale
                scope_context = f"""
                <div class="scope-stats">
                  <span><strong>Ideal:</strong> {html.escape(str(scope_fit.get("ideal_step_count", "")))} steps</span>
                  <span><strong>Actual:</strong> {html.escape(str(scope_fit.get("actual_step_count", "")))} steps</span>
                  <span><strong>Difference:</strong> {html.escape(str(scope_fit.get("difference", "")))}</span>
                </div>
                """
            criteria_rows.append(f"""
            <tr>
              <td><strong>{html.escape(_display_name(name))}</strong></td>
              <td class="dimension-score" style="color:{_score_color(float(value))}"><strong>{value:.1f}/10</strong></td>
              <td>{html.escape(str(rationale))}{scope_context}</td>
            </tr>
            """)
        issue_items = []
        for issue in quality.get("issues", []):
            issue_type = str(issue.get("type", "issue"))
            explanation = str(issue.get("explanation", ""))
            if issue_type == "poor_scope_fit" and scope_fit.get("reason"):
                explanation = str(scope_fit["reason"])
            issue_items.append(
                f"<li>{_severity_badge(str(issue.get('severity', 'medium')))} "
                f"<div><strong>{html.escape(_display_name(issue_type))}</strong>"
                f"<p>{html.escape(explanation)}</p></div></li>"
            )
        issues = "".join(issue_items) or '<li class="empty-state">No structure quality issues reported.</li>'
        recommendations = "".join(
            f"<li>{html.escape(str(rec))}</li>"
            for rec in quality.get("recommendations", [])
        ) or '<li class="empty-state">No recommendations reported.</li>'
        steps = "".join(
            f"<li><strong>{html.escape(str(step.get('label', '')))}</strong>: {html.escape(str(step.get('description', '')))}</li>"
            for step in item["roadmap"].get("steps", [])
        )
        details.append(f"""
        <details class="case-detail">
          <summary>{idx}. {html.escape(item["name"])} — structure quality {score:.1f}/10</summary>
          <div class="detail-grid">
            <div>
              <h3>Roadmap</h3>
              <ol>{steps}</ol>
            </div>
            <div class="quality-column">
              <div class="quality-heading">
                <div>
                  <h3>Structure Quality</h3>
                  <span class="score-band">Band: {html.escape(str(quality.get("score_band", "")))}</span>
                </div>
                <strong class="quality-score" style="color:{_score_color(score)}">{score:.1f}<small>/10</small></strong>
              </div>

              <section class="assessment-block">
                <h4>Overall Assessment</h4>
                <p>{html.escape(str(quality.get("reason", "")))}</p>
              </section>

              <section>
                <h4>Dimension Assessment</h4>
                <table class="dimension-table">
                  <thead><tr><th>Dimension</th><th>Score</th><th>Evaluation</th></tr></thead>
                  <tbody>{''.join(criteria_rows)}</tbody>
                </table>
              </section>

              <div class="findings-grid">
                <section class="findings-block issues-block">
                  <h4>Issues <span class="count-badge">{quality.get("issue_count", 0)}</span></h4>
                  <ul class="issue-list">{issues}</ul>
                </section>
                <section class="findings-block recommendations-block">
                  <h4>Recommendations</h4>
                  <ul>{recommendations}</ul>
                </section>
              </div>
            </div>
          </div>
        </details>
        """)

    for idx, item in enumerate(payload.get("schema_results", []), 1):
        schema = item["schema_validity"]
        validation = item["validation"]
        score = float(schema.get("score", 0))
        violations = schema.get("violations", [])
        violation_items = "".join(
            f"<li>{html.escape(str(violation))}</li>"
            for violation in violations
        ) or "<li>No schema violations reported.</li>"
        expected_range = validation.get("expected_schema_range", [])
        expected_text = (
            f"{expected_range[0]:.1f}-{expected_range[1]:.1f}"
            if len(expected_range) == 2
            else ""
        )
        schema_rows.append(f"""
        <tr>
          <td>{idx}</td>
          <td><strong>{html.escape(item["name"])}</strong><br><span>{html.escape(item["expected_level"])}</span></td>
          <td style="color:{_score_color(score)}"><strong>{score:.1f}/10</strong></td>
          <td>{expected_text}</td>
          <td>{schema.get("passed", 0)}/{schema.get("total_checks", 0)}</td>
          <td><ul>{violation_items}</ul></td>
          <td>{_status_badge(bool(validation["passed"]))}</td>
        </tr>
        """)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Structure Metrics Controlled Validation</title>
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
    details.case-detail {{ background:#fff; margin-bottom:12px; padding:12px 16px; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,.06); }}
    details.case-detail > summary {{ cursor:pointer; font-weight:600; }}
    .detail-grid {{ display:grid; grid-template-columns: minmax(280px, .85fr) minmax(520px, 1.4fr); gap:28px; margin-top:18px; }}
    li {{ margin-bottom:6px; }}
    .quality-column h3, .quality-column h4 {{ margin-top:0; }}
    .quality-heading {{ display:flex; align-items:flex-start; justify-content:space-between; gap:16px; padding-bottom:14px; border-bottom:1px solid #eee; }}
    .quality-heading h3 {{ margin-bottom:4px; }}
    .score-band {{ color:#666; font-size:12px; }}
    .quality-score {{ font-size:30px; line-height:1; white-space:nowrap; }}
    .quality-score small {{ font-size:14px; color:#777; }}
    .assessment-block {{ background:#f7f9fc; border-left:4px solid #1677ff; border-radius:4px; padding:12px 14px; margin:16px 0 20px; }}
    .assessment-block h4 {{ margin-bottom:6px; }}
    .assessment-block p, .secondary-note, .issue-list p {{ margin:0; line-height:1.45; }}
    .dimension-table {{ box-shadow:none; border:1px solid #eee; margin:8px 0 20px; }}
    .dimension-table th:nth-child(1) {{ width:145px; }}
    .dimension-table th:nth-child(2) {{ width:72px; text-align:right; }}
    .dimension-score {{ text-align:right; white-space:nowrap; }}
    .scope-stats {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:9px; }}
    .scope-stats span {{ background:#f0f2f5; border-radius:12px; padding:3px 8px; font-size:12px; }}
    .secondary-note {{ color:#666; margin-top:8px; }}
    .findings-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
    .findings-block {{ border:1px solid #eee; border-radius:7px; padding:14px; }}
    .findings-block h4 {{ margin-bottom:10px; }}
    .findings-block ul {{ margin:0; padding-left:20px; }}
    .issue-list {{ list-style:none; padding-left:0 !important; }}
    .issue-list li:not(.empty-state) {{ display:flex; align-items:flex-start; gap:8px; }}
    .severity-badge {{ color:#fff; border-radius:10px; padding:2px 7px; font-size:10px; font-weight:700; flex:none; margin-top:1px; }}
    .count-badge {{ background:#f0f2f5; border-radius:10px; padding:2px 7px; font-size:11px; color:#555; }}
    .empty-state {{ color:#777; font-style:italic; }}
    @media (max-width: 1050px) {{
      .detail-grid {{ grid-template-columns:1fr; }}
    }}
    @media (max-width: 700px) {{
      body {{ padding:12px; }}
      .findings-grid {{ grid-template-columns:1fr; }}
      .dimension-table {{ display:block; overflow-x:auto; }}
    }}
  </style>
</head>
<body>
  <h1>Structure Metrics Controlled Validation</h1>
  <div class="meta">Generated: {html.escape(payload["generated_at"])} | Model: {html.escape(payload["model"])}</div>
  <div class="cards">
    <div class="card"><strong>{payload["case_count"]}</strong>Total Cases</div>
    <div class="card"><strong>{payload["quality_case_count"]}</strong>Quality Cases</div>
    <div class="card"><strong>{payload["schema_case_count"]}</strong>Schema Cases</div>
    <div class="card"><strong style="color:#52c41a">{payload["passed_count"]}</strong>Expected Results</div>
    <div class="card"><strong style="color:#f5222d">{payload["failed_count"]}</strong>Needs Review</div>
    <div class="card"><strong>{payload["pass_rate"]:.0%}</strong>Expected Match Rate</div>
  </div>
  <h2>Structure Quality Overview</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Case</th>
        <th>Category</th>
        <th>Structure Quality</th>
        <th>Band</th>
        <th>Issues</th>
        <th>Schema Validity</th>
        <th>Expected Check</th>
      </tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Schema Validity Overview</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Case</th>
        <th>Schema Validity</th>
        <th>Expected Range</th>
        <th>Checks Passed</th>
        <th>Violations</th>
        <th>Expected Check</th>
      </tr>
    </thead>
    <tbody>{''.join(schema_rows)}</tbody>
  </table>
  <h2>Case Details</h2>
  {''.join(details)}
</body>
</html>
"""
    path.write_text(html_doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate structure metrics with controlled roadmaps.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(config.reports_dir, "structure_metric_validation"),
        help="Directory where JSON and HTML results will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _run_cases()
    json_path = output_dir / "structure_metric_validation.json"
    html_path = output_dir / "structure_metric_validation.html"
    _write_json(payload, json_path)
    _write_html(payload, html_path)

    print("Structure metric controlled validation complete")
    print(f"  cases: {payload['case_count']}")
    print(f"  quality cases: {payload['quality_case_count']} ({payload['quality_passed_count']} expected)")
    print(f"  schema cases: {payload['schema_case_count']} ({payload['schema_passed_count']} expected)")
    print(f"  passed: {payload['passed_count']}")
    print(f"  failed/review: {payload['failed_count']}")
    print(f"  json: {json_path}")
    print(f"  html: {html_path}")


if __name__ == "__main__":
    main()
