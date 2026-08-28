"""Controlled validation for the dedicated Grounding metric."""

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
from evaluation.metrics.grounding_judge import GroundingJudge
from src.llm_provider import active_model_name


def _step(index: int, label: str, description: str) -> dict[str, Any]:
    return {
        "id": f"step_{index}",
        "label": label,
        "description": description,
        "type": "inicio" if index == 1 else "proceso",
        "key_points": [],
    }


def _context(index: int, text: str) -> dict[str, str]:
    return {"id": f"context_{index}", "text": text}


@dataclass(frozen=True)
class GroundingCase:
    name: str
    question: str
    category: str
    roadmap: dict[str, Any]
    contexts: list[dict[str, str]]
    score_range: tuple[float, float]
    claim_support_range: tuple[float, float]
    step_grounded_range: tuple[float, float]
    expected_problem_steps: frozenset[str]
    expected_contradiction_steps: frozenset[str] = frozenset()
    expected_problem_statuses: frozenset[str] = frozenset()


CASES = [
    GroundingCase(
        "good_fully_supported_dax",
        "How do I create a year-over-year sales measure in DAX?",
        "dax_power_bi",
        {"title": "Supported YoY", "steps": [
            _step(1, "Create the date table", "Create a continuous Date table, mark it as the date table, and relate it to Sales[OrderDate]."),
            _step(2, "Create total sales", "Create [Total Sales] with SUM(Sales[SalesAmount])."),
            _step(3, "Create prior-year sales", "Create [Sales PY] with CALCULATE([Total Sales], SAMEPERIODLASTYEAR(Date[Date]))."),
        ]},
        [
            _context(1, "Time-intelligence models require a continuous calendar table marked as the date table and related to the fact-table date column, such as Sales[OrderDate] in this model."),
            _context(2, "Define Total Sales as SUM(Sales[SalesAmount])."),
            _context(3, "Prior-year sales can be calculated with CALCULATE([Total Sales], SAMEPERIODLASTYEAR(Date[Date]))."),
        ],
        (8, 10), (100, 100), (100, 100), frozenset(),
    ),
    GroundingCase(
        "medium_one_invented_detail",
        "How do I create a sales measure in DAX?",
        "dax_power_bi",
        {"title": "One Unsupported Detail", "steps": [
            _step(1, "Create total sales", "Create [Total Sales] with SUM(Sales[SalesAmount])."),
            _step(2, "Format total sales", "Format [Total Sales] as Currency and enable the AutoClean property to repair transaction outliers automatically."),
            _step(3, "Display sales", "Place [Total Sales] in a card visual."),
        ]},
        [
            _context(1, "Define Total Sales as SUM(Sales[SalesAmount])."),
            _context(2, "Format the Total Sales measure as Currency to control how its values are displayed."),
            _context(3, "A card visual can display a single measure such as Total Sales."),
        ],
        (5, 7.9), (50, 80), (50, 80), frozenset({"step_2"}),
        expected_problem_statuses=frozenset({"unsupported"}),
    ),
    GroundingCase(
        "medium_whole_step_without_evidence",
        "How do I build and publish a Power BI sales report?",
        "power_bi",
        {"title": "Unsupported Deployment Step", "steps": [
            _step(1, "Load sales data", "Use Get Data to load the Sales table from SQL Server."),
            _step(2, "Build the report", "Create a bar chart using Product and Total Sales."),
            _step(3, "Deploy with Fabric", "Create a Fabric deployment pipeline with development, test, and production stages and assign workspace rules."),
        ]},
        [
            _context(1, "Power BI Desktop can load a Sales table from SQL Server through Get Data."),
            _context(2, "A bar chart can compare Product with the Total Sales measure."),
        ],
        (5, 7.9), (50, 80), (50, 80), frozenset({"step_3"}),
        expected_problem_statuses=frozenset({"unsupported"}),
    ),
    GroundingCase(
        "poor_multiple_unsupported_claims",
        "How do I automate customer synchronization in n8n?",
        "n8n_automation",
        {"title": "Mostly Unsupported Automation", "steps": [
            _step(1, "Trigger the workflow", "Use a Schedule Trigger to run every hour."),
            _step(2, "Fetch customers", "Use the Quantum CRM node to retrieve encrypted customer batches."),
            _step(3, "Transform records", "Use the AutoSchema node to infer and repair every field automatically."),
            _step(4, "Write customers", "Use the Universal Sync node to upsert all records without field mapping."),
        ]},
        [_context(1, "An n8n Schedule Trigger can run a workflow at an hourly interval.")],
        (0, 4.9), (0, 40), (0, 40),
        frozenset({"step_2", "step_3", "step_4"}),
        expected_problem_statuses=frozenset({"unsupported"}),
    ),
    GroundingCase(
        "poor_explicit_contradiction",
        "How should I calculate total sales in DAX?",
        "dax_power_bi",
        {"title": "Contradicted Sales Measure", "steps": [
            _step(1, "Create total sales", "Create [Total Sales] with AVERAGE(Sales[SalesAmount]) because averaging rows returns the total sales amount."),
            _step(2, "Display the result", "Place [Total Sales] in a card visual."),
        ]},
        [
            _context(1, "Total Sales must use SUM(Sales[SalesAmount]). AVERAGE returns the mean transaction value and must not be used for total sales."),
            _context(2, "A card visual can display the Total Sales measure."),
        ],
        (0, 4.9), (30, 70), (30, 70), frozenset({"step_1"}),
        frozenset({"step_1"}), frozenset({"contradicted"}),
    ),
    GroundingCase(
        "poor_irrelevant_context",
        "How do I build an n8n API-to-Sheets workflow?",
        "n8n_automation",
        {"title": "Workflow Against Irrelevant Context", "steps": [
            _step(1, "Call the API", "Use an HTTP Request node with GET and API credentials."),
            _step(2, "Map records", "Map API fields to spreadsheet columns with Edit Fields."),
            _step(3, "Append rows", "Use Google Sheets Append Row to write the mapped records."),
        ]},
        [_context(1, "Power BI themes define report colors, fonts, and visual formatting defaults.")],
        (0, 4.9), (0, 10), (0, 10),
        frozenset({"step_1", "step_2", "step_3"}),
        expected_problem_statuses=frozenset({"unsupported"}),
    ),
    GroundingCase(
        "orthogonal_incomplete_but_grounded",
        "How do I create a complete year-over-year sales solution in DAX?",
        "dax_power_bi",
        {"title": "Incomplete but Supported", "steps": [
            _step(1, "Create the date table", "Create a continuous Date table and mark it as the date table."),
            _step(2, "Create total sales", "Create [Total Sales] with SUM(Sales[SalesAmount])."),
        ]},
        [
            _context(1, "Time intelligence requires a continuous calendar table marked as the date table."),
            _context(2, "Define Total Sales as SUM(Sales[SalesAmount])."),
        ],
        (8, 10), (100, 100), (100, 100), frozenset(),
    ),
    GroundingCase(
        "poor_plausible_but_not_supported",
        "How do I compare current and prior-year sales in DAX?",
        "dax_power_bi",
        {"title": "Plausible Without Evidence", "steps": [
            _step(1, "Create prior-year sales", "Use SAMEPERIODLASTYEAR(Date[Date]) inside CALCULATE to shift Total Sales by one year."),
            _step(2, "Create YoY percentage", "Use DIVIDE([Total Sales] - [Sales PY], [Sales PY]) to calculate year-over-year percentage change."),
        ]},
        [_context(1, "Time intelligence is the general practice of comparing business measures across different time periods.")],
        (0, 4.9), (0, 30), (0, 30), frozenset({"step_1", "step_2"}),
        expected_problem_statuses=frozenset({"insufficient_evidence", "unsupported"}),
    ),
]

SMOKE_NAMES = {
    "good_fully_supported_dax",
    "medium_one_invented_detail",
    "poor_explicit_contradiction",
}


def _validate(case: GroundingCase, value: dict[str, Any]) -> dict[str, Any]:
    score = float(value["support_score"])
    claim_pct = float(value["claim_support_pct"])
    step_pct = float(value["step_grounded_pct"])
    problem_steps = {str(item["step_id"]) for item in value["unsupported_claims"]}
    contradiction_steps = {str(item["step_id"]) for item in value["contradictions"]}
    problem_statuses = {str(item["status"]) for item in value["unsupported_claims"]}
    score_ok = case.score_range[0] <= score <= case.score_range[1]
    claim_pct_ok = case.claim_support_range[0] <= claim_pct <= case.claim_support_range[1]
    step_pct_ok = case.step_grounded_range[0] <= step_pct <= case.step_grounded_range[1]
    problem_steps_ok = problem_steps == set(case.expected_problem_steps)
    contradictions_ok = contradiction_steps == set(case.expected_contradiction_steps)
    statuses_ok = (
        not case.expected_problem_statuses
        or (
            bool(problem_statuses)
            and problem_statuses <= set(case.expected_problem_statuses)
        )
    )
    return {
        "passed": all((score_ok, claim_pct_ok, step_pct_ok, problem_steps_ok, contradictions_ok, statuses_ok)),
        "score_ok": score_ok,
        "claim_support_ok": claim_pct_ok,
        "step_grounded_ok": step_pct_ok,
        "problem_steps_ok": problem_steps_ok,
        "contradictions_ok": contradictions_ok,
        "statuses_ok": statuses_ok,
        "expected_score_range": list(case.score_range),
        "expected_claim_support_range": list(case.claim_support_range),
        "expected_step_grounded_range": list(case.step_grounded_range),
        "expected_problem_steps": sorted(case.expected_problem_steps),
        "observed_problem_steps": sorted(problem_steps),
        "expected_contradiction_steps": sorted(case.expected_contradiction_steps),
        "observed_contradiction_steps": sorted(contradiction_steps),
        "observed_problem_statuses": sorted(problem_statuses),
    }


def _run(cases: list[GroundingCase], suite: str) -> dict[str, Any]:
    judge = GroundingJudge()
    started = time.perf_counter()
    results = []
    for index, case in enumerate(cases, 1):
        print(f"  [{index}/{len(cases)}] {case.name}", flush=True)
        case_started = time.perf_counter()
        evaluated = judge.evaluate(case.roadmap, case.contexts, question=case.question)
        value = evaluated["grounding"]
        results.append({
            "name": case.name,
            "question": case.question,
            "category": case.category,
            "roadmap": case.roadmap,
            "contexts": case.contexts,
            "grounding": value,
            "validation": _validate(case, value),
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


def _color(score: float) -> str:
    return "#52c41a" if score >= 8 else "#fa8c16" if score >= 5 else "#f5222d"


def _badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{html.escape(text)}</span>'


def _display(value: str) -> str:
    acronyms = {"dax": "DAX", "bi": "BI", "n8n": "n8n"}
    return " ".join(acronyms.get(word.lower(), word.capitalize()) for word in value.replace("_", " ").split())


def _case_html(item: dict[str, Any], index: int) -> str:
    value = item["grounding"]
    score = float(value["support_score"])
    steps = "".join(
        f"<li><strong>{html.escape(step['label'])}</strong>: {html.escape(step['description'])}</li>"
        for step in item["roadmap"]["steps"]
    )
    contexts = "".join(
        f"<li><code>{html.escape(context['id'])}</code>: {html.escape(context['text'])}</li>"
        for context in item["contexts"]
    )
    claim_rows = "".join(
        f"<tr><td><code>{html.escape(claim['step_id'])}</code></td><td>{html.escape(claim['claim'])}</td>"
        f"<td>{html.escape(claim['importance'].title())}</td><td>{_badge(claim['status'].replace('_',' ').upper(), {'supported':'#52c41a','contradicted':'#f5222d','unsupported':'#fa8c16','insufficient_evidence':'#1677ff'}.get(claim['status'],'#888'))}</td>"
        f"<td>{html.escape(', '.join(claim['context_ids']) or '—')}</td><td>{html.escape(claim['evidence'] or '—')}</td><td>{html.escape(claim['explanation'])}</td></tr>"
        for claim in value["claims"]
    )
    step_rows = "".join(
        f"<tr><td><code>{html.escape(step['step_id'])}</code></td><td>{html.escape(step['evaluability'].replace('_',' ').title())}</td>"
        f"<td>{html.escape(step['status'].replace('_',' ').title())}</td><td>{step['supported_claim_count']}</td><td>{step['unsupported_claim_count']}</td><td>{html.escape(step['reason'])}</td></tr>"
        for step in value["step_assessments"]
    )
    strengths = "".join(f"<li>{html.escape(text)}</li>" for text in value["strengths"]) or "<li class='empty'>None reported.</li>"
    recommendations = "".join(f"<li>{html.escape(text)}</li>" for text in value["recommendations"]) or "<li class='empty'>None reported.</li>"
    notes = ""
    if value["normalization_notes"]:
        notes = "<details class='notes'><summary>Processing notes</summary><ul>" + "".join(f"<li>{html.escape(note)}</li>" for note in value["normalization_notes"]) + "</ul></details>"
    return f"""
    <details class="case-detail" id="case-{index}"><summary>{index}. {html.escape(_display(item['name']))} — grounding {score:.1f}/10</summary>
    <div class="detail-grid"><div><h3>Roadmap</h3><p><strong>Question:</strong> {html.escape(item['question'])}</p><ol>{steps}</ol><h3>Retrieved Contexts</h3><ul>{contexts}</ul></div>
    <div><div class="metric-heading"><div><h3>Grounding</h3><span class="muted">Band: {value['score_band']}</span></div><strong class="metric-score" style="color:{_color(score)}">{score:.1f}<small>/10</small></strong></div>
    <section class="assessment"><h4>Overall Assessment</h4><p>{html.escape(value['reason'])}</p></section>
    <div class="stats"><span><strong>Claim support:</strong> {value['claim_support_pct']:.0f}%</span><span><strong>Grounded steps:</strong> {value['step_grounded_pct']:.0f}%</span><span><strong>Claims:</strong> {value['evaluable_claim_count']}</span><span><strong>Unsupported:</strong> {value['unsupported_claim_count']}</span><span><strong>Contradictions:</strong> {value['contradiction_count']}</span></div>
    <div class="findings"><section><h4>Strengths</h4><ul>{strengths}</ul></section><section><h4>Recommendations</h4><ul>{recommendations}</ul></section></div>{notes}</div></div>
    <section class="diagnostics"><h3>Step Assessments</h3><div class="scroll"><table class="nested"><thead><tr><th>Step</th><th>Evaluability</th><th>Status</th><th>Supported</th><th>Unsupported</th><th>Reason</th></tr></thead><tbody>{step_rows}</tbody></table></div>
    <h3>Claim Evidence</h3><div class="scroll"><table class="nested"><thead><tr><th>Step</th><th>Claim</th><th>Importance</th><th>Status</th><th>Contexts</th><th>Evidence</th><th>Evaluation</th></tr></thead><tbody>{claim_rows}</tbody></table></div></section></details>"""


def _save(payload: dict[str, Any], output_dir: str, stem: str) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{stem}.json"
    html_path = directory / f"{stem}.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = "".join(
        f"<tr><td>{index}</td><td><a href='#case-{index}'><strong>{html.escape(_display(item['name']))}</strong></a><br><span class='muted'>{html.escape(_display(item['category']))}</span></td>"
        f"<td style='color:{_color(item['grounding']['support_score'])}'><strong>{item['grounding']['support_score']:.1f}/10</strong></td>"
        f"<td>{_badge(item['grounding']['verdict'].replace('_',' '), _color(item['grounding']['support_score']))}</td><td>{item['grounding']['claim_support_pct']:.0f}%</td><td>{item['grounding']['step_grounded_pct']:.0f}%</td>"
        f"<td>{item['grounding']['unsupported_claim_count']}</td><td>{item['grounding']['contradiction_count']}</td><td>{item['duration_seconds']:.1f}s</td>"
        f"<td>{_badge('EXPECTED RESULT' if item['validation']['passed'] else 'NEEDS REVIEW', '#52c41a' if item['validation']['passed'] else '#f5222d')}</td></tr>"
        for index, item in enumerate(payload["results"], 1)
    )
    details = "".join(_case_html(item, index) for index, item in enumerate(payload["results"], 1))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Grounding Controlled Validation</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f5f5f5;color:#222;padding:24px}}h1{{margin-bottom:4px}}h3,h4{{margin-top:0}}.meta,.muted{{color:#666}}.meta{{margin-bottom:20px}}.cards{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px}}.card{{background:#fff;border-radius:8px;padding:16px 20px;box-shadow:0 2px 8px #00000014;min-width:150px}}.card strong{{display:block;font-size:26px;margin-bottom:4px}}table{{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px #00000014;margin-bottom:24px}}th,td{{padding:10px 12px;border-bottom:1px solid #eee;text-align:left;vertical-align:top;font-size:13px}}th{{background:#fafafa}}.badge{{color:#fff;padding:2px 8px;border-radius:12px;font-size:12px;white-space:nowrap}}a{{color:#222;text-decoration:none}}.case-detail{{background:#fff;margin-bottom:12px;padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px #0000000f}}.case-detail>summary{{cursor:pointer;font-weight:600}}.detail-grid{{display:grid;grid-template-columns:minmax(320px,.9fr) minmax(560px,1.5fr);gap:28px;margin-top:18px}}li{{margin-bottom:7px}}.metric-heading{{display:flex;justify-content:space-between;padding-bottom:14px;border-bottom:1px solid #eee}}.metric-score{{font-size:30px}}.metric-score small{{font-size:14px;color:#777}}.assessment{{background:#f7f9fc;border-left:4px solid #1677ff;padding:12px 14px;margin:16px 0}}.assessment p{{margin:0}}.stats{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}}.stats span{{background:#f0f2f5;border-radius:12px;padding:4px 9px;font-size:12px}}.findings{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.findings section{{border:1px solid #eee;border-radius:7px;padding:14px}}.diagnostics{{margin-top:24px;padding-top:20px;border-top:1px solid #eee}}.nested{{box-shadow:none;border:1px solid #eee}}.scroll{{overflow-x:auto}}.notes{{margin-top:14px;border:1px solid #eee;padding:10px}}.empty{{color:#777;font-style:italic}}code{{background:#f0f2f5;padding:1px 4px}}@media(max-width:1100px){{.detail-grid{{grid-template-columns:1fr}}}}@media(max-width:700px){{body{{padding:12px}}.findings{{grid-template-columns:1fr}}.overview{{display:block;overflow-x:auto}}}}
</style></head><body><h1>Grounding Controlled Validation</h1><p class="meta">Generated: {payload['generated_at']} | Model: {html.escape(payload['model'])} | Suite: {payload['suite']} | Ollama seed: {payload['ollama_seed']}</p>
<div class="cards"><div class="card"><strong>{payload['case_count']}</strong>Total Cases</div><div class="card"><strong style="color:#52c41a">{payload['passed_count']}</strong>Expected Results</div><div class="card"><strong style="color:#f5222d">{payload['failed_count']}</strong>Needs Review</div><div class="card"><strong>{payload['pass_rate']:.0%}</strong>Expected Match Rate</div><div class="card"><strong>{payload['duration_seconds']/60:.1f}</strong>Total Minutes</div></div>
<h2>Grounding Overview</h2><table class="overview"><thead><tr><th>#</th><th>Case</th><th>Score</th><th>Metric Result</th><th>Claim Support</th><th>Grounded Steps</th><th>Unsupported</th><th>Contradictions</th><th>Duration</th><th>Score Check</th></tr></thead><tbody>{rows}</tbody></table><h2>Case Details</h2>{details}</body></html>"""
    html_path.write_text(page, encoding="utf-8")
    return json_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Grounding with controlled roadmaps.")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output-dir", default=os.path.join(config.reports_dir, "grounding_metric_validation"))
    args = parser.parse_args()
    selected = CASES
    if args.case:
        names = set(args.case)
        selected = [case for case in CASES if case.name in names]
        unknown = names - {case.name for case in CASES}
        if unknown:
            raise SystemExit(f"Unknown Grounding cases: {', '.join(sorted(unknown))}")
    elif args.suite == "smoke":
        selected = [case for case in CASES if case.name in SMOKE_NAMES]
    payload = _run(selected, args.suite)
    stem = f"grounding_metric_validation_{args.suite}"
    json_path, html_path = _save(payload, args.output_dir, stem)
    print(f"Grounding validation complete: {payload['passed_count']}/{payload['case_count']}")
    print(f"  json: {json_path.resolve()}")
    print(f"  html: {html_path.resolve()}")


if __name__ == "__main__":
    main()
