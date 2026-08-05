"""Generate canonical JSON and HTML evaluation reports."""

from __future__ import annotations

import html
import json
import math
import os
from datetime import datetime
from typing import Any

from evaluation.metric_contract import metric_score
from src.llm_provider import active_model_name


_METRIC_LABELS = {
    "grounding": "Grounding",
    "completeness": "Completeness",
    "actionability": "Actionability",
    "logical_order": "Logical Order",
    "structure_quality": "Structure Quality",
    "step_distinctness": "Step Distinctness",
}


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    print(f"  JSON saved: {path}")


def _score_color(score: float) -> str:
    if score >= 8:
        return "#52c41a"
    if score >= 5:
        return "#fa8c16"
    return "#f5222d"


def _status_color(status: str) -> str:
    return {
        "READY": "#52c41a",
        "NEEDS_REVIEW": "#fa8c16",
        "FAIL": "#f5222d",
        "PASS": "#52c41a",
    }.get(status, "#8c8c8c")


def _badge(value: Any, color: str | None = None) -> str:
    text = str(value).replace("_", " ")
    return (
        f'<span class="badge" style="background:{color or _status_color(str(value))}">'
        f"{html.escape(text)}</span>"
    )


def _bar(score: float) -> str:
    width = max(0, min(100, score * 10))
    return (
        '<div class="bar"><div style="width:'
        f'{width:.0f}%;background:{_score_color(score)}"></div></div>'
    )


def _display(value: str) -> str:
    acronyms = {"dax": "DAX", "bi": "BI", "n8n": "n8n"}
    return " ".join(
        acronyms.get(word.lower(), word.capitalize())
        for word in value.replace("_", " ").split()
    )


def _page(title: str, body: str) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
*{{box-sizing:border-box}}body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f5f5f5;color:#262626;margin:0;padding:24px}}h1{{margin:0 0 4px}}h2{{margin:28px 0 12px;border-bottom:2px solid #e8e8e8;padding-bottom:7px}}h3{{margin:18px 0 8px}}.meta,.muted{{color:#777;font-size:13px}}.cards,.metric-grid{{display:flex;gap:14px;flex-wrap:wrap}}.card,.metric-card,.case{{background:#fff;border-radius:9px;box-shadow:0 2px 8px #00000012}}.card{{padding:18px;min-width:155px}}.card strong{{display:block;font-size:27px}}.metric-card{{padding:14px;min-width:155px;flex:1}}.metric-card strong{{font-size:24px}}.metric-card p{{font-size:12px;color:#666;margin:7px 0}}table{{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px #00000012}}th,td{{padding:10px 11px;text-align:left;vertical-align:top;border-bottom:1px solid #eee;font-size:13px}}th{{background:#fafafa;white-space:nowrap}}.scroll{{overflow-x:auto;margin-bottom:20px}}.badge{{display:inline-block;color:#fff;border-radius:12px;padding:2px 8px;font-size:11px;white-space:nowrap}}.bar{{height:8px;background:#eee;border-radius:4px;overflow:hidden;margin-top:7px}}.bar div{{height:100%}}.case{{padding:14px 17px;margin-bottom:12px}}.case>summary{{cursor:pointer;font-weight:700}}.case-grid{{display:grid;grid-template-columns:minmax(280px,.8fr) minmax(520px,1.5fr);gap:24px;margin-top:18px}}.steps{{padding-left:20px}}.steps li,.issues li{{margin-bottom:6px}}.diagnostic{{border:1px solid #e8e8e8;border-radius:7px;padding:10px 12px;margin-top:9px}}.diagnostic>summary{{cursor:pointer;font-weight:600}}.issue-list{{padding-left:17px;margin:0}}.good{{color:#389e0d}}.review{{color:#d46b08}}.fail{{color:#cf1322}}code{{background:#f0f2f5;padding:1px 4px;border-radius:3px}}@media(max-width:1000px){{.case-grid{{grid-template-columns:1fr}}}}@media(max-width:650px){{body{{padding:12px}}}}
</style></head><body><h1>{html.escape(title)}</h1>
<div class="meta">Generated: {generated} | Model: {html.escape(active_model_name())}</div>
{body}</body></html>'''


def _summary_status(samples: list[dict[str, Any]]) -> str:
    statuses = [sample.get("roadmap_readiness", {}).get("status") for sample in samples]
    if "FAIL" in statuses:
        return "FAIL"
    if "NEEDS_REVIEW" in statuses:
        return "NEEDS_REVIEW"
    if statuses and all(status == "READY" for status in statuses):
        return "READY"
    return "NOT_EVALUATED"


def _readiness_issues(readiness: dict[str, Any]) -> str:
    issues = readiness.get("issues", [])
    if not issues:
        return '<span class="good">All key dimensions are above threshold.</span>'
    return '<ul class="issue-list">' + "".join(
        f'<li><strong style="color:{_status_color("FAIL" if issue.get("severity") == "critical" else "NEEDS_REVIEW")}">'
        f'{html.escape(str(issue.get("label", issue.get("metric", "Issue"))))}:</strong> '
        f'{html.escape(str(issue.get("reason", "Review required")))}</li>'
        for issue in issues
    ) + "</ul>"


def _metric_cards(sample: dict[str, Any], names: list[str]) -> str:
    cards = []
    for name in names:
        result = sample.get(name, {})
        score = metric_score(name, result)
        cards.append(
            f'<div class="metric-card"><div class="muted">{_METRIC_LABELS[name]}</div>'
            f'<strong style="color:{_score_color(score)}">{score:.1f}/10</strong>'
            f'<p>{html.escape(str(result.get("reason", result.get("score_rationale", ""))))}</p>'
            f'{_bar(score)}</div>'
        )
    return "".join(cards)


def _grounding_detail(value: dict[str, Any]) -> str:
    gaps = value.get("unsupported_claims", [])
    rows = "".join(
        f'<tr><td><code>{html.escape(str(item.get("step_id", "")))}</code></td>'
        f'<td>{html.escape(str(item.get("claim", "")))}</td>'
        f'<td>{_badge(str(item.get("status", "")), _status_color("FAIL" if item.get("status") == "contradicted" else "NEEDS_REVIEW"))}</td>'
        f'<td>{html.escape(str(item.get("explanation", "")))}</td></tr>'
        for item in gaps
    ) or '<tr><td colspan="4" class="good">All claims are supported.</td></tr>'
    return f'<p class="muted">Claim support: {value.get("claim_support_pct", 0):.0f}% | Grounded steps: {value.get("step_grounded_pct", 0):.0f}%</p><div class="scroll"><table><thead><tr><th>Step</th><th>Claim gap</th><th>Status</th><th>Assessment</th></tr></thead><tbody>{rows}</tbody></table></div>'


def _completeness_detail(value: dict[str, Any]) -> str:
    gaps = value.get("missing_elements", [])
    rows = "".join(
        f'<tr><td>{html.escape(str(item.get("name", item.get("element", ""))))}</td>'
        f'<td>{html.escape(str(item.get("importance", "")))}</td>'
        f'<td>{html.escape(str(item.get("coverage_status", "")))}</td>'
        f'<td>{html.escape(str(item.get("recommendation", "")))}</td></tr>'
        for item in gaps
    ) or '<tr><td colspan="4" class="good">No material coverage gaps.</td></tr>'
    return f'<p class="muted">Coverage: {value.get("coverage_pct", 0):.0f}%</p><div class="scroll"><table><thead><tr><th>Expected element</th><th>Importance</th><th>Coverage</th><th>Recommendation</th></tr></thead><tbody>{rows}</tbody></table></div>'


def _actionability_detail(value: dict[str, Any]) -> str:
    steps = value.get("weak_action_steps", [])
    rows = "".join(
        f'<tr><td><code>{html.escape(str(item.get("step_id", "")))}</code></td>'
        f'<td>{float(item.get("score", 0)):.1f}/10</td>'
        f'<td>{html.escape(str(item.get("reason", "")))}</td></tr>'
        for item in steps
    ) or '<tr><td colspan="3" class="good">All steps are actionable.</td></tr>'
    return f'<p class="muted">Actionability density: {value.get("actionability_density", 0):.0%}</p><div class="scroll"><table><thead><tr><th>Step</th><th>Score</th><th>Clarification needed</th></tr></thead><tbody>{rows}</tbody></table></div>'


def _logical_order_detail(value: dict[str, Any]) -> str:
    violations = value.get("dependency_violations", [])
    rows = "".join(
        f'<tr><td><code>{html.escape(str(item.get("dependent_step_id", "")))}</code></td>'
        f'<td><code>{html.escape(str(item.get("required_predecessor_step_id", "")))}</code></td>'
        f'<td>{html.escape(str(item.get("dependency_type", "")))}</td>'
        f'<td>{html.escape(str(item.get("explanation", "")))}</td></tr>'
        for item in violations
    ) or '<tr><td colspan="4" class="good">No dependency violations.</td></tr>'
    return f'<div class="scroll"><table><thead><tr><th>Dependent</th><th>Required first</th><th>Type</th><th>Explanation</th></tr></thead><tbody>{rows}</tbody></table></div>'


def _structure_detail(value: dict[str, Any], schema: dict[str, Any]) -> str:
    criteria = value.get("criteria", {})
    rationales = value.get("criteria_rationales", {})
    rows = "".join(
        f'<tr><td>{_display(name)}</td><td style="color:{_score_color(float(score))}">{float(score):.1f}/10</td>'
        f'<td>{html.escape(str(rationales.get(name, "")))}</td></tr>'
        for name, score in criteria.items()
    )
    structure_table = f'<div class="scroll"><table><thead><tr><th>Dimension</th><th>Score</th><th>Assessment</th></tr></thead><tbody>{rows}</tbody></table></div>'
    if not schema:
        return structure_table
    violations = schema.get("violations", [])
    schema_text = "".join(f"<li>{html.escape(str(item))}</li>" for item in violations) or '<li class="good">Schema contract passed.</li>'
    return f'{structure_table}<h4>Schema Validity: {float(schema.get("score", 0)):.1f}/10 {_badge(schema.get("verdict", "N/A"))}</h4><ul class="issues">{schema_text}</ul>'


def _schema_detail(schema: dict[str, Any]) -> str:
    violations = schema.get("violations", [])
    items = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in violations
    ) or '<li class="good">Schema contract passed.</li>'
    return f'<p><strong>{float(schema.get("score", 0)):.1f}/10</strong> {_badge(schema.get("verdict", "N/A"))}</p><ul class="issues">{items}</ul>'


def _step_detail(distinctness: dict[str, Any], overlap: dict[str, Any]) -> str:
    weak = distinctness.get("weak_steps", [])
    pairs = overlap.get("overlapping_pairs", [])
    weak_items = "".join(
        f'<li><strong>{html.escape(str(item.get("label", item.get("step_id", "Step"))))}:</strong> {html.escape(str(item.get("reason", "")))}</li>'
        for item in weak
    ) or '<li class="good">No weak steps.</li>'
    pair_items = "".join(
        f'<li><strong>{html.escape(str(item.get("steps", [])))}:</strong> {html.escape(str(item.get("explanation", "")))}</li>'
        for item in pairs
    ) or '<li class="good">No overlapping pairs.</li>'
    return f'<h4>Weak steps</h4><ul class="issues">{weak_items}</ul><h4>Overlap ({html.escape(str(overlap.get("overall_severity", "none")))})</h4><ul class="issues">{pair_items}</ul>'


def _case_details(sample: dict[str, Any], index: int, names: list[str]) -> str:
    readiness = sample.get("roadmap_readiness", {})
    status = readiness.get("status", "NOT_EVALUATED")
    steps = "".join(
        f'<li><strong>{html.escape(str(step.get("label", "")))}</strong>: {html.escape(str(step.get("description", "")))}</li>'
        for step in sample.get("roadmap", {}).get("steps", []) if isinstance(step, dict)
    )
    expected = "".join(f"<li>{html.escape(str(item))}</li>" for item in sample.get("expected_steps", []))
    diagnostics = []
    if "grounding" in names:
        diagnostics.append(f'<details class="diagnostic"><summary>Grounding evidence</summary>{_grounding_detail(sample["grounding"])}</details>')
    if "completeness" in names:
        diagnostics.append(f'<details class="diagnostic"><summary>Completeness gaps</summary>{_completeness_detail(sample["completeness"])}</details>')
    if "actionability" in names:
        diagnostics.append(f'<details class="diagnostic"><summary>Actionability by step</summary>{_actionability_detail(sample["actionability"])}</details>')
    if "logical_order" in names:
        diagnostics.append(f'<details class="diagnostic"><summary>Logical Order dependencies</summary>{_logical_order_detail(sample["logical_order"])}</details>')
    if "structure_quality" in names:
        diagnostics.append(f'<details class="diagnostic"><summary>Structure dimensions</summary>{_structure_detail(sample["structure_quality"], sample.get("schema_validity", {}))}</details>')
    elif "schema_validity" in sample:
        diagnostics.append(f'<details class="diagnostic"><summary>Schema Validity</summary>{_schema_detail(sample["schema_validity"])}</details>')
    if "step_distinctness" in names:
        diagnostics.append(f'<details class="diagnostic"><summary>Step Distinctness and Overlap</summary>{_step_detail(sample["step_distinctness"], sample.get("step_overlap", {}))}</details>')
    return f'''<details class="case" id="case-{index}"><summary>{index}. {html.escape(sample.get("question", ""))} — {_badge(status)}</summary><div class="case-grid"><div><h3>Roadmap Output</h3><ol class="steps">{steps}</ol>{f'<h3>Expected Sequence</h3><ol class="steps">{expected}</ol>' if expected else ''}<p class="muted">Refined prompt: {html.escape(sample.get("refined_question", "") or "N/A")}</p></div><div><div class="metric-grid">{_metric_cards(sample, names)}</div><h3>Readiness</h3>{_readiness_issues(readiness)}{''.join(diagnostics)}</div></div></details>'''


def _judge_html(results: dict[str, Any]) -> str:
    samples = results.get("per_sample", [])
    aggregates = results.get("aggregated", {})
    available = [name for name in _METRIC_LABELS if name in aggregates.get("metrics", {})]
    has_schema = bool(aggregates.get("schema_validity"))
    status = _summary_status(samples)
    readiness = aggregates.get("readiness", {})
    cards = [
        f'<div class="card"><strong style="color:{_status_color(status)}">{html.escape(status.replace("_", " "))}</strong><span>Batch Status</span></div>',
        f'<div class="card"><strong>{len(samples)}</strong><span>Evaluated Roadmaps</span></div>',
    ]
    for name in available:
        score = float(aggregates["metrics"][name].get("mean_score", 0))
        cards.append(f'<div class="card"><strong style="color:{_score_color(score)}">{score:.1f}</strong><span>{_METRIC_LABELS[name]}</span>{_bar(score)}</div>')
    if has_schema:
        score = float(aggregates["schema_validity"].get("mean_score", 0))
        cards.append(f'<div class="card"><strong style="color:{_score_color(score)}">{score:.1f}</strong><span>Schema Validity</span>{_bar(score)}</div>')

    header_metrics = "".join(f"<th>{_METRIC_LABELS[name]}</th>" for name in available)
    schema_header = "<th>Schema</th>" if has_schema else ""
    rows = []
    for index, sample in enumerate(samples, 1):
        metric_cells = "".join(
            f'<td style="color:{_score_color(metric_score(name, sample.get(name, {})))}"><strong>{metric_score(name, sample.get(name, {})):.1f}</strong></td>'
            for name in available
        )
        schema = sample.get("schema_validity", {})
        schema_cell = f'<td>{float(schema.get("score", 0)):.1f}</td>' if has_schema else ""
        rows.append(f'<tr><td>{index}</td><td><a href="#case-{index}">{html.escape(sample.get("question", ""))}</a><br><span class="muted">{html.escape(_display(sample.get("category", "")))}</span></td><td>{_badge(sample.get("roadmap_readiness", {}).get("status", "NOT_EVALUATED"))}</td>{metric_cells}{schema_cell}<td>{_readiness_issues(sample.get("roadmap_readiness", {}))}</td></tr>')

    details = "".join(_case_details(sample, index, available) for index, sample in enumerate(samples, 1))
    body = f'''<h2>General Summary</h2><div class="cards">{''.join(cards)}</div><p class="muted">Ready: {readiness.get("counts", {}).get("READY", 0)} | Needs Review: {readiness.get("counts", {}).get("NEEDS_REVIEW", 0)} | Failed: {readiness.get("counts", {}).get("FAIL", 0)}</p><h2>Batch Overview</h2><p class="muted">Compare canonical scores and all detected readiness issues across cases.</p><div class="scroll"><table><thead><tr><th>#</th><th>Question</th><th>Status</th>{header_metrics}{schema_header}<th>Quality Signals</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><h2>Case Details</h2>{details}'''
    return _page("NotionMap — Roadmap Evaluation", body)


def _ragas_cell(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return '<td class="muted">N/A</td>'
    if math.isnan(numeric):
        return '<td class="muted">N/A</td>'
    score = max(0, min(1, numeric)) * 10
    return f'<td style="color:{_score_color(score)}"><strong>{score:.2f}/10</strong></td>'


def _ragas_html(results: dict[str, Any]) -> str:
    rows = "".join(
        f'<tr><td>{html.escape(sample.get("question", ""))}</td><td>{html.escape(_display(sample.get("category", "")))}</td>{_ragas_cell(sample.get("scores", {}).get("faithfulness"))}{_ragas_cell(sample.get("scores", {}).get("answer_relevancy"))}{_ragas_cell(sample.get("scores", {}).get("context_precision"))}{_ragas_cell(sample.get("scores", {}).get("context_recall"))}</tr>'
        for sample in results.get("per_sample", [])
    )
    body = f'<h2>RAGAS — Additional Diagnostics</h2><p class="muted">Explicitly requested technical support. These values do not override canonical roadmap metrics or Readiness.</p><div class="scroll"><table><thead><tr><th>Question</th><th>Category</th><th>Faithfulness</th><th>Answer Relevancy</th><th>Context Precision</th><th>Context Recall</th></tr></thead><tbody>{rows}</tbody></table></div>'
    return _page("NotionMap — RAGAS Diagnostics", body)


def _corpus_html(results: dict[str, Any]) -> str:
    aggregate = results.get("aggregated", {})
    diversity = results.get("semantic_diversity", {}) or {}
    cards = [
        ("Corpus Score", results.get("overall_corpus_score", 0)),
        ("Average Quality", aggregate.get("avg_quality", 0)),
        ("Coherence", aggregate.get("avg_coherencia", 0)),
        ("Technical Density", aggregate.get("avg_densidad_tecnica", 0)),
        ("RAG Usefulness", aggregate.get("avg_utilidad_rag", 0)),
    ]
    card_html = "".join(f'<div class="card"><strong style="color:{_score_color(float(score))}">{float(score):.1f}</strong><span>{label}</span>{_bar(float(score))}</div>' for label, score in cards)
    body = f'<h2>Corpus Diagnostics</h2><div class="cards">{card_html}</div><p>Chunks evaluated: {results.get("n_chunks_evaluated", 0)}/{results.get("n_chunks_total", 0)} | Semantic diversity: {diversity.get("diversity_score", "N/A")} | Redundant pairs: {diversity.get("redundant_pairs", 0)}</p>'
    return _page("NotionMap — Corpus Diagnostics", body)


def save_html(
    ragas_results: dict | None,
    judge_results: dict | None,
    path: str,
    corpus_results: dict | None = None,
) -> None:
    """Write one mode-specific report; optional analyses remain separate."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if judge_results:
        content = _judge_html(judge_results)
    elif ragas_results:
        content = _ragas_html(ragas_results)
    elif corpus_results:
        content = _corpus_html(corpus_results)
    else:
        content = _page("NotionMap — Evaluation", "<p>No evaluation results.</p>")
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"  HTML saved: {path}")
