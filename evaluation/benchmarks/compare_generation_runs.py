"""Generate a standalone side-by-side report for two generation benchmarks."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _roadmap(benchmark_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    artifact = benchmark_path.parent / result["roadmap_artifact"]
    payload = _load(artifact)
    return payload.get("roadmap", payload)


def _timing(result: dict[str, Any], name: str) -> float:
    return float(result.get("generation_trace", {}).get("timings", {}).get(name, 0))


def _trace_number(result: dict[str, Any], name: str) -> float:
    return float(result.get("generation_trace", {}).get(name, 0))


def _summary(payload: dict[str, Any]) -> dict[str, float]:
    results = payload["results"]
    return {
        "total": mean(float(item["observed_total_s"]) for item in results),
        "generation": mean(_timing(item, "roadmap_generation_s") for item in results),
        "prompt": mean(_trace_number(item, "generation_prompt_characters") for item in results),
        "output": mean(_trace_number(item, "roadmap_output_characters") for item in results),
        "steps": mean(_trace_number(item, "roadmap_step_count") for item in results),
        "within": sum(bool(item.get("within_target")) for item in results),
        "schema": sum(
            item.get("schema_validity", {}).get("verdict") == "PASS"
            for item in results
        ),
        "count": len(results),
    }


def _pct_change(before: float, after: float) -> float:
    return ((after - before) / before * 100) if before else 0.0


def _metric_row(label: str, before: float, after: float, unit: str = "") -> str:
    delta = _pct_change(before, after)
    delta_class = "good" if delta < 0 else "bad" if delta > 0 else "neutral"
    suffix = f" {unit}" if unit else ""
    return (
        "<tr>"
        f"<th>{html.escape(label)}</th>"
        f"<td>{before:,.2f}{suffix}</td>"
        f"<td>{after:,.2f}{suffix}</td>"
        f'<td class="{delta_class}">{delta:+.1f}%</td>'
        "</tr>"
    )


def _schema(result: dict[str, Any]) -> str:
    schema = result.get("schema_validity", {})
    verdict = str(schema.get("verdict", "UNKNOWN"))
    score = float(schema.get("score", 0))
    css = "pass" if verdict == "PASS" else "fail"
    violations = schema.get("violations", [])
    detail = ""
    if violations:
        items = "".join(f"<li>{html.escape(str(value))}</li>" for value in violations)
        detail = f"<details><summary>View violations</summary><ul>{items}</ul></details>"
    return f'<span class="badge {css}">{score:.2f}/10 {verdict}</span>{detail}'


def _roadmap_html(roadmap: dict[str, Any], result: dict[str, Any]) -> str:
    trace = result.get("generation_trace", {})
    retrieval = result.get("retrieval", {})
    steps = roadmap.get("steps", [])
    rendered_steps = []
    for position, step in enumerate(steps, 1):
        points = step.get("key_points", [])
        point_list = "".join(f"<li>{html.escape(str(point))}</li>" for point in points)
        rendered_steps.append(
            '<article class="step">'
            f'<div class="step-title"><span>{position}</span>'
            f'<strong>{html.escape(str(step.get("label", "Missing label")))}</strong></div>'
            f'<div class="step-meta">{html.escape(str(step.get("type", "missing type")))}'
            f' · source: {html.escape(str(step.get("source", "unassigned")))}</div>'
            f'<p>{html.escape(str(step.get("description", "Missing description")))}</p>'
            f'<ul>{point_list}</ul>'
            "</article>"
        )
    return f"""
    <div class="roadmap-meta">
      <div><b>Total:</b> {float(result['observed_total_s']):.2f}s</div>
      <div><b>Generation:</b> {_timing(result, 'roadmap_generation_s'):.2f}s</div>
      <div><b>Prompt:</b> {_trace_number(result, 'generation_prompt_characters'):,.0f} chars</div>
      <div><b>Output:</b> {_trace_number(result, 'roadmap_output_characters'):,.0f} chars</div>
      <div><b>Steps:</b> {len(steps)}</div>
      <div><b>Retrieval:</b> {html.escape(str(retrieval.get('mode', 'unknown')))}</div>
    </div>
    <div class="schema">{_schema(result)}</div>
    <h4>{html.escape(str(roadmap.get('title', 'Missing title')))}</h4>
    <details class="query"><summary>Refined query</summary>
      <p>{html.escape(str(result.get('refined_question', '')))}</p>
    </details>
    <div class="steps">{''.join(rendered_steps)}</div>
    """


def _review_form(case_id: int) -> str:
    criteria = (
        ("Completeness", "Covers the necessary stages without critical omissions."),
        ("Actionability", "Steps explain what to do and provide usable technical detail."),
        ("Logical order", "Dependencies and progression remain coherent."),
        ("Structure and scope", "The number and framing of steps fit the question."),
        ("Step distinctness", "Steps have separate responsibilities without duplication."),
        ("Grounding", "Technical claims remain supported by the available evidence."),
    )
    rows = []
    for index, (name, guidance) in enumerate(criteria):
        field = f"case-{case_id}-{index}"
        rows.append(
            "<tr>"
            f"<th>{html.escape(name)}<small>{html.escape(guidance)}</small></th>"
            f'<td><select data-review="{field}">'
            '<option value="">Not reviewed</option>'
            '<option value="better">Equivalent or better</option>'
            '<option value="minor">Minor acceptable regression</option>'
            '<option value="major">Material regression</option>'
            "</select></td></tr>"
        )
    return (
        '<details class="manual-review"><summary>Manual quality checklist</summary>'
        f'<table><tbody>{"".join(rows)}</tbody></table>'
        f'<textarea data-review="case-{case_id}-notes" placeholder="Review notes"></textarea>'
        "</details>"
    )


def generate(
    before_path: Path,
    after_path: Path,
    output_path: Path,
    before_label: str,
    after_label: str,
) -> None:
    before = _load(before_path)
    after = _load(after_path)
    before_results = {int(item["sample_index"]): item for item in before["results"]}
    after_results = {int(item["sample_index"]): item for item in after["results"]}
    case_ids = sorted(set(before_results) & set(after_results))
    if not case_ids:
        raise ValueError("The benchmarks do not contain matching sample indices.")

    old = _summary(before)
    new = _summary(after)
    summary_rows = "".join((
        _metric_row("Mean total time", old["total"], new["total"], "s"),
        _metric_row("Mean final generation", old["generation"], new["generation"], "s"),
        _metric_row("Mean prompt size", old["prompt"], new["prompt"], "chars"),
        _metric_row("Mean output size", old["output"], new["output"], "chars"),
        _metric_row("Mean step count", old["steps"], new["steps"]),
    ))

    case_rows = []
    case_sections = []
    for case_id in case_ids:
        prior = before_results[case_id]
        current = after_results[case_id]
        delta = _pct_change(float(prior["observed_total_s"]), float(current["observed_total_s"]))
        case_rows.append(
            "<tr>"
            f"<td>{case_id}</td><td>{html.escape(str(current['question']))}</td>"
            f"<td>{float(prior['observed_total_s']):.2f}s</td>"
            f"<td>{float(current['observed_total_s']):.2f}s</td>"
            f'<td class="{"good" if delta < 0 else "bad"}">{delta:+.1f}%</td>'
            f"<td>{int(_trace_number(prior, 'roadmap_step_count'))} → "
            f"{int(_trace_number(current, 'roadmap_step_count'))}</td>"
            "</tr>"
        )
        before_roadmap = _roadmap(before_path, prior)
        after_roadmap = _roadmap(after_path, current)
        case_sections.append(f"""
        <section class="case" id="case-{case_id}">
          <h2>Case {case_id}: {html.escape(str(current['question']))}</h2>
          <div class="comparison">
            <div class="version"><h3>{html.escape(before_label)}</h3>{_roadmap_html(before_roadmap, prior)}</div>
            <div class="version optimized"><h3>{html.escape(after_label)}</h3>{_roadmap_html(after_roadmap, current)}</div>
          </div>
          {_review_form(case_id)}
        </section>
        """)

    generated = datetime.now().isoformat(timespec="seconds")
    total_reduction = -_pct_change(old["total"], new["total"])
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prompt Optimization Comparison</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#d8deea;--soft:#f5f7fb;--green:#16803c;--red:#c62828;--blue:#155eef}}
*{{box-sizing:border-box}} body{{font-family:Inter,Segoe UI,Arial,sans-serif;margin:0;color:var(--ink);background:#eef2f7}}
main{{max-width:1500px;margin:auto;background:white;padding:28px}} h1,h2,h3,h4{{margin-top:0}} .subtitle{{color:var(--muted)}}
.hero{{display:grid;grid-template-columns:1fr auto;gap:20px;align-items:center;border-bottom:1px solid var(--line);padding-bottom:20px}}
.saving{{font-size:32px;font-weight:800;color:var(--green);text-align:right}} .saving small{{display:block;font-size:13px;color:var(--muted)}}
table{{width:100%;border-collapse:collapse;margin:16px 0 28px}} th,td{{border:1px solid var(--line);padding:9px;text-align:left;vertical-align:top}} th{{background:var(--soft)}}
.good{{color:var(--green);font-weight:700}} .bad{{color:var(--red);font-weight:700}} .neutral{{color:var(--muted)}}
.note{{padding:12px 14px;border-left:4px solid #f0a000;background:#fff8e6;margin:18px 0}}
.case{{border-top:4px solid #243b64;padding-top:24px;margin-top:42px}} .comparison{{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}}
.version{{border:1px solid var(--line);border-radius:10px;padding:18px;min-width:0}} .optimized{{border-color:#70bf8a;background:#fbfffc}}
.roadmap-meta{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-size:13px;background:var(--soft);padding:10px;border-radius:7px}}
.schema{{margin:12px 0}} .badge{{display:inline-block;border-radius:999px;padding:4px 9px;font-weight:700;font-size:12px}} .pass{{background:#dff7e7;color:var(--green)}} .fail{{background:#ffe3e3;color:var(--red)}}
.step{{border-left:3px solid #aab7cf;padding:10px 12px;margin:12px 0;background:white}} .step-title{{display:flex;gap:8px;align-items:start}} .step-title span{{background:#243b64;color:white;border-radius:50%;min-width:23px;height:23px;text-align:center;padding-top:2px}}
.step-meta{{font-size:12px;color:var(--muted);margin:4px 0 4px 31px}} .step p{{line-height:1.45}} .step li{{margin:4px 0}}
details summary{{cursor:pointer;color:var(--blue)}} .manual-review{{margin-top:16px;padding:14px;background:var(--soft);border-radius:8px}} .manual-review small{{display:block;color:var(--muted);font-weight:400;margin-top:3px}}
select,textarea{{width:100%;padding:7px;border:1px solid #aeb8ca;border-radius:5px;background:white}} textarea{{min-height:75px}}
@media(max-width:900px){{.comparison{{grid-template-columns:1fr}}.hero{{grid-template-columns:1fr}}.saving{{text-align:left}}.roadmap-meta{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<header class="hero"><div><h1>Prompt Optimization Comparison</h1>
<p class="subtitle">{html.escape(before_label)} vs. {html.escape(after_label)} · Generated {generated}</p></div>
<div class="saving">{total_reduction:.1f}% faster<small>mean end-to-end generation</small></div></header>

<h2>Executive comparison</h2>
<table><thead><tr><th>Indicator</th><th>{html.escape(before_label)}</th><th>{html.escape(after_label)}</th><th>Change</th></tr></thead><tbody>{summary_rows}</tbody></table>
<div class="note"><b>Reliability:</b> Schema Validity passed {int(old['schema'])}/{int(old['count'])} before and {int(new['schema'])}/{int(new['count'])} after. The optimized run used web successfully in both required cases. Manual semantic review is still required because shorter output is not automatically equivalent quality.</div>

<h2>Case timing overview</h2>
<table><thead><tr><th>Case</th><th>Question</th><th>Previous</th><th>Optimized</th><th>Time change</th><th>Steps</th></tr></thead><tbody>{''.join(case_rows)}</tbody></table>

<h2>How to review content</h2>
<p>Read each pair side by side, then use the checklist below it. The optimized roadmap should preserve necessary coverage and technical usefulness while removing verbosity—not remove essential work.</p>
{''.join(case_sections)}
</main>
<script>
const key='roadmap-prompt-comparison-review';
const fields=[...document.querySelectorAll('[data-review]')];
const saved=JSON.parse(localStorage.getItem(key)||'{{}}');
fields.forEach(el=>{{el.value=saved[el.dataset.review]||'';el.addEventListener('change',()=>{{saved[el.dataset.review]=el.value;localStorage.setItem(key,JSON.stringify(saved));}});}});
</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--before-label", default="Previous version")
    parser.add_argument("--after-label", default="Optimized version")
    args = parser.parse_args()
    generate(args.before, args.after, args.output, args.before_label, args.after_label)
    print(f"Comparison report saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
