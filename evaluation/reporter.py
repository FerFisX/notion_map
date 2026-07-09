"""Genera las salidas de evaluación: eval_report.json, eval_report.html y human_review.csv."""

import os
import json
import csv
import html as html_lib
import math
from datetime import datetime

from evaluation.config import config
from src.llm_provider import active_model_name


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  JSON guardado: {path}")


def save_human_review_csv(judge_results: dict, path: str):
    """Tabla que el humano llena offline con sus notas y comentarios."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N°", "Pregunta", "Prompt mejorado", "Intención", "Estrategia rewrite", "Categoría",
            "Pasos generados (en orden)",
            "Pasos esperados",
            "Readiness", "Readiness reason",
            "Score Resumen", "Grounding", "Completeness", "Logical Order", "Actionability", "Step Distinctness",
            "Score Secuencia", "Secuencia válida (IA)",
            "Pasos fuera de orden (IA)", "Sugerencia de reorden (IA)",
            # columnas para el humano
            "Nota Humana Secuencia (0-10)", "Nota Humana General (0-10)",
            "¿Aprobar? (Si/No)", "Comentarios del Revisor",
        ])

        for i, s in enumerate(judge_results.get("per_sample", []), 1):
            seq     = s["sequence_eval"]
            mese    = s["mese"]
            step_distinctness = s.get("step_distinctness", {})
            readiness = s.get("roadmap_readiness", {})
            steps   = "\n".join([f"{j+1}. {p}" for j, p in enumerate(s.get("steps", []))])
            exp     = "\n".join([f"{j+1}. {p}" for j, p in enumerate(s.get("expected_steps", []))])
            oor     = "; ".join(seq.get("out_of_order_steps", []))

            writer.writerow([
                i,
                s["question"],
                s.get("refined_question", ""),
                _intent_label(s),
                s.get("rewrite_strategy", s.get("judge_context_strategy", "")),
                s.get("category", ""),
                steps,
                exp,
                readiness.get("status", ""),
                readiness.get("reason", ""),
                round(mese["composite"], 2),
                round(mese["mapping"], 2),
                round(mese["exhaustiveness"], 2),
                round(mese["sequence"], 2),
                round(mese["experience"], 2),
                round(step_distinctness.get("score", 0), 2),
                round(seq["score"], 2),
                "Sí" if seq["is_valid"] else "No",
                oor or "—",
                seq.get("suggested_fix", "—"),
                # columnas vacías para el humano
                "", "", "", "",
            ])

    print(f"  CSV revisión humana: {path}")


def _score_color(score: float) -> str:
    if score >= 7:  return "#52c41a"
    if score >= 5:  return "#fa8c16"
    return "#f5222d"

def _ragas_score_cell(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return '<td style="text-align:center;color:#999">N/A</td>'
    if math.isnan(numeric):
        return '<td style="text-align:center;color:#999">N/A</td>'
    score = round(max(0.0, min(1.0, numeric)) * 10, 2)
    return f'<td style="text-align:center;color:{_score_color(score)}">{score:.2f}/10</td>'

def _badge(text: str, color: str) -> str:
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:12px;font-size:12px">{text}</span>'

def _readiness_color(status: str) -> str:
    if status in ("READY", "Ready", "Approved"):
        return "#52c41a"
    if status in ("NEEDS_REVIEW", "Needs Review"):
        return "#fa8c16"
    return "#f5222d"

def _readiness_label(status: str) -> str:
    labels = {
        "READY": "Ready",
        "NEEDS_REVIEW": "Needs Review",
        "FAIL": "Failed",
    }
    return labels.get(status, status or "N/A")

def _summary_readiness_status(judge_results: dict, readiness_agg: dict) -> str:
    samples = judge_results.get("per_sample", []) if judge_results else []
    statuses = [
        s.get("roadmap_readiness", {}).get("status")
        for s in samples
        if s.get("roadmap_readiness", {}).get("status")
    ]
    if statuses:
        if any(status == "FAIL" for status in statuses):
            return "FAIL"
        if any(status == "NEEDS_REVIEW" for status in statuses):
            return "NEEDS_REVIEW"
        return "READY"
    if readiness_agg.get("fail_rate", 0) > 0:
        return "FAIL"
    if readiness_agg.get("needs_review_rate", 0) > 0:
        return "NEEDS_REVIEW"
    if readiness_agg.get("ready_rate", 0) > 0:
        return "READY"
    return "N/A"

def _bar(score: float, max_score: float = 10) -> str:
    pct = max(0, min(100, (score / max_score) * 100))
    color = _score_color(score)
    return f'''<div style="background:#f0f0f0;border-radius:4px;height:10px;width:100%">
      <div style="background:{color};width:{pct:.0f}%;height:10px;border-radius:4px"></div>
    </div>'''



def _overlap_details_html(step_overlap: dict) -> str:
    pairs = step_overlap.get("overlapping_pairs", []) or []
    if not pairs:
        if step_overlap.get("evaluated"):
            return "<li style='color:#52c41a'>No overlapping weak steps detected</li>"
        return "<li style='color:#52c41a'>No weak steps detected; overlap analysis was not required</li>"
    return "".join(
        "<li>"
        f"Steps {', '.join(str(s) for s in p.get('steps', []))}: "
        f"<strong>{html_lib.escape(str(p.get('overlap_type', 'overlap')))}</strong> "
        f"({html_lib.escape(str(p.get('severity', 'medium')))}). "
        f"{html_lib.escape(str(p.get('explanation', '')))} "
        f"<em>{html_lib.escape(str(p.get('recommendation', '')))}</em>"
        "</li>"
        for p in pairs
    )


def _readiness_counts(judge_results: dict) -> dict:
    samples = judge_results.get("per_sample", []) if judge_results else []
    counts = {"READY": 0, "NEEDS_REVIEW": 0, "FAIL": 0}
    for sample in samples:
        status = sample.get("roadmap_readiness", {}).get("status", "N/A")
        if status in counts:
            counts[status] += 1
    return counts


def _weakest_metric(mese: dict, step_distinctness: dict, struct: dict) -> tuple[str, float]:
    metrics = {
        "Grounding": float(mese.get("mapping", 0)),
        "Completeness": float(mese.get("exhaustiveness", 0)),
        "Logical Order": float(mese.get("sequence", 0)),
        "Actionability": float(mese.get("experience", 0)),
        "Step Distinctness": float(step_distinctness.get("score", 0)),
        "Structure": float(struct.get("score", 0)),
    }
    return min(metrics.items(), key=lambda item: item[1])


def _main_issue(mese: dict, step_distinctness: dict, struct: dict, readiness: dict) -> str:
    reason = readiness.get("reason", "")
    if reason:
        return reason
    metric, score = _weakest_metric(mese, step_distinctness, struct)
    if score >= 7:
        return "No major issue detected"
    return f"{metric} needs review"


def _batch_issues_html(mese: dict, step_distinctness: dict, struct: dict) -> str:
    checks = [
        ("Grounding", float(mese.get("mapping", 0)), "weak support against retrieved context", "mostly supported, worth checking evidence"),
        ("Completeness", float(mese.get("exhaustiveness", 0)), "missing important expected scope", "covers the core topic but misses useful depth"),
        ("Actionability", float(mese.get("experience", 0)), "steps are too abstract to execute", "steps need clearer actions or deliverables"),
        ("Step Distinctness", float(step_distinctness.get("score", 0)), "steps do not contribute distinct learning value", "some steps need clearer unique contribution"),
        ("Structure", float(struct.get("score", 0)), "structural validation failed", "structure needs review"),
    ]
    issues = []
    for label, score, severe_text, review_text in checks:
        if score < 5:
            issues.append((label, score, severe_text))
        elif score < 7:
            issues.append((label, score, review_text))

    if not issues:
        return '<span style="color:#52c41a">All key dimensions are above threshold.</span>'

    return "<ul class='issue-list'>" + "".join(
        f'<li><strong style="color:{_score_color(score)}">{html_lib.escape(label)}:</strong> {html_lib.escape(text)}</li>'
        for label, score, text in issues
    ) + "</ul>"


def _query_intent(sample: dict) -> dict:
    value = sample.get("query_intent") or {}
    if isinstance(value, dict):
        return value
    return {"intent": str(value)}


def _intent_label(sample: dict) -> str:
    return sample.get("intent") or _query_intent(sample).get("intent", "")


def save_html(ragas_results: dict, judge_results: dict, path: str,
              corpus_results: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    ragas_agg  = ragas_results.get("aggregated", {})  if ragas_results  else {}
    judge_agg  = judge_results.get("aggregated", {})  if judge_results  else {}
    corpus_agg = corpus_results.get("aggregated", {}) if corpus_results else {}
    readiness_agg = judge_agg.get("readiness", {}) if judge_agg else {}
    roadmap_agg = judge_agg.get("roadmap", {}) if judge_agg else {}
    grounding_agg = judge_agg.get("grounding", {}) if judge_agg else {}
    readiness_status = _summary_readiness_status(judge_results or {}, readiness_agg)
    readiness_label = _readiness_label(readiness_status)
    readiness_color = _readiness_color(readiness_status)
    samples = (judge_results or {}).get("per_sample", [])
    readiness_counts = _readiness_counts(judge_results or {})

    # tarjetas resumen
    summary_cards = ""
    all_cards = []
    if judge_agg:
        all_cards += [
            ("Grounding",       grounding_agg.get("support_score", 0)),
            ("Completeness",    roadmap_agg.get("completeness", 0)),
            ("Logical Order",   roadmap_agg.get("logical_order", 0)),
            ("Actionability",   roadmap_agg.get("actionability", 0)),
            ("Step Distinctness", roadmap_agg.get("step_distinctness", 0)),
            ("Structure",       judge_agg.get("structure", {}).get("mean_score", 0)),
        ]
    if judge_agg and len(samples) > 1:
        total_samples = max(1, len(samples))
        ready_pct = readiness_counts["READY"] / total_samples * 100
        review_pct = readiness_counts["NEEDS_REVIEW"] / total_samples * 100
        fail_pct = readiness_counts["FAIL"] / total_samples * 100
        summary_cards += f'''
        <div style="background:#fff;border-radius:8px;padding:20px;
                    box-shadow:0 2px 8px rgba(0,0,0,.1);min-width:240px">
          <div style="font-size:18px;font-weight:700;color:#333">Roadmap Status</div>
          <div style="font-size:13px;color:#666;margin-top:4px">{total_samples} evaluated roadmaps</div>
          <div style="display:flex;height:10px;border-radius:4px;overflow:hidden;background:#f0f0f0;margin-top:10px">
            <div title="Ready" style="background:#52c41a;width:{ready_pct:.0f}%"></div>
            <div title="Needs Review" style="background:#fa8c16;width:{review_pct:.0f}%"></div>
            <div title="Failed" style="background:#f5222d;width:{fail_pct:.0f}%"></div>
          </div>
          <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;font-size:12px">
            <span style="color:#52c41a">Ready: {readiness_counts["READY"]}</span>
            <span style="color:#fa8c16">Review: {readiness_counts["NEEDS_REVIEW"]}</span>
            <span style="color:#f5222d">Failed: {readiness_counts["FAIL"]}</span>
          </div>
        </div>'''
    elif judge_agg:
        summary_cards += f'''
        <div style="background:#fff;border-radius:8px;padding:20px;text-align:center;
                    box-shadow:0 2px 8px rgba(0,0,0,.1);min-width:170px">
          <div style="font-size:28px;font-weight:700;color:{readiness_color}">{readiness_label}</div>
          <div style="font-size:13px;color:#666;margin-top:4px">Roadmap Status</div>
          <div style="background:#f0f0f0;border-radius:4px;height:10px;width:100%;margin-top:8px">
            <div style="background:{readiness_color};width:100%;height:10px;border-radius:4px"></div>
          </div>
        </div>'''
    for label, val in all_cards:
        color = _score_color(val)
        summary_cards += f'''
        <div style="background:#fff;border-radius:8px;padding:20px;text-align:center;
                    box-shadow:0 2px 8px rgba(0,0,0,.1);min-width:140px">
          <div style="font-size:32px;font-weight:700;color:{color}">{val:.1f}</div>
          <div style="font-size:13px;color:#666;margin-top:4px">{label}</div>
          {_bar(val)}
        </div>'''

    # tabla RAGAS
    ragas_rows = ""
    for s in (ragas_results or {}).get("per_sample", []):
        sc = s.get("scores", {})
        ragas_rows += f'''<tr>
          <td style="max-width:250px;word-break:break-word">{s["question"]}</td>
          <td>{_badge(s.get("category",""), "#1890ff")}</td>
          {_ragas_score_cell(sc.get("faithfulness"))}
          {_ragas_score_cell(sc.get("answer_relevancy"))}
          {_ragas_score_cell(sc.get("context_precision"))}
          {_ragas_score_cell(sc.get("context_recall"))}
        </tr>'''

    # tabla humana (LLM Judge + revision)
    batch_rows = ""
    human_rows = ""
    for i, s in enumerate((judge_results or {}).get("per_sample", []), 1):
        seq    = s["sequence_eval"]
        mese   = s["mese"]
        step_distinctness = s.get("step_distinctness", {})
        step_overlap = s.get("step_overlap", {})
        overlap_evaluated = bool(step_overlap.get("evaluated"))
        overlap_issue_count = int(step_overlap.get("issue_count", 0))
        overlap_label = "Evaluated" if overlap_evaluated else "Not required"
        overlap_color = _score_color(10 if not overlap_evaluated and overlap_issue_count == 0 else max(0, 10 - overlap_issue_count * 2))
        readiness = s.get("roadmap_readiness", {})
        struct = s.get("structure", {})
        steps  = "".join(f"<li>{html_lib.escape(str(p))}</li>" for p in s.get("steps", []))
        exp    = "".join(f"<li>{html_lib.escape(str(p))}</li>" for p in s.get("expected_steps", []))
        oor    = seq.get("out_of_order_steps", [])
        oor_html = "".join(f'<li style="color:#f5222d">{p}</li>' for p in oor) if oor else "<li style='color:#52c41a'>None</li>"
        verdict_badge  = _badge(s["verdict"],      "#52c41a" if s["verdict"]=="PASS"      else "#f5222d")
        mese_badge     = _badge(s["mese_verdict"],  "#52c41a" if s["mese_verdict"]=="PASS"  else "#f5222d")
        seq_badge      = _badge("Valid" if seq["is_valid"] else "Invalid",
                                "#52c41a" if seq["is_valid"] else "#f5222d")
        struct_verdict = struct.get("verdict", "N/A")
        struct_badge   = _badge(struct_verdict, "#52c41a" if struct_verdict=="PASS" else "#f5222d")
        readiness_status = readiness.get("status", "N/A")
        readiness_badge = _badge(readiness_status, _readiness_color(readiness_status))
        readiness_reason_html = html_lib.escape(readiness.get("reason", ""))
        grounding_source = html_lib.escape(
            s.get("grounding", {}).get("source")
            or s.get("ragas_alignment", {}).get("grounding_source", "judge_fallback")
        )
        struct_violations_html = "".join(
            f'<li style="color:#f5222d;font-size:11px">{v}</li>'
            for v in struct.get("violations", [])
        ) or "<li style='color:#52c41a;font-size:11px'>Sin violaciones</li>"
        resp_time = s.get("response_time", 0)
        question_html = html_lib.escape(s.get("question", ""))
        refined_html = html_lib.escape(s.get("refined_question", ""))
        intent = _query_intent(s)
        intent_html = html_lib.escape(_intent_label(s))
        intent_goal_html = html_lib.escape(intent.get("roadmap_goal", ""))
        rewrite_strategy_html = html_lib.escape(s.get("rewrite_strategy", s.get("judge_context_strategy", "")))
        refined_block = (
            f'''<details style="margin-top:8px">
              <summary style="cursor:pointer;color:#1890ff;font-size:12px">View refined prompt</summary>
              <div style="font-size:12px;color:#555;margin-top:4px;line-height:1.35">{refined_html}</div>
            </details>'''
            if refined_html else
            '<small style="color:#999">Prompt mejorado no disponible</small>'
        )
        intent_goal_block = (
            f'''<details style="margin-top:4px">
              <summary style="cursor:pointer;color:#666;font-size:12px">View goal/intent</summary>
              <div style="font-size:12px;color:#555;margin-top:4px;line-height:1.35">{intent_goal_html}</div>
            </details>'''
            if intent_goal_html else ""
        )

        issues_html = _batch_issues_html(mese, step_distinctness, struct)

        batch_rows += f'''
        <tr>
          <td style="text-align:center;font-weight:bold">{i}</td>
          <td style="max-width:280px">
            <strong>{question_html}</strong><br>
            {_badge(s.get("category",""), "#722ed1")}
            <small style="color:#666"> {intent_html or "N/A"}</small>
          </td>
          <td>{readiness_badge}</td>
          <td style="color:{_score_color(mese["mapping"])}">{mese["mapping"]:.1f}</td>
          <td style="color:{_score_color(mese["exhaustiveness"])}">{mese["exhaustiveness"]:.1f}</td>
          <td style="color:{_score_color(mese["experience"])}">{mese["experience"]:.1f}</td>
          <td style="color:{_score_color(step_distinctness.get("score", 0))}">{step_distinctness.get("score", 0):.1f}</td>
          <td style="color:{_score_color(struct.get("score", 0))}">{struct.get("score", 0):.1f}</td>
          <td>{issues_html}</td>
        </tr>'''

        human_rows += f'''
        <tr id="row-{i}">
          <td style="text-align:center;font-weight:bold">{i}</td>
          <td class="question-cell">
            <strong>{question_html}</strong><br>
            {_badge(s.get("category",""), "#722ed1")}<br>
            <small style="color:#666">intent={intent_html or "N/A"}</small><br>
            <small style="color:#666">rewrite={rewrite_strategy_html or "N/A"}</small><br>
            <small style="color:#999">t={resp_time:.1f}s</small>
            {refined_block}
            {intent_goal_block}
          </td>
          <td class="steps-cell">
            <details open>
              <summary style="cursor:pointer;color:#1890ff">View generated steps ({len(s.get("steps",[]))})</summary>
              <ol class="steps-list">{steps}</ol>
            </details>
            <details open style="margin-top:8px">
              <summary style="cursor:pointer;color:#52c41a">View expected steps</summary>
              <ol class="steps-list">{exp}</ol>
            </details>
          </td>
          <td class="sequence-cell">
            <strong>Sequence</strong> {seq_badge}<br>
            <small>{_bar(seq["score"])} {seq["score"]:.1f}/10</small><br>
            <details style="margin-top:4px">
              <summary style="cursor:pointer;font-size:12px">Out-of-order steps</summary>
              <ul style="font-size:12px;padding-left:16px;margin:4px 0">{oor_html}</ul>
              <div style="font-size:12px;color:#666;margin-top:4px">
                <strong>Suggestion:</strong> {seq.get("suggested_fix","—")}
              </div>
            </details>
          </td>
          <td class="structure-cell">
            {struct_badge} {struct.get("score", 0):.1f}/10<br>
            <small style="color:#666">{struct.get("passed",0)}/{struct.get("total_checks",0)} checks</small>
            <details style="margin-top:4px">
              <summary style="cursor:pointer;font-size:12px">View violations</summary>
              <ul style="padding-left:14px;margin:4px 0">{struct_violations_html}</ul>
            </details>
          </td>
          <td class="overlap-cell">
            <strong style="color:{overlap_color}">{overlap_label}</strong><br>
            <small style="color:#666;display:block">
              Evaluated: {"yes" if overlap_evaluated else "no"}<br>
              Trigger: {html_lib.escape(str(step_overlap.get("trigger", "no_weak_steps")))}<br>
              Issues: {overlap_issue_count}
            </small>
            <details style="margin-top:8px">
              <summary style="cursor:pointer;font-size:12px">Overlapping pairs</summary>
              <ul style="font-size:12px;padding-left:16px;margin:4px 0">{_overlap_details_html(step_overlap)}</ul>
            </details>
          </td>
          <td class="readiness-cell">
            {readiness_badge}<br>
            <small style="color:#666">{readiness_reason_html}</small>
          </td>
        </tr>'''


    # seccion Corpus Judge
    corpus_section = ""
    if corpus_results and "overall_corpus_score" in corpus_results:
        ca   = corpus_results.get("aggregated", {})
        sd   = corpus_results.get("semantic_diversity", {})
        cv   = corpus_results.get("coverage", {})
        cs   = corpus_results["overall_corpus_score"]
        cv_d = corpus_results["verdict"]
        prob = corpus_results.get("problematic_chunks", [])

        prob_rows = "".join(
            f'<tr><td style="font-size:12px;max-width:300px">{p["preview"]}</td>'
            f'<td style="font-size:12px;color:#f5222d">{p["problem"]}</td></tr>'
            for p in prob
        ) or '<tr><td colspan="2" style="color:#52c41a;font-size:12px">No problematic chunks</td></tr>'

        chunk_rows = "".join(
            f'<tr>'
            f'<td style="font-size:11px;max-width:250px;color:#666">{r.get("chunk_preview","")[:80]}...</td>'
            f'<td style="text-align:center">{r.get("coherencia",0)}</td>'
            f'<td style="text-align:center">{r.get("densidad_tecnica",0)}</td>'
            f'<td style="text-align:center">{r.get("utilidad_rag",0)}</td>'
            f'<td style="text-align:center;font-weight:bold;color:{_score_color(r.get("avg_score",0))}">'
            f'{r.get("avg_score",0):.1f}</td>'
            f'<td style="font-size:11px;color:#666">{r.get("tema_principal","")}</td>'
            f'</tr>'
            for r in corpus_results.get("per_chunk", [])
        )

        corpus_section = f'''
<h2>Corpus Judge — Knowledge Base Quality</h2>
<p style="font-size:13px;color:#888;margin-bottom:12px">
  Evaluation of {corpus_results["n_chunks_total"]} chunks in ChromaDB
  ({corpus_results["n_chunks_evaluated"]} evaluated with an LLM).
  Inspired by Rothman (2024): semantic cosine similarity + LLM judge for coherence and usefulness.
</p>

<div class="cards" style="margin-bottom:16px">
  <div style="background:#fff;border-radius:8px;padding:20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.1);min-width:140px">
    <div style="font-size:32px;font-weight:700;color:{_score_color(cs)}">{cs:.1f}</div>
    <div style="font-size:13px;color:#666;margin-top:4px">Score Corpus</div>
    {_bar(cs)} {_badge(cv_d, "#52c41a" if cv_d=="PASS" else "#f5222d")}
  </div>
  <div style="background:#fff;border-radius:8px;padding:20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.1);min-width:140px">
    <div style="font-size:32px;font-weight:700;color:{_score_color(ca.get("avg_quality",0))}">{ca.get("avg_quality",0):.1f}</div>
    <div style="font-size:13px;color:#666;margin-top:4px">Average Quality</div>
    {_bar(ca.get("avg_quality",0))}
  </div>
  <div style="background:#fff;border-radius:8px;padding:20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.1);min-width:140px">
    <div style="font-size:32px;font-weight:700;color:{'#52c41a' if (sd.get('diversity_score') or 0)>0.5 else '#fa8c16'}">{(sd.get("diversity_score") or 0):.2f}</div>
    <div style="font-size:13px;color:#666;margin-top:4px">Semantic Diversity</div>
    {_bar((sd.get("diversity_score") or 0)*10)}
  </div>
  <div style="background:#fff;border-radius:8px;padding:20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.1);min-width:140px">
    <div style="font-size:32px;font-weight:700;color:{'#52c41a' if sd.get('redundant_pairs',0)==0 else '#f5222d'}">{sd.get("redundant_pairs",0)}</div>
    <div style="font-size:13px;color:#666;margin-top:4px">Redundant Pairs</div>
  </div>
</div>

<div style="display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap">
  <div style="background:#fff;border-radius:8px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.08);flex:1;min-width:200px">
    <strong>Aggregated Metrics</strong>
    <table style="margin-top:8px;font-size:13px;width:100%;box-shadow:none">
      <tr><td>Average coherence</td><td style="text-align:right;color:{_score_color(ca.get("avg_coherencia",0))}">{ca.get("avg_coherencia",0):.2f}/10</td></tr>
      <tr><td>Technical density</td><td style="text-align:right;color:{_score_color(ca.get("avg_densidad_tecnica",0))}">{ca.get("avg_densidad_tecnica",0):.2f}/10</td></tr>
      <tr><td>RAG usefulness</td><td style="text-align:right;color:{_score_color(ca.get("avg_utilidad_rag",0))}">{ca.get("avg_utilidad_rag",0):.2f}/10</td></tr>
      <tr><td>Average length</td><td style="text-align:right">{ca.get("avg_chunk_length",0)} chars</td></tr>
      <tr><td>Chunks with issues</td><td style="text-align:right;color:{'#f5222d' if ca.get('chunks_with_issues',0)>0 else '#52c41a'}">{ca.get("chunks_with_issues",0)}</td></tr>
    </table>
  </div>
  <div style="background:#fff;border-radius:8px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.08);flex:1;min-width:200px">
    <strong>Topic Coverage</strong>
    <table style="margin-top:8px;font-size:13px;width:100%;box-shadow:none">
      <tr><td>Breadth</td><td style="text-align:right;color:{_score_color(cv.get("amplitud",5))}">{cv.get("amplitud","N/A")}/10</td></tr>
      <tr><td>Depth</td><td style="text-align:right;color:{_score_color(cv.get("profundidad",5))}">{cv.get("profundidad","N/A")}/10</td></tr>
      <tr><td>Topic coherence</td><td style="text-align:right;color:{_score_color(cv.get("coherencia_tematica",5))}">{cv.get("coherencia_tematica","N/A")}/10</td></tr>
      <tr><td>Estimated unique topics</td><td style="text-align:right">{cv.get("temas_unicos_estimados","N/A")}</td></tr>
    </table>
    <div style="font-size:12px;color:#666;margin-top:8px;font-style:italic">
      {cv.get("observacion","")}
    </div>
  </div>
</div>

<details>
  <summary style="cursor:pointer;color:#1890ff;margin-bottom:8px">View per-chunk evaluation</summary>
  <div style="overflow-x:auto">
  <table>
    <thead><tr>
      <th>Chunk (preview)</th><th>Coherence</th><th>Density</th><th>Usefulness</th><th>Avg</th><th>Topic</th>
    </tr></thead>
    <tbody>{chunk_rows}</tbody>
  </table>
  </div>
</details>

{"<h3 style='color:#f5222d;margin-top:16px'>Problematic Chunks</h3><table><thead><tr><th>Preview</th><th>Detected problem</th></tr></thead><tbody>" + prob_rows + "</tbody></table>" if prob else ""}
'''

    html = f'''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NotionMap — Evaluation Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f5f5f5; color: #333; padding: 24px; }}
  h1 {{ font-size: 24px; margin-bottom: 4px; }}
  h2 {{ font-size: 18px; margin: 24px 0 12px; color: #444; border-bottom: 2px solid #e8e8e8;
        padding-bottom: 6px; }}
  h3 {{ font-size: 15px; margin: 16px 0 8px; color: #555; }}
  .meta {{ color: #888; font-size: 13px; margin-bottom: 24px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff;
           border-radius: 8px; overflow: hidden;
           box-shadow: 0 2px 8px rgba(0,0,0,.08); margin-bottom: 24px; }}
  th {{ background: #fafafa; padding: 10px 12px; text-align: left;
        font-size: 13px; border-bottom: 2px solid #e8e8e8; }}
  td {{ padding: 10px 12px; font-size: 13px; border-bottom: 1px solid #f0f0f0;
        vertical-align: top; }}
  tr:hover td {{ background: #fafafa; }}
  .question-cell {{ min-width: 170px; max-width: 230px; }}
  .steps-cell {{ min-width: 260px; max-width: 340px; }}
  .sequence-cell {{ min-width: 170px; max-width: 240px; }}
  .structure-cell {{ min-width: 200px; max-width: 280px; }}
  .overlap-cell {{ min-width: 170px; max-width: 240px; }}
  .readiness-cell {{ min-width: 160px; max-width: 220px; }}
  .steps-list {{ padding-left: 18px; margin: 6px 0 0; line-height: 1.35; }}
  .steps-list li {{ margin-bottom: 4px; }}
  .issue-list {{ padding-left: 16px; margin: 0; line-height: 1.35; }}
  .issue-list li {{ margin-bottom: 4px; }}
  .mini-table {{ font-size:12px; width:100%; box-shadow:none; margin:6px 0; border-radius:4px; }}
  .mini-table td {{ padding:5px 6px; }}
  details summary::-webkit-details-marker {{ display:none; }}
  .export-btn {{ background: #1890ff; color: #fff; border: none; padding: 8px 20px;
                 border-radius: 6px; cursor: pointer; font-size: 14px; margin-top: 16px; }}
  .export-btn:hover {{ background: #096dd9; }}
</style>
</head>
<body>

<h1>NotionMap — Evaluation Report</h1>
<div class="meta">Generated: {now} &nbsp;|&nbsp; Model: {active_model_name()}</div>

<!-- Tarjetas resumen -->
<h2>General Summary</h2>
<div class="cards">{summary_cards}</div>

<h2>Batch Overview</h2>
<p style="font-size:13px;color:#888;margin-bottom:12px">
  Comparative view to quickly identify which questions are ready, which need review, and which dimensions need attention per case.
</p>
<div style="overflow-x:auto">
<table class="batch-table">
  <thead>
    <tr>
      <th>#</th>
      <th>Question</th>
      <th>Status</th>
      <th>Grounding</th>
      <th>Completeness</th>
      <th>Actionability</th>
      <th>Step Distinctness</th>
      <th>Structure</th>
      <th>Quality Signals</th>
    </tr>
  </thead>
  <tbody>{batch_rows}</tbody>
</table>
</div>

<!-- LLM Judge + Tabla Humana -->
<h2>Case Details</h2>
<p style="font-size:13px;color:#888;margin-bottom:12px">
  Per-question drill-down with generated output, sequence analysis, and technical diagnostics needed to explain each case.
</p>
<h3>Roadmap Output + Evaluation Insights</h3>
<div style="overflow-x:auto">
<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Question</th>
      <th>Steps / Expected Sequence</th>
      <th>Sequence Analysis</th>
      <th>Structure</th>
      <th>Step Overlap</th>
      <th>Readiness</th>
    </tr>
  </thead>
  <tbody>{human_rows}</tbody>
</table>
</div>

</body>
</html>'''

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML guardado: {path}")

    if ragas_rows or corpus_section:
        base, ext = os.path.splitext(path)
        technical_path = f"{base}_ragas_corpus{ext or '.html'}"
        technical_html = f'''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NotionMap — RAGAS + Corpus Diagnostics</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f5f5f5; color: #333; padding: 24px; }}
  h1 {{ font-size: 24px; margin-bottom: 4px; }}
  h2 {{ font-size: 18px; margin: 24px 0 12px; color: #444; border-bottom: 2px solid #e8e8e8;
        padding-bottom: 6px; }}
  .meta {{ color: #888; font-size: 13px; margin-bottom: 24px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff;
           border-radius: 8px; overflow: hidden;
           box-shadow: 0 2px 8px rgba(0,0,0,.08); margin-bottom: 24px; }}
  th {{ background: #fafafa; padding: 10px 12px; text-align: left;
        font-size: 13px; border-bottom: 2px solid #e8e8e8; }}
  td {{ padding: 10px 12px; font-size: 13px; border-bottom: 1px solid #f0f0f0;
        vertical-align: top; }}
  tr:hover td {{ background: #fafafa; }}
  details summary::-webkit-details-marker {{ display:none; }}
</style>
</head>
<body>
<h1>NotionMap — RAGAS + Corpus Diagnostics</h1>
<div class="meta">Generated: {now} &nbsp;|&nbsp; Model: {active_model_name()}</div>
<p style="font-size:13px;color:#888;margin-bottom:12px">
  Separate technical report for automatic RAGAS metrics and corpus diagnostics.
</p>
{"<h2>RAGAS — Automatic Metrics</h2><table><thead><tr><th>Question</th><th>Category</th><th>Faithfulness</th><th>Relevancy</th><th>Precision</th><th>Recall</th></tr></thead><tbody>" + ragas_rows + "</tbody></table>" if ragas_rows else ""}
{corpus_section}
</body>
</html>'''
        with open(technical_path, "w", encoding="utf-8") as f:
            f.write(technical_html)
        print(f"  HTML técnico RAGAS/Corpus guardado: {technical_path}")
