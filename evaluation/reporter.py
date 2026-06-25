"""Genera las salidas de evaluación: eval_report.json, eval_report.html y human_review.csv."""

import os
import json
import csv
from datetime import datetime

from evaluation.config import config


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
            "N°", "Pregunta", "Categoría",
            "Pasos generados (en orden)",
            "Pasos esperados",
            "Score MESE", "Score Secuencia", "Secuencia válida (IA)",
            "Pasos fuera de orden (IA)", "Sugerencia de reorden (IA)",
            # columnas para el humano
            "Nota Humana Secuencia (0-10)", "Nota Humana General (0-10)",
            "¿Aprobar? (Si/No)", "Comentarios del Revisor",
        ])

        for i, s in enumerate(judge_results.get("per_sample", []), 1):
            seq     = s["sequence_eval"]
            mese    = s["mese"]
            steps   = "\n".join([f"{j+1}. {p}" for j, p in enumerate(s.get("steps", []))])
            exp     = "\n".join([f"{j+1}. {p}" for j, p in enumerate(s.get("expected_steps", []))])
            oor     = "; ".join(seq.get("out_of_order_steps", []))

            writer.writerow([
                i,
                s["question"],
                s.get("category", ""),
                steps,
                exp,
                round(mese["composite"], 2),
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

def _badge(text: str, color: str) -> str:
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:12px;font-size:12px">{text}</span>'

def _bar(score: float, max_score: float = 10) -> str:
    pct = max(0, min(100, (score / max_score) * 100))
    color = _score_color(score)
    return f'''<div style="background:#f0f0f0;border-radius:4px;height:10px;width:100%">
      <div style="background:{color};width:{pct:.0f}%;height:10px;border-radius:4px"></div>
    </div>'''


def save_html(ragas_results: dict, judge_results: dict, path: str,
              corpus_results: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    ragas_agg  = ragas_results.get("aggregated", {})  if ragas_results  else {}
    judge_agg  = judge_results.get("aggregated", {})  if judge_results  else {}
    corpus_agg = corpus_results.get("aggregated", {}) if corpus_results else {}
    pass_rate  = judge_results.get("pass_rate", 0)    if judge_results  else 0
    mese_rate  = judge_results.get("mese_pass_rate", 0) if judge_results else 0

    # tarjetas resumen
    summary_cards = ""
    all_cards = []
    if judge_agg:
        all_cards += [
            ("Score General",   judge_agg.get("overall_score", 0)),
            ("MESE Compuesto",  judge_agg.get("mese", {}).get("composite", 0)),
            ("Secuencia",       judge_agg.get("sequence", {}).get("mean_score", 0)),
            ("Estructura",      judge_agg.get("structure", {}).get("mean_score", 0)),
        ]
    if corpus_results and "overall_corpus_score" in corpus_results:
        all_cards.append(("Corpus",  corpus_results["overall_corpus_score"]))
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
          <td style="text-align:center;color:{_score_color(sc.get("faithfulness",0))}">
            {sc.get("faithfulness",0):.2f}</td>
          <td style="text-align:center;color:{_score_color(sc.get("answer_relevancy",0))}">
            {sc.get("answer_relevancy",0):.2f}</td>
          <td style="text-align:center;color:{_score_color(sc.get("context_precision",0))}">
            {sc.get("context_precision",0):.2f}</td>
          <td style="text-align:center;color:{_score_color(sc.get("context_recall",0))}">
            {sc.get("context_recall",0):.2f}</td>
        </tr>'''

    # tabla humana (LLM Judge + revision)
    human_rows = ""
    for i, s in enumerate((judge_results or {}).get("per_sample", []), 1):
        seq    = s["sequence_eval"]
        mese   = s["mese"]
        struct = s.get("structure", {})
        steps  = "".join(f"<li>{p}</li>" for p in s.get("steps", []))
        exp    = "".join(f"<li>{p}</li>" for p in s.get("expected_steps", []))
        oor    = seq.get("out_of_order_steps", [])
        oor_html = "".join(f'<li style="color:#f5222d">{p}</li>' for p in oor) if oor else "<li style='color:#52c41a'>Ninguno</li>"
        verdict_badge  = _badge(s["verdict"],      "#52c41a" if s["verdict"]=="PASS"      else "#f5222d")
        mese_badge     = _badge(s["mese_verdict"],  "#52c41a" if s["mese_verdict"]=="PASS"  else "#f5222d")
        seq_badge      = _badge("Válida" if seq["is_valid"] else "Inválida",
                                "#52c41a" if seq["is_valid"] else "#f5222d")
        struct_verdict = struct.get("verdict", "N/A")
        struct_badge   = _badge(struct_verdict, "#52c41a" if struct_verdict=="PASS" else "#f5222d")
        struct_violations_html = "".join(
            f'<li style="color:#f5222d;font-size:11px">{v}</li>'
            for v in struct.get("violations", [])
        ) or "<li style='color:#52c41a;font-size:11px'>Sin violaciones</li>"
        resp_time = s.get("response_time", 0)

        human_rows += f'''
        <tr id="row-{i}">
          <td style="text-align:center;font-weight:bold">{i}</td>
          <td style="max-width:200px">
            <strong>{s["question"]}</strong><br>
            {_badge(s.get("category",""), "#722ed1")}<br>
            <small style="color:#999">t={resp_time:.1f}s</small>
          </td>
          <td>
            <details>
              <summary style="cursor:pointer;color:#1890ff">Ver pasos generados ({len(s.get("steps",[]))})</summary>
              <ol style="padding-left:16px;margin:4px 0">{steps}</ol>
            </details>
            <details style="margin-top:4px">
              <summary style="cursor:pointer;color:#52c41a">Ver pasos esperados</summary>
              <ol style="padding-left:16px;margin:4px 0">{exp}</ol>
            </details>
          </td>
          <td>
            <strong>Secuencia</strong> {seq_badge}<br>
            <small>{_bar(seq["score"])} {seq["score"]:.1f}/10</small><br>
            <details style="margin-top:4px">
              <summary style="cursor:pointer;font-size:12px">Pasos fuera de orden</summary>
              <ul style="font-size:12px;padding-left:16px;margin:4px 0">{oor_html}</ul>
              <div style="font-size:12px;color:#666;margin-top:4px">
                <strong>Sugerencia:</strong> {seq.get("suggested_fix","—")}
              </div>
            </details>
          </td>
          <td>
            <table style="font-size:12px;width:100%">
              <tr><td>Mapping</td><td style="color:{_score_color(mese["mapping"])}">{mese["mapping"]:.1f}</td></tr>
              <tr><td>Exhaustividad</td><td style="color:{_score_color(mese["exhaustiveness"])}">{mese["exhaustiveness"]:.1f}</td></tr>
              <tr><td>Secuencia</td><td style="color:{_score_color(mese["sequence"])}">{mese["sequence"]:.1f}</td></tr>
              <tr><td>Experiencia</td><td style="color:{_score_color(mese["experience"])}">{mese["experience"]:.1f}</td></tr>
              <tr style="font-weight:bold;border-top:1px solid #eee">
                <td>MESE</td>
                <td style="color:{_score_color(mese["composite"])}">{mese["composite"]:.1f}</td>
              </tr>
            </table>
            {mese_badge}
          </td>
          <td>
            {struct_badge} {struct.get("score", 0):.1f}/10<br>
            <small style="color:#666">{struct.get("passed",0)}/{struct.get("total_checks",0)} checks</small>
            <details style="margin-top:4px">
              <summary style="cursor:pointer;font-size:12px">Ver violaciones</summary>
              <ul style="padding-left:14px;margin:4px 0">{struct_violations_html}</ul>
            </details>
          </td>
          <td>
            <strong style="color:{_score_color(s["overall_score"])}">{s["overall_score"]:.1f}/10</strong>
            {verdict_badge}<br>
            <small style="color:#666">{s.get("classic",{}).get("summary","")[:120]}...</small>
          </td>
          <td style="background:#fffbe6">
            <label style="font-size:12px;display:block">Nota Secuencia (0-10):</label>
            <input type="number" min="0" max="10" step="0.5"
                   style="width:60px;padding:2px 4px;margin-bottom:4px"
                   id="hs-{i}" placeholder="—">
            <label style="font-size:12px;display:block">Nota General (0-10):</label>
            <input type="number" min="0" max="10" step="0.5"
                   style="width:60px;padding:2px 4px;margin-bottom:4px"
                   id="hg-{i}" placeholder="—">
            <label style="font-size:12px;display:block">¿Aprobar?</label>
            <select id="ha-{i}" style="padding:2px 4px;margin-bottom:4px">
              <option value="">—</option>
              <option value="Si">Sí</option>
              <option value="No">No</option>
            </select>
            <label style="font-size:12px;display:block">Comentarios:</label>
            <textarea id="hc-{i}" rows="2"
                      style="width:100%;font-size:12px;resize:vertical"
                      placeholder="Observaciones..."></textarea>
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
        ) or '<tr><td colspan="2" style="color:#52c41a;font-size:12px">Sin chunks problemáticos</td></tr>'

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
<h2>Corpus Judge — Calidad de la Base de Conocimiento</h2>
<p style="font-size:13px;color:#888;margin-bottom:12px">
  Evaluación de los {corpus_results["n_chunks_total"]} chunks en ChromaDB
  (se evaluaron {corpus_results["n_chunks_evaluated"]} con LLM).
  Inspirado en Rothman (2024): cosine similarity semántica + juez LLM de coherencia y utilidad.
</p>

<div class="cards" style="margin-bottom:16px">
  <div style="background:#fff;border-radius:8px;padding:20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.1);min-width:140px">
    <div style="font-size:32px;font-weight:700;color:{_score_color(cs)}">{cs:.1f}</div>
    <div style="font-size:13px;color:#666;margin-top:4px">Score Corpus</div>
    {_bar(cs)} {_badge(cv_d, "#52c41a" if cv_d=="PASS" else "#f5222d")}
  </div>
  <div style="background:#fff;border-radius:8px;padding:20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.1);min-width:140px">
    <div style="font-size:32px;font-weight:700;color:{_score_color(ca.get("avg_quality",0))}">{ca.get("avg_quality",0):.1f}</div>
    <div style="font-size:13px;color:#666;margin-top:4px">Calidad Media</div>
    {_bar(ca.get("avg_quality",0))}
  </div>
  <div style="background:#fff;border-radius:8px;padding:20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.1);min-width:140px">
    <div style="font-size:32px;font-weight:700;color:{'#52c41a' if (sd.get('diversity_score') or 0)>0.5 else '#fa8c16'}">{(sd.get("diversity_score") or 0):.2f}</div>
    <div style="font-size:13px;color:#666;margin-top:4px">Diversidad Semántica</div>
    {_bar((sd.get("diversity_score") or 0)*10)}
  </div>
  <div style="background:#fff;border-radius:8px;padding:20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.1);min-width:140px">
    <div style="font-size:32px;font-weight:700;color:{'#52c41a' if sd.get('redundant_pairs',0)==0 else '#f5222d'}">{sd.get("redundant_pairs",0)}</div>
    <div style="font-size:13px;color:#666;margin-top:4px">Pares Redundantes</div>
  </div>
</div>

<div style="display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap">
  <div style="background:#fff;border-radius:8px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.08);flex:1;min-width:200px">
    <strong>Métricas Agregadas</strong>
    <table style="margin-top:8px;font-size:13px;width:100%;box-shadow:none">
      <tr><td>Coherencia media</td><td style="text-align:right;color:{_score_color(ca.get("avg_coherencia",0))}">{ca.get("avg_coherencia",0):.2f}/10</td></tr>
      <tr><td>Densidad técnica</td><td style="text-align:right;color:{_score_color(ca.get("avg_densidad_tecnica",0))}">{ca.get("avg_densidad_tecnica",0):.2f}/10</td></tr>
      <tr><td>Utilidad RAG</td><td style="text-align:right;color:{_score_color(ca.get("avg_utilidad_rag",0))}">{ca.get("avg_utilidad_rag",0):.2f}/10</td></tr>
      <tr><td>Longitud media</td><td style="text-align:right">{ca.get("avg_chunk_length",0)} chars</td></tr>
      <tr><td>Chunks con issues</td><td style="text-align:right;color:{'#f5222d' if ca.get('chunks_with_issues',0)>0 else '#52c41a'}">{ca.get("chunks_with_issues",0)}</td></tr>
    </table>
  </div>
  <div style="background:#fff;border-radius:8px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.08);flex:1;min-width:200px">
    <strong>Cobertura Temática</strong>
    <table style="margin-top:8px;font-size:13px;width:100%;box-shadow:none">
      <tr><td>Amplitud</td><td style="text-align:right;color:{_score_color(cv.get("amplitud",5))}">{cv.get("amplitud","N/A")}/10</td></tr>
      <tr><td>Profundidad</td><td style="text-align:right;color:{_score_color(cv.get("profundidad",5))}">{cv.get("profundidad","N/A")}/10</td></tr>
      <tr><td>Coherencia temática</td><td style="text-align:right;color:{_score_color(cv.get("coherencia_tematica",5))}">{cv.get("coherencia_tematica","N/A")}/10</td></tr>
      <tr><td>Temas únicos</td><td style="text-align:right">{cv.get("temas_unicos_estimados","N/A")}</td></tr>
    </table>
    <div style="font-size:12px;color:#666;margin-top:8px;font-style:italic">
      {cv.get("observacion","")}
    </div>
  </div>
</div>

<details>
  <summary style="cursor:pointer;color:#1890ff;margin-bottom:8px">Ver evaluación por chunk</summary>
  <div style="overflow-x:auto">
  <table>
    <thead><tr>
      <th>Chunk (preview)</th><th>Coherencia</th><th>Densidad</th><th>Utilidad</th><th>Avg</th><th>Tema</th>
    </tr></thead>
    <tbody>{chunk_rows}</tbody>
  </table>
  </div>
</details>

{"<h3 style='color:#f5222d;margin-top:16px'>Chunks Problemáticos</h3><table><thead><tr><th>Preview</th><th>Problema detectado</th></tr></thead><tbody>" + prob_rows + "</tbody></table>" if prob else ""}
'''

    html = f'''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NotionMap — Reporte de Evaluación</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f5f5f5; color: #333; padding: 24px; }}
  h1 {{ font-size: 24px; margin-bottom: 4px; }}
  h2 {{ font-size: 18px; margin: 24px 0 12px; color: #444; border-bottom: 2px solid #e8e8e8;
        padding-bottom: 6px; }}
  .meta {{ color: #888; font-size: 13px; margin-bottom: 24px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  .rates {{ display: flex; gap: 16px; margin-bottom: 24px; }}
  .rate-box {{ background:#fff; border-radius:8px; padding:16px 24px;
               box-shadow:0 2px 8px rgba(0,0,0,.08); }}
  table {{ width: 100%; border-collapse: collapse; background: #fff;
           border-radius: 8px; overflow: hidden;
           box-shadow: 0 2px 8px rgba(0,0,0,.08); margin-bottom: 24px; }}
  th {{ background: #fafafa; padding: 10px 12px; text-align: left;
        font-size: 13px; border-bottom: 2px solid #e8e8e8; }}
  td {{ padding: 10px 12px; font-size: 13px; border-bottom: 1px solid #f0f0f0;
        vertical-align: top; }}
  tr:hover td {{ background: #fafafa; }}
  details summary::-webkit-details-marker {{ display:none; }}
  .export-btn {{ background: #1890ff; color: #fff; border: none; padding: 8px 20px;
                 border-radius: 6px; cursor: pointer; font-size: 14px; margin-top: 16px; }}
  .export-btn:hover {{ background: #096dd9; }}
</style>
</head>
<body>

<h1>NotionMap — Reporte de Evaluación</h1>
<div class="meta">Generado: {now} &nbsp;|&nbsp; Modelo: {config.bedrock_model_id}</div>

<!-- Tarjetas resumen -->
<h2>Resumen General</h2>
<div class="cards">{summary_cards}</div>

<div class="rates">
  <div class="rate-box">
    <div style="font-size:22px;font-weight:700;color:{'#52c41a' if pass_rate>=0.7 else '#f5222d'}">
      {pass_rate:.0%}</div>
    <div style="font-size:13px;color:#666">Pass Rate (Judge)</div>
  </div>
  <div class="rate-box">
    <div style="font-size:22px;font-weight:700;color:{'#52c41a' if mese_rate>=0.7 else '#f5222d'}">
      {mese_rate:.0%}</div>
    <div style="font-size:13px;color:#666">MESE Pass Rate</div>
  </div>
  <div class="rate-box">
    <div style="font-size:22px;font-weight:700;color:{'#52c41a' if judge_agg.get('sequence',{}).get('valid_pct',0)>=0.7 else '#f5222d'}">
      {judge_agg.get("sequence",{}).get("valid_pct",0):.0%}</div>
    <div style="font-size:13px;color:#666">Secuencias Válidas</div>
  </div>
</div>

<!-- RAGAS -->
{"<h2>RAGAS — Métricas Automáticas</h2><table><thead><tr><th>Pregunta</th><th>Categoría</th><th>Faithfulness</th><th>Relevancy</th><th>Precision</th><th>Recall</th></tr></thead><tbody>" + ragas_rows + "</tbody></table>" if ragas_rows else ""}

<!-- LLM Judge + Tabla Humana -->
<h2>LLM Judge + Revisión Humana</h2>
<p style="font-size:13px;color:#888;margin-bottom:12px">
  Completa las columnas amarillas y exporta con el botón.
</p>
<div style="overflow-x:auto">
<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Pregunta</th>
      <th>Pasos / Secuencia esperada</th>
      <th>Análisis de Secuencia</th>
      <th>MESE</th>
      <th>Estructura</th>
      <th>Score General</th>
      <th style="background:#fffbe6">Revisión Humana</th>
    </tr>
  </thead>
  <tbody>{human_rows}</tbody>
</table>
</div>

<button class="export-btn" onclick="exportCSV()">Exportar revisión humana (CSV)</button>

{corpus_section}

<script>
function exportCSV() {{
  const n = document.querySelectorAll("tr[id^='row-']").length;
  let csv = "N°,Nota Secuencia,Nota General,Aprobar,Comentarios\\n";
  for (let i = 1; i <= n; i++) {{
    const hs = document.getElementById("hs-"+i)?.value || "";
    const hg = document.getElementById("hg-"+i)?.value || "";
    const ha = document.getElementById("ha-"+i)?.value || "";
    const hc = (document.getElementById("hc-"+i)?.value || "").replace(/"/g,'""');
    csv += `${{i}},${{hs}},${{hg}},${{ha}},"${{hc}}"\\n`;
  }}
  const blob = new Blob([csv], {{type:"text/csv"}});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = "human_review_filled.csv"; a.click();
}}
</script>

</body>
</html>'''

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML guardado: {path}")
