"""
LLM-as-a-Judge para los roadmaps. Evalúa cada roadmap con criterios clásicos
(relevancia, completitud, coherencia, precisión, fidelidad), las cuatro
dimensiones MESE (Mapping, Exhaustiveness, Sequence, Experience), un análisis
de secuencia dedicado, validación de estructura y similitud query-respuesta.

Fundamento — principio MECE (Minto, 1987):
  Las cuatro dimensiones se diseñaron bajo el principio MECE
  (Mutually Exclusive, Collectively Exhaustive):
    - Mutuamente excluyentes: cada dimensión mide algo que SOLO ella mide
      (no se solapan). Se fuerza en el prompt del juez.
    - Colectivamente exhaustivas: juntas cubren toda la calidad del roadmap
      (correcto + completo + ordenado + claro), sin dejar nada fuera.
  El acrónimo MESE corresponde a las iniciales de las cuatro dimensiones
  (Mapping-Exhaustiveness-Sequence-Experience); el principio que las sustenta
  es MECE.
"""
from __future__ import annotations

import json
import time
import traceback
from typing import List

from pydantic import BaseModel, Field

from evaluation.config import config, EvalConfig
from evaluation.dataset import EvalSample
from evaluation.rag_adapter import RagAdapter
from evaluation.structure_validator import StructureValidator
from evaluation.similarity_metrics import query_answer_relevance
from src.llm_provider import get_llm, active_model_name


class CriterionScore(BaseModel):
    score:         int = Field(ge=0, le=10, description="Puntuación 0-10")
    justification: str = Field(description="Justificación en 1-2 oraciones")


class ClassicScores(BaseModel):
    """Criterios clásicos independientes del MESE."""
    relevance:          CriterionScore
    completeness:       CriterionScore
    coherence:          CriterionScore
    technical_accuracy: CriterionScore
    context_fidelity:   CriterionScore
    summary:            str = Field(description="Evaluación general en 2-3 oraciones")


class SequenceEval(BaseModel):
    """
    Rúbrica dedicada de secuencia.
    Alimenta la dimensión S del MESE y la tabla de revisión humana.
    """
    score:              int        = Field(ge=0, le=10)
    is_valid:           bool       = Field(description="True si el orden es aceptable")
    out_of_order_steps: List[str]  = Field(description="Labels exactos de pasos fuera de orden")
    suggested_fix:      str        = Field(description="Reordenamiento sugerido o 'Orden correcto'")
    explanation:        str        = Field(description="Análisis de dependencias causales entre pasos")


class MESEScores(BaseModel):
    """
    Cuatro dimensiones MESE diseñadas bajo el principio MECE de Minto:
    Mutually Exclusive (cada una mide algo que SOLO ella mide) y
    Collectively Exhaustive (juntas cubren toda la calidad del roadmap).
    """
    mapping:               int = Field(ge=0, le=10,
        description="M — ¿Es factualmente preciso cada paso vs el contexto? NO evalúes cobertura ni orden ni claridad")
    mapping_justification: str = Field(description="Evidencia específica de errores o aciertos factuales")

    exhaustiveness:               int = Field(ge=0, le=10,
        description="E — ¿Están TODOS los pasos necesarios? ¿Falta algo? NO evalúes precisión ni orden ni claridad")
    exhaustiveness_justification: str = Field(description="Pasos faltantes detectados, o 'Cobertura completa'")

    sequence:               int = Field(ge=0, le=10,
        description="S — ¿El ORDEN respeta dependencias causales? NO evalúes qué hay ni precisión ni claridad")
    sequence_justification: str = Field(description="Dependencias causales violadas o 'Dependencias correctas'")

    experience:               int = Field(ge=0, le=10,
        description="E — ¿Es CLARO y accionable para el usuario? NO evalúes precisión, cobertura ni orden")
    experience_justification: str = Field(description="Pasos ambiguos detectados o 'Todos los pasos son accionables'")

    composite: float = Field(default=0.0, description="Score ponderado (calculado automáticamente)")


_PROMPT_TEMPLATE = """\
Eres un evaluador experto en sistemas RAG y roadmaps técnicos.
Responde ÚNICAMENTE con JSON válido, sin markdown, sin texto adicional.

═══════════════════════════════════
PREGUNTA DEL USUARIO
═══════════════════════════════════
{question}

═══════════════════════════════════
CONSULTA REFINADA USADA PARA RECUPERAR CONTEXTO
═══════════════════════════════════
{refined_question}

═══════════════════════════════════
CONTEXTO RECUPERADO (RAG)
═══════════════════════════════════
{context}

═══════════════════════════════════
ROADMAP GENERADO
═══════════════════════════════════
{roadmap}

═══════════════════════════════════
RESPUESTA ESPERADA (ground truth)
═══════════════════════════════════
{ground_truth}

═══════════════════════════════════
PASOS DEL ROADMAP EN ORDEN
═══════════════════════════════════
{steps_list}

═══════════════════════════════════
INSTRUCCIONES
═══════════════════════════════════

SECCIÓN 1 — CRITERIOS CLÁSICOS (0-10):
  relevance:          ¿Responde directamente la pregunta?
  completeness:       ¿Incluye todos los pasos esenciales?
  coherence:          ¿Los pasos están en orden lógico?
  technical_accuracy: ¿Los detalles técnicos son correctos?
  context_fidelity:   ¿Se basa en el contexto recuperado, que es el mismo contexto usado por el generador?
  Escala: 0-3 deficiente | 4-6 aceptable | 7-8 bueno | 9-10 excelente

SECCIÓN 2 — ANÁLISIS DE SECUENCIA (detallado):
  Regla de dependencia causal: el paso N NO puede requerir el resultado del paso N+1.
  Ejemplos de violación: "testear" antes de "crear", "configurar" antes de "instalar".
  - Lista los pasos fuera de orden por su label EXACTO
  - Si el orden está bien, pon out_of_order_steps: [] y suggested_fix: "Orden correcto"

SECCIÓN 3 — MESE bajo principio MECE (CRÍTICO — cada dimensión es MUTUAMENTE EXCLUSIVA y, en conjunto, COLECTIVAMENTE EXHAUSTIVA):
  MAPPING (M):
    Evalúa SOLO si el CONTENIDO de cada paso es factualmente correcto.
    Pregúntate: ¿dice algo falso o incorrecto según el contexto?
    NO penalices por pasos faltantes, ni por orden, ni por claridad.

  EXHAUSTIVENESS (E):
    Evalúa SOLO si HAY todos los pasos necesarios.
    Pregúntate: ¿falta algún paso que debería estar?
    NO penalices si los pasos existentes tienen errores, están desordenados, o son confusos.

  SEQUENCE (S):
    Evalúa SOLO el ORDEN de los pasos.
    Pregúntate: ¿el paso i depende del paso j que viene después?
    NO penalices si el contenido es incorrecto o si faltan pasos.

  EXPERIENCE (E):
    Evalúa SOLO si el usuario puede ENTENDER Y EJECUTAR cada paso sin confusión.
    Pregúntate: ¿hay pasos vagos como "configurar el sistema" sin especificar cómo?
    NO penalices por precisión factual, cobertura ni orden.

JSON esperado (sin markdown):
{{
  "classic": {{
    "relevance":          {{"score": <0-10>, "justification": "<texto>"}},
    "completeness":       {{"score": <0-10>, "justification": "<texto>"}},
    "coherence":          {{"score": <0-10>, "justification": "<texto>"}},
    "technical_accuracy": {{"score": <0-10>, "justification": "<texto>"}},
    "context_fidelity":   {{"score": <0-10>, "justification": "<texto>"}},
    "summary": "<evaluación general 2-3 frases>"
  }},
  "sequence_eval": {{
    "score": <0-10>,
    "is_valid": <true|false>,
    "out_of_order_steps": ["<label exacto>", ...],
    "suggested_fix": "<orden corregido o 'Orden correcto'>",
    "explanation": "<análisis de dependencias causales>"
  }},
  "mese": {{
    "mapping":                <0-10>,
    "mapping_justification":  "<evidencia factual>",
    "exhaustiveness":                <0-10>,
    "exhaustiveness_justification":  "<pasos faltantes o 'Cobertura completa'>",
    "sequence":                <0-10>,
    "sequence_justification":  "<dependencias violadas o 'Dependencias correctas'>",
    "experience":                <0-10>,
    "experience_justification":  "<pasos ambiguos o 'Todos accionables'>",
    "composite": 0
  }}
}}
"""


class LLMJudgeEvaluator:

    def __init__(self, cfg: EvalConfig = config):
        self.cfg = cfg
        self.llm = get_llm(temperature=cfg.judge_temperature, max_tokens=4096)
        self.structure_validator = StructureValidator()

    def _judge_single(
        self,
        question:     str,
        roadmap_text: str,
        contexts:     List[str],
        ground_truth: str,
        steps_list:   str,
        refined_question: str = "",
    ) -> dict:
        context_text = "\n---\n".join(contexts) if contexts else "Sin contexto."
        prompt = _PROMPT_TEMPLATE.format(
            question=question,
            refined_question=refined_question or "(No disponible)",
            context=context_text[:3500],
            roadmap=roadmap_text[:2500],
            ground_truth=ground_truth,
            steps_list=steps_list,
        )
        raw = self.llm.invoke(prompt).content.strip()

        # Limpiar bloques markdown si los hubiera
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    def _mese_composite(self, mese: dict) -> float:
        w = self.cfg.mese_weights
        return round(
            mese["mapping"]        * w["mapping"] +
            mese["exhaustiveness"] * w["exhaustiveness"] +
            mese["sequence"]       * w["sequence"] +
            mese["experience"]     * w["experience"],
            2,
        )

    @staticmethod
    def _print_verbose(i: int, total: int, sample_result: dict) -> None:
        SEP  = "-" * 60
        SEP2 = "=" * 60
        q    = sample_result["question"]
        cl   = sample_result["classic"]
        mese = sample_result["mese"]
        seq  = sample_result["sequence_eval"]
        ov   = sample_result["overall_score"]
        verd = sample_result["verdict"]
        mv   = sample_result["mese_verdict"]

        PASS_SYM = "[PASS]"
        FAIL_SYM = "[FAIL]"

        def score_label(s):
            if s >= 7:  return f"{s:5.1f}  [OK]"
            if s >= 5:  return f"{s:5.1f}  [WARN]"
            return             f"{s:5.1f}  [LOW]"

        print(f"\n{SEP2}")
        print(f"  MUESTRA {i}/{total}")
        print(f"{SEP2}")
        print(f"  Pregunta : {q}")
        print(f"  Categoria: {sample_result.get('category','')}")

        # Pasos generados
        steps = sample_result.get("steps", [])
        print(f"\n  PASOS GENERADOS ({len(steps)}):")
        for j, s in enumerate(steps, 1):
            print(f"    {j}. {s}")

        # Pasos esperados
        exp = sample_result.get("expected_steps", [])
        if exp:
            print(f"\n  PASOS ESPERADOS ({len(exp)}):")
            for j, s in enumerate(exp, 1):
                print(f"    {j}. {s}")

        # Criterios clásicos
        print(f"\n  {SEP}")
        print(f"  CRITERIOS CLASICOS")
        print(f"  {SEP}")
        classic_labels = {
            "relevance":          "Relevancia       ",
            "completeness":       "Completitud      ",
            "coherence":          "Coherencia       ",
            "technical_accuracy": "Precision Tecnica",
            "context_fidelity":   "Fidelidad Context",
        }
        total_cl = 0
        for key, label in classic_labels.items():
            s   = cl[key]["score"]
            j   = cl[key]["justification"]
            total_cl += s
            print(f"    {label}  {score_label(s)}")
            print(f"      -> {j}")
        avg_cl = total_cl / len(classic_labels)
        print(f"    {'PROMEDIO CLASICO  ':20} {score_label(avg_cl)}")
        print(f"\n  Resumen: {cl.get('summary','')}")

        # Secuencia
        print(f"\n  {SEP}")
        print(f"  ANALISIS DE SECUENCIA")
        print(f"  {SEP}")
        sv = PASS_SYM if seq["is_valid"] else FAIL_SYM
        print(f"    Score          : {score_label(seq['score'])}")
        print(f"    Orden valido   : {sv}")
        oor = seq.get("out_of_order_steps", [])
        if oor:
            print(f"    Pasos fuera de orden:")
            for p in oor:
                print(f"      ! {p}")
            print(f"    Sugerencia: {seq.get('suggested_fix','')}")
        else:
            print(f"    Pasos fuera de orden: ninguno")
        print(f"    Explicacion: {seq.get('explanation','')}")

        # MESE
        print(f"\n  {SEP}")
        print(f"  MESE  (principio MECE: Mutually Exclusive, Collectively Exhaustive)")
        print(f"  {SEP}")
        mese_labels = [
            ("mapping",        "M - Mapping        ", "Precision factual de cada paso vs contexto"),
            ("exhaustiveness",  "E - Exhaustividad  ", "Cobertura: faltan pasos?"),
            ("sequence",       "S - Secuencia      ", "Orden logico y dependencias causales"),
            ("experience",     "E - Experiencia    ", "Claridad y usabilidad"),
        ]
        for key, label, desc in mese_labels:
            s   = mese[key]
            jk  = key + "_justification"
            jt  = mese.get(jk, "")
            print(f"    {label}  {score_label(s)}")
            print(f"      Dimension: {desc}")
            print(f"      Razon    : {jt}")
        print(f"    {'MESE COMPUESTO     ':20}  {mese['composite']:5.2f}  [{mv}]")

        # Veredicto final
        print(f"\n  {SEP2}")
        verd_sym = PASS_SYM if verd == "PASS" else FAIL_SYM
        print(f"  SCORE GENERAL: {ov:.2f}/10  {verd_sym}")
        print(f"{SEP2}\n")

    def evaluate(self, adapter: RagAdapter, samples: List[EvalSample],
                 verbose: bool = False) -> dict:
        per_sample   = []
        classic_keys = ["relevance", "completeness", "coherence", "technical_accuracy", "context_fidelity"]
        mese_keys    = ["mapping", "exhaustiveness", "sequence", "experience"]

        print(f"\n[LLM Judge] {len(samples)} muestras | modelo: {active_model_name()}")
        if verbose:
            print(f"  Modo verbose activado — mostrando scores y justificaciones\n")
        for i, sample in enumerate(samples, 1):
            print(f"  [{i}/{len(samples)}] {sample.question[:65]}...")
            try:
                t0        = time.time()
                result    = adapter.query(sample.question)
                response_time = round(time.time() - t0, 2)
                steps_str = "\n".join(f"  {j+1}. {s}" for j, s in enumerate(adapter.steps_as_list(result["roadmap"])))

                raw = self._judge_single(
                    question=sample.question,
                    roadmap_text=result["answer"],
                    contexts=result["contexts"],
                    ground_truth=sample.ground_truth,
                    steps_list=steps_str,
                    refined_question=result.get("refined_question", ""),
                )

                # Calcular derivados
                raw["mese"]["composite"] = self._mese_composite(raw["mese"])
                overall = sum(raw["classic"][k]["score"] for k in classic_keys) / len(classic_keys)
                verdict       = "PASS" if overall      >= self.cfg.pass_threshold      else "FAIL"
                mese_verdict  = "PASS" if raw["mese"]["composite"] >= self.cfg.mese_pass_threshold else "FAIL"
                seq_valid     = raw["sequence_eval"].get("is_valid", False)

                # Validación de estructura (determinística, sin LLM)
                struct_result = self.structure_validator.validate(result["roadmap"])

                # similitud query-respuesta (metrica automatica)
                similarity = query_answer_relevance(sample.question, result["answer"])

                sample_result = {
                    "question":       sample.question,
                    "query_intent":   result.get("query_intent", {}),
                    "refined_question": result.get("refined_question", ""),
                    "category":       sample.category,
                    "ground_truth":   sample.ground_truth,
                    "answer":         result["answer"],
                    "contexts":       result["contexts"],
                    "retrieval":      result.get("retrieval", {}),
                    "judge_context_strategy": result.get("judge_context_strategy", ""),
                    "roadmap":        result["roadmap"],
                    "steps":          adapter.steps_as_list(result["roadmap"]),
                    "expected_steps": sample.expected_step_order,
                    "classic":        raw["classic"],
                    "sequence_eval":  raw["sequence_eval"],
                    "mese":           raw["mese"],
                    "structure":      struct_result,
                    "similarity":     similarity,
                    "response_time":  response_time,
                    "overall_score":  round(overall, 2),
                    "verdict":        verdict,
                    "mese_verdict":   mese_verdict,
                }
                per_sample.append(sample_result)

                if verbose:
                    self._print_verbose(i, len(samples), sample_result)

            except Exception:
                print(f"  Error en muestra {i}:")
                traceback.print_exc()

        if not per_sample:
            raise RuntimeError("No se generó ningún resultado válido.")

        aggregated     = self._aggregate(per_sample, classic_keys, mese_keys)
        pass_rate      = sum(1 for s in per_sample if s["verdict"] == "PASS") / len(per_sample)
        mese_pass_rate = sum(1 for s in per_sample if s["mese_verdict"] == "PASS") / len(per_sample)
        seq_valid_pct  = sum(1 for s in per_sample if s["sequence_eval"].get("is_valid")) / len(per_sample)

        print(f"  Pass rate: {pass_rate:.0%} | MESE pass: {mese_pass_rate:.0%} | Secuencias OK: {seq_valid_pct:.0%}")
        return {
            "per_sample":     per_sample,
            "aggregated":     aggregated,
            "pass_rate":      round(pass_rate, 4),
            "mese_pass_rate": round(mese_pass_rate, 4),
            "seq_valid_pct":  round(seq_valid_pct, 4),
        }

    @staticmethod
    def _aggregate(per_sample: list, classic_keys: list, mese_keys: list) -> dict:
        n = len(per_sample)

        classic_agg = {
            k: round(sum(s["classic"][k]["score"] for s in per_sample) / n, 2)
            for k in classic_keys
        }
        mese_agg = {
            k: round(sum(s["mese"][k] for s in per_sample) / n, 2)
            for k in mese_keys + ["composite"]
        }
        seq_scores    = [s["sequence_eval"]["score"] for s in per_sample]
        struct_scores = [s["structure"]["score"]     for s in per_sample]
        resp_times    = [s.get("response_time", 0)   for s in per_sample]
        sim_semantic  = [s["similarity"]["semantic"] for s in per_sample if "similarity" in s]
        sim_tfidf     = [s["similarity"]["tfidf"]    for s in per_sample if "similarity" in s]

        return {
            "overall_score": round(sum(s["overall_score"] for s in per_sample) / n, 2),
            "classic":       classic_agg,
            "mese":          mese_agg,
            "sequence": {
                "mean_score": round(sum(seq_scores) / n, 2),
                "valid_pct":  round(sum(1 for s in per_sample if s["sequence_eval"].get("is_valid")) / n, 2),
            },
            "structure": {
                "mean_score": round(sum(struct_scores) / n, 2),
                "pass_rate":  round(sum(1 for s in per_sample if s["structure"]["verdict"] == "PASS") / n, 2),
            },
            "similarity": {
                "mean_semantic": round(sum(sim_semantic) / len(sim_semantic), 4) if sim_semantic else 0,
                "mean_tfidf":    round(sum(sim_tfidf) / len(sim_tfidf), 4)       if sim_tfidf    else 0,
            },
            "response_time": {
                "mean_s": round(sum(resp_times) / n, 2),
                "max_s":  round(max(resp_times), 2),
                "min_s":  round(min(resp_times), 2),
            },
        }


def run_judge(adapter: RagAdapter, samples: List[EvalSample],
              verbose: bool = False) -> dict:
    """Wrapper funcional para runner.py."""
    return LLMJudgeEvaluator().evaluate(adapter, samples, verbose=verbose)
