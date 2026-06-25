"""
Evalúa la calidad del knowledge base: calidad de cada chunk (LLM), diversidad
semántica entre chunks (cosine similarity) y cobertura temática. Ref: Rothman (2024).
"""

from __future__ import annotations

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from evaluation.config import config
from src.llm_provider import get_llm, active_model_name


_CHUNK_EVAL_PROMPT = """\
Eres un experto en sistemas RAG. Evalúa el siguiente fragmento de texto
que forma parte de una base de conocimiento técnica.

FRAGMENTO:
\"\"\"
{chunk}
\"\"\"

Evalúa en 3 dimensiones (0-10 cada una):

1. coherencia: ¿Es el fragmento un texto coherente y completo? ¿Tiene sentido por sí solo?
   0 = texto roto/ilegible, 10 = unidad de información completa y clara

2. densidad_tecnica: ¿Qué tan rico en información técnica útil es?
   0 = genérico sin valor, 10 = lleno de conceptos, datos, comandos o procedimientos específicos

3. utilidad_rag: ¿Qué tan útil sería este fragmento para responder preguntas técnicas?
   0 = inútil, 10 = altamente informativo para construir respuestas precisas

Responde SOLO con JSON válido, sin markdown:
{{
  "coherencia": <0-10>,
  "densidad_tecnica": <0-10>,
  "utilidad_rag": <0-10>,
  "tema_principal": "<tema en 5 palabras máximo>",
  "problema_detectado": "<si hay un problema de calidad, describelo en 1 oración; si no, escribe 'ninguno'>"
}}
"""

_COVERAGE_PROMPT = """\
Eres un experto en sistemas RAG. Tienes los siguientes temas principales
extraídos de los fragmentos de la base de conocimiento:

{topics}

Evalúa la cobertura temática (0-10):

  amplitud: ¿Qué tan amplio es el rango de temas? ¿Hay diversidad?
  profundidad: ¿Los temas tienen suficiente detalle técnico en conjunto?
  coherencia_tematica: ¿Los temas están relacionados entre sí o son completamente dispares?

Responde SOLO con JSON válido:
{{
  "amplitud": <0-10>,
  "profundidad": <0-10>,
  "coherencia_tematica": <0-10>,
  "temas_unicos_estimados": <numero entero>,
  "observacion": "<recomendación sobre el corpus en 2 oraciones>"
}}
"""


class CorpusJudge:
    """Evalúa la calidad del knowledge base almacenado en ChromaDB."""

    def __init__(self):
        self.llm = get_llm(temperature=0.0, max_tokens=512)

    def _eval_chunk(self, chunk: str, idx: int, total: int) -> dict:
        print(f"    [{idx}/{total}] evaluando chunk ({len(chunk)} chars)...")
        try:
            raw = self.llm.invoke(
                _CHUNK_EVAL_PROMPT.format(chunk=chunk[:1200])
            ).content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw.strip())
            result["chunk_preview"] = chunk[:120].replace("\n", " ")
            result["chunk_length"]  = len(chunk)
            result["avg_score"] = round(
                (result.get("coherencia", 0) +
                 result.get("densidad_tecnica", 0) +
                 result.get("utilidad_rag", 0)) / 3, 2
            )
            return result
        except Exception as e:
            return {
                "coherencia": 0, "densidad_tecnica": 0, "utilidad_rag": 0,
                "avg_score": 0,
                "tema_principal": "error",
                "problema_detectado": str(e)[:100],
                "chunk_preview": chunk[:120],
                "chunk_length": len(chunk),
            }

    @staticmethod
    def _semantic_diversity(chunks: list[str]) -> dict:
        """Diversidad por cosine similarity entre chunks (0=idénticos, 1=distintos)."""
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            model     = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(chunks, show_progress_bar=False)
            sim_matrix = cosine_similarity(embeddings)

            n = len(chunks)
            pairs_high = []
            similarities = []
            for i in range(n):
                for j in range(i + 1, n):
                    sim = float(sim_matrix[i][j])
                    similarities.append(sim)
                    if sim > 0.85:
                        pairs_high.append({
                            "chunk_a": chunks[i][:80],
                            "chunk_b": chunks[j][:80],
                            "similarity": round(sim, 4),
                        })

            avg_sim    = float(np.mean(similarities)) if similarities else 0.0
            diversity  = round(1.0 - avg_sim, 4)

            return {
                "diversity_score":   diversity,
                "avg_similarity":    round(avg_sim, 4),
                "redundant_pairs":   len(pairs_high),
                "redundant_examples": pairs_high[:3],
                "interpretation": (
                    "Alta diversidad — buen corpus"  if diversity > 0.6 else
                    "Diversidad media"               if diversity > 0.4 else
                    "Baja diversidad — muchos chunks similares"
                ),
            }
        except ImportError:
            return {
                "diversity_score": None,
                "error": "sentence-transformers no disponible",
            }

    def _eval_coverage(self, topics: list[str]) -> dict:
        topics_str = "\n".join(f"  - {t}" for t in topics)
        try:
            raw = self.llm.invoke(
                _COVERAGE_PROMPT.format(topics=topics_str)
            ).content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            return {"error": str(e)}

    def evaluate(self, chunks: list[str], max_chunks: int = 20) -> dict:
        """Evalúa la calidad del knowledge base sobre hasta max_chunks fragmentos."""
        if not chunks:
            return {"error": "No hay chunks en el vectorstore."}

        sample = chunks[:max_chunks]
        total  = len(sample)
        print(f"\n  [Corpus Judge] Evaluando {total} chunks con {active_model_name()}...")

        chunk_results = [self._eval_chunk(c, i + 1, total) for i, c in enumerate(sample)]

        print("  [Corpus Judge] Calculando diversidad semántica...")
        diversity = self._semantic_diversity(sample)

        topics = [r.get("tema_principal", "desconocido") for r in chunk_results]
        print("  [Corpus Judge] Evaluando cobertura temática...")
        coverage = self._eval_coverage(topics)

        scores_coh  = [r.get("coherencia", 0)       for r in chunk_results]
        scores_den  = [r.get("densidad_tecnica", 0)  for r in chunk_results]
        scores_rag  = [r.get("utilidad_rag", 0)      for r in chunk_results]
        scores_avg  = [r.get("avg_score", 0)          for r in chunk_results]
        lengths     = [r.get("chunk_length", 0)       for r in chunk_results]
        problems    = [r for r in chunk_results if r.get("problema_detectado", "ninguno") != "ninguno"]

        avg_quality  = round(sum(scores_avg) / len(scores_avg), 2) if scores_avg else 0
        avg_length   = round(sum(lengths)    / len(lengths),    0) if lengths    else 0
        div_score    = diversity.get("diversity_score") or 0

        corpus_score = round(
            avg_quality       * 0.50 +
            div_score * 10    * 0.25 +
            coverage.get("profundidad", 5) * 0.15 +
            coverage.get("amplitud",    5) * 0.10,
            2
        )

        print(f"  [Corpus Judge] Score corpus: {corpus_score}/10\n")

        return {
            "overall_corpus_score": corpus_score,
            "verdict":  "PASS" if corpus_score >= 6.0 else "FAIL",
            "n_chunks_evaluated": total,
            "n_chunks_total":     len(chunks),
            "aggregated": {
                "avg_quality":         avg_quality,
                "avg_coherencia":      round(sum(scores_coh) / len(scores_coh), 2),
                "avg_densidad_tecnica":round(sum(scores_den) / len(scores_den), 2),
                "avg_utilidad_rag":    round(sum(scores_rag) / len(scores_rag), 2),
                "avg_chunk_length":    int(avg_length),
                "chunks_with_issues":  len(problems),
            },
            "semantic_diversity": diversity,
            "coverage":           coverage,
            "per_chunk":          chunk_results,
            "problematic_chunks": [
                {"preview": r["chunk_preview"], "problem": r["problema_detectado"]}
                for r in problems
            ][:5],
        }


def run_corpus_judge(vectorstore_path: str = None, max_chunks: int = 20) -> dict:
    """Carga los chunks desde ChromaDB y los evalúa."""
    import os
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from dotenv import load_dotenv

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(base_dir, ".env"))

    db_path = vectorstore_path or os.path.join(base_dir, "vectorstore", "chroma_db")

    print(f"  Cargando chunks desde: {db_path}")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)

    # Obtener todos los documentos del vectorstore
    all_docs = db.get()
    chunks   = all_docs.get("documents", [])
    print(f"  Total chunks en vectorstore: {len(chunks)}")

    judge   = CorpusJudge()
    results = judge.evaluate(chunks, max_chunks=max_chunks)
    return results
