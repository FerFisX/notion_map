import os
import json
import time
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.llm_provider import get_generation_llm, active_model_name

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DB_PATH = os.path.join(BASE_DIR, "vectorstore", "chroma_db")

# Retrieval / re-ranking. RERANK_METHOD: mmr | crossencoder | none
RERANK_METHOD = os.getenv("RERANK_METHOD", "mmr").lower().strip()
POOL_SIZE     = int(os.getenv("RETRIEVAL_POOL_SIZE", "10"))
TOP_N         = int(os.getenv("RETRIEVAL_TOP_N", "5"))

# Fallback a búsqueda web cuando el corpus local no cubre la consulta.
# Efímero: lo encontrado se usa solo para esa consulta, no se guarda en ChromaDB.
# Apagado por defecto para que la evaluación siga siendo reproducible.
WEB_FALLBACK     = os.getenv("WEB_FALLBACK", "false").lower().strip() in ("true", "1", "yes")
WEB_FALLBACK_MIN = float(os.getenv("WEB_FALLBACK_THRESHOLD", "0.25"))
# Banda híbrida: si el mejor score del corpus cae entre el umbral de fallback y
# el híbrido, el corpus cubre parcialmente y se complementa con búsqueda web.
#   score >= WEB_HYBRID_MIN                  -> solo corpus
#   WEB_FALLBACK_MIN <= score < WEB_HYBRID_MIN -> híbrido (corpus + web)
#   score < WEB_FALLBACK_MIN                 -> mayormente web
WEB_HYBRID_MIN = float(os.getenv("WEB_HYBRID_THRESHOLD", "0.45"))

# Query refinement experiments.
# Apagar esto permite reproducir el refinamiento genérico anterior.
QUERY_INTENT_ENABLED = os.getenv("QUERY_INTENT_ENABLED", "true").lower().strip() in ("true", "1", "yes")
QUERY_PREPROCESSING_MODE = os.getenv(
    "QUERY_PREPROCESSING_MODE", "sequential"
).lower().strip()
if QUERY_PREPROCESSING_MODE not in {"sequential", "merged"}:
    raise ValueError("QUERY_PREPROCESSING_MODE must be 'sequential' or 'merged'")
GENERATION_PROMPT_VERSION = "concise-v2"


class RoadmapStep(BaseModel):
    id: str = Field(description="Unique short ID, for example 'step_1'.")
    label: str = Field(
        description="Concise action verb plus a specific object."
    )
    description: str = Field(
        description="One or two concise sentences explaining what to do, how, and the expected outcome."
    )
    type: str = Field(description="One of: 'inicio', 'proceso', 'decision', 'fin'.")
    key_points: List[str] = Field(
        description="One to three concrete commands, settings, cautions, tools, or verifiable outcomes."
    )

class Roadmap(BaseModel):
    title: str = Field(description="Concise title for the complete roadmap.")
    steps: List[RoadmapStep] = Field(
        description="Smallest complete ordered set of steps appropriate to the user's objective."
    )

# --- MOTOR RAG ---
class RagEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = Chroma(persist_directory=DB_PATH, embedding_function=self.embeddings)

        print(f"  Modelo: {active_model_name()}")
        self.llm = get_generation_llm(temperature=0.1, max_tokens=4096)
        self.parser = JsonOutputParser(pydantic_object=Roadmap)
        self._last_build_diagnostics = {}

    def classify_query_intent(self, raw_query: str) -> dict:
        """
        Clasifica la intención de la consulta para orientar retrieval y generación.

        La decisión queda en manos del LLM usando categorías generales; no depende
        de reglas rígidas por palabras clave.
        """
        prompt = f"""\
        Eres un analista de intención para un sistema RAG que genera roadmaps técnicos.
        Clasifica la consulta del usuario y define cómo debe transformarse para construir
        un roadmap útil.

        Categorías disponibles:
        - conceptual_learning: el usuario pregunta qué es/qué son/para qué sirve algo y necesita convertirlo en un proceso de aprendizaje aplicable.
        - implementation: el usuario quiere construir, configurar, implementar o aplicar algo.
        - troubleshooting: el usuario tiene un error, bloqueo o comportamiento inesperado.
        - comparison: el usuario quiere comparar alternativas, enfoques o herramientas.
        - optimization: el usuario quiere mejorar rendimiento, calidad, costos o precisión.
        - exploratory: la intención es amplia o ambigua y necesita exploración ordenada.

        Responde SOLO con JSON válido:
        {{
        "intent": "<una categoría>",
        "roadmap_goal": "<objetivo accionable del roadmap en el idioma de la consulta>",
        "retrieval_focus": "<qué conocimiento debe buscarse en la base para responder bien>",
        "generation_guidance": "<cómo debe comportarse el generador del roadmap>",
        "confidence": <número entre 0 y 1>
        }}

        Consulta del usuario:
        {raw_query}
        """
        default = {
            "intent": "exploratory",
            "roadmap_goal": raw_query,
            "retrieval_focus": raw_query,
            "generation_guidance": "Construye un roadmap técnico, ordenado y accionable.",
            "confidence": 0.0,
        }
        try:
            raw = self.llm.invoke(prompt).content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            if not raw.startswith("{"):
                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end > start:
                    raw = raw[start:end + 1]
            intent = json.loads(raw)
            result = {**default, **intent}
            result["confidence"] = float(result.get("confidence") or 0.0)
        except Exception as e:
            print(f"  [Query Intent] No se pudo clasificar ({e}); usando intención exploratoria.")
            result = default

        print(
            "  [Query Intent]\n"
            f"    Intent   : {result.get('intent')}\n"
            f"    Objetivo : {result.get('roadmap_goal')}\n"
            f"    Confianza: {result.get('confidence'):.2f}"
        )
        return result

    def analyze_and_rewrite_query(self, raw_query: str) -> tuple[dict, str]:
        """Classify intent and produce the retrieval query in one LLM call."""
        prompt = f"""\
        Eres un analista de consultas para un sistema RAG que genera roadmaps técnicos.
        En una sola operación, clasifica la intención y crea una consulta refinada para
        recuperar el conocimiento necesario y orientar un roadmap útil.

        Categorías disponibles:
        - conceptual_learning: aprender, practicar y aplicar un concepto.
        - implementation: construir, configurar, implementar o aplicar algo.
        - troubleshooting: diagnosticar un error o comportamiento inesperado.
        - comparison: comparar alternativas mediante criterios y casos de uso.
        - optimization: mejorar rendimiento, calidad, costo o precisión.
        - exploratory: explorar de forma ordenada una intención amplia o ambigua.

        Reglas para refined_query:
        - Hazla específica, técnica y adecuada para búsqueda semántica.
        - Mantén el idioma original y el objetivo real del usuario.
        - Conserva literalmente versiones, fechas, IDs de modelos, APIs y parámetros.
        - No inventes expansiones ni significados para nombres propios o identificadores.
        - Añade solo términos que ayuden a recuperar evidencia relevante.

        Responde SOLO con JSON válido:
        {{
          "intent": "<una categoría>",
          "roadmap_goal": "<objetivo accionable>",
          "retrieval_focus": "<conocimiento que debe recuperarse>",
          "generation_guidance": "<dirección que debe seguir el roadmap>",
          "refined_query": "<consulta técnica refinada>",
          "confidence": <número entre 0 y 1>
        }}

        Consulta original:
        {raw_query}
        """
        default = {
            "intent": "exploratory",
            "roadmap_goal": raw_query,
            "retrieval_focus": raw_query,
            "generation_guidance": "Construye un roadmap técnico, ordenado y accionable.",
            "confidence": 0.0,
        }
        refined_query = raw_query
        try:
            raw = str(self.llm.invoke(prompt).content).strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            if not raw.startswith("{"):
                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end > start:
                    raw = raw[start:end + 1]
            payload = json.loads(raw)
            refined_query = str(payload.pop("refined_query", "")).strip() or raw_query
            result = {**default, **payload}
            result["confidence"] = max(
                0.0, min(1.0, float(result.get("confidence") or 0.0))
            )
        except Exception as exc:
            print(
                "  [Query Analysis/Merged] No se pudo analizar "
                f"({exc}); usando consulta original."
            )
            result = default

        print(
            "  [Query Analysis/Merged]\n"
            f"    Intent   : {result.get('intent')}\n"
            f"    Objetivo : {result.get('roadmap_goal')}\n"
            f"    Original : {raw_query}\n"
            f"    Mejorada : {refined_query}\n"
            f"    Confianza: {result.get('confidence'):.2f}"
        )
        return result, refined_query

    def rewrite_query(self, raw_query: str, query_intent: dict = None) -> str:
        """Expande la consulta del usuario para mejorar el retrieval y el roadmap."""
        if QUERY_INTENT_ENABLED:
            query_intent = query_intent or self.classify_query_intent(raw_query)
            intent_section = (
                "INTENCIÓN CLASIFICADA:\n"
                f"- Tipo: {query_intent.get('intent', 'exploratory')}\n"
                f"- Objetivo del roadmap: {query_intent.get('roadmap_goal', raw_query)}\n"
                f"- Foco de retrieval: {query_intent.get('retrieval_focus', raw_query)}\n"
                f"- Guía de generación: {query_intent.get('generation_guidance', '')}\n\n"
            )
            intent_rules = (
                "- Si la intención es conceptual_learning, orienta la consulta hacia aprender, practicar y aplicar el concepto\n"
                "- Si la intención es implementation, orienta la consulta hacia pasos de construcción o configuración\n"
                "- Si la intención es troubleshooting, orienta la consulta hacia diagnóstico, causas y validaciones\n"
                "- Si la intención es comparison, orienta la consulta hacia criterios de decisión, diferencias y casos de uso\n"
            )
        else:
            query_intent = query_intent or {"intent": "disabled", "confidence": 0.0}
            intent_section = ""
            intent_rules = ""

        prompt = (
            "Eres un experto en sistemas RAG técnicos. "
            "Reescribe la siguiente consulta del usuario para hacerla más específica, técnica "
            "y adecuada para búsqueda semántica en una base de conocimiento.\n\n"
            f"{intent_section}"
            "REGLAS:\n"
            "- Añade terminología técnica relevante del dominio\n"
            "- Especifica el objetivo final que el usuario quiere lograr\n"
            "- Expande siglas o términos ambiguos\n"
            f"{intent_rules}"
            "- Mantén el idioma original\n"
            "- Responde SOLO con la consulta mejorada, sin explicaciones ni prefijos\n\n"
            f"Consulta original: {raw_query}\n\n"
            "Consulta mejorada:"
        )
        rewritten = self.llm.invoke(prompt).content.strip()
        # Limpiar prefijos que el modelo pueda agregar
        for prefix in ("Consulta mejorada:", "Aquí", "La consulta"):
            if rewritten.startswith(prefix):
                rewritten = rewritten[len(prefix):].strip()
        print(f"  [Query Rewriting]\n    Original : {raw_query}\n    Mejorada : {rewritten}")
        return rewritten

    _cross_encoder = None

    def _get_cross_encoder(self):
        """Carga perezosa del cross-encoder. Devuelve None si no está disponible."""
        if RagEngine._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                RagEngine._cross_encoder = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
            except Exception as e:
                print(f"  [Re-rank] Cross-encoder no disponible ({e}); usando MMR.")
                RagEngine._cross_encoder = False
        return RagEngine._cross_encoder or None

    def retrieve_contexts_scored(
        self,
        query:     str,
        pool_size: int  = POOL_SIZE,
        top_n:     int  = TOP_N,
        method:    str  = None,
    ) -> dict:
        """
        Recupera un pool de candidatos, los re-rankea y devuelve los top_n.
        Devuelve {method, pool, selected} con scores para inspección.
        """
        method = (method or RERANK_METHOD).lower().strip()

        # Pool inicial con scores del vector store
        scored = self.vector_db.similarity_search_with_relevance_scores(query, k=pool_size)
        pool = []
        for i, (doc, score) in enumerate(scored, 1):
            pool.append({
                "text":         doc.page_content,
                "vector_score": round(float(score), 4),
                "init_rank":    i,
                "source":       doc.metadata.get("source", "?"),
            })

        if not pool:
            return {"method": method, "pool": [], "selected": []}

        if method == "none":
            ranked = pool

        elif method == "crossencoder" and self._get_cross_encoder():
            ce     = self._get_cross_encoder()
            scores = ce.predict([(query, c["text"]) for c in pool])
            for c, s in zip(pool, scores):
                c["rerank_score"] = round(float(s), 4)
            ranked = sorted(pool, key=lambda c: c["rerank_score"], reverse=True)

        else:  # "mmr" — relevancia + diversidad, sin descargas adicionales
            method   = "mmr"
            mmr_docs = self.vector_db.max_marginal_relevance_search(
                query, k=min(top_n, len(pool)), fetch_k=pool_size
            )
            by_text = {c["text"]: c for c in pool}
            ranked  = []
            for doc in mmr_docs:
                ranked.append(by_text.get(doc.page_content, {
                    "text":         doc.page_content,
                    "vector_score": None,
                    "init_rank":    None,
                    "source":       doc.metadata.get("source", "?"),
                }))
            for c in pool:
                if c["text"] not in {r["text"] for r in ranked}:
                    ranked.append(c)

        selected = []
        for r, c in enumerate(ranked[:top_n], 1):
            item = dict(c)
            item["final_rank"]  = r
            item["final_score"] = c.get("rerank_score", c.get("vector_score"))
            selected.append(item)

        return {"method": method, "pool": pool, "selected": selected}

    def retrieve_contexts(self, query: str) -> list:
        """Returns raw text of retrieved chunks (re-ranked). Used by evaluators."""
        result = self.retrieve_contexts_scored(query)
        return [c["text"] for c in result["selected"]]

    def get_contexts_with_sources(self, refined_query: str) -> dict:
        """
        Recupera contexto y decide la fuente según el mejor score del corpus:
          - score >= WEB_HYBRID_MIN              -> solo corpus
          - WEB_FALLBACK_MIN <= score < HYBRID   -> híbrido (corpus + web)
          - score < WEB_FALLBACK_MIN             -> mayormente web
        Devuelve los contextos separados por origen para poder atribuir cada nodo.
        """
        retrieval_started = time.perf_counter()
        retrieval = self.retrieve_contexts_scored(refined_query)
        retrieval_seconds = round(time.perf_counter() - retrieval_started, 4)
        corpus_contexts = [c["text"] for c in retrieval["selected"]]
        scores = [c.get("vector_score") for c in retrieval.get("pool", []) if c.get("vector_score") is not None]
        best   = max(scores) if scores else 0.0

        web_contexts = []
        web_search_seconds = 0.0
        if best >= WEB_HYBRID_MIN:
            mode = "corpus"
        elif WEB_FALLBACK:
            mode = "hybrid" if best >= WEB_FALLBACK_MIN else "web"
            print(f"  [Web Fallback/{mode}] mejor score del corpus={best:.3f}. Complementando con internet...")
            from src.web_search import search_web
            web_started = time.perf_counter()
            web_contexts = search_web(refined_query, max_results=4)
            web_search_seconds = round(time.perf_counter() - web_started, 4)
            if not web_contexts:
                mode = "corpus"  # la búsqueda falló: seguimos solo con corpus
        else:
            mode = "corpus"  # fallback desactivado

        return {
            "contexts":        web_contexts + corpus_contexts,  # web primero
            "corpus_contexts": corpus_contexts,
            "web_contexts":    web_contexts,
            "best_score":      round(best, 3),
            "mode":            mode,
            "timings": {
                "retrieval_s": retrieval_seconds,
                "web_search_s": web_search_seconds,
            },
        }

    def _cosine(self, a, b) -> float:
        import numpy as np
        a, b = np.array(a), np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(a.dot(b) / denom) if denom else 0.0

    def attribute_step_sources(self, roadmap: dict, corpus_contexts: list, web_contexts: list) -> dict:
        """
        Marca cada paso con su fuente predominante ('corpus' o 'web') comparando
        la similitud semántica del paso contra cada conjunto de contextos.
        Devuelve los porcentajes globales.
        """
        steps = roadmap.get("steps", [])
        if not steps:
            return {"corpus_pct": 0, "web_pct": 0}

        # Casos triviales: una sola fuente
        if not web_contexts:
            for s in steps:
                s["source"] = "corpus"
            return {"corpus_pct": 100, "web_pct": 0}
        if not corpus_contexts:
            for s in steps:
                s["source"] = "web"
            return {"corpus_pct": 0, "web_pct": 100}

        corpus_emb = self.embeddings.embed_documents(corpus_contexts)
        web_emb    = self.embeddings.embed_documents(web_contexts)

        corpus_n = web_n = 0
        for s in steps:
            text = f"{s.get('label','')}. {s.get('description','')}"
            e = self.embeddings.embed_query(text)
            best_corpus = max(self._cosine(e, c) for c in corpus_emb)
            best_web    = max(self._cosine(e, w) for w in web_emb)
            src = "web" if best_web > best_corpus else "corpus"
            s["source"] = src
            if src == "corpus":
                corpus_n += 1
            else:
                web_n += 1

        total = len(steps)
        return {
            "corpus_pct": round(corpus_n / total * 100),
            "web_pct":    round(web_n / total * 100),
        }

    def generate_roadmap_with_trace(self, query: str) -> dict:
        """
        Genera el roadmap y devuelve la traza completa del RAG.

        Esta traza permite que evaluadores externos (LLM Judge, RAGAS, reportes)
        usen exactamente la misma consulta refinada y los mismos contextos que
        vio el generador, evitando evaluar contra un retrieval distinto.
        """
        total_started = time.perf_counter()
        stage_timings = {
            "query_preprocessing_s": 0.0,
            "intent_classification_s": 0.0,
            "query_refinement_s": 0.0,
            "retrieval_s": 0.0,
            "web_search_s": 0.0,
            "roadmap_generation_s": 0.0,
            "source_attribution_s": 0.0,
        }
        try:
            if QUERY_INTENT_ENABLED and QUERY_PREPROCESSING_MODE == "merged":
                preprocessing_started = time.perf_counter()
                query_intent, refined_query = self.analyze_and_rewrite_query(query)
                stage_timings["query_preprocessing_s"] = round(
                    time.perf_counter() - preprocessing_started, 4
                )
            else:
                intent_started = time.perf_counter()
                query_intent = (
                    self.classify_query_intent(query)
                    if QUERY_INTENT_ENABLED
                    else {"intent": "disabled", "confidence": 0.0}
                )
                stage_timings["intent_classification_s"] = round(
                    time.perf_counter() - intent_started, 4
                )
                refinement_started = time.perf_counter()
                refined_query = self.rewrite_query(query, query_intent=query_intent)
                stage_timings["query_refinement_s"] = round(
                    time.perf_counter() - refinement_started, 4
                )
            print(f"  [Retrieval] Buscando con consulta refinada...")
            src     = self.get_contexts_with_sources(refined_query)
            stage_timings.update(src.get("timings", {}))
            generation_started = time.perf_counter()
            roadmap = self.build_roadmap(query, refined_query, src["contexts"], query_intent=query_intent)
            stage_timings["roadmap_generation_s"] = round(
                time.perf_counter() - generation_started, 4
            )

            # Atribuir fuente a cada nodo y calcular porcentajes globales
            attribution_started = time.perf_counter()
            pct = self.attribute_step_sources(roadmap, src["corpus_contexts"], src["web_contexts"])
            stage_timings["source_attribution_s"] = round(
                time.perf_counter() - attribution_started, 4
            )
            roadmap["sources"] = {
                "corpus_pct":      pct["corpus_pct"],
                "web_pct":         pct["web_pct"],
                "mode":            src["mode"],
                "best_score":      src["best_score"],
                "n_corpus_chunks": len(src["corpus_contexts"]),
                "n_web_chunks":    len(src["web_contexts"]),
            }
            print(f"  [Fuentes] {pct['corpus_pct']}% corpus / {pct['web_pct']}% web  (modo: {src['mode']})")
            generation_trace = {
                "timings": {
                    **stage_timings,
                    "total_generation_s": round(
                        time.perf_counter() - total_started, 4
                    ),
                },
                "provider": os.getenv("LLM_PROVIDER", "bedrock").lower().strip(),
                "model": active_model_name(),
                "generation_reasoning_mode": os.getenv(
                    "LLM_GENERATION_REASONING_MODE", "provider_default"
                ).lower().strip(),
                "query_preprocessing_mode": QUERY_PREPROCESSING_MODE,
                "query_intent_enabled": QUERY_INTENT_ENABLED,
                "rerank_method": RERANK_METHOD,
                "retrieval_pool_size": POOL_SIZE,
                "retrieval_top_n": TOP_N,
                "web_fallback_enabled": WEB_FALLBACK,
                "corpus_context_count": len(src["corpus_contexts"]),
                "web_context_count": len(src["web_contexts"]),
                "total_context_count": len(src["contexts"]),
                "context_characters": sum(len(value) for value in src["contexts"]),
                **self._last_build_diagnostics,
            }
            timing_summary = " | ".join(
                f"{name.removesuffix('_s')}={value:.2f}s"
                for name, value in generation_trace["timings"].items()
            )
            print(f"  [Generation timing] {timing_summary}")
            return {
                "question":            query,
                "query_intent":        query_intent,
                "refined_question":    refined_query,
                "contexts":            src["contexts"],
                "corpus_contexts":     src["corpus_contexts"],
                "web_contexts":        src["web_contexts"],
                "retrieval": {
                    "mode":            src["mode"],
                    "best_score":      src["best_score"],
                    "n_contexts":      len(src["contexts"]),
                    "n_corpus_chunks": len(src["corpus_contexts"]),
                    "n_web_chunks":    len(src["web_contexts"]),
                },
                "generation_trace":   generation_trace,
                "roadmap":             roadmap,
            }

        except Exception as e:
            print(f"Error generando roadmap: {e}")
            roadmap = {
                "title": "Error",
                "steps": [{
                    "id": "err", "label": "Error de Sistema",
                    "description": str(e)[:200], "type": "decision", "key_points": []
                }],
                "sources": {"corpus_pct": 0, "web_pct": 0, "mode": "error"},
            }
            return {
                "question":         query,
                "query_intent":     {"intent": "error", "confidence": 0.0},
                "refined_question": "",
                "contexts":         [],
                "corpus_contexts":  [],
                "web_contexts":     [],
                "retrieval":        {"mode": "error", "best_score": 0, "n_contexts": 0},
                "generation_trace": {
                    "timings": {
                        **stage_timings,
                        "total_generation_s": round(
                            time.perf_counter() - total_started, 4
                        ),
                    },
                    "provider": os.getenv("LLM_PROVIDER", "bedrock").lower().strip(),
                    "model": active_model_name(),
                    "generation_reasoning_mode": os.getenv(
                        "LLM_GENERATION_REASONING_MODE", "provider_default"
                    ).lower().strip(),
                    "query_preprocessing_mode": QUERY_PREPROCESSING_MODE,
                    "error_type": type(e).__name__,
                },
                "roadmap":          roadmap,
            }

    def generate_roadmap(self, query: str):
        """Mantiene compatibilidad: devuelve solo el roadmap."""
        return self.generate_roadmap_with_trace(query)["roadmap"]

    def build_roadmap(self, original_query: str, refined_query: str, contexts: list, query_intent: dict = None):
        """Construye el roadmap con la consulta refinada y los contextos ya recuperados."""
        self._last_build_diagnostics = {}
        try:
            query_intent = query_intent or {
                "intent": "exploratory",
                "roadmap_goal": original_query,
                "generation_guidance": "Construye un roadmap técnico, ordenado y accionable.",
            }
            if QUERY_INTENT_ENABLED and query_intent.get("intent") not in ("disabled", "error"):
                intent_context = (
                    f"Intent: {query_intent.get('intent', 'exploratory')}\n"
                    f"Goal: {query_intent.get('roadmap_goal', original_query)}\n"
                    f"Guidance: {query_intent.get('generation_guidance', '')}"
                )
                intent_section = intent_context
            else:
                intent_context = ""
                intent_section = "No additional intent guidance."
            context = "\n\n---\n\n".join(contexts) if contexts else (
                "No hay contexto específico recuperado. "
                "Genera pasos basados en conocimiento técnico general del dominio."
            )

            template = """\
Create a technical, actionable roadmap in the same language as the user's query.

Original query: {original_query}
Refined query: {refined_query}
Intent guidance:
{intent_section}

Retrieved evidence:
{context}

Requirements:
- Use the smallest complete number of steps appropriate to the objective; do not target a fixed count.
- Order steps by real dependencies. Keep responsibilities distinct and avoid duplicated work.
- Use exactly one `inicio` as the first step and one `fin` as the last step. Use `decision` only for a real branch.
- Make every label specific and every description concise, actionable, and verifiable.
- Include only useful key points such as commands, settings, cautions, tools, or expected outcomes.
- Prefer retrieved evidence. Prefix unsupported but necessary steps with `[inferido]`.
- Every step must contain `id`, `label`, `description`, `type`, and `key_points`; never emit an empty or partial step.
- Complete and close the full JSON object before finishing the response.
- Return only valid JSON matching the schema. Do not add commentary.

{format_instructions}
"""
            prompt = PromptTemplate(
                template=template,
                input_variables=["original_query", "refined_query", "intent_section", "context"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()},
            )

            prompt_inputs = {
                "original_query": original_query,
                "refined_query":  refined_query,
                "intent_section": intent_section,
                "context":        context,
            }
            rendered_prompt = prompt.format_prompt(**prompt_inputs).to_string()
            self._last_build_diagnostics = {
                "generation_prompt_version": GENERATION_PROMPT_VERSION,
                "generation_prompt_characters": len(rendered_prompt),
                "generation_context_characters": len(context),
            }
            chain = prompt | self.llm | self.parser
            result = chain.invoke(prompt_inputs)
            self._last_build_diagnostics.update({
                "roadmap_output_characters": len(
                    json.dumps(result, ensure_ascii=False)
                ),
                "roadmap_step_count": len(result.get("steps", [])),
            })
            print(f"  [Generación] {len(result.get('steps', []))} pasos generados.")
            return result

        except Exception as e:
            print(f"Error generando roadmap: {e}")
            return {
                "title": "Error",
                "steps": [{
                    "id": "err", "label": "Error de Sistema",
                    "description": str(e)[:200], "type": "decision", "key_points": []
                }]
            }
