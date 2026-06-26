import os
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.llm_provider import get_llm, active_model_name

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


class RoadmapStep(BaseModel):
    id: str = Field(description="ID corto único, ej: 'step_1'")
    label: str = Field(
        description="Verbo de acción + objeto específico. Ej: 'Verificar unicidad de atributos' NO 'Revisar el sistema'"
    )
    description: str = Field(
        description=(
            "Mínimo 2 oraciones. Explica QUÉ se hace, POR QUÉ es necesario y CÓMO se ejecuta. "
            "Incluye herramientas, conceptos técnicos o configuraciones reales del contexto. "
            "PROHIBIDO: frases genéricas como 'este paso es importante' o 'se debe configurar el sistema'."
        )
    )
    type: str = Field(description="Tipo del nodo: 'inicio', 'proceso', 'decision', 'fin'")
    key_points: List[str] = Field(
        description=(
            "Entre 3 y 5 items OBLIGATORIOS. Cada item es UNO de: "
            "comando exacto con parámetros, valor de configuración específico, "
            "advertencia técnica crítica, herramienta con versión, o resultado verificable esperado. "
            "PROHIBIDO: items vagos como 'tener cuidado' o 'revisar la documentación'."
        )
    )

class Roadmap(BaseModel):
    title: str = Field(description="Título conciso del proceso completo")
    steps: List[RoadmapStep] = Field(
        description="Lista ordenada de 6 a 12 pasos. Cada paso debe ser independiente y verificable."
    )

# --- MOTOR RAG ---
class RagEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = Chroma(persist_directory=DB_PATH, embedding_function=self.embeddings)

        print(f"  Modelo: {active_model_name()}")
        self.llm = get_llm(temperature=0.1, max_tokens=4096)
        self.parser = JsonOutputParser(pydantic_object=Roadmap)

    def rewrite_query(self, raw_query: str) -> str:
        """Expande la consulta del usuario para mejorar el retrieval y el roadmap."""
        prompt = (
            "Eres un experto en sistemas RAG técnicos. "
            "Reescribe la siguiente consulta del usuario para hacerla más específica, técnica "
            "y adecuada para búsqueda semántica en una base de conocimiento.\n\n"
            "REGLAS:\n"
            "- Añade terminología técnica relevante del dominio\n"
            "- Especifica el objetivo final que el usuario quiere lograr\n"
            "- Expande siglas o términos ambiguos\n"
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

    def corpus_covers(self, retrieval: dict) -> bool:
        """True si el corpus local cubre la consulta (hay algún fragmento con score suficiente)."""
        scores = [c.get("vector_score") for c in retrieval.get("pool", []) if c.get("vector_score") is not None]
        return bool(scores) and max(scores) >= WEB_FALLBACK_MIN

    def get_contexts(self, refined_query: str) -> list:
        """
        Recupera contexto del corpus local. Si está activado el fallback web y el
        corpus no cubre la consulta, complementa con búsqueda en internet (efímera).
        """
        retrieval = self.retrieve_contexts_scored(refined_query)
        contexts  = [c["text"] for c in retrieval["selected"]]

        if WEB_FALLBACK and not self.corpus_covers(retrieval):
            best = max((c.get("vector_score") or 0) for c in retrieval["pool"]) if retrieval["pool"] else 0
            print(f"  [Web Fallback] Corpus insuficiente (mejor score={best:.3f} < {WEB_FALLBACK_MIN}). Buscando en internet...")
            from src.web_search import search_web
            web_contexts = search_web(refined_query, max_results=4)
            if web_contexts:
                contexts = web_contexts + contexts  # prioriza lo recién encontrado
        return contexts

    def generate_roadmap(self, query: str):
        try:
            refined_query = self.rewrite_query(query)
            print(f"  [Retrieval] Buscando con consulta refinada...")
            contexts = self.get_contexts(refined_query)
            return self.build_roadmap(query, refined_query, contexts)

        except Exception as e:
            print(f"Error generando roadmap: {e}")
            return {
                "title": "Error",
                "steps": [{
                    "id": "err", "label": "Error de Sistema",
                    "description": str(e)[:200], "type": "decision", "key_points": []
                }]
            }

    def build_roadmap(self, original_query: str, refined_query: str, contexts: list):
        """Construye el roadmap con la consulta refinada y los contextos ya recuperados."""
        try:
            context = "\n\n---\n\n".join(contexts) if contexts else (
                "No hay contexto específico recuperado. "
                "Genera pasos basados en conocimiento técnico general del dominio."
            )

            template = """\
Eres un arquitecto técnico senior con experiencia en documentación de procesos empresariales.
Tu tarea es crear un roadmap TÉCNICO, ESPECÍFICO y ACCIONABLE.

═══════════════════════════════════════════════════
CONSULTA ORIGINAL DEL USUARIO
═══════════════════════════════════════════════════
{original_query}

═══════════════════════════════════════════════════
CONSULTA ENRIQUECIDA (usa esta para el roadmap)
═══════════════════════════════════════════════════
{refined_query}

═══════════════════════════════════════════════════
CONTEXTO TÉCNICO RECUPERADO DE LA BASE DE CONOCIMIENTO
═══════════════════════════════════════════════════
{context}

═══════════════════════════════════════════════════
ESTÁNDARES DE CALIDAD OBLIGATORIOS
═══════════════════════════════════════════════════

PARA CADA PASO — EXIGENCIAS MÍNIMAS:

  label:
    ✅ "Verificar la propiedad de unicidad en cada columna candidata"
    ❌ "Verificar datos"

  description (mínimo 2 oraciones):
    ✅ "Se comprueba que ningún valor se repite en la columna candidata usando
        la restricción UNIQUE. Esta propiedad garantiza que cada fila pueda
        ser identificada de forma inequívoca sin depender de otras columnas."
    ❌ "Este paso es importante para el proceso."

  key_points (3 a 5 items, cada uno concreto):
    ✅ ["Consulta SQL: SELECT col, COUNT(*) FROM tabla GROUP BY col HAVING COUNT(*) > 1",
        "CUIDADO: NULL no viola unicidad en algunos motores (PostgreSQL, MySQL)",
        "Resultado esperado: cero filas duplicadas en la columna candidata"]
    ❌ ["Tener cuidado", "Revisar documentación", "Es importante"]

REGLAS GENERALES:
  - Genera entre 6 y 12 pasos (ni muy pocos ni demasiados)
  - Prioriza información del contexto recuperado sobre conocimiento genérico
  - Si el contexto no cubre un paso, indícalo: "[inferido]" al inicio del label
  - Cada paso debe ser ejecutable de forma independiente y verificable
  - NUNCA uses frases de relleno ni pasos obvios sin detalle técnico

{format_instructions}
"""
            prompt = PromptTemplate(
                template=template,
                input_variables=["original_query", "refined_query", "context"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()},
            )

            chain = prompt | self.llm | self.parser
            result = chain.invoke({
                "original_query": original_query,
                "refined_query":  refined_query,
                "context":        context,
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