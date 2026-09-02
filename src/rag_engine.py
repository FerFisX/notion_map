import os
import json
import re
import sys
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Los print() de debug emiten texto generado por el LLM (emojis, flechas "\u2192",
# caracteres no-latinos). En Windows con stdout redirigido (python ... | tee) el
# codec por defecto es cp1252 y un solo caracter no representable hacía CRASHAR el
# roadmap entero (mode=error) en medio de la evaluación. Con errors="replace" esos
# caracteres se degradan a "?" en consola y el flujo sigue. No afecta los datos:
# el contenido real viaja en los dicts de retorno, no por stdout.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.llm_provider import get_llm, active_model_name
from src.intent_classifier import IntentClassifier, policy_for_intent

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

DB_PATH = os.path.join(BASE_DIR, "vectorstore", "chroma_db")

# Retrieval / re-ranking. RERANK_METHOD: crossencoder | mmr | none
# Cambiado a "crossencoder" por defecto: es el que habilita Anclaje Estricto
# (scores 0-1 comparables entre preguntas, a diferencia del score vectorial
# crudo de Chroma, que da valores altos para cualquier pregunta que comparta
# vocabulario con un corpus chico y tematicamente concentrado).
RERANK_METHOD = os.getenv("RERANK_METHOD", "crossencoder").lower().strip()
POOL_SIZE     = int(os.getenv("RETRIEVAL_POOL_SIZE", "10"))
TOP_N         = int(os.getenv("RETRIEVAL_TOP_N", "5"))

# --- ANCLAJE ESTRICTO (Source-Grounded RAG con filtro de confianza) ---
# Umbrales sobre el score del cross-encoder (0-1, con activacion sigmoid --
# ver _get_cross_encoder). Igual que NotebookLM: alta confianza = KB-only
# SIN pasar por el LLM (100% determinista, cero varianza); confianza media =
# zona gris donde SI vale la pena que el LLM lea el contenido (assess_corpus_
# coverage); por debajo del piso, el fragmento ni siquiera entra al contexto.
CE_STRICT_MIN = float(os.getenv("CE_STRICT_MIN", "0.70"))  # >= esto -> candidato fuerte
CE_FLOOR         = float(os.getenv("CE_FLOOR", "0.25"))         # KB-only (intent 1): piso permisivo
CE_FLOOR_STRICT  = float(os.getenv("CE_FLOOR_STRICT", "0.35"))  # intents con web: piso estricto (evita contaminar case2/case3)

# Conciliación web-vs-corpus: si está apagada, el modo híbrido usa los
# fragmentos web tal cual vienen de la búsqueda (comportamiento anterior).
WEB_RECONCILE_ENABLED = os.getenv("WEB_RECONCILE_ENABLED", "true").lower().strip() in ("true", "1", "yes")

# --- ADAPTACIÓN RUNTIME (política del lado respuesta, NUNCA clasificación) ---
# Si el intent inferido confiaba en la KB (intent 1 / 2) pero la KB en realidad
# NO la cubre (vacía o casi vacía), el orquestador adapta la política de fuente
# "en vivo" y avisa al usuario con KB_NOT_COVERED_MESSAGE. La clasificación NO
# se re-ejecuta: la adaptación se registra como coverage_adaptation en la traza
# y nunca alimenta al clasificador de intent.
ADAPT_EMPTY_KB_MIN_CHUNKS = int(os.getenv("ADAPT_EMPTY_KB_MIN_CHUNKS", "2"))
KB_NOT_COVERED_MESSAGE = (
    "La Knowledge Base no cubre esta información. La respuesta se genera "
    "con información externa."
)

# --- DETECCIÓN DE INTENT: delegada 100% a src/intent_classifier.py ---
# El clasificador de intención (4 intents) vive en su módulo propio y es
# independiente de la generación de roadmaps. Aquí solo queda el ruido léxico
# que se elimina al condensar la consulta web (la empresa como marca no es
# información de búsqueda -- INTENT_CLASSIFIER co.INTERNAL_ANCHOR_TERMS /
# EXTERNAL_ANCHOR_TERMS / SOFT_ADVISORY_TERMS son HINTS del clasificador, no
# cortes deterministas).
_WEB_QUERY_NOISE = [
    t.strip().lower() for t in os.getenv(
        "WEB_QUERY_NOISE", "bamboo,bambootec,bambotec,nuestra,nuestro,"
        "nuestras,nuestros,interno,interna,internos,internas,nuestro kb,"
        "kb interno,corpus interno,base de conocimiento"
    ).split(",") if t.strip()
]

# Disponibilidad de búsqueda web (no política de fuente). El clasificador de
# intent decide CUÁNDO se usa; esta bandera solo dice si el buscador existe.
# Mantiene compatibilidad con el viejo WEB_FALLBACK del .env.
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", os.getenv("WEB_FALLBACK", "false")).lower().strip() in ("true", "1", "yes")


def _strip_web_noise(text: str) -> str:
    lowered = text.lower()
    for term in _WEB_QUERY_NOISE:
        lowered = lowered.replace(term, " ")
    return re.sub(r"\s{2,}", " ", lowered).strip(" ,-–")


MODE_LABELS = {
    "corpus":       "Knowledge Base",
    "hybrid":       "Hybrid (KB + Web)",
    "web":          "Web",
    "clarify":      "Clarify",
    "no_retrieval": "No Retrieval",
    "error":        "Error",
}


QUERY_INTENT_ENABLED = os.getenv("QUERY_INTENT_ENABLED", "true").lower().strip() in ("true", "1", "yes")


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
    source: str = Field(
        description=(
            "'corpus' si el hecho o criterio CONCRETO que usa este paso viene principalmente "
            "del BLOQUE CORPUS (base interna); 'web' si viene principalmente del BLOQUE WEB "
            "(búsqueda externa). Decide por el CONTENIDO factual que usaste para escribir este "
            "paso, no por el estilo de redacción. Si el paso combina ambos en partes similares, "
            "o no hay bloque web en este contexto, usa 'corpus'."
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
    
        self.llm_light = get_llm(temperature=0.0, max_tokens=2048)
        self.llm_main = get_llm(temperature=0.0, max_tokens=4096)
        self.parser = JsonOutputParser(pydantic_object=Roadmap)

        # Clasificador de intención de fuente (SRC standalone). Independiente de la
        # generación de roadmaps: selo decide QUÉ fuente espera el usuario y cuándo
        # conviene clarificar (Intent 4) / no hacer retrieval.
        self.classifier = IntentClassifier()

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
            raw = self.llm_light.invoke(prompt).content.strip()
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
        rewritten = self.llm_light.invoke(prompt).content.strip()
        # Si el modelo antepuso razonamiento antes de la etiqueta (ej. "¿Intención:
        # comparison. Objetivo final: ...\n\nConsulta mejorada:\n<texto>"), nos
        # quedamos solo con lo que sigue después de la última aparición de la etiqueta.
        marker = "consulta mejorada:"
        idx = rewritten.lower().rfind(marker)
        if idx != -1:
            rewritten = rewritten[idx + len(marker):].strip()
        # Limpiar prefijos residuales que el modelo pueda agregar
        for prefix in ("Consulta mejorada:", "Aquí", "La consulta"):
            if rewritten.startswith(prefix):
                rewritten = rewritten[len(prefix):].strip()

        if not rewritten:
            # Modelos de razonamiento (ej. gpt-5-nano) a veces consumen todo
            # max_tokens "pensando" y no dejan nada para la salida visible.
            # Sin este fallback, un refined_query vacío se propaga a retrieval
            # y a build_web_search_query, que termina inventando una consulta
            # de búsqueda web sin relación con la pregunta real.
            print("  [Query Rewriting] Respuesta vacía del LLM (probable agotamiento de tokens en razonamiento); usando la pregunta original sin reescribir.")
            rewritten = raw_query

        print(f"  [Query Rewriting]\n    Original : {raw_query}\n    Mejorada : {rewritten}")
        return rewritten

    _cross_encoder = None

    def _get_cross_encoder(self):
        """Carga perezosa del cross-encoder. Devuelve None si no está disponible."""
        if RagEngine._cross_encoder is None:
            try:
                import torch
                from sentence_transformers import CrossEncoder
                # activation_fn=Sigmoid: sin esto, ms-marco-MiniLM-L-6-v2 devuelve
                # logits crudos sin acotar (ej. -8.3, 8.6), no 0-1. Los umbrales
                # de Anclaje Estricto (CE_STRICT_MIN/CE_FLOOR) solo tienen sentido
                # con la salida normalizada.
                RagEngine._cross_encoder = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    activation_fn=torch.nn.Sigmoid(),
                )
            except Exception as e:
                print(f"  [Re-rank] Cross-encoder no disponible ({e}); usando MMR.")
                RagEngine._cross_encoder = False
        return RagEngine._cross_encoder or None

    def retrieve_contexts_scored(
        self,
        query:        str,
        pool_size:    int  = POOL_SIZE,
        top_n:        int  = TOP_N,
        method:       str  = None,
        rerank_query: str  = None,
        extra_query:  str  = None,
    ) -> dict:
        """
        Recupera un pool de candidatos, los re-rankea y devuelve los top_n.
        Devuelve {method, pool, selected} con scores para inspección.

        rerank_query: si se especifica, el cross-encoder puntua los
        fragmentos contra este texto en vez de contra `query`. `query` se usa
        SOLO para la busqueda inicial en el vector store (mejor recall con una
        consulta reescrita/expandida); `rerank_query` es contra lo que se
        calcula el score real de confianza -- tipicamente la pregunta
        original del usuario, corta y estable, en vez de la reescritura del
        LLM (que puede ser mucho mas larga y diluir el score del
        cross-encoder aunque el fragmento si responda la pregunta).

        extra_query: si se especifica, se hace una SEGUNDA busqueda en Chroma
        con este texto y se une al pool de `query` (sin duplicados) antes de
        re-rankear. Pensado para pasar aqui `decision_query` (original_query,
        estable) mientras `query` sigue siendo `refined_query` (reescrita,
        mejor recall) -- asi el pool nunca depende SOLO de una reescritura
        que cambia de redaccion en cada corrida. Sin esto, un fragmento
        relevante podia quedar afuera del pool en corridas donde la
        reescritura se alejaba demasiado del vocabulario del corpus, sin que
        ningun re-rank posterior lo pudiera rescatar (no estaba ni siquiera
        en el pool). Ver hallazgo del 02-ago-2026: la misma pregunta original
        daba best_ce_score=0.884 en una corrida y 0.001 en otra.
        """
        method = (method or RERANK_METHOD).lower().strip()
        rerank_query = rerank_query or query

        # Pool inicial con scores del vector store
        scored = self.vector_db.similarity_search_with_relevance_scores(query, k=pool_size)
        pool = []
        seen_texts = set()
        for i, (doc, score) in enumerate(scored, 1):
            pool.append({
                "text":         doc.page_content,
                "vector_score": round(float(score), 4),
                "init_rank":    i,
                "source":       doc.metadata.get("source", "?"),
            })
            seen_texts.add(doc.page_content)

        # Segundo pool (estable) para no depender solo de query si esta varia
        # de redaccion entre corridas -- se une sin duplicados.
        if extra_query and extra_query != query:
            extra_scored = self.vector_db.similarity_search_with_relevance_scores(extra_query, k=pool_size)
            for i, (doc, score) in enumerate(extra_scored, 1):
                if doc.page_content in seen_texts:
                    continue
                pool.append({
                    "text":         doc.page_content,
                    "vector_score": round(float(score), 4),
                    "init_rank":    i,
                    "source":       doc.metadata.get("source", "?"),
                })
                seen_texts.add(doc.page_content)

        if not pool:
            return {"method": method, "pool": [], "selected": []}

        if method == "none":
            ranked = pool

        elif method == "crossencoder" and self._get_cross_encoder():
            ce     = self._get_cross_encoder()
            scores = ce.predict([(rerank_query, c["text"]) for c in pool])
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

    def reconcile_web_with_corpus(self, refined_query: str, corpus_contexts: list, web_contexts: list, coverage: str = "parcial", gap_description: str = "", raw_query: str = None) -> list:
        """
        Concilia los fragmentos web contra el corpus ANTES de que lleguen al
        generador del roadmap.

        ESTRICTA vs LAXA (regla determinista, NO depende del LLM): el estándar
        se decide por la COBERTURA ya evaluada por assess_corpus_coverage (que
        sí leyó el contenido real): cobertura "parcial" -> estándar MODERADO
        (el fragmento debe atacar el tema central de la pregunta y aportar algo
        específico que el corpus no tiene); cobertura "nula" -> estándar LAXO
        (basta con tocar el mismo tema). En case1/case2 el sesgo esperado es
        KB > Web y la conciliación modera; en case3 es Web > KB y la laxitud
        permite completar.

        Por cada fragmento web, el LLM decide "descartar" | "adjuntar" | "mantener"
        -- el LLM sigue siendo necesario aquí porque juzgar si un fragmento
        aporta contenido nuevo SÍ requiere leer el texto (a diferencia de
        detectar un ancla, que es una búsqueda literal). Lo que se elimina de
        raíz es la varianza en el ESTÁNDAR aplicado, no el juicio de contenido
        en sí.
        """
        # NOTA: el switch estricto/laxo ya NO depende del ancla textual (INTERNAL_
        # ANCHOR_TERMS). Se probó y falló: preguntas de case3 (web-dominante,
        # territorio nuevo) TAMBIÉN mencionan "Bamboo" -- es natural, la pregunta
        # es sobre desplegar algo EN el contexto de la empresa, no implica que el
        # KB lo documente. Usar el ancla como señal de "debe dominar el corpus"
        # confundía "esto es sobre nuestra empresa" con "esto ya está documentado
        # en nuestro KB". La señal correcta es la cobertura YA evaluada por
        # assess_corpus_coverage, que sí leyó el contenido real recuperado.
        moderate_mode = (coverage == "parcial")

        # Fallback cuando el JSON parseo bien pero el campo "decision" vino con un
        # valor invalido: se favorece conservar en ambos niveles -- el costo de un
        # fragmento de mas es bajo frente al costo de perder cobertura real.
        default_decision = "adjuntar"
        gap_line = (
            f"HUECO ESPECIFICO IDENTIFICADO EN EL CORPUS: {gap_description}\n\n"
            if gap_description else ""
        )

        if moderate_mode:
            # ESTANDAR MODERADO (cobertura "parcial"): ni imposible de pasar (la
            # version anterior exigia que el fragmento resolviera el HUECO EXACTO
            # con la misma terminologia -- eso descartaba incluso documentacion
            # oficial directamente relevante, ej. "Elegir o eliminar columnas -
            # Power Query | Microsoft Learn" para una pregunta sobre exactamente
            # esa comparacion), ni tan laxo que diluya el sesgo hacia KB esperado
            # en case2. El estandar: ¿esta fuente es sobre el TEMA CENTRAL de la
            # pregunta y aporta algo especifico (una tecnica, comparacion,
            # criterio, cifra, ejemplo) que el corpus no tiene? Si si, adjuntar.
            decision_options = (
                "- \"descartar\": contradice al corpus, o es generico/publicitario/"
                "fuera de tema respecto al TEMA CENTRAL de la pregunta (no menciona "
                "ese tema en absoluto, solo palabras sueltas).\n"
                "            - \"adjuntar\": es sobre el TEMA CENTRAL de la pregunta "
                "(ej. una fuente oficial o tecnica que trata directamente ese tema) Y "
                "aporta al menos UN detalle especifico -- una tecnica, criterio, cifra, "
                "comando o ejemplo -- que el corpus no tiene. NO hace falta que resuelva "
                "el hueco por completo ni que use la misma terminologia exacta.\n"
                "            - \"mantener\": corrobora un hecho del corpus y suma un "
                "detalle menor util, sin contradecir nada."
            )
            tie_break_rule = (
                "La cobertura del corpus es 'parcial' -- el sesgo esperado es que el "
                "KB siga dominando, pero SI corresponde sumar contenido web que aporte "
                "algo concreto sobre el tema central (no exijas que cite la misma frase "
                "exacta del hueco). Si la fuente es tecnica/oficial y trata directamente "
                "el tema de la pregunta, el default es \"adjuntar\" salvo que sea "
                "genuinamente generica o publicitaria. Ante duda entre \"descartar\" y "
                "\"adjuntar\" para una fuente que SI es del tema pero con enfoque "
                "distinto, responde \"adjuntar\"."
            )
        else:
            decision_options = (
                "- \"descartar\": contradice al corpus, o es sobre un tema completamente "
                "distinto al hueco identificado.\n"
                "            - \"adjuntar\": toca el mismo tema/concepto que el hueco "
                "especifico, aunque sea de forma parcial o general -- NO hace falta que "
                "lo resuelva por completo.\n"
                "            - \"mantener\": corrobora un hecho del corpus y suma un "
                "detalle menor util, sin contradecir nada."
            )
            tie_break_rule = (
                "El corpus NO cubre este tema en absoluto (cobertura nula), asi que no "
                "hay nada con que ser \"redundante\". Ante duda, responde \"adjuntar\" -- "
                "solo descarta si el fragmento es claramente incorrecto, irrelevante al "
                "tema, o de muy baja calidad."
            )

        print(f"  [Reconciliación] Modo: {'MODERADO (cobertura parcial)' if moderate_mode else 'LAXO (cobertura nula)'}")

        corpus_block = "\n---\n".join(corpus_contexts)
        kept = []
        for i, snippet in enumerate(web_contexts, 1):
            prompt = f"""\
            Eres un validador de consistencia para un sistema RAG. El corpus interno
            (la base de conocimiento propia) es la fuente de verdad preferida frente
            a internet. Compara el fragmento web contra el corpus y decide su destino.

            CONSULTA: {refined_query}

            {gap_line}
            CORPUS (fuente de verdad, cobertura evaluada: {coverage}):
            {corpus_block}

            FRAGMENTO WEB:
            {snippet}

            Decide UNA opcion:
            {decision_options}

            Regla de desempate: {tie_break_rule}

            Responde SOLO con JSON: {{"decision": "descartar"|"adjuntar"|"mantener", "razon": "<breve, una frase, di explicitamente si toca el hueco o no>"}}
            """
            # Fallback SOLO para fallas de formato/parseo (no para un juicio real de
            # la LLM): perder un fragmento porque el JSON vino mal formado es un bug
            # de robustez, no una decision de contenido -- se favorece conservar en
            # ambos modos.
            decision = "mantener"
            razon = "sin respuesta parseable (fallback de robustez, no juicio de contenido)"
            try:
                raw = self.llm_light.invoke(prompt).content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()
                start, end = raw.find("{"), raw.rfind("}")
                parsed = False
                if start >= 0 and end > start:
                    try:
                        data = json.loads(raw[start:end + 1])
                        decision = str(data.get("decision", default_decision)).lower().strip()
                        razon = str(data.get("razon", "")).strip()
                        if decision not in ("descartar", "adjuntar", "mantener"):
                            decision = default_decision
                        parsed = True
                    except Exception:
                        parsed = False
                if not parsed:
                    # Respaldo por regex antes de rendirse: a veces el modelo casi
                    # arma el JSON pero deja una coma colgante o comillas sin cerrar.
                    import re
                    m = re.search(r'"decision"\s*:\s*"(descartar|adjuntar|mantener)"', raw)
                    if m:
                        decision = m.group(1)
                        razon = "extraido por regex (JSON mal formado)"
                    # si tampoco matchea, se queda con el fallback "mantener" de arriba
            except Exception as e:
                decision = "mantener"
                print(f"  [Reconciliación] Fragmento #{i} sin veredicto claro ({e}); se usa el fallback de robustez ({decision}).")
                razon = f"error de invocacion: {e}"

            print(f"  [Reconciliación] Web #{i}: {decision} — {razon}")
            print(f"       snippet: {snippet[:160]}...")
            if decision in ("adjuntar", "mantener"):
                kept.append(snippet)

        print(f"  [Reconciliación] {len(kept)}/{len(web_contexts)} fragmentos web conservados tras contrastar con el corpus.")
        return kept

    def build_web_search_query(self, refined_query: str, gap_description: str = "") -> str:
        """
        Los buscadores web rinden mal con consultas de un párrafo cargadas de
        jerga (el refined_query está optimizado para retrieval semántico, no
        para un buscador). Se condensa a una consulta corta en lenguaje natural
        antes de llamar a search_web.

        gap_description: cuando viene el hueco puntual identificado en el corpus
        (la justificación de assess_corpus_coverage), la consulta corta se enfoca
        en ESE hueco específico y no en el tema genérico -- es lo que evita que
        el modo híbrido busque "lo mismo que ya está en el KB" en vez del dato
        que falta.
        """
        gap_line = (
            "Objetivo: completar ESPECÍFICAMENTE este hueco detectado en la base "
            "interna de la empresa, no explicar el tema en general:\n"
            f"{gap_description}\n"
            if gap_description else ""
        )
        prompt = (
            "Convierte esta consulta técnica extensa en una consulta CORTA de "
            "búsqueda web, como la escribiría una persona en Google: 6 a 10 "
            "palabras, en lenguaje natural, sin listas de sinónimos ni jerga "
            "acumulada. NO incluyas el nombre de la empresa ni palabras como "
            "'interno', 'nuestro' o 'Bamboo': el buscador no conoce la jerga "
            "corporativa, solo busca la técnica genérica del tema. Responde SOLO "
            "con la consulta corta, sin comillas ni explicaciones.\n\n"
            f"{gap_line}"
            f"Consulta extensa:\n{refined_query}\n\n"
            "Consulta corta:"
        )
        try:
            short = self.llm_light.invoke(prompt).content.strip().strip('"')
        except Exception:
            short = refined_query[:80]
        # Respaldo determinista por si la LLM dejó la jerga corporativa en la
        # consulta: se eliminan el nombre de la empresa y los posesivos/marcadores
        # de contexto interno, que solo ensucian los resultados del buscador.
        short = _strip_web_noise(short)
        print(f"  [Web Query] Consulta condensada para buscador: {short}")
        return short

    def assess_corpus_coverage(self, query: str, corpus_contexts: list) -> dict:
        """
        En vez de inferir cobertura solo del score vectorial (que con un corpus
        chico y tematicamente concentrado da scores altos para CUALQUIER
        pregunta que comparta vocabulario, aunque el contenido real no la
        responda), se le pide al LLM que LEA los fragmentos recuperados y
        juzgue si responden la pregunta. Esto evita que "normalizar hasta 3FN"
        o "criterio de desempate por estabilidad" se queden en modo corpus
        solo porque el corpus tiene un documento sobre llaves candidatas.

        Devuelve {"coverage": "completa"|"parcial"|"nula", "justificacion": str}

        HARNESS (18-ago-2026): antes se le pedía al LLM que eligiera "completa"/
        "parcial"/"nula" directamente en una sola decisión difusa -- ahí es donde
        vivía la varianza (la misma pregunta, con el mismo corpus, podía salir
        distinta entre corridas). Ahora el LLM solo marca HECHOS ATÓMICOS (¿el
        tema central aparece en el corpus? ¿cada elemento puntual que la pregunta
        necesita está o no?) y la clasificación completa/parcial/nula se CALCULA
        en Python a partir de esas marcas -- el LLM ya no elige el label final.
        Sigue siendo UNA sola llamada (no self-consistency): no agrega latencia,
        solo mueve el límite de decisión del LLM al código, igual que se hizo con
        attribute_step_sources -> RoadmapStep.source.
        """
        if not corpus_contexts:
            return {"coverage": "nula", "justificacion": "no hay contexto recuperado"}

        corpus_block = "\n---\n".join(corpus_contexts)
        prompt = f"""\
        You are evaluating whether an internal knowledge base (corpus) has enough
        information to fully answer a user's question. You will be given the
        retrieved corpus context and the question. Do NOT judge writing quality,
        completeness of prose, or whether the corpus explicitly spells out the
        final answer in a single sentence — judge only whether the FACTS needed
        to construct the answer are present in the corpus.

        Return your analysis in two parts:

        1. topic_present_in_corpus (bool): Does the corpus address the core
        subject/technology of the question at all?

        2. required_elements: Break the question down into 1 to 4 concrete
        facts, values, or criteria the question needs answered. For each,
        mark found=true if the corpus contains that fact — even if it is
        not phrased as an explicit answer to the question.

        CRITICAL CALIBRATION RULES — read carefully, these are the most common
        source of false negatives:

        - COMPARISON questions ("how does X compare to Y", "which is cheaper,
          X or Y"): if the corpus documents X and Y separately with enough
          detail to derive the comparison (e.g., a table listing both with
          their values), mark found=true for each element. Do NOT require the
          corpus to state the comparison itself in prose — deriving "A is
          cheaper than B" from two rows of the same table counts as found.

          Example (Spanish source material):
          Question: "¿Cómo se compara el costo de un merge/join frente a una
          selección de columnas en Power Query?"
          Corpus contains a table row: "Selección de columnas | O(1)" and a
          separate row: "Merge/Join | O(n log n) - O(n^2)".
          → topic_present_in_corpus: true
          → required_elements: [
              {{"element": "costo de selección de columnas", "found": true}},
              {{"element": "costo de merge/join", "found": true}}
            ]
          → coverage should resolve to "completa", NOT "parcial".

        - STEP-LIST or SOLUTION questions ("what steps...", "how did X solve
          Y", "what sequence..."): if the corpus contains a complete list or
          a stated solution, mark found=true even if there is no further
          elaboration or justification for each item. A bulleted list that
          fully answers "what are the steps" or "what was the fix" is
          sufficient — do not require narrative explanation beyond the list.

          Example (Spanish source material):
          Question: "¿Cómo resolvió Bamboo el problema de desalineación de
          zonas horarias al sincronizar Notion con Google Calendar?"
          Corpus states: "La solución fue: Definir timezone única, Prohibir
          offsets manuales, Especificar flujos de actualización claros en el
          prompt."
          → topic_present_in_corpus: true
          → required_elements: [
              {{"element": "solución al problema de zonas horarias", "found": true}}
            ]
          → coverage should resolve to "completa", NOT "parcial".

        - INTERNAL ORGANIZATION questions ("what does [company] document...",
          "how did [company] solve..."): if the corpus is an internal
          knowledge base belonging to the organization mentioned in the
          question, then ANY document in the corpus that addresses the topic
          counts as the organization's own documentation — even if the
          document does not explicitly repeat the company name in every
          paragraph. The corpus IS the company's documentation by
          definition. Do NOT require the text to say "Bamboo documented
          that..." or "According to Bamboo's internal case..." — if the
          content answers the question and comes from the internal corpus,
          mark found=true.

          Example:
          Question: "¿Qué regla documenta Bamboo sobre la separación entre
          decisión y ejecución?"
          Corpus contains a document titled "Creación y testeo de flujos con
          agentes IA en N8N" that states: "Separación estricta entre
          decisión y ejecución → El agente decide, Las tools ejecutan."
          → The corpus IS Bamboo's internal documentation. The rule is
          present. Mark found=true, NOT found=false just because the
          paragraph does not say "Bamboo".

        - TABLE-ROW DERIVATION: when the question asks "what is the value
          for X" or "how does X compare to Y", and the corpus contains a
          table where each row lists an item with its corresponding value
          (e.g., complexity notation, price, rating), then each individual
          row counts as evidence for that item's value. The LLM must NOT
          require the corpus to state the answer in prose — the table row
          IS the answer. If the question asks about "selección de columnas"
          and the corpus has a row "Selección de columnas | O(1)", that is
          found=true.

        - Do NOT invent required_elements beyond what the question literally
          asks. A single-fact question (e.g., "what complexity is assigned to
          operation X?") has exactly ONE requiredElement, not several.

        - Reserve found=false for genuine gaps: the corpus does not mention
          the fact/criterion at all, or only mentions the general topic
          without the specific value/rule/step the question asks for.

        Return ONLY valid JSON in this exact shape, nothing else:

        {{
          "topic_present_in_corpus": true|false,
          "required_elements": [
            {{"element": "<short description>", "found": true|false}},
            ...
          ]
        }}

        Corpus context:
        {corpus_block}

        Question:
        {query}
        """
        coverage, justificacion, tema_central = "parcial", "sin respuesta parseable", ""
        try:
            raw = self.llm_light.invoke(prompt).content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            start_i, end_i = raw.find("{"), raw.rfind("}")
            if start_i >= 0 and end_i > start_i:
                data = json.loads(raw[start_i:end_i + 1])
                tema_central = str(data.get("tema_central", "")).strip()
                justificacion = str(data.get("justificacion", "")).strip()
                topic_present = bool(data.get("topic_present_in_corpus", False))
                elements = [e for e in (data.get("required_elements") or []) if isinstance(e, dict)]

                if not topic_present:
                    coverage = "nula"
                elif not elements or all(bool(e.get("found")) for e in elements):
                    coverage = "completa"
                else:
                    coverage = "parcial"
                if not justificacion:
                    missing = [e.get("element", "") for e in elements if not e.get("found")]
                    justificacion = (
                        f"tema central {'presente' if topic_present else 'ausente'}; "
                        f"{len([e for e in elements if e.get('found')])}/{len(elements)} elementos encontrados"
                        + (f"; faltan: {', '.join(missing)}" if missing else "")
                    )
        except Exception as e:
            coverage, justificacion = "parcial", f"error de evaluacion: {e}"

        print(f"  [Cobertura del corpus] {coverage}" + (f" — tema='{tema_central}'" if tema_central else "") + f" — {justificacion}")
        return {"coverage": coverage, "justificacion": justificacion}

    def get_contexts_with_sources(self, refined_query: str, original_query: str = None, classification: dict = None) -> dict:
        """
        Decide la fuente según el INTENT del usuario (clasificado ANTES,
        pre-retrieval). Es la política de fuentes del sistema.

        Contrato de entrada:
          classification: dict salido de IntentClassifier.classify() -- o de la
            continuación de un Intent 4 ya clarificado (intent 1/2/3 seleccionado
            por el usuario). Campos usados: intent, decision, source_policy
            (kb/external/web_search/bias_target).

        PRINCIPIO ÚNICO (reemplaza la tabla de fusión "coverage x modality"):
          la POLÍTICA DEL INTENT manda el uso de la web:
            - intent 1 (KB Only)       -> web_search=false: SOLO corpus.
            - intent 2 (KB + External) -> web_search=true: SIEMPRE se busca web y
              el resultado es KB-dominante (sesgo corpus >= 70%), incluso con
              cobertura completa del corpus.
            - intent 3 (External + KB) -> web_search=true: SIEMPRE se busca web y
              el resultado es web-dominante (sesgo web >= 60%).
          La cobertura del corpus (assess_corpus_coverage) ya NO decide si se usa
          web; solo (a) enfoca la consulta web en el HUECO y (b) dispara la
          ADAPTACIÓN RUNTIME cuando un intent que confiaba en la KB es traicionado
          por una KB vacía o casi vacía (coverage_adaptation).

        ADAPTACIÓN RUNTIME (política del lado respuesta, NUNCA clasificación):
          Si el intent inferido confiaba en la KB (intent 1/2) pero la KB no la
          cubre, la política se adapta "en vivo" hacia case3 (web dominante), con
          el mensaje KB_NOT_COVERED_MESSAGE para el usuario:
            - 0 chunks anclados -> 100% web (corpus vacío al generador).
            - cobertura "nula" o <= ADAPT_EMPTY_KB_MIN_CHUNKS chunks -> KB (casi)
              vacía -> case3. Ambos casos con el MISMO mensaje.
          La adaptación se registra en `coverage_adaptation` y NUNCA re-ejecuta el
          clasificador.

        Retorno: mantiene los campos históricos (mode, modality, bias_target,
        coverage, best_score, contexts, corpus_contexts, web_contexts) para no
        romper evaluadores/frontend, y agrega intent, coverage_adaptation y notice.
        """
        _INTENT_MODALITY = {1: "kb_only", 2: "kb_dominant", 3: "web_dominant"}
        decision_query = original_query or refined_query
        intent = (classification or {}).get("intent")
        source_policy = (classification or {}).get("source_policy") or {
            "kb": True, "external": False, "web_search": False,
            "bias_target": {"favor": "corpus", "min_pct": 100},
        }
        web_allowed = bool(source_policy.get("web_search", False)) and WEB_SEARCH_ENABLED
        bias_target = dict(source_policy.get("bias_target") or {"favor": "corpus", "min_pct": 100})
        modality = _INTENT_MODALITY.get(intent, "ambiguous")

        # Pool combinado: refined_query aporta recall amplio (sinonimos,
        # terminos que la reescritura agrega), decision_query (original_query)
        # garantiza un piso estable. El cross-encoder re-rankea el pool unido,
        # siempre contra decision_query.
        retrieval = self.retrieve_contexts_scored(
            refined_query,
            method="crossencoder",
            rerank_query=decision_query,
            extra_query=decision_query,
        )

        # Piso de anclaje: un intent que PERMITE web exige estándar más estricto
        # (evita contaminar case2/case3 con chunks marginales); intent 1 (KB-only)
        # es permisivo porque el usuario pidió explícitamente solo contenido interno.
        _floor = CE_FLOOR if not web_allowed else CE_FLOOR_STRICT
        anchored = [c for c in retrieval["selected"] if (c.get("rerank_score") or 0.0) >= _floor]
        corpus_contexts = [c["text"] for c in anchored]

        ce_scores = [c.get("rerank_score") for c in retrieval.get("pool", []) if c.get("rerank_score") is not None]
        best = max(ce_scores) if ce_scores else 0.0
        print(f"  [Score de referencia] best_ce_score={best:.3f} sobre: \"{decision_query[:70]}\"")
        if best >= CE_STRICT_MIN:
            print(f"  [Anclaje Estricto] score={best:.3f} >= {CE_STRICT_MIN} -> candidato fuerte, verificando cobertura completa del pedido...")

        # Cobertura (LLM): se mantiene aunque ya no gatea la web -- enfoca la
        # búsqueda en el hueco y dispara la adaptación runtime.
        coverage_result = self.assess_corpus_coverage(decision_query, corpus_contexts)
        coverage = coverage_result["coverage"]

        web_contexts = []
        mode = "corpus"
        coverage_adaptation = {"triggered": False, "reason": None, "message": None}
        notice = None

        # --- ADAPTACIÓN RUNTIME (KB vacía / casi vacía) ---
        trusted_kb = intent in (1, 2)  # el intent confiaba en la KB
        n_anchored = len(corpus_contexts)
        kb_effectively_empty = (
            coverage == "nula"
            or n_anchored == 0
            or (coverage == "parcial" and n_anchored <= ADAPT_EMPTY_KB_MIN_CHUNKS)
        )
        if trusted_kb and kb_effectively_empty and WEB_SEARCH_ENABLED:
            coverage_adaptation = {
                "triggered": True,
                "reason": (
                    f"intent {intent} esperaba respaldo de KB, pero la KB no lo "
                    f"cubre (coverage={coverage}, {n_anchored} chunks anclados) -> case3"
                ),
                "message": KB_NOT_COVERED_MESSAGE,
            }
            notice = KB_NOT_COVERED_MESSAGE
            web_allowed = True
            bias_target = {"favor": "web", "min_pct": 60}
            if n_anchored == 0:
                mode = "web"
                corpus_contexts = []  # 100% web: nada del corpus llega al generador
            else:
                mode = "hybrid"
            print(f"  [Adaptación KB vacía] {coverage_adaptation['reason']}")

        if not web_allowed:
            # Intent 1 sin adaptación: SOLO corpus, sin buscar ni confirmar nada.
            mode = "corpus"
        elif mode == "corpus":
            # Intent 2/3: la web se busca SIEMPRE; la cobertura solo regula el sesgo.
            mode = "hybrid"

        if mode in ("hybrid", "web"):
            print(f"  [Fuentes/{MODE_LABELS.get(mode, mode)}] intent={intent}, cobertura={coverage}, sesgo={bias_target}. Complementando con internet...")
            from src.web_search import search_web
            # La búsqueda apunta al HUECO específico del corpus (no al tema
            # genérico): es lo que separa "cubrir el dato que falta" de
            # "recuperar lo mismo que ya tiene el KB".
            web_query = self.build_web_search_query(refined_query, gap_description=coverage_result.get("justificacion", ""))
            web_contexts = search_web(web_query, max_results=4)

            if web_contexts and WEB_RECONCILE_ENABLED:
                web_contexts = self.reconcile_web_with_corpus(refined_query, corpus_contexts, web_contexts, coverage=coverage, gap_description=coverage_result.get("justificacion", ""), raw_query=decision_query)

            if web_contexts and bias_target and bias_target.get("favor") == "corpus":
                # Intent 2 (KB-dominante): el bloque web llega al generador LIMITADO
                # a lo mínimo -- a lo sumo 2 fragmentos truncados -- para que el
                # corpus siga dominando el roadmap.
                web_contexts = [w[:300] for w in web_contexts][:2]

            if not web_contexts:
                # La web no aportó nada (búsqueda vacía o descartada en la
                # conciliación). Se mantiene el modo por honestidad del sesgo
                # elegido; el evaluador valida el % real contra el sesgo.
                mode = "web" if coverage == "nula" else "hybrid"

        # Orden de los bloques en el contexto del generador: sigue el sesgo. Con
        # favor=corpus (intent 1/2) el corpus va PRIMERO y es la base visible del
        # roadmap; con favor=web (intent 3 / adaptación) la web va primero.
        if bias_target and bias_target.get("favor") == "corpus":
            ordered_contexts = corpus_contexts + web_contexts
        else:
            ordered_contexts = web_contexts + corpus_contexts

        return {
            "contexts":            ordered_contexts,
            "corpus_contexts":     corpus_contexts,
            "web_contexts":        web_contexts,
            "best_score":          round(best, 3),
            "coverage":            coverage,
            "mode":                mode,
            "modality":            modality,
            "bias_target":         bias_target,
            "intent":              intent,
            "coverage_adaptation": coverage_adaptation,
            "notice":              notice,
        }


    def _cosine(self, a, b) -> float:
        import numpy as np
        a, b = np.array(a), np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(a.dot(b) / denom) if denom else 0.0

    def _compute_source_pct(self, roadmap: dict, corpus_contexts: list, web_contexts: list) -> dict:
        """
        Cuenta corpus_pct/web_pct a partir del campo 'source' que cada paso ya
        trae desde build_roadmap() -- ver RoadmapStep.source. Antes esto era
        una SEGUNDA llamada al LLM (attribute_step_sources, eliminada) que
        releía el roadmap ya escrito y adivinaba la procedencia por su cuenta,
        sin ver la instrucción de sesgo (bias_section) que build_roadmap sí
        recibió. Esas dos IAs -- la que escribe y la que mide -- nunca se
        hablaban entre sí, lo que producía desacoples: el generador podía
        intentar seguir el sesgo pedido y el auditor, leyendo a ciegas,
        clasificar distinto. Fusionar la atribución en la misma llamada que
        escribe cada paso elimina ese desacople y una llamada completa a LLM.

        Los casos triviales (una sola fuente) se fuerzan aquí en vez de
        confiar en lo que el LLM haya puesto en 'source': si no hubo bloque
        web en el prompt, el campo no es informativo y se corrige a "corpus"
        (e igual al revés) -- mismo comportamiento que tenía la versión vieja.
        """
        steps = roadmap.get("steps", [])
        if not steps:
            return {"corpus_pct": 0, "web_pct": 0}

        # Caso "ninguna fuente aportó nada": ni corpus ni web tenían contenido
        # (típico de coverage="nula" + search_web() vacío -- rate-limit u otro
        # fallo silencioso). Se distingue de "no hizo falta buscar en la web"
        # (solo web_contexts vacío) para no reportar "100% corpus" cuando el
        # corpus tampoco aportó nada real. Ver 10.3 del handoff.
        if not corpus_contexts and not web_contexts:
            for s in steps:
                s["source"] = "sin_fuente"
            return {"corpus_pct": 0, "web_pct": 0}
        if not web_contexts:
            for s in steps:
                s["source"] = "corpus"
            return {"corpus_pct": 100, "web_pct": 0}
        if not corpus_contexts:
            for s in steps:
                s["source"] = "web"
            return {"corpus_pct": 0, "web_pct": 100}

        corpus_n = sum(1 for s in steps if s.get("source") != "web")
        total = len(steps)
        return {
            "corpus_pct": round(corpus_n / total * 100),
            "web_pct":    round((total - corpus_n) / total * 100),
        }

    def _build_clarify_response(self, query: str, classification: dict) -> dict:
        """
        Intent 4 -- Builds the CLARIFY response when the classifier decided
        "clarify". No retrieval, no roadmap generation: the same dict shape as
        generate_roadmap_with_trace so adapters/evaluators/frontends do not need
        a special branch, but with mode="clarify", 0 steps, the CONTEXTUAL
        clarifying question and the intent options so the user can CONTINUE:
          - type = "source_ambiguous": options [1, 2, 3].
          - type = "external_two_way": options [2, 3].
          - type = "no_roadmap":       no options (conversational reply; actually
            served by _build_no_retrieval_response).

        The question and the option labels come from the classifier's English
        payload (single source of truth); typed_options just restates them
        normalized for the public contract.
        """
        clarify = (classification or {}).get("clarify") or {}
        ctype = clarify.get("type", "source_ambiguous")
        fallback_questions = {
            "source_ambiguous": (
                "I could answer this with Bamboo's internal documentation, with "
                "external information, or with a combination of both. How would "
                "you like me to generate the roadmap?"
            ),
            "external_two_way": (
                "You asked for external or updated information. Do you want me to "
                "use it as a complement to Bamboo's Knowledge Base, or as the "
                "main basis of the answer?"
            ),
            "no_roadmap": (
                "This doesn't look like a request to generate a technical roadmap."
            ),
        }
        question = clarify.get("question") or fallback_questions.get(ctype) or fallback_questions["source_ambiguous"]
        # Options arrive as ints (classifier payload) or dicts {id, label,
        # description} — normalize to ints for the public contract.
        raw_options = clarify.get("options") or []
        options = [
            o["id"] if isinstance(o, dict) and "id" in o else (o if isinstance(o, int) else str(o))
            for o in raw_options
        ]
        label_by_id = {
            o["id"]: o["label"] for o in raw_options
            if isinstance(o, dict) and "id" in o and "label" in o
        }
        typed_options = [
            {"intent": opt, "label": label_by_id.get(opt, str(opt))}
            for opt in sorted(options)
        ]
        roadmap = {
            "title": "Clarification needed before generating the roadmap",
            "question": query,
            "steps": [],
            "clarifying_question": question,
            "clarify": {
                "type":          ctype,
                "question":      question,
                "options":       sorted(options),
                "typed_options": typed_options,
            },
            "sources": {
                "corpus_pct":      0,
                "web_pct":         0,
                "mode":            "clarify",
                "modality":        "ambiguous",
                "best_score":      0.0,
                "intent":          4,
                "n_corpus_chunks": 0,
                "n_web_chunks":    0,
            },
        }
        print(f"  [Intent 4 / Clarify] type={ctype} -- {question}")
        return {
            "question":         query,
            "query_intent":     {"intent": "ambiguous", "confidence": classification.get("confidence", 0.0)},
            "source_intent":    classification,
            "refined_question": "",
            "contexts":         [],
            "corpus_contexts":  [],
            "web_contexts":     [],
            "retrieval": {
                "mode":            "clarify",
                "modality":        "ambiguous",
                "best_score":      0.0,
                "n_contexts":      0,
                "n_corpus_chunks": 0,
                "n_web_chunks":    0,
            },
            "roadmap": roadmap,
        }

    def _build_no_retrieval_response(self, query: str, classification: dict) -> dict:
        """
        Intent 4 (type no_roadmap) -- Builds the conversational reply for queries
        that need NO retrieval (greetings, off-topic, personal questions; decided
        by the classifier pre-gate or by the LLM). No source question, no roadmap
        generation: the same dict shape as generate_roadmap_with_trace with
        mode="no_retrieval" so adapters/evaluators do not need a special branch.
        """
        clarify = (classification or {}).get("clarify") or {}
        message = clarify.get("question") or (
            "This query does not require searching for technical knowledge in the "
            "knowledge base. What process or topic would you like help with?"
        )
        roadmap = {
            "title": "Non-technical query",
            "steps": [{
                "id": "no_retrieval",
                "label": "Conversational reply",
                "description": message,
                "type": "proceso",
                "key_points": ["Knowledge base was not queried", "No roadmap was generated"],
            }],
            "sources": {
                "corpus_pct":      0,
                "web_pct":         0,
                "mode":            "no_retrieval",
                "modality":        "no_retrieval",
                "best_score":      0.0,
                "intent":          4,
                "n_corpus_chunks": 0,
                "n_web_chunks":    0,
            },
        }
        print(f"  [Intent 4 / No Retrieval] Non-technical query -- no retrieval, no source question.")
        return {
            "question":         query,
            "query_intent":     {"intent": "no_retrieval", "confidence": classification.get("confidence", 1.0)},
            "source_intent":    classification,
            "refined_question": "",
            "contexts":         [],
            "corpus_contexts":  [],
            "web_contexts":     [],
            "retrieval": {
                "mode":            "no_retrieval",
                "modality":        "no_retrieval",
                "best_score":      0.0,
                "n_contexts":      0,
                "n_corpus_chunks": 0,
                "n_web_chunks":    0,
            },
            "roadmap": roadmap,
        }

    def generate_roadmap_with_trace(self, query: str, intent: int = None) -> dict:
        """
        Genera el roadmap y devuelve la traza completa del RAG.

        FLUJO (el clasificador de intent de fuente es INDEPENDIENTE de la
        generación de roadmaps):
          1. Clasificación pre-retrieval: decide intent 1/2/3 (proceed), Intent 4
             (clarify) o no_retrieval. Si `intent` viene 1/2/3, el usuario está
             CONTINUANDO un Intent 4 ya clarificado (seleccionó la fuente en el
             frontend): se construye la clasificación desde ese intent y se omite
             el re-clasificado.
          2. decision == "no_retrieval" -> _build_no_retrieval_response (sin
             retrieval, sin generar).
          3. decision == "clarify"      -> _build_clarify_response (sin retrieval,
             respuesta contextual con opciones de intent para continuar).
          4. proceed -> planificación de generación (classify_query_intent +
             rewrite_query, DESACOPLADO del clasificador), retrieval con política
             de fuente del intent (get_contexts_with_sources), adaptación runtime
             si la KB no la cubre (coverage_adaptation + notice) y build_roadmap
             con el sesgo de la política.

        La traza expone source_intent (clasificador de fuente), query_intent
        (planificación de generación), coverage_adaptation y notice, para que los
        evaluadores externos y el frontend usen exactamente lo que el generador vio.
        """
        classification = None
        src = None
        try:
            # --- Paso 1: clasificación de intención de fuente (pre-retrieval) ---
            if intent in (1, 2, 3):
                classification = {
                    "intent": intent,
                    "label": {1: "kb_only", 2: "kb_plus_external", 3: "external_plus_kb"}[intent],
                    "confidence": 1.0,
                    "rationale": "intent seleccionado por el usuario (continuación de aclaración)",
                    "decision": "proceed",
                    "retrieval_necessity": True,
                    "clarify": None,
                    "source_policy": policy_for_intent(intent),
                    "pre_gate": False,
                    "continuation": True,
                }
            else:
                classification = self.classifier.classify(query)

            decision = classification.get("decision", "clarify")
            if decision == "no_retrieval":
                return self._build_no_retrieval_response(query, classification)
            if decision == "clarify":
                return self._build_clarify_response(query, classification)

            # --- Paso 2: planificación de generación (independiente del intent) ---
            query_intent = (
                self.classify_query_intent(query)
                if QUERY_INTENT_ENABLED
                else {"intent": "disabled", "confidence": 0.0}
            )
            refined_query = self.rewrite_query(query, query_intent=query_intent)
            print(f"  [Retrieval] Buscando con consulta refinada...")
            src = self.get_contexts_with_sources(refined_query, original_query=query, classification=classification)

            roadmap = self.build_roadmap(
                query, refined_query, src["corpus_contexts"], src["web_contexts"],
                query_intent=query_intent, bias_target=src.get("bias_target"),
            )

            # Contar fuente por paso -- ya viene anotada desde build_roadmap (RoadmapStep.source)
            pct = self._compute_source_pct(roadmap, src["corpus_contexts"], src["web_contexts"])
            roadmap["sources"] = {
                "corpus_pct":      pct["corpus_pct"],
                "web_pct":         pct["web_pct"],
                "mode":            src["mode"],
                "modality":        src.get("modality", "ambiguous"),
                "best_score":      src["best_score"],
                "intent":          src.get("intent"),
                "n_corpus_chunks": len(src["corpus_contexts"]),
                "n_web_chunks":    len(src["web_contexts"]),
            }
            if src.get("notice"):
                roadmap["notice"] = src["notice"]

            print(f"  [Fuentes] {pct['corpus_pct']}% corpus / {pct['web_pct']}% web  (modo: {MODE_LABELS.get(src['mode'], src['mode'])}, intent: {classification.get('intent')})")
            return {
                "question":            query,
                "query_intent":        query_intent,
                "source_intent":       classification,
                "refined_question":    refined_query,
                "contexts":            src["contexts"],
                "corpus_contexts":     src["corpus_contexts"],
                "web_contexts":        src["web_contexts"],
                "coverage_adaptation": src.get("coverage_adaptation"),
                "retrieval": {
                    "mode":            src["mode"],
                    "modality":        src.get("modality", "ambiguous"),
                    "best_score":      src["best_score"],
                    "n_contexts":      len(src["contexts"]),
                    "n_corpus_chunks": len(src["corpus_contexts"]),
                    "n_web_chunks":    len(src["web_contexts"]),
                },
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
                "source_intent":    classification if isinstance(classification, dict) else {},
                "refined_question": "",
                "contexts":         [],
                "corpus_contexts":  [],
                "web_contexts":     [],
                "coverage_adaptation": (src or {}).get("coverage_adaptation"),
                "retrieval":        {"mode": "error", "best_score": 0, "n_contexts": 0},
                "roadmap":          roadmap,
            }

    def generate_roadmap(self, query: str, intent: int = None):
        """Mantiene compatibilidad: devuelve solo el roadmap. `intent` (1/2/3)
        se usa para CONTINUAR un Intent 4 clarificado (fuente elegida por el
        usuario en el frontend)."""
        return self.generate_roadmap_with_trace(query, intent=intent)["roadmap"]

    def build_roadmap(self, original_query: str, refined_query: str, corpus_contexts: list, web_contexts: list, query_intent: dict = None, bias_target: dict = None):
        """Construye el roadmap con la consulta refinada y los contextos ya recuperados.

        bias_target: orientación de fuente que el generador debe respetar, viene de
        la tabla de fusión (get_contexts_with_sources). {"favor": "corpus"|"web",
        "min_pct": N} -- controla el peso relativo de cada bloque en la generación
        y por tanto los porcentajes corpus_pct/web_pct que mide el evaluador.

        corpus_contexts/web_contexts van SEPARADOS (no ya mezclados en una sola
        lista) para poder etiquetar cada bloque en el prompt (BLOQUE CORPUS /
        BLOQUE WEB) y pedirle a esta misma llamada que anote la procedencia de
        cada paso (RoadmapStep.source) mientras lo escribe -- antes esto lo
        hacía attribute_step_sources() en una SEGUNDA llamada que releía el
        roadmap ya terminado a ciegas, sin ver bias_target ni el detalle de qué
        contexto usó el generador en cada paso. Ver _compute_source_pct()."""
        try:
            query_intent = query_intent or {
                "intent": "exploratory",
                "roadmap_goal": original_query,
                "generation_guidance": "Construye un roadmap técnico, ordenado y accionable.",
            }
            if QUERY_INTENT_ENABLED and query_intent.get("intent") not in ("disabled", "error"):
                intent_context = (
                    f"Tipo de intención: {query_intent.get('intent', 'exploratory')}\n"
                    f"Objetivo accionable: {query_intent.get('roadmap_goal', original_query)}\n"
                    f"Guía para el roadmap: {query_intent.get('generation_guidance', '')}"
                )
                intent_section = f"""\
═══════════════════════════════════════════════════
INTENCIÓN Y DIRECCIÓN DEL ROADMAP
═══════════════════════════════════════════════════
{intent_context}

"""
            else:
                intent_context = ""
                intent_section = ""

            # Orientación de fuente (bias) inyectada por la tabla de fusión.
            bias_section = ""
            if bias_target:
                favor = bias_target.get("favor")
                if favor == "web":
                    bias_section = """\
═══════════════════════════════════════════════════
ORIENTACIÓN DE FUENTE (la web manda)
═══════════════════════════════════════════════════
El bloque WEB es la fuente PRIORITARIA de este roadmap: apóyate en él para el
detalle técnico, comandos y criterios principales. Usa el bloque corpus solo si
aporta un hecho específico de la empresa; no fuerces contenido interno que no esté
presente en el contexto.

"""
                elif favor == "corpus":
                    min_pct = bias_target.get("min_pct", 70)
                    target_pct = min(min_pct + 5, 90)
                    bias_section = f"""\
═══════════════════════════════════════════════════
ORIENTACIÓN DE FUENTE (el corpus es la base)
═══════════════════════════════════════════════════
El bloque CORPUS es la BASE de este roadmap: usa sus pasos, hechos y criterios
como fuente principal. Al menos el {target_pct}% de los pasos debe derivarse del
corpus, y la CONCLUSIÓN o decisión central debe apoyarse en los hechos que el
corpus documenta (no en la web).
Escribe los pasos en este orden: primero todos los que derivan de los hechos,
criterios y reglas del corpus, y solo hacia el final incluye el detalle puntual
del hueco que venga del bloque WEB (márcale "[inferido]"). No reescribas lo que
el corpus ya documenta con contenido externo, y no dejes que un fragmento web
único se convierta en la mitad del roadmap.

"""
            if web_contexts:
                corpus_block = "\n\n---\n\n".join(corpus_contexts) if corpus_contexts else "(sin contenido del corpus para esta consulta)"
                web_block = "\n\n---\n\n".join(web_contexts)
                if bias_target and bias_target.get("favor") == "web":
                    context = (
                        f"BLOQUE WEB (búsqueda externa):\n{web_block}\n\n"
                        f"BLOQUE CORPUS (base interna):\n{corpus_block}"
                    )
                else:
                    context = (
                        f"BLOQUE CORPUS (base interna):\n{corpus_block}\n\n"
                        f"BLOQUE WEB (búsqueda externa):\n{web_block}"
                    )
            elif corpus_contexts:
                context = "BLOQUE CORPUS (base interna):\n" + "\n\n---\n\n".join(corpus_contexts)
            else:
                context = (
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
{intent_section}{bias_section}═══════════════════════════════════════════════════
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

  source (obligatorio, uno por paso):
    Decide "corpus" o "web" según de dónde sacaste el hecho/criterio CONCRETO
    que usaste para ESTE paso -- no por el estilo de redacción (todo se
    escribe en el mismo registro técnico-formal, eso no es señal de
    procedencia). Si el paso mezcla ambos bloques en partes similares, o no
    hay BLOQUE WEB en el contexto de arriba, usa "corpus".

REGLAS GENERALES:
  - Genera entre 6 y 12 pasos (ni muy pocos ni demasiados)
  - Prioriza información del contexto recuperado sobre conocimiento genérico
  - Si el contexto no cubre un paso, indícalo: "[inferido]" al inicio del label
  - Cada paso debe ser ejecutable de forma independiente y verificable
  - NUNCA uses frases de relleno ni pasos obvios sin detalle técnico

ANCLAJE ESTRICTO (grounding):
  - El CONTEXTO TÉCNICO de arriba ya fue filtrado y validado como relevante para
    esta consulta -- básate en él como fuente principal, no en conocimiento
    genérico del dominio que no esté respaldado ahí.
  - Todo hecho, cifra, comando o criterio específico que uses debe poder
    rastrearse a algo presente en el CONTEXTO. Si necesitas un dato puntual que
    el contexto no menciona, marca ese key_point o esa parte del label con
    "[inferido]" -- NUNCA presentes una inferencia como si fuera un hecho
    documentado.
  - No inventes nombres de documentos, cifras exactas o afirmaciones atribuidas
    al corpus si no aparecen literalmente o de forma claramente derivable en el
    CONTEXTO TÉCNICO.

{format_instructions}
"""
            prompt = PromptTemplate(
                template=template,
                input_variables=["original_query", "refined_query", "intent_section", "bias_section", "context"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()},
            )

            chain = prompt | self.llm_main | self.parser
            result = chain.invoke({
                "original_query": original_query,
                "refined_query":  refined_query,
                "intent_section": intent_section,
                "bias_section":   bias_section,
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