"""
kb_coverage.py — Coverage de KB para el clasificador CRAG (Interpretación A).

Encapsula el acceso al vector store (ChromaDB) de la KB REAL y expone un
`coverage_score`: similitud semántica entre una intención core y los
documentos/chunks recuperados de la KB. Es la entrada del CRAG check.

Principios:
  * DI (inyección de dependencia): la instancia se construye con `db_path` y
    `model_name` explicitos, y se puede apuntar a un dataset de prueba (para el
    evaluador aislado / métrica de FP) sin tocar la KB de produccion.
  * Lazy singleton: el indice/embeddings se cargan una sola vez por proceso.
  * Ruta canónica: apunta SIEMPRE a `vectorstore/chroma_db/` (por defecto),
    NUNCA a `data/chroma/` (remanente obsoleto). Esto es un punto de
    configuracion de ruta, no de datos.
"""

import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "vectorstore", "chroma_db")
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class KBCoverage:
    """Computa el coverage de una intencion core contra la KB real (ChromaDB)."""

    _shared_embedders = {}

    def __init__(self, db_path: str | None = None, model_name: str | None = None,
                 top_k: int = 10):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.model_name = model_name or DEFAULT_MODEL
        self.top_k = top_k
        self._db = None

    # -- lazy load ---------------------------------------------------------
    def _ensure_db(self):
        """Carga (lazy) ChromaDB + embeddings una sola vez por (db_path, model)."""
        if self._db is not None:
            return self._db
        key = (self.db_path, self.model_name)
        if key not in KBCoverage._shared_embedders:
            from langchain_huggingface import HuggingFaceEmbeddings
            KBCoverage._shared_embedders[key] = HuggingFaceEmbeddings(
                model_name=self.model_name
            )
        embeddings = KBCoverage._shared_embedders[key]
        from langchain_chroma import Chroma
        self._db = Chroma(persist_directory=self.db_path, embedding_function=embeddings)
        return self._db

    # -- API ---------------------------------------------------------------
    def raw_scores(self, core: str, k: int | None = None) -> list[float]:
        """Similitudes (vector_score de ChromaDB) para los top-k chunks
        recuperados contra `core`. Lista vacia si no hay KB/chunks."""
        core = (core or "").strip()
        if not core:
            return []
        db = self._ensure_db()
        try:
            scored = db.similarity_search_with_relevance_scores(core, k=k or self.top_k)
        except Exception:
            return []
        return [round(float(s), 4) for _, s in scored]

    def coverage_score(self, core: str, agg: str = "max", k: int | None = None) -> float:
        """Coverage agregado de la intencion core contra la KB.

        agg: "max" (recomendado, el tema esta minimamente representado),
             "mean", o "p90". Devuelve 0.0 si no hay chunks/score.
        """
        scores = self.raw_scores(core, k=k)
        if not scores:
            return 0.0
        if agg == "max":
            return max(scores)
        if agg == "mean":
            return round(sum(scores) / len(scores), 4)
        if agg == "p90":
            import statistics
            return round(statistics.quantiles(scores, n=10)[8], 4)
        return max(scores)


# -- extractor de core (intencion tematica pura, sin clausulas de fuente) ---
# Recorta prefijos de clausula de fuente conocidos del inicio de la query,
# dejando la "intencion core" (el tema real) sobre el que se mide la cobertura.
# La salvaguarda: si no se detecta ninguna clausula conocida, se usa la query
# completa tal cual (nunca se fabrica un core que no exista en el original).
_SOURCE_PREFIX_RE = re.compile(
    r"^\s*(?:"
    # "according to X," / "per X," / "based on X," / "según X," (con coma)
    r"(?:according|per|según|segun)\s+(?:to\s+)?[^,.]*?,\s*"
    r"|based\s+(?:on|off)\s+[^,.]*?,\s*"
    # "going only by X," / "sticking strictly to X," / "using nothing but X,"
    r"(?:going|sticking|using|starting|grounding)\s+.*?,\s*"
    # frases con verbo inicial tipo "our internal notes say ..."
    r"(?:\w+\s+)*?our (?:internal )?(?:notes|docs|documentation|runbook|materials)\s+"
    r"(?:say|stated?|claims?|indicate|sug|\w+)?\s+"
    r")",
    re.IGNORECASE,
)
# Expresiones fuente en medio (p. ej. "según", "as our docs state") y su
# recorte: no las eliminamos si rompen la gramatica del core; para fase 1 solo
# atacamos prefijos, que es donde el benchmark mostro que vive la mayoria.
_KNOWN_SOURCE_PREFIXES = (
    "according to our documentation,",
    "according to our internal documentation,",
    "per our internal runbook,",
    "our internal notes say",
    "our internal notes on this",
    "our notes say",
    "going only by our own written materials,",
    "going only by what we have documented,",
    "sticking strictly to what's already recorded internally,",
    "sticking strictly to our documented",
    "using nothing but what we have documented for this,",
    "using only what our",
    "based on our internal",
    "as our internal",
    "as our documented",
    "per our documented",
)


def extract_core(query: str) -> tuple[str, bool]:
    """Devuelve (core, extraido). `extraido` es True si se recorto algo.

    El core es la intencion tematica de la query sin clausulas de fuente
    iniciales ("per our internal runbook,", "according to our documentation,").
    Si no hay clausulas reconocibles, el core es la query original y
    `extraido` es False (fase 1: no inventamos recortes no seguros).
    """
    q = (query or "").strip()
    if not q:
        return q, False
    low = q.lower()
    for prefix in _KNOWN_SOURCE_PREFIXES:
        if low.startswith(prefix) and len(q) > len(prefix) + 4:
            return q[len(prefix):].strip(" .,;:-").strip(), True
    m = _SOURCE_PREFIX_RE.match(q)
    if m and len(q) - m.end() >= 5:
        return q[m.end():].strip(" .,;:-").strip(), True
    return q, False


# -- handle inyectable (forma funcional, para el clasificador) -------------
def make_coverage_retriever(db_path: str | None = None, model_name: str | None = None,
                            agg: str = "max", top_k: int = 10, threshold: float = 0.0):
    """Devuelve un callable `coverage(core: str) -> float` listo para inyectar
    en IntentClassifier. Default tolerante si algo falla: coverage=1.0 (caso 1),
    para no romper tests/uso sin KB durante la migracion."""
    impl = KBCoverage(db_path=db_path, model_name=model_name, top_k=top_k)

    def coverage(core: str) -> float:
        try:
            return impl.coverage_score(core, agg=agg)
        except Exception:
            return 1.0

    return coverage
