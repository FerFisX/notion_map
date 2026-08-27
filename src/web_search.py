"""
Búsqueda web de respaldo (fallback) cuando el corpus local no cubre la consulta.

Usa DuckDuckGo (sin API key). El contenido recuperado es efímero: se usa solo
para responder la consulta actual y NO se guarda en la base de conocimiento.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._-]{2,}", re.IGNORECASE)
_DOCUMENTATION_HOST_PREFIXES = (
    "docs.",
    "developer.",
    "developers.",
    "learn.",
    "support.",
)
_DOCUMENTATION_PATH_MARKERS = (
    "/api/",
    "/docs/",
    "/documentation/",
    "/reference/",
    "/userguide/",
)
_SECONDARY_PATH_MARKERS = ("/article/", "/articles/", "/blog/", "/blogs/")

def prepare_web_query(query: str, max_chars: int = 300) -> str:
    """Return a compact, provider-independent query suitable for a web engine.

    The caller should pass the original user question instead of an LLM-expanded
    retrieval query. Normalization is deliberately deterministic so web search
    behavior does not depend on the configured LLM provider or model.
    """
    normalized = " ".join(str(query or "").split())
    if not normalized or len(normalized) <= max_chars:
        return normalized

    truncated = normalized[:max_chars]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.rstrip(" ,;:-")


def rank_web_results(query: str, results: list[dict]) -> list[dict]:
    """Rank web results using deterministic source-quality and relevance signals."""
    query_tokens = set(_TOKEN_RE.findall(query.lower()))
    current_information_requested = bool(
        query_tokens
        & {"actual", "actualizada", "current", "latest", "updated", "vigente"}
    )

    def score(result: dict) -> float:
        title = str(result.get("title", "")).lower()
        body = str(result.get("body", "")).lower()
        href = str(result.get("href", ""))
        parsed = urlparse(href)
        host = parsed.netloc.lower().removeprefix("www.")
        path = parsed.path.lower()

        value = 0.0
        if parsed.scheme == "https":
            value += 0.25
        if host.startswith(_DOCUMENTATION_HOST_PREFIXES):
            value += 4.0
        if any(marker in f"{path}/" for marker in _DOCUMENTATION_PATH_MARKERS):
            value += 2.0

        host_matches = sum(token in host for token in query_tokens)
        title_matches = sum(token in title for token in query_tokens)
        body_matches = sum(token in body for token in query_tokens)
        value += min(host_matches * 1.5, 3.0)
        value += min(title_matches * 0.35, 2.5)
        value += min(body_matches * 0.08, 0.8)

        if any(marker in f"{path}/" for marker in _SECONDARY_PATH_MARKERS):
            value -= 1.0
        if "legacy" in title or "legacy" in path or "heredado" in title:
            value -= 7.0 if current_information_requested else 1.0
        return value

    deduplicated = []
    seen_urls = set()
    for result in results:
        href = str(result.get("href", "")).strip()
        key = href.rstrip("/").lower()
        if not key or key in seen_urls:
            continue
        seen_urls.add(key)
        deduplicated.append(result)
    return sorted(deduplicated, key=score, reverse=True)


def search_web(query: str, max_results: int = 4, fetch_full: bool = True) -> list[str]:
    """
    Busca en internet y devuelve fragmentos de texto listos para usar como contexto.

    Cada fragmento incluye el título y la URL de origen para trazabilidad.
    Si fetch_full está activo, intenta descargar y convertir la página completa
    con markitdown; si falla, usa el resumen que devuelve el buscador.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        print("  [Web] Librería 'ddgs' no instalada. Corre: pip install ddgs")
        return []

    try:
        pool_size = max(8, max_results * 3)
        results = list(DDGS().text(query, max_results=pool_size))
    except Exception as e:
        print(f"  [Web] Error en la búsqueda: {e}")
        return []

    if not results:
        print("  [Web] Sin resultados.")
        return []

    results = rank_web_results(query, results)[:max_results]

    contexts = []
    for r in results:
        title = r.get("title", "")
        href  = r.get("href", "")
        body  = r.get("body", "")
        content = body

        if fetch_full and href:
            full = _fetch_page(href)
            if full:
                content = full

        contexts.append(f"[Fuente web: {title} — {href}]\n{content}".strip())

    print(f"  [Web] {len(contexts)} fuentes recuperadas de internet (efímeras).")
    return contexts


def _fetch_page(url: str, max_chars: int = 3000) -> str:
    """Descarga una página y la convierte a texto con markitdown. '' si falla."""
    try:
        from markitdown import MarkItDown
        text = MarkItDown().convert(url).text_content or ""
        return text.strip()[:max_chars]
    except Exception:
        return ""
