"""
Búsqueda web de respaldo (fallback) cuando el corpus local no cubre la consulta.

Usa DuckDuckGo (sin API key). El contenido recuperado es efímero: se usa solo
para responder la consulta actual y NO se guarda en la base de conocimiento.
"""

from __future__ import annotations

import re
import unicodedata
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
_QUERY_STOPWORDS = {
    "a", "al", "ante", "como", "con", "considerando", "cual", "cuales",
    "cuando", "de", "del", "desde", "donde", "el", "en", "entre", "es",
    "esta", "estas", "este", "estos", "hacer", "la", "las", "lo", "los",
    "manera", "me", "mediante", "mi", "para", "pero", "por", "puedo", "que",
    "se", "segun", "ser", "sin", "sobre", "su", "sus", "un", "una", "usar",
    "and", "can", "do", "for", "from", "how", "in", "into", "is", "of",
    "on", "or", "the", "to", "using", "what", "when", "with",
}

_CURRENT_INFORMATION_PHRASES = (
    "actualmente",
    "vigente",
    "documentacion vigente",
    "informacion vigente",
    "mas reciente",
    "modelo vigente",
    "modelos vigentes",
    "precio vigente",
    "precios vigentes",
    "version actual",
    "version vigente",
    "current version",
    "currently available",
    "latest release",
    "latest update",
    "up to date",
)


def _search_normalized_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    without_accents = "".join(
        character for character in decomposed
        if not unicodedata.combining(character)
    )
    return " ".join(without_accents.casefold().split())


def requires_current_web_evidence(query: str) -> bool:
    """Return whether the request explicitly requires time-sensitive evidence."""
    normalized = _search_normalized_text(query)
    return any(phrase in normalized for phrase in _CURRENT_INFORMATION_PHRASES)

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


def compact_web_query(query: str, max_terms: int = 14) -> str:
    """Reduce a natural-language question to stable search terms.

    The transformation is deterministic and preserves product names, acronyms,
    versions, and technical terms without depending on the active LLM.
    """
    normalized = prepare_web_query(query)
    tokens = re.findall(r"[^\W_]+(?:[.+#/-][^\W_]+)*", normalized, re.UNICODE)
    selected: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = "".join(
            character
            for character in unicodedata.normalize("NFKD", token)
            if not unicodedata.combining(character)
        ).casefold()
        if key in _QUERY_STOPWORDS or key in seen:
            continue
        if len(key) < 2 and not any(character.isdigit() for character in key):
            continue
        selected.append(token)
        seen.add(key)
        if len(selected) >= max_terms:
            break
    return " ".join(selected)


def build_web_query_candidates(
    primary_query: str,
    refined_query: str | None = None,
    max_variants: int = 3,
) -> list[str]:
    """Build ordered, provider-independent search fallbacks."""
    raw_candidates = [
        prepare_web_query(primary_query),
        compact_web_query(primary_query),
        compact_web_query(refined_query or ""),
    ]
    candidates: list[str] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        key = candidate.casefold().strip()
        if not key or key in seen:
            continue
        candidates.append(candidate)
        seen.add(key)
        if len(candidates) >= max_variants:
            break
    return candidates


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


def search_web_sources(
    query: str,
    max_results: int = 4,
    fetch_full: bool = True,
) -> list[dict[str, str]]:
    """Return structured, traceable web evidence.

    Search snippets are used only when full-page extraction fails.  The caller
    can distinguish that weaker evidence through ``content_origin``.
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
    sources: list[dict[str, str]] = []
    for index, result in enumerate(results, 1):
        title = str(result.get("title", "")).strip()
        url = str(result.get("href", "")).strip()
        snippet = str(result.get("body", "")).strip()
        full_content = _fetch_page(url) if fetch_full and url else ""
        text = full_content or snippet
        if not url or not text:
            continue
        sources.append({
            "id": f"web_{index}",
            "source_type": "web",
            "title": title or url,
            "url": url,
            "text": text,
            "content_origin": "page" if full_content else "search_snippet",
        })

    print(f"  [Web] {len(sources)} fuentes recuperadas de internet (efímeras).")
    return sources


def search_web(query: str, max_results: int = 4, fetch_full: bool = True) -> list[str]:
    """
    Busca en internet y devuelve fragmentos de texto listos para usar como contexto.

    Cada fragmento incluye el título y la URL de origen para trazabilidad.
    Si fetch_full está activo, intenta descargar y convertir la página completa
    con markitdown; si falla, usa el resumen que devuelve el buscador.
    """
    return [
        f"[Fuente web: {source['title']} — {source['url']}]\n{source['text']}"
        for source in search_web_sources(query, max_results, fetch_full)
    ]


def _fetch_page(url: str, max_chars: int = 3000) -> str:
    """Descarga una página y la convierte a texto con markitdown. '' si falla."""
    try:
        from markitdown import MarkItDown
        text = MarkItDown().convert(url).text_content or ""
        return text.strip()[:max_chars]
    except Exception:
        return ""
