"""
Búsqueda web de respaldo (fallback) cuando el corpus local no cubre la consulta.

Usa DuckDuckGo (sin API key). El contenido recuperado es efímero: se usa solo
para responder la consulta actual y NO se guarda en la base de conocimiento.
"""

from __future__ import annotations


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
        results = list(DDGS().text(query, max_results=max_results))
    except Exception as e:
        print(f"  [Web] Error en la búsqueda: {e}")
        return []

    if not results:
        print("  [Web] Sin resultados.")
        return []

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
