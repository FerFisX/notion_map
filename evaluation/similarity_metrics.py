"""
similarity_metrics.py — Métricas de similitud query↔respuesta.

Basado en Rothman (2024), cap. 1-2:
  "Evaluating the output with cosine similarity"

Implementa DOS formas que el libro contrasta directamente:
  1. TF-IDF cosine          — basado en frecuencia de palabras (rápido, superficial)
  2. Sentence-Transformer   — basado en embeddings semánticos (capta significado)

El libro muestra que para la misma consulta:
  TF-IDF              -> ~0.13  (no entiende sinónimos ni contexto)
  Sentence-Transformer -> ~0.74  (capta relación semántica real)

Sirve como métrica AUTOMÁTICA sin necesidad de ground truth:
mide cuánto se relaciona la respuesta generada con la pregunta del usuario.
"""

from __future__ import annotations

import warnings

# Modelo compartido (lazy load para no pagar el costo si no se usa)
_st_model = None


def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


# ── 1. TF-IDF cosine (baseline del libro) ─────────────────────────────────────

def tfidf_cosine(text_a: str, text_b: str) -> float:
    """
    Similitud coseno basada en TF-IDF.
    Mide solapamiento de palabras, NO significado.
    Retorna 0.0 - 1.0
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not text_a.strip() or not text_b.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([text_a, text_b])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
        return round(float(sim[0][0]), 4)
    except ValueError:
        # Ocurre si tras quitar stopwords no queda vocabulario común
        return 0.0


# ── 2. Sentence-Transformer cosine (semántico) ────────────────────────────────

def semantic_cosine(text_a: str, text_b: str) -> float:
    """
    Similitud coseno basada en embeddings semánticos (all-MiniLM-L6-v2).
    Capta significado, sinónimos y contexto.
    Retorna 0.0 - 1.0
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if not text_a.strip() or not text_b.strip():
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _get_st_model()
        emb_a = model.encode(text_a)
        emb_b = model.encode(text_b)
    sim = cosine_similarity([emb_a], [emb_b])
    return round(float(sim[0][0]), 4)


# ── Métrica combinada query↔respuesta ─────────────────────────────────────────

def query_answer_relevance(question: str, answer: str) -> dict:
    """
    Evalúa qué tan relacionada está la RESPUESTA generada con la PREGUNTA.
    Métrica automática (no necesita ground truth).

    Returns:
        {
          "tfidf":           float 0-1,   # baseline superficial
          "semantic":        float 0-1,   # semántico (el que importa)
          "semantic_0_10":   float 0-10,  # escalado para combinar con otras métricas
          "gain":            float,        # cuánto mejora semantic vs tfidf
          "interpretation":  str,
        }
    """
    tf = tfidf_cosine(question, answer)
    se = semantic_cosine(question, answer)

    if se >= 0.6:
        interp = "Alta relevancia — la respuesta aborda directamente la pregunta"
    elif se >= 0.4:
        interp = "Relevancia media — la respuesta se relaciona pero podría estar más enfocada"
    else:
        interp = "Baja relevancia — la respuesta se desvía de la pregunta"

    return {
        "tfidf":          tf,
        "semantic":       se,
        "semantic_0_10":  round(se * 10, 2),
        "gain":           round(se - tf, 4),
        "interpretation": interp,
    }
