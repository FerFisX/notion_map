"""
Métricas de similitud query-respuesta: TF-IDF cosine y Sentence-Transformer.
Métrica automática que no necesita ground truth. Ref: Rothman (2024), cap. 1-2.
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


def tfidf_cosine(text_a: str, text_b: str) -> float:
    """Similitud coseno basada en TF-IDF (solapamiento de palabras). Retorna 0.0-1.0."""
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


def semantic_cosine(text_a: str, text_b: str) -> float:
    """Similitud coseno con embeddings semánticos (all-MiniLM-L6-v2). Retorna 0.0-1.0."""
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


def query_answer_relevance(question: str, answer: str) -> dict:
    """Evalúa qué tan relacionada está la respuesta generada con la pregunta."""
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
