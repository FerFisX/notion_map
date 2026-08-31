"""Zero-cost calibration report for roadmap source-selection thresholds.

This module uses only the persisted Chroma database and the local embedding
model. It never invokes an LLM or performs a web request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from evaluation.dataset import EVAL_SAMPLES
from src.rag_engine import DB_PATH
from src.web_search import requires_current_web_evidence


SAMPLE_INDICES = range(15, 45)


def _best_score(vector_db: Chroma, question: str) -> float:
    results = vector_db.similarity_search_with_relevance_scores(question, k=10)
    return max((float(score) for _, score in results), default=0.0)


def _candidate_pairs() -> list[tuple[float, float]]:
    values = [round(value / 100, 2) for value in range(30, 71, 5)]
    return [(low, high) for low in values for high in values if low < high]


def _print_benchmark_deltas(vector_db: Chroma, paths: list[str]) -> None:
    if not paths:
        return
    print("\nORIGINAL VS REFINED SCORES")
    print("INDEX  ORIGINAL  REFINED   DELTA")
    for raw_path in paths:
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        for result in payload.get("results", []):
            original = _best_score(vector_db, str(result.get("question", "")))
            refined = _best_score(
                vector_db, str(result.get("refined_question", ""))
            )
            print(
                f"{int(result['sample_index']):>5}  {original:.4f}    "
                f"{refined:.4f}   {refined - original:+.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-json", action="append", default=[])
    args = parser.parse_args()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )
    rows: list[dict] = []
    for index in SAMPLE_INDICES:
        sample = EVAL_SAMPLES[index - 1]
        rows.append({
            "index": index,
            "requires_web": bool(sample.requires_web),
            "score": _best_score(vector_db, sample.question),
            "category": sample.category,
        })

    print("INDEX  LABEL   SCORE   CATEGORY")
    for row in sorted(rows, key=lambda value: value["score"]):
        label = "WEB" if row["requires_web"] else "CORPUS"
        print(
            f"{row['index']:>5}  {label:<6}  {row['score']:.4f}  "
            f"{row['category']}"
        )

    corpus_scores = [row["score"] for row in rows if not row["requires_web"]]
    web_scores = [row["score"] for row in rows if row["requires_web"]]
    print(
        "\nCORPUS "
        f"n={len(corpus_scores)} min={min(corpus_scores):.4f} "
        f"mean={mean(corpus_scores):.4f} max={max(corpus_scores):.4f}"
    )
    print(
        "WEB    "
        f"n={len(web_scores)} min={min(web_scores):.4f} "
        f"mean={mean(web_scores):.4f} max={max(web_scores):.4f}"
    )

    guard_false_positives = [
        row["index"] for row in rows
        if not row["requires_web"]
        and requires_current_web_evidence(
            EVAL_SAMPLES[row["index"] - 1].question
        )
    ]
    guard_false_negatives = [
        row["index"] for row in rows
        if row["requires_web"]
        and not requires_current_web_evidence(
            EVAL_SAMPLES[row["index"] - 1].question
        )
    ]
    print(
        "CURRENTNESS GUARD "
        f"false_positives={guard_false_positives} "
        f"false_negatives={guard_false_negatives}"
    )
    if guard_false_positives or guard_false_negatives:
        raise AssertionError(
            "Current-information routing guard does not match the controlled "
            "30-case source labels"
        )

    candidates = []
    for web_threshold, corpus_threshold in _candidate_pairs():
        corpus_sent_to_web = sum(
            not row["requires_web"] and row["score"] < web_threshold
            for row in rows
        )
        web_sent_to_corpus = sum(
            row["requires_web"] and row["score"] >= corpus_threshold
            for row in rows
        )
        hybrid_count = sum(
            web_threshold <= row["score"] < corpus_threshold for row in rows
        )
        candidates.append((
            corpus_sent_to_web + web_sent_to_corpus,
            web_sent_to_corpus,
            corpus_sent_to_web,
            hybrid_count,
            web_threshold,
            corpus_threshold,
        ))

    print("\nBEST THRESHOLD PAIRS (label errors, then hybrid volume)")
    for total, web_bad, corpus_bad, hybrid, low, high in sorted(candidates)[:10]:
        print(
            f"web<{low:.2f} hybrid<{high:.2f} corpus>={high:.2f} | "
            f"errors={total} web_to_corpus={web_bad} "
            f"corpus_to_web={corpus_bad} hybrid={hybrid}"
        )
    _print_benchmark_deltas(vector_db, args.benchmark_json)


if __name__ == "__main__":
    main()
