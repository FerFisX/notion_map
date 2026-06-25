"""
config.py — Configuración centralizada del sistema de evaluación.
Cambia aquí pesos, modelos y umbrales sin tocar el resto del código.
"""

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


@dataclass
class EvalConfig:
    # ── LLM ───────────────────────────────────────────────────────────────
    bedrock_model_id: str = field(
        default_factory=lambda: os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-5-haiku-20241022-v1:0"
        )
    )
    aws_region: str = field(
        default_factory=lambda: os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    )
    judge_temperature: float = 0.0

    # ── RAGAS ─────────────────────────────────────────────────────────────
    ragas_metrics: List[str] = field(default_factory=lambda: [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ])

    # ── LLM Judge — criterios clásicos (0-10) ─────────────────────────────
    judge_criteria: List[str] = field(default_factory=lambda: [
        "faithfulness",
        "relevance",
        "completeness",
        "clarity",
        "technical_accuracy",
    ])

    # ── MESE — Mutually Exclusive, Simultaneously Exhaustive ──────────────
    # M = Mapping        | precisión factual de cada paso vs contexto
    # E = Exhaustiveness | cobertura: ¿falta algún paso importante?
    # S = Sequence       | orden lógico y dependencias causales  ← mayor peso
    # E = Experience     | claridad y usabilidad para el usuario final
    #
    # Son mutuamente exclusivos: cada uno falla de forma independiente
    # Son simultáneamente exhaustivos: juntos cubren toda la calidad del roadmap
    mese_weights: dict = field(default_factory=lambda: {
        "mapping":        0.25,
        "exhaustiveness": 0.20,
        "sequence":       0.35,
        "experience":     0.20,
    })

    # ── Umbrales ──────────────────────────────────────────────────────────
    pass_threshold:      float = 6.0
    mese_pass_threshold: float = 6.0
    sequence_min_score:  float = 5.0

    # ── Rutas ─────────────────────────────────────────────────────────────
    reports_dir:     str = os.path.join(BASE_DIR, "evaluation", "reports")
    vectorstore_dir: str = os.path.join(BASE_DIR, "vectorstore", "chroma_db")


config = EvalConfig()
