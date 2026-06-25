"""Configuración centralizada del sistema de evaluación: pesos, modelos y umbrales."""

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


@dataclass
class EvalConfig:
    # LLM
    bedrock_model_id: str = field(
        default_factory=lambda: os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-5-haiku-20241022-v1:0"
        )
    )
    aws_region: str = field(
        default_factory=lambda: os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    )
    judge_temperature: float = 0.0

    # RAGAS
    ragas_metrics: List[str] = field(default_factory=lambda: [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ])

    # LLM Judge: criterios clasicos (0-10)
    judge_criteria: List[str] = field(default_factory=lambda: [
        "faithfulness",
        "relevance",
        "completeness",
        "clarity",
        "technical_accuracy",
    ])

    # MESE: Mapping, Exhaustiveness, Sequence (mayor peso), Experience
    mese_weights: dict = field(default_factory=lambda: {
        "mapping":        0.25,
        "exhaustiveness": 0.20,
        "sequence":       0.35,
        "experience":     0.20,
    })

    # Umbrales
    pass_threshold:      float = 6.0
    mese_pass_threshold: float = 6.0
    sequence_min_score:  float = 5.0

    # Rutas
    reports_dir:     str = os.path.join(BASE_DIR, "evaluation", "reports")
    vectorstore_dir: str = os.path.join(BASE_DIR, "vectorstore", "chroma_db")


config = EvalConfig()
