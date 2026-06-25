"""Puente entre RagEngine y los evaluadores: expone contextos y convierte el roadmap a texto."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.rag_engine import RagEngine


class RagAdapter:
    """Envuelve RagEngine y expone lo que necesitan RAGAS y LLM Judge."""

    def __init__(self):
        print("  Cargando RagEngine (embeddings + ChromaDB + Bedrock)...")
        self.engine = RagEngine()
        print("  RagEngine listo.")

    def query(self, question: str) -> dict:
        """Retorna question, answer (texto), contexts y roadmap (dict)."""
        contexts = self.engine.retrieve_contexts(question)
        roadmap  = self.engine.generate_roadmap(question)
        answer   = self.roadmap_to_text(roadmap)

        return {
            "question": question,
            "answer":   answer,
            "contexts": contexts,
            "roadmap":  roadmap,
        }

    @staticmethod
    def roadmap_to_text(roadmap: dict) -> str:
        """Convierte el JSON del roadmap a texto legible para los evaluadores."""
        if not roadmap or "steps" not in roadmap:
            return "Error: roadmap vacío"

        lines = [f"# {roadmap.get('title', 'Roadmap')}", ""]
        for i, step in enumerate(roadmap.get("steps", []), 1):
            lines.append(f"{i}. [{step.get('type','proceso').upper()}] {step.get('label','')}")
            if step.get("description"):
                lines.append(f"   {step['description']}")
            for kp in step.get("key_points", []):
                lines.append(f"   • {kp}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def steps_as_list(roadmap: dict) -> list[str]:
        """Devuelve solo los labels de los pasos, en orden."""
        return [s.get("label", "") for s in roadmap.get("steps", [])]
