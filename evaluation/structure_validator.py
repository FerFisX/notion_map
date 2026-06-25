"""
structure_validator.py — Validación determinística de la estructura del roadmap.

Verifica FORMA, no contenido. No usa LLM.
Cada check es pass/fail con peso; el score final es 0-10.

Checks implementados:
  step_count              — entre 6 y 12 pasos
  has_title               — el roadmap tiene título
  unique_ids              — todos los step.id son únicos
  single_inicio           — exactamente un nodo tipo 'inicio'
  single_fin              — exactamente un nodo tipo 'fin'
  starts_with_inicio      — primer paso es 'inicio'
  ends_with_fin           — último paso es 'fin'
  valid_types             — todos los tipos son inicio/proceso/decision/fin
  key_points_count        — cada paso tiene 3-5 key_points
  label_action_verb       — cada label empieza con verbo de acción
  description_min_length  — cada descripción tiene >= 2 oraciones y >= 60 chars
  no_empty_fields         — ningún campo obligatorio está vacío
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List


# Verbos de acción aceptados en labels (español e inglés)
_ACTION_VERBS = {
    # Español
    "verificar", "validar", "crear", "configurar", "instalar", "analizar",
    "definir", "implementar", "diseñar", "ejecutar", "revisar", "establecer",
    "comprobar", "generar", "obtener", "conectar", "agregar", "eliminar",
    "actualizar", "optimizar", "probar", "documentar", "migrar", "transformar",
    "calcular", "seleccionar", "identificar", "evaluar", "construir", "aplicar",
    "normalizar", "indexar", "mapear", "registrar", "iniciar", "finalizar",
    "desplegar", "integrar", "preparar", "exportar", "importar", "limpiar",
    "asegurar", "monitorear", "depurar", "configurar", "habilitar", "deshabilitar",
    "detectar", "corregir", "publicar", "completar", "determinar", "establecer",
    "asignar", "cargar", "descargar", "instanciar", "inicializar", "terminar",
    # Inglés (por si el LLM responde en inglés)
    "verify", "validate", "create", "configure", "install", "analyze",
    "define", "implement", "design", "execute", "review", "establish",
    "check", "generate", "obtain", "connect", "add", "remove", "update",
    "optimize", "test", "document", "migrate", "transform", "calculate",
    "select", "identify", "evaluate", "build", "apply", "normalize",
    "index", "map", "register", "start", "finalize", "deploy", "integrate",
    "prepare", "export", "import", "clean", "ensure", "monitor", "debug",
    "enable", "disable", "detect", "fix", "publish", "complete", "determine",
    "assign", "load", "download", "instantiate", "initialize", "terminate",
}

_VALID_TYPES = {"inicio", "proceso", "decision", "fin"}


@dataclass
class _Check:
    name:    str
    passed:  bool
    detail:  str
    weight:  float = 1.0


class StructureValidator:
    """Valida la estructura de un roadmap generado por el RAG."""

    PASS_THRESHOLD = 7.0  # score mínimo para PASS

    def validate(self, roadmap: dict) -> dict:
        """
        Returns:
            {
                "score":        float 0-10,
                "verdict":      "PASS" | "FAIL",
                "passed":       int,
                "total_checks": int,
                "pass_rate":    float 0-1,
                "checks":       list[{name, passed, detail, weight}],
                "violations":   list[str],    # solo los que fallaron
            }
        """
        steps: List[dict] = roadmap.get("steps", [])
        checks: List[_Check] = []

        # ── 1. Cantidad de pasos (peso alto) ──────────────────────────────────
        n = len(steps)
        checks.append(_Check(
            "step_count", 6 <= n <= 12,
            f"{n} pasos (requiere 6-12)",
            weight=1.5,
        ))

        # ── 2. Título presente ────────────────────────────────────────────────
        title = roadmap.get("title", "").strip()
        checks.append(_Check(
            "has_title", bool(title),
            f"Título: '{title[:60]}'" if title else "Sin título",
        ))

        # ── 3. IDs únicos ─────────────────────────────────────────────────────
        ids = [s.get("id", "") for s in steps]
        dups = [id_ for id_ in set(ids) if ids.count(id_) > 1]
        checks.append(_Check(
            "unique_ids", not dups,
            "IDs únicos" if not dups else f"IDs duplicados: {dups}",
        ))

        # ── 4. Exactamente un nodo 'inicio' ───────────────────────────────────
        inicio_n = sum(1 for s in steps if s.get("type") == "inicio")
        checks.append(_Check(
            "single_inicio", inicio_n == 1,
            f"Nodos 'inicio': {inicio_n} (debe ser 1)",
            weight=1.2,
        ))

        # ── 5. Exactamente un nodo 'fin' ──────────────────────────────────────
        fin_n = sum(1 for s in steps if s.get("type") == "fin")
        checks.append(_Check(
            "single_fin", fin_n == 1,
            f"Nodos 'fin': {fin_n} (debe ser 1)",
            weight=1.2,
        ))

        # ── 6. Primer paso es 'inicio' ────────────────────────────────────────
        if steps:
            first_type = steps[0].get("type", "")
            checks.append(_Check(
                "starts_with_inicio", first_type == "inicio",
                f"Primer paso tipo '{first_type}' (debe ser 'inicio')",
            ))

        # ── 7. Último paso es 'fin' ───────────────────────────────────────────
        if steps:
            last_type = steps[-1].get("type", "")
            checks.append(_Check(
                "ends_with_fin", last_type == "fin",
                f"Último paso tipo '{last_type}' (debe ser 'fin')",
            ))

        # ── 8. Tipos válidos en todos los pasos ───────────────────────────────
        bad_types = [
            f"'{s.get('label','?')}' → tipo='{s.get('type')}'"
            for s in steps
            if s.get("type") not in _VALID_TYPES
        ]
        checks.append(_Check(
            "valid_types", not bad_types,
            "Todos los tipos son válidos" if not bad_types
                else f"Tipos inválidos: {bad_types[:3]}",
        ))

        # ── 9. key_points: entre 3 y 5 por paso (peso alto) ──────────────────
        kp_bad = [
            f"'{s.get('label','?')}': {len(s.get('key_points', []))} key_points"
            for s in steps
            if not (3 <= len(s.get("key_points", [])) <= 5)
        ]
        checks.append(_Check(
            "key_points_count", not kp_bad,
            "Todos tienen 3-5 key_points" if not kp_bad
                else f"Fuera de rango: {kp_bad[:3]}",
            weight=1.5,
        ))

        # ── 10. Label empieza con verbo de acción (peso medio) ────────────────
        label_bad = []
        for s in steps:
            raw_label = s.get("label", "").strip()
            first = raw_label.lower().split()[0] if raw_label.split() else ""
            # También aceptar si cualquier verbo conocido es prefijo de la primera palabra
            has_verb = first in _ACTION_VERBS or any(
                first.startswith(v) for v in _ACTION_VERBS if len(v) >= 4
            )
            if not has_verb:
                label_bad.append(raw_label[:60])
        checks.append(_Check(
            "label_action_verb", not label_bad,
            "Todos los labels usan verbo de acción" if not label_bad
                else f"Sin verbo de acción: {label_bad[:3]}",
            weight=1.2,
        ))

        # ── 11. Description >= 2 oraciones y >= 60 caracteres ─────────────────
        desc_bad = []
        for s in steps:
            desc = s.get("description", "")
            sentences = [p.strip() for p in re.split(r"[.!?]+", desc) if p.strip()]
            if len(sentences) < 2 or len(desc) < 60:
                desc_bad.append(
                    f"'{s.get('label','?')}' ({len(sentences)} oraciones, {len(desc)} chars)"
                )
        checks.append(_Check(
            "description_min_length", not desc_bad,
            "Todas las descripciones tienen >=2 oraciones" if not desc_bad
                else f"Descripciones cortas: {desc_bad[:3]}",
            weight=1.3,
        ))

        # ── 12. Sin campos vacíos ─────────────────────────────────────────────
        empty_bad = []
        for s in steps:
            for field in ("id", "label", "description", "type"):
                if not str(s.get(field, "")).strip():
                    empty_bad.append(f"paso '{s.get('id','?')}' campo '{field}'")
        checks.append(_Check(
            "no_empty_fields", not empty_bad,
            "Sin campos vacíos" if not empty_bad else f"Vacíos: {empty_bad[:3]}",
        ))

        # ── Score ponderado ────────────────────────────────────────────────────
        total_w    = sum(c.weight for c in checks)
        passed_w   = sum(c.weight for c in checks if c.passed)
        score      = round((passed_w / total_w) * 10, 2) if total_w else 0.0
        passed_n   = sum(1 for c in checks if c.passed)

        return {
            "score":        score,
            "verdict":      "PASS" if score >= self.PASS_THRESHOLD else "FAIL",
            "passed":       passed_n,
            "total_checks": len(checks),
            "pass_rate":    round(passed_n / len(checks), 4) if checks else 0.0,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail, "weight": c.weight}
                for c in checks
            ],
            "violations": [c.detail for c in checks if not c.passed],
        }
