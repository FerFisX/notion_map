"""Compact, complete payloads for metric judge prompts.

Metric prompts must never slice serialized JSON by character count: doing so
can hide later roadmap steps or leave an invalid fragment.  These helpers keep
the fields needed by semantic judges while preserving every roadmap step.
"""

from __future__ import annotations

import json
from typing import Any


def compact_roadmap(roadmap: dict[str, Any]) -> dict[str, Any]:
    """Return the complete roadmap using only judge-relevant fields."""

    raw_steps = roadmap.get("steps", [])
    steps = raw_steps if isinstance(raw_steps, list) else []
    compact_steps: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        compact_steps.append({
            "id": str(step.get("id", "")).strip(),
            "type": str(step.get("type", "")).strip(),
            "label": str(step.get("label", "")).strip(),
            "description": str(step.get("description", "")).strip(),
            "key_points": (
                step.get("key_points", [])
                if isinstance(step.get("key_points", []), list)
                else []
            ),
        })
    return {
        "title": str(roadmap.get("title", "")).strip(),
        "step_count": len(compact_steps),
        "steps": compact_steps,
    }


def compact_contexts(contexts: list[Any] | None) -> list[dict[str, str]]:
    """Normalize contexts without cutting serialized JSON fragments."""

    normalized: list[dict[str, str]] = []
    for index, value in enumerate(contexts or [], 1):
        if isinstance(value, dict):
            context_id = str(value.get("id", "")).strip() or f"context_{index}"
            text = str(value.get("text", value.get("content", ""))).strip()
        else:
            context_id = f"context_{index}"
            text = str(value).strip()
        if text:
            normalized.append({"id": context_id, "text": text})
    return normalized


def prompt_json(value: Any) -> str:
    """Serialize a prompt payload compactly while retaining valid JSON."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
