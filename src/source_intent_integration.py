"""Bridge the source-intent classifier with request-scoped source modes.

The classifier describes the source preference expressed in the query.  The
retrieval engine remains responsible for checking whether that preference is
viable with the evidence actually available at runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.source_modes import SourceMode, normalize_source_mode


@dataclass(frozen=True)
class SourceIntentPlan:
    """Provider-neutral instructions consumed by the retrieval layer."""

    requested_mode: SourceMode
    strategy: str
    preference: str
    classifier_applied: bool
    reason: str
    classification: dict[str, Any] | None = None
    no_retrieval: bool = False

    def to_trace(self) -> dict[str, Any]:
        value = asdict(self)
        value["requested_mode"] = self.requested_mode.value
        return value


def build_source_intent_plan(
    source_mode: SourceMode | str | None,
    classification: dict[str, Any] | None = None,
) -> SourceIntentPlan:
    """Combine explicit mode selection and inferred source preference.

    Explicit ``corpus`` and ``web`` selections always win.  Classification is
    advisory only in ``auto`` mode; ambiguous, disabled, or malformed results
    fall back to the existing evidence-coverage router.
    """

    requested_mode = normalize_source_mode(source_mode)
    if requested_mode is SourceMode.CORPUS:
        return SourceIntentPlan(
            requested_mode=requested_mode,
            strategy="corpus",
            preference="corpus",
            classifier_applied=False,
            reason="explicit_corpus_mode",
        )
    if requested_mode is SourceMode.WEB:
        return SourceIntentPlan(
            requested_mode=requested_mode,
            strategy="web",
            preference="web",
            classifier_applied=False,
            reason="explicit_web_mode",
        )

    result = classification if isinstance(classification, dict) else {}
    decision = str(result.get("decision", "")).strip().lower()
    label = str(result.get("label", "")).strip().lower()
    intent = result.get("intent")

    if decision == "no_retrieval" or label == "no_retrieval":
        return SourceIntentPlan(
            requested_mode=requested_mode,
            strategy="none",
            preference="none",
            classifier_applied=True,
            reason="classifier_no_roadmap",
            classification=result,
            no_retrieval=True,
        )
    if decision == "proceed" and (intent == 1 or label == "kb_only"):
        return SourceIntentPlan(
            requested_mode=requested_mode,
            strategy="corpus",
            preference="corpus",
            classifier_applied=True,
            reason="classifier_kb_only",
            classification=result,
        )
    if decision == "proceed" and (intent == 2 or label == "kb_plus_external"):
        return SourceIntentPlan(
            requested_mode=requested_mode,
            strategy="hybrid",
            preference="corpus",
            classifier_applied=True,
            reason="classifier_kb_plus_external",
            classification=result,
        )
    if decision == "proceed" and (intent == 3 or label == "external_plus_kb"):
        return SourceIntentPlan(
            requested_mode=requested_mode,
            strategy="hybrid",
            preference="web",
            classifier_applied=True,
            reason="classifier_external_plus_kb",
            classification=result,
        )

    return SourceIntentPlan(
        requested_mode=requested_mode,
        strategy="route",
        preference="evidence",
        classifier_applied=bool(result),
        reason="coverage_router_fallback",
        classification=result or None,
    )
