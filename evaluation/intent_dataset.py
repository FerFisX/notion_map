"""Focused fixtures for the isolated source-intent classifier.

These cases are intentionally separate from ``evaluation.dataset.EVAL_SAMPLES``:
the latter evaluates generated roadmaps, while this module evaluates only the
pre-retrieval source preference contract introduced by the intent-classifier PR.
"""

from enum import Enum


class QueryCase(Enum):
    KB_ONLY = 1
    KB_DOMINANT = 2
    WEB_DOMINANT = 3
    CLARIFY = 4

    @property
    def expected_intent(self) -> int:
        return self.value

    @property
    def label(self) -> str:
        return {
            QueryCase.KB_ONLY: "Knowledge Base only",
            QueryCase.KB_DOMINANT: "Knowledge Base with external support",
            QueryCase.WEB_DOMINANT: "External sources with KB support",
            QueryCase.CLARIFY: "Ambiguous source preference",
        }[self]

    @classmethod
    def from_id(cls, value) -> "QueryCase":
        if isinstance(value, cls):
            return value
        if isinstance(value, str) and not value.isdigit():
            return cls[value]
        return cls(int(value))


PRE_GATE_SAMPLES = [
    "Hello",
    "What should I cook tonight?",
]

ADVERSARIAL_SAMPLES = [
    {"question": "Use Bamboo context but rely on current industry best practices.",
     "assert": "literal_intent3"},
    {"question": "Hello, use only our internal documentation for the roadmap.",
     "assert": "still_gate"},
]

PARAPHRASE_SAMPLES = [
    {"question": "Going only by our own written material, build the process.",
     "query_case": QueryCase.KB_ONLY, "assert_source": "semantic"},
    {"question": "Start with our notes and enrich them with current practices.",
     "query_case": QueryCase.KB_DOMINANT, "assert_source": "semantic"},
    {"question": "Treat external guidance as authoritative and compare our notes.",
     "query_case": QueryCase.WEB_DOMINANT, "assert_source": "semantic"},
]

INTENT_SAMPLES = [
    {"question": "Use only our internal documentation to define the roadmap.",
     "query_case": QueryCase.KB_ONLY, "signal_kind": "literal"},
    {"question": "Use our internal runbook as the base and complement it with current web practices.",
     "query_case": QueryCase.KB_DOMINANT, "signal_kind": "literal"},
    {"question": "Use current industry best practices as the source and compare our internal notes.",
     "query_case": QueryCase.WEB_DOMINANT, "signal_kind": "literal"},
    {"question": "Create a deployment roadmap.",
     "query_case": QueryCase.CLARIFY, "signal_kind": "semantic",
     "expected_clarify_type": "source_ambiguous"},
]

MINIMAL_PAIRS = [
    {"group": "source_priority", "frontier": "1_2", "intent": 1,
     "question": "Use only our documentation for this roadmap."},
    {"group": "source_priority", "frontier": "1_2", "intent": 2,
     "question": "Use our documentation as the base and add current web practices."},
    {"group": "source_priority", "frontier": "2_3", "intent": 3,
     "question": "Use current web practices as the base and compare our documentation."},
]

OOD_SAMPLES = [
    {"question": "What should I cook tonight?", "ood_tier": "far",
     "expected_rejection": True},
    {"question": "Recommend a movie for the weekend.", "ood_tier": "far",
     "expected_rejection": True},
]
