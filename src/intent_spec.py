"""
intent_spec.py — Sprint 2: structured intent model.

Centralizes each intent's semantic definition (IntentSpec), a constraint
engine that derives the compatible candidate intents from detected signals,
a confusion graph of the critical boundaries, and helpers to map the
structured attribute dimensions (internal_relation / external_relation /
ambiguity) onto the four intents.

This module is deliberately dependency-free (no torch / embeddings / LLM) so
the constraint engine can run before any embedding computation. The classifier
(IntentClassifier) is free to use it; the evaluator can use it to report
whether a decision respects the declared constraints.

Dimensions (Sprint 2 point 4):
  * internal_relation — relationship to the knowledge base
  * external_relation — relationship to external / web information
  * ambiguity         — how clear the source commitment is
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Structured attribute dimensions
# ---------------------------------------------------------------------------
class InternalRelation:
    NONE = "NONE"          # no KB involvement
    ONLY = "ONLY"          # KB as the only source
    BASE = "BASE"          # KB is the primary base
    COMPLEMENT = "COMPLEMENT"  # KB is the complement
    DISTRUSTED = "DISTRUSTED"  # KB distrusted / not to be relied on


class ExternalRelation:
    NONE = "NONE"          # no external involvement
    BASE = "BASE"          # external is the primary base
    COMPLEMENT = "COMPLEMENT"  # external complements the KB
    REQUESTED = "REQUESTED"    # external requested, base-vs-complement unclear
    UNSPECIFIED = "UNSPECIFIED"  # external not explicitly requested


class Ambiguity:
    CLEAR = "CLEAR"
    SOURCE_AMBIGUOUS = "SOURCE_AMBIGUOUS"
    EXTERNAL_BASE_AMBIGUOUS = "EXTERNAL_BASE_AMBIGUOUS"
    NO_ROADMAP = "NO_ROADMAP"


# ---------------------------------------------------------------------------
# IntentSpec
# ---------------------------------------------------------------------------
@dataclass
class IntentSpec:
    id: int
    name: str
    description: str
    positive_examples: list = field(default_factory=list)
    hard_negatives: list = field(default_factory=list)
    required_signals: list = field(default_factory=list)
    forbidden_signals: list = field(default_factory=list)
    confusable_with: list = field(default_factory=list)
    internal_relation: str = InternalRelation.NONE
    external_relation: str = ExternalRelation.NONE
    ambiguity: str = Ambiguity.CLEAR
    decision: str = "proceed"      # or "clarify"
    clarify_type: str = None       # for intent 4


# Dimension -> intent registry (Sprint 2 point 4)
DIMENSION_TO_INTENT = {
    (InternalRelation.ONLY, ExternalRelation.NONE, Ambiguity.CLEAR): 1,
    (InternalRelation.BASE, ExternalRelation.COMPLEMENT, Ambiguity.CLEAR): 2,
    (InternalRelation.COMPLEMENT, ExternalRelation.BASE, Ambiguity.CLEAR): 3,
    (InternalRelation.DISTRUSTED, ExternalRelation.BASE, Ambiguity.CLEAR): 3,
    (InternalRelation.NONE, ExternalRelation.REQUESTED, Ambiguity.EXTERNAL_BASE_AMBIGUOUS): 4,
    (InternalRelation.BASE, ExternalRelation.UNSPECIFIED, Ambiguity.SOURCE_AMBIGUOUS): 4,
}


# The four canonical intents as IntentSpec (Sprint 2 point 7)
INTENT_SPECS: dict = {
    1: IntentSpec(
        id=1,
        name="kb_only",
        description=("Internal knowledge base is the only source; no external "
                     "information requested."),
        required_signals=["internal"],
        forbidden_signals=["external", "distrust", "as_answer"],
        confusable_with=[2],
        internal_relation=InternalRelation.ONLY,
        external_relation=ExternalRelation.NONE,
        ambiguity=Ambiguity.CLEAR,
        decision="proceed",
    ),
    2: IntentSpec(
        id=2,
        name="kb_plus_external",
        description=("Internal knowledge base is the primary basis and external "
                     "information is used as a complement."),
        required_signals=["internal"],
        forbidden_signals=["distrust"],
        confusable_with=[1, 3, 4],
        internal_relation=InternalRelation.BASE,
        external_relation=ExternalRelation.COMPLEMENT,
        ambiguity=Ambiguity.CLEAR,
        decision="proceed",
    ),
    3: IntentSpec(
        id=3,
        name="external_plus_kb",
        description=("External / updated information is the primary basis, using "
                     "the knowledge base as a complement (or distrusting it)."),
        required_signals=["external", "distrust", "as_answer"],
        forbidden_signals=["internal_only"],
        confusable_with=[2, 4],
        internal_relation=InternalRelation.COMPLEMENT,
        external_relation=ExternalRelation.BASE,
        ambiguity=Ambiguity.CLEAR,
        decision="proceed",
    ),
    4: IntentSpec(
        id=4,
        name="ambiguous",
        description=("Source commitment is unclear or a non-technical request; "
                     "clarify before proceeding."),
        required_signals=[],
        forbidden_signals=["internal_only"],
        confusable_with=[2, 3],
        internal_relation=InternalRelation.NONE,
        external_relation=ExternalRelation.UNSPECIFIED,
        ambiguity=Ambiguity.SOURCE_AMBIGUOUS,
        decision="clarify",
        clarify_type="source_ambiguous",
    ),
}

# Semantic cluster -> intent (mirrors _semantic_decision mappings)
CLUSTER_TO_INTENT = {
    "distrust": 3,
    "as_answer": 3,
    "internal_and_external": 2,
    "internal": 1,
    # "external" -> 4 (clarify external_two_way)
    # "soft"     -> 4 (clarify source_ambiguous)
    # "off_topic"-> 4 (clarify no_roadmap)
}


# ---------------------------------------------------------------------------
# Confusion graph (Sprint 2 point 8)
# ---------------------------------------------------------------------------
# Edges are the boundaries that need the most scrutiny. `weight` records the
# observed confusion intensity at baseline (from the Sprint 1 --no-llm run).
CONFUSION_GRAPH: dict = {
    1: {"neighbors": [2]},
    2: {"neighbors": [1, 3, 4]},
    3: {"neighbors": [2, 4]},
    4: {"neighbors": [2, 3]},
}

# The critical frontiers (Sprint 1 / point 20) — evaluated in isolation.
CRITICAL_FRONTIERS = ["2_3", "2_4", "3_4"]


def get_candidate_pairs(intent: int) -> list:
    """Returns the (a, b) frontier partners for a given intent (e.g. the
    reranker only needs to compare against these, not all 4 classes)."""
    pairs = []
    for n in CONFUSION_GRAPH.get(intent, {}).get("neighbors", []):
        pair = tuple(sorted((intent, n)))
        if pair not in pairs:
            pairs.append(pair)
    return pairs


# ---------------------------------------------------------------------------
# Constraint engine (Sprint 2 point 5)
# ---------------------------------------------------------------------------
def derive_candidate_intents(signals: dict) -> list:
    """Given detected signals, returns the ordered list of intent candidates
    that are COMPATIBLE with the evidence (before embedding retrieval).

    `signals` keys (from _detect_signals): internal, external, distrust,
    as_answer, company.

    Rules (business logic, explicit and validated):
      * distrust -> intents 1 and 2 are impossible (KB not reliable as base);
                    candidates {3, 4}.
      * as_answer -> external asked as the answer -> candidates {3, 4}.
      * internal AND external (no distrust) -> KB base + external complement
                    -> candidate {2}.
      * internal only (no external) -> candidate {1}.
      * external only (no internal) -> clarify external_two_way -> {4}.
      * no signal at all -> ambiguous -> {4}.
    Returns a list of intents (may be a single element)."""
    flags = {
        "internal":  bool(signals.get("internal")),
        "external":  bool(signals.get("external")),
        "distrust":  bool(signals.get("distrust")),
        "as_answer": bool(signals.get("as_answer")),
    }

    if flags["distrust"]:
        return [3, 4]
    if flags["as_answer"]:
        return [3, 4]
    if flags["internal"] and flags["external"]:
        return [2]
    if flags["internal"]:
        return [1]
    if flags["external"]:
        return [4]
    return [4]


def intent_allowed(intent: int, signals: dict) -> bool:
    """Returns whether `intent` is compatible with the detected signals."""
    return intent in derive_candidate_intents(signals)


def dimensions_to_intent(internal: str, external: str, ambiguity: str):
    return DIMENSION_TO_INTENT.get((internal, external, ambiguity))
