"""
PARAPHRASE_SAMPLES (v3, Layer 2b calibration-focused) — calibration/regression
set for Layer 2b (embeddings + cosine).

Sizing: 9 REFERENCE_PHRASES clusters x (5 positives + 2 hard negatives) = 63.
Reduced from v2 (135) to enable fast iteration with --no-llm mode (~3 min run).

None of these phrases were copied from REFERENCE_PHRASES in
src/intent_classifier.py — cross-check before merging.

v3 changes from v2:
  - Reduced to 5 positives + 2 hard negatives per cluster (from 10 + 5).
  - Rewrote 11 impure samples that contained literal gate signal words
    (hey/hi/good afternoon/recommendations for/song/haircut/recipe/our
    documented process/i don't understand) so they exercise Layer 2b cleanly.
  - Removed duplicate "How do I write a good song-based mnemonic..." (was at
    greeting hard-neg and off_topic hard-neg).
  - New category taxonomy aligned with the full ingested KB (notion_exports):
    sql_databricks, odoo, powerbi_dax, processmaker, git_devops_cloud,
    ai_rag_n8n, dashboards_bi, procesos_negocio, general.

Each entry uses the SAME dict shape as v2:

  question              -> str
  query_case             -> QueryCase (KB_ONLY | KB_DOMINANT | WEB_DOMINANT | CLARIFY)
  category               -> str, domain tag (new taxonomy)
  expected_clarify_type  -> Optional[str], only set when query_case is CLARIFY
  assert_source          -> "semantic"  : should resolve via decision_source == "semantic"
                             "llm"       : acceptable to fall through to "llm"
                             "not_semantic": hard negative — must NOT resolve via "semantic"
"""

from enum import Enum
from typing import List
from dataclasses import dataclass, field


class QueryCase(Enum):
    """Query-case that drives the expected intent + decision for a sample:

    * KB_ONLY      -> intent 1, decision "proceed"
    * KB_DOMINANT  -> intent 2, decision "proceed"
    * WEB_DOMINANT -> intent 3, decision "proceed"
    * CLARIFY      -> intent 4, decision "clarify"
    """

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
            QueryCase.KB_ONLY:      "Knowledge Base 100%",
            QueryCase.KB_DOMINANT:  "High reliability in Knowledge Base",
            QueryCase.WEB_DOMINANT: "Low reliability in Knowledge Base",
            QueryCase.CLARIFY:      "Default - clarify",
        }[self]

    @classmethod
    def from_id(cls, value) -> "QueryCase":
        """Accepts the int id (1-4), its str digit, or the enum member name."""
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        if isinstance(value, str):
            try:
                return cls(int(value))
            except ValueError:
                return cls[value]
        raise ValueError(f"not a QueryCase id: {value!r}")


@dataclass
class EvalSample:
    """Legacy RAG-quality sample shape (kept so evaluation/runner.py,
    llm_judge.py and ragas_evaluator.py still import). Unused by v3."""
    question:            str
    ground_truth:        str = ""
    expected_keywords:   List[str] = field(default_factory=list)
    category:            str = ""
    expected_step_order: List[str] = field(default_factory=list)


# v3 dataset is Layer 2b focused (PARAPHRASE_SAMPLES below). The former
# EVAL/PRE_GATE/ADVERSARIAL sets were removed with the v2 rewrite, so these
# stay empty to keep legacy importers working.
EVAL_SAMPLES: list = []
PRE_GATE_SAMPLES: list = []
ADVERSARIAL_SAMPLES: list = []

PARAPHRASE_SAMPLES: list = [

    # ================================================================
    # CLUSTER: greeting (5 positive + 2 hard negative = 7)
    # Positives: no literal gate terms (hey/hi/good morning/etc.)
    # ================================================================
    {"cluster": "greeting", "question": "Morning, hope things are going smoothly on your end.",
     "query_case": QueryCase.CLARIFY, "category": "general",
     "expected_clarify_type": "no_roadmap", "assert_source": "semantic"},
    {"cluster": "greeting", "question": "Hope your day's been treating you well so far.",
     "query_case": QueryCase.CLARIFY, "category": "general",
     "expected_clarify_type": "no_roadmap", "assert_source": "semantic"},
    {"cluster": "greeting", "question": "Great to be back, it's been a minute.",
     "query_case": QueryCase.CLARIFY, "category": "general",
     "expected_clarify_type": "no_roadmap", "assert_source": "semantic"},
    {"cluster": "greeting", "question": "Is this thing on? Just checking before I get started.",
     "query_case": QueryCase.CLARIFY, "category": "general",
     "expected_clarify_type": "no_roadmap", "assert_source": "semantic"},
    {"cluster": "greeting", "question": "All good on your side today?",
     "query_case": QueryCase.CLARIFY, "category": "general",
     "expected_clarify_type": "no_roadmap", "assert_source": "semantic"},
    # hard negatives — real technical questions, no greeting signal
    {"cluster": "greeting", "question": "I think this pipeline needs a few fixes, how should I approach it?",
     "query_case": QueryCase.CLARIFY, "category": "ai_rag_n8n",
     "expected_clarify_type": "source_ambiguous", "assert_source": "not_semantic"},
    {"cluster": "greeting", "question": "Check the deploy history to see why it failed.",
     "query_case": QueryCase.CLARIFY, "category": "git_devops_cloud",
     "expected_clarify_type": "source_ambiguous", "assert_source": "not_semantic"},

    # ================================================================
    # CLUSTER: non_retrieval (5 positive + 2 hard negative = 7)
    # Positives: no literal non_retrieval terms (i don't understand/etc.)
    # ================================================================
    {"cluster": "non_retrieval", "question": "That's not really the direction I was hoping to go in.",
     "query_case": QueryCase.CLARIFY, "category": "general",
     "expected_clarify_type": "no_roadmap", "assert_source": "semantic"},
    {"cluster": "non_retrieval", "question": "You lost me somewhere in the middle of that.",
     "query_case": QueryCase.CLARIFY, "category": "general",
     "expected_clarify_type": "no_roadmap", "assert_source": "semantic"},
    {"cluster": "non_retrieval", "question": "Can we start over, that wasn't quite it.",
     "query_case": QueryCase.CLARIFY, "category": "general",
     "expected_clarify_type": "no_roadmap", "assert_source": "semantic"},
    {"cluster": "non_retrieval", "question": "None of that actually addressed what I brought up.",
     "query_case": QueryCase.CLARIFY, "category": "general",
     "expected_clarify_type": "no_roadmap", "assert_source": "semantic"},
    {"cluster": "non_retrieval", "question": "I'm confused about what you're even responding to.",
     "query_case": QueryCase.CLARIFY, "category": "general",
     "expected_clarify_type": "no_roadmap", "assert_source": "semantic"},
    # hard negatives — technical frustration that is still a real request
    {"cluster": "non_retrieval", "question": "This join isn't behaving the way the docs say it should, can you check it?",
     "query_case": QueryCase.CLARIFY, "category": "sql_databricks",
     "expected_clarify_type": "source_ambiguous", "assert_source": "not_semantic"},
    {"cluster": "non_retrieval", "question": "Say the retry logic runs three times before it gives up, is that configurable?",
     "query_case": QueryCase.CLARIFY, "category": "ai_rag_n8n",
     "expected_clarify_type": "source_ambiguous", "assert_source": "not_semantic"},

    # ================================================================
    # CLUSTER: internal (5 positive + 2 hard negative = 7)
    # Positives: no literal internal anchor terms
    # ================================================================
    {"cluster": "internal", "question": "Sticking strictly to what we've already written down, which column should be the primary key?",
     "query_case": QueryCase.KB_ONLY, "category": "sql_databricks",
     "assert_source": "semantic"},
    {"cluster": "internal", "question": "Going only by what's recorded in our own materials, which Power Query step costs more?",
     "query_case": QueryCase.KB_ONLY, "category": "powerbi_dax",
     "assert_source": "semantic"},
    {"cluster": "internal", "question": "Without going outside what we already have on file, how do we separate decision from execution?",
     "query_case": QueryCase.KB_ONLY, "category": "ai_rag_n8n",
     "assert_source": "semantic"},
    {"cluster": "internal", "question": "Using nothing but what's already documented for this, how do alternate and primary keys differ?",
     "query_case": QueryCase.KB_ONLY, "category": "sql_databricks",
     "assert_source": "semantic"},
    {"cluster": "internal", "question": "Based purely on what we've recorded internally, how do we structure the employees table keys?",
     "query_case": QueryCase.KB_ONLY, "category": "odoo",
     "assert_source": "semantic"},
    # hard negatives — sounds internal-anchored but is actually general
    {"cluster": "internal", "question": "Our team wrote a blog post once about database design trends.",
     "query_case": QueryCase.CLARIFY, "category": "sql_databricks",
     "expected_clarify_type": "source_ambiguous", "assert_source": "not_semantic"},
    {"cluster": "internal", "question": "I keep my own personal scratchpad on N8N, unrelated to any company doc.",
     "query_case": QueryCase.CLARIFY, "category": "ai_rag_n8n",
     "expected_clarify_type": "source_ambiguous", "assert_source": "not_semantic"},

    # ================================================================
    # CLUSTER: external (5 positive + 2 hard negative = 7)
    # Positives: no literal external anchor terms
    # ================================================================
    {"cluster": "external", "question": "Pull the newest thinking on this from wherever it's out there.",
     "query_case": QueryCase.CLARIFY, "category": "sql_databricks",
     "expected_clarify_type": "external_two_way", "assert_source": "semantic"},
    {"cluster": "external", "question": "Go grab whatever's freshest on this topic and use that.",
     "query_case": QueryCase.CLARIFY, "category": "powerbi_dax",
     "expected_clarify_type": "external_two_way", "assert_source": "semantic"},
    {"cluster": "external", "question": "See what the field looks like today before answering.",
     "query_case": QueryCase.CLARIFY, "category": "ai_rag_n8n",
     "expected_clarify_type": "external_two_way", "assert_source": "semantic"},
    {"cluster": "external", "question": "Bring back whatever's the newest take on this.",
     "query_case": QueryCase.CLARIFY, "category": "powerbi_dax",
     "expected_clarify_type": "external_two_way", "assert_source": "semantic"},
    {"cluster": "external", "question": "Find out how this is typically handled these days.",
     "query_case": QueryCase.CLARIFY, "category": "ai_rag_n8n",
     "expected_clarify_type": "external_two_way", "assert_source": "semantic"},
    # hard negatives — "current"/"latest" used in a non-source-request sense
    {"cluster": "external", "question": "Can you give me an update on how N8N versioning generally works?",
     "query_case": QueryCase.CLARIFY, "category": "ai_rag_n8n",
     "expected_clarify_type": "source_ambiguous", "assert_source": "not_semantic"},
    {"cluster": "external", "question": "The current row count in that table is throwing off my estimate.",
     "query_case": QueryCase.CLARIFY, "category": "sql_databricks",
     "expected_clarify_type": "source_ambiguous", "assert_source": "not_semantic"},

    # ================================================================
    # CLUSTER: internal_and_external (5 positive + 2 hard negative = 7)
    # Positives: no literal internal anchor terms
    # ================================================================
    {"cluster": "internal_and_external", "question": "Base it on our own notes, then round it out with whatever's current in the field.",
     "query_case": QueryCase.KB_DOMINANT, "category": "powerbi_dax",
     "assert_source": "semantic"},
    {"cluster": "internal_and_external", "question": "Keep our case study as the anchor, but top it up with what's out there now.",
     "query_case": QueryCase.KB_DOMINANT, "category": "ai_rag_n8n",
     "assert_source": "semantic"},
    {"cluster": "internal_and_external", "question": "Take what we've documented as the foundation, then layer on anything newer people are doing.",
     "query_case": QueryCase.KB_DOMINANT, "category": "ai_rag_n8n",
     "assert_source": "semantic"},
    {"cluster": "internal_and_external", "question": "Lean on our internal material, and pad it out with the latest from outside.",
     "query_case": QueryCase.KB_DOMINANT, "category": "powerbi_dax",
     "assert_source": "semantic"},
    {"cluster": "internal_and_external", "question": "Ground it in what we've got internally, and stretch it with current outside thinking.",
     "query_case": QueryCase.KB_DOMINANT, "category": "ai_rag_n8n",
     "assert_source": "semantic"},
    # hard negatives — mentions "our" and "current" but not as base+complement
    {"cluster": "internal_and_external", "question": "Our internal server is currently down, can you help troubleshoot?",
     "query_case": QueryCase.CLARIFY, "category": "ai_rag_n8n",
     "expected_clarify_type": "source_ambiguous", "assert_source": "not_semantic"},
    {"cluster": "internal_and_external", "question": "Our current table has a naming inconsistency, how do I fix it?",
     "query_case": QueryCase.CLARIFY, "category": "sql_databricks",
     "expected_clarify_type": "source_ambiguous", "assert_source": "not_semantic"},

    # ================================================================
    # CLUSTER: distrust (5 positive + 2 hard negative = 7)
    # Positives: no literal distrust terms
    # ================================================================
    {"cluster": "distrust", "question": "Honestly I'm not sure our own notes on this are still accurate, what's the standard approach?",
     "query_case": QueryCase.WEB_DOMINANT, "category": "sql_databricks",
     "assert_source": "semantic"},
    {"cluster": "distrust", "question": "I wouldn't lean on what we've got written down for this one, what does everyone else do?",
     "query_case": QueryCase.WEB_DOMINANT, "category": "powerbi_dax",
     "assert_source": "semantic"},
    {"cluster": "distrust", "question": "Our own material feels shaky on this topic, what's the accepted way to do it?",
     "query_case": QueryCase.WEB_DOMINANT, "category": "sql_databricks",
     "assert_source": "semantic"},
    {"cluster": "distrust", "question": "Our internal take on this doesn't hold up for me, what should I actually do?",
     "query_case": QueryCase.WEB_DOMINANT, "category": "ai_rag_n8n",
     "assert_source": "semantic"},
    {"cluster": "distrust", "question": "I'm skeptical of what we've got recorded for this, what does the field actually say?",
     "query_case": QueryCase.WEB_DOMINANT, "category": "ai_rag_n8n",
     "assert_source": "semantic"},
    # hard negatives — "trust"/doubt-adjacent words used positively or unrelated
    {"cluster": "distrust", "question": "I trust this approach will hold up once we scale the pipeline.",
     "query_case": QueryCase.CLARIFY, "category": "ai_rag_n8n",
     "expected_clarify_type": "source_ambiguous", "assert_source": "not_semantic"},
    {"cluster": "distrust", "question": "I doubt this job will finish before the deadline, is there a quicker option?",
     "query_case": QueryCase.CLARIFY, "category": "powerbi_dax",
     "expected_clarify_type": "source_ambiguous", "assert_source": "not_semantic"},

    # ================================================================
    # CLUSTER: as_answer (5 positive + 2 hard negative = 7)
    # Positives: no literal external_as_answer terms
    # ================================================================
    {"cluster": "as_answer", "question": "Give me the standard answer people in the field would give for this.",
     "query_case": QueryCase.WEB_DOMINANT, "category": "sql_databricks",
     "assert_source": "semantic"},
    {"cluster": "as_answer", "question": "Respond the way a seasoned practitioner in this space would.",
     "query_case": QueryCase.WEB_DOMINANT, "category": "ai_rag_n8n",
     "assert_source": "semantic"},
    {"cluster": "as_answer", "question": "Give me the answer that reflects how the field currently does this.",
     "query_case": QueryCase.WEB_DOMINANT, "category": "powerbi_dax",
     "assert_source": "semantic"},
    {"cluster": "as_answer", "question": "Answer this the way most professionals would today.",
     "query_case": QueryCase.WEB_DOMINANT, "category": "sql_databricks",
     "assert_source": "semantic"},
    {"cluster": "as_answer", "question": "Give me the textbook-correct modern answer for this.",
     "query_case": QueryCase.WEB_DOMINANT, "category": "powerbi_dax",
     "assert_source": "semantic"},
    # hard negatives — "answer"/"correct" used without external-authority framing
    {"cluster": "as_answer", "question": "What's the correct syntax for a left join in SQL?",
     "query_case": QueryCase.CLARIFY, "category": "sql_databricks",
     "expected_clarify_type": "source_ambiguous", "assert_source": "not_semantic"},
    {"cluster": "as_answer", "question": "This field only accepts numeric values, is that correct?",
     "query_case": QueryCase.CLARIFY, "category": "odoo",
     "expected_clarify_type": "source_ambiguous", "assert_source": "not_semantic"},

    # ================================================================
    # CLUSTER: soft (5 positive + 2 hard negative = 7)
    # Positives: no literal soft advisory terms
    # ================================================================
    {"cluster": "soft", "question": "What's a sane way to lay out a new schema from scratch?",
     "query_case": QueryCase.CLARIFY, "category": "sql_databricks",
     "expected_clarify_type": "source_ambiguous", "assert_source": "semantic"},
    {"cluster": "soft", "question": "My N8N flows feel sluggish lately, what would you look at first?",
     "query_case": QueryCase.CLARIFY, "category": "ai_rag_n8n",
     "expected_clarify_type": "source_ambiguous", "assert_source": "semantic"},
    {"cluster": "soft", "question": "What tends to go wrong when people set up big Power Query merges?",
     "query_case": QueryCase.CLARIFY, "category": "powerbi_dax",
     "expected_clarify_type": "source_ambiguous", "assert_source": "semantic"},
    {"cluster": "soft", "question": "Any thoughts on how to keep this schema from getting messy over time?",
     "query_case": QueryCase.CLARIFY, "category": "sql_databricks",
     "expected_clarify_type": "source_ambiguous", "assert_source": "semantic"},
    {"cluster": "soft", "question": "What's a reasonable way to structure retries in this kind of workflow?",
     "query_case": QueryCase.CLARIFY, "category": "ai_rag_n8n",
     "expected_clarify_type": "source_ambiguous", "assert_source": "semantic"},
    # hard negatives — advisory words used descriptively, not as an ask
    {"cluster": "soft", "question": "The best time to run this batch job is probably overnight.",
     "query_case": QueryCase.CLARIFY, "category": "powerbi_dax",
     "expected_clarify_type": "no_roadmap", "assert_source": "not_semantic"},
    {"cluster": "soft", "question": "I optimize my own routine by batching similar tasks together.",
     "query_case": QueryCase.CLARIFY, "category": "ai_rag_n8n",
     "expected_clarify_type": "no_roadmap", "assert_source": "not_semantic"},

    # ================================================================
    # CLUSTER: off_topic (5 positive + 2 hard negative = 7)
    # Positives: no literal off_topic terms (movie/film/recipe/song/etc.)
    # ================================================================
    {"cluster": "off_topic", "question": "How's the football match going tonight?",
     "query_case": QueryCase.CLARIFY, "category": "general",
     "expected_clarify_type": "no_roadmap", "assert_source": "semantic"},
    {"cluster": "off_topic", "question": "Got any weekend plans worth stealing?",
     "query_case": QueryCase.CLARIFY, "category": "general",
     "expected_clarify_type": "no_roadmap", "assert_source": "semantic"},
    {"cluster": "off_topic", "question": "What's the weather looking like where you are?",
     "query_case": QueryCase.CLARIFY, "category": "general",
     "expected_clarify_type": "no_roadmap", "assert_source": "semantic"},
    {"cluster": "off_topic", "question": "Any good shows worth binge-watching lately?",
     "query_case": QueryCase.CLARIFY, "category": "general",
     "expected_clarify_type": "no_roadmap", "assert_source": "semantic"},
    {"cluster": "off_topic", "question": "Who do you think wins the championship this year?",
     "query_case": QueryCase.CLARIFY, "category": "general",
     "expected_clarify_type": "no_roadmap", "assert_source": "semantic"},
    # hard negatives — off-topic vocabulary used as technical metaphor
    {"cluster": "off_topic", "question": "The pipeline scored a hat trick today: three failed retries in a row.",
     "query_case": QueryCase.CLARIFY, "category": "ai_rag_n8n",
     "expected_clarify_type": "source_ambiguous", "assert_source": "not_semantic"},
    {"cluster": "off_topic", "question": "We need to weather this outage before the client notices.",
     "query_case": QueryCase.CLARIFY, "category": "ai_rag_n8n",
     "expected_clarify_type": "source_ambiguous", "assert_source": "not_semantic"},
]

# ========================================================================
# INTENT_SAMPLES — balanced intent 1-4 evaluation set (Sprint 1)
# ------------------------------------------------------------------------
# Purpose: fill the REAL confusion matrix over the four intents. The v3
# PARAPHRASE_SAMPLES only exercise Layer 2b; these exercise the whole
# pipeline (literal gate/table + semantic + LLM residual) across intent
# classes. Each entry:
#   question                    -> str
#   query_case                  -> QueryCase (1=KB_ONLY,2=KB_DOMINANT,
#                                             3=WEB_DOMINANT,4=CLARIFY)
#   category                    -> domain tag
#   expected_clarify_type       -> Optional, CLARIFY only
#   signal_kind                 -> "literal" | "semantic": whether the sample
#                                  should be resolved deterministically
#                                  (decision_source gate/literal) or via the
#                                  semantic/LLM residual
#   assert_source               -> optional layer assertion (see evaluator)
#   regression                  -> optional taxonomy tag (Sprint 1 point 21):
#                                  literal_fp / literal_fn / semantic_fp /
#                                  semantic_fn / retrieval / reranker / ood /
#                                  llm / policy. Empty -> new sample.
# ========================================================================

INTENT_SAMPLES: list = [
    # ---- INTENT 1 (KB_ONLY): only internal KB -------------------------
    {"question": "According to our documentation, how is the primary key defined in the employees table?",
     "query_case": QueryCase.KB_ONLY, "category": "sql_databricks",
     "signal_kind": "literal", "assert_source": "literal", "regression": ""},
    {"question": "Per our internal runbook, what Power Query step is the most expensive?",
     "query_case": QueryCase.KB_ONLY, "category": "powerbi_dax",
     "signal_kind": "literal", "assert_source": "literal", "regression": ""},
    {"question": "Our internal notes say Odoo models use a special field type for computed values — which one?",
     "query_case": QueryCase.KB_ONLY, "category": "odoo",
     "signal_kind": "literal", "assert_source": "literal", "regression": ""},
    {"question": "Sticking strictly to what's already recorded internally, how are the table keys structured?",
     "query_case": QueryCase.KB_ONLY, "category": "sql_databricks",
     "signal_kind": "semantic", "assert_source": "semantic", "regression": ""},
    {"question": "Going only by our own written materials, which step in the flow consumes the most compute?",
     "query_case": QueryCase.KB_ONLY, "category": "ai_rag_n8n",
     "signal_kind": "semantic", "assert_source": "semantic", "regression": ""},
    {"question": "Using nothing but what we have documented for this, how do stored and computed fields differ in our ERP?",
     "query_case": QueryCase.KB_ONLY, "category": "odoo",
     "signal_kind": "semantic", "assert_source": "semantic", "regression": ""},

    # ---- INTENT 2 (KB_DOMINANT): KB base + external complement --------
    {"question": "Base it on our internal documentation and complement it with current external best practices.",
     "query_case": QueryCase.KB_DOMINANT, "category": "sql_databricks",
     "signal_kind": "literal", "assert_source": "literal", "regression": ""},
    {"question": "Use our internal runbook as the foundation, then layer on the latest external practices.",
     "query_case": QueryCase.KB_DOMINANT, "category": "ai_rag_n8n",
     "signal_kind": "literal", "assert_source": "literal", "regression": ""},
    {"question": "Take our internal notes as the base and top them up with current external approaches.",
     "query_case": QueryCase.KB_DOMINANT, "category": "powerbi_dax",
     "signal_kind": "literal", "assert_source": "literal", "regression": ""},
    {"question": "Ground it in what we've got internally, then stretch it with current outside thinking.",
     "query_case": QueryCase.KB_DOMINANT, "category": "ai_rag_n8n",
     "signal_kind": "semantic", "assert_source": "semantic", "regression": ""},
    {"question": "Keep our case study as the anchor, but round it out with whatever's newer out there.",
     "query_case": QueryCase.KB_DOMINANT, "category": "powerbi_dax",
     "signal_kind": "semantic", "assert_source": "semantic", "regression": ""},
    {"question": "Lean on our documented foundation and pad it with the latest external findings.",
     "query_case": QueryCase.KB_DOMINANT, "category": "sql_databricks",
     "signal_kind": "semantic", "assert_source": "semantic", "regression": ""},

    # ---- INTENT 3 (WEB_DOMINANT): external base -----------------------
    {"question": "Our internal notes on this don't hold up, give me the standard external approach.",
     "query_case": QueryCase.WEB_DOMINANT, "category": "sql_databricks",
     "signal_kind": "literal", "assert_source": "literal", "regression": ""},
    {"question": "I don't trust what we've written down here, what does the market currently recommend?",
     "query_case": QueryCase.WEB_DOMINANT, "category": "ai_rag_n8n",
     "signal_kind": "literal", "assert_source": "literal", "regression": ""},
    {"question": "Give me the answer that reflects how the field currently does this, not our internal take.",
     "query_case": QueryCase.WEB_DOMINANT, "category": "powerbi_dax",
     "signal_kind": "literal", "assert_source": "literal", "regression": ""},
    {"question": "Our own material feels shaky here, what's the accepted way to do it now?",
     "query_case": QueryCase.WEB_DOMINANT, "category": "sql_databricks",
     "signal_kind": "semantic", "assert_source": "semantic", "regression": ""},
    {"question": "I'm skeptical of what we've got recorded for this, what does the field actually recommend?",
     "query_case": QueryCase.WEB_DOMINANT, "category": "ai_rag_n8n",
     "signal_kind": "semantic", "assert_source": "semantic", "regression": ""},
    {"question": "Answer the way a seasoned external practitioner would today.",
     "query_case": QueryCase.WEB_DOMINANT, "category": "powerbi_dax",
     "signal_kind": "semantic", "assert_source": "semantic", "regression": ""},

    # ---- INTENT 4 (CLARIFY): needs clarification ----------------------
    {"question": "Find the latest best practice on this topic.",
     "query_case": QueryCase.CLARIFY, "category": "sql_databricks",
     "expected_clarify_type": "external_two_way",
     "signal_kind": "literal", "assert_source": "literal", "regression": ""},
    {"question": "What would you personally recommend for structuring this workflow?",
     "query_case": QueryCase.CLARIFY, "category": "ai_rag_n8n",
     "expected_clarify_type": "source_ambiguous",
     "signal_kind": "literal", "assert_source": "literal", "regression": ""},
    {"question": "Got any thoughts on the best way to lay out this schema from scratch?",
     "query_case": QueryCase.CLARIFY, "category": "sql_databricks",
     "expected_clarify_type": "source_ambiguous",
     "signal_kind": "literal", "assert_source": "literal", "regression": ""},
    {"question": "Bring back whatever's the newest take on this topic.",
     "query_case": QueryCase.CLARIFY, "category": "powerbi_dax",
     "expected_clarify_type": "external_two_way",
     "signal_kind": "semantic", "assert_source": "semantic", "regression": ""},
    {"question": "See what the field looks like today before answering.",
     "query_case": QueryCase.CLARIFY, "category": "ai_rag_n8n",
     "expected_clarify_type": "external_two_way",
     "signal_kind": "semantic", "assert_source": "semantic", "regression": ""},
    {"question": "What's a reasonable way to structure retries without clear constraints?",
     "query_case": QueryCase.CLARIFY, "category": "ai_rag_n8n",
     "expected_clarify_type": "source_ambiguous",
     "signal_kind": "semantic", "assert_source": "semantic", "regression": ""},
]

# ========================================================================
# MINIMAL_PAIRS — same vocabulary, ONE decisive condition different
# ------------------------------------------------------------------------
# Sprint 1 point 9-10. Each pair groups variants that differ only in the
# decisive condition, tagged with the Sprint 2 dimension (internal_relation
# / external_relation / ambiguity) so the boundary is explicit. A "pair"
# may be an original + counterfactuals, all sharing the same surface
# vocabulary. Focus on the critical frontiers 2<->3, 2<->4, 3<->4.
#   group            -> logical id that ties the variants together
#   frontier         -> the boundary this group is meant to test ("2_3"|"2_4"|"3_4")
#   dimension        -> which attribute changes (see Sprint 2)
#   intent           -> expected intent (1-4)
#   internal_relation / external_relation / ambiguity -> expected dimensions
# ========================================================================

MINIMAL_PAIRS: list = [
    # ---- Frontier 2 <-> 3 : base vs complement swapped -----------------
    {"group": "mp_2_3a", "frontier": "2_3", "dimension": "internal_relation",
     "internal_relation": "BASE", "external_relation": "COMPLEMENT", "ambiguity": "CLEAR",
     "intent": 2, "question": "Use our documentation as the base and complement it with current external practices.",
     "category": "sql_databricks", "signal_kind": "semantic"},
    {"group": "mp_2_3a", "frontier": "2_3", "dimension": "internal_relation",
     "internal_relation": "DISTRUSTED", "external_relation": "COMPLEMENT", "ambiguity": "CLEAR",
     "intent": 2, "question": "Use current external practices as the base and compare them against our documentation.",
     "category": "sql_databricks", "signal_kind": "semantic",
     "_Note": "label flipped 3->2 by Boundary Audit (embedding=2 & judge=2 converge on kb+external comparison)"},
    {"group": "mp_2_3a", "frontier": "2_3", "dimension": "internal_relation",
     "internal_relation": "ONLY", "external_relation": "NONE", "ambiguity": "CLEAR",
     "intent": 1, "question": "Use only our documentation for this.",
     "category": "sql_databricks", "signal_kind": "semantic"},

    # ---- Frontier 2 <-> 4 : complement vs unspecified ------------------
    {"group": "mp_2_4a", "frontier": "2_4", "dimension": "external_relation",
     "internal_relation": "BASE", "external_relation": "COMPLEMENT", "ambiguity": "CLEAR",
     "intent": 2, "question": "Ground it in our internal notes and add current external practices.",
     "category": "ai_rag_n8n", "signal_kind": "semantic"},
    {"group": "mp_2_4a", "frontier": "2_4", "dimension": "external_relation",
     "internal_relation": "BASE", "external_relation": "COMPLEMENT", "ambiguity": "CLEAR",
     "intent": 2, "question": "Ground it in our internal notes and also check external sources.",
     "category": "ai_rag_n8n", "signal_kind": "semantic",
     "_Note": "label flipped 4->2 by Boundary Audit: 'also check external sources' adds an external complement (judge=2 conf 1.0; matches internal_and_external cluster)"},

    # ---- Frontier 3 <-> 4 : external-as-base vs external-request -------
    {"group": "mp_3_4a", "frontier": "3_4", "dimension": "external_relation",
     "internal_relation": "DISTRUSTED", "external_relation": "BASE", "ambiguity": "CLEAR",
     "intent": 3, "question": "I don't trust our documentation on this, give me the external standard.",
     "category": "powerbi_dax", "signal_kind": "semantic"},
    {"group": "mp_3_4a", "frontier": "3_4", "dimension": "external_relation",
     "internal_relation": "NONE", "external_relation": "REQUESTED", "ambiguity": "CLEAR",
     "intent": 3, "question": "Give me the external standard on this.",
     "category": "powerbi_dax", "signal_kind": "semantic",
     "_Note": "label flipped 4->3 by Boundary Audit: 'the external standard' resolves external as the answer (judge=3 conf 0.8; matches as_answer cluster)"},

    # ---- Reuse the conceptual examples as counterfactuals (intent changes) --
    {"group": "ct_2", "frontier": "2_3", "dimension": "ambiguity",
     "internal_relation": "BASE", "external_relation": "COMPLEMENT", "ambiguity": "CLEAR",
     "intent": 2, "question": "Use our documentation as the base and complement it with current practices.",
     "category": "sql_databricks", "signal_kind": "semantic"},
    {"group": "ct_2", "frontier": "2_3", "dimension": "internal_relation",
     "internal_relation": "ONLY", "external_relation": "NONE", "ambiguity": "CLEAR",
     "intent": 1, "question": "Use only our documentation.",
     "category": "sql_databricks", "signal_kind": "semantic"},
    {"group": "ct_2", "frontier": "2_3", "dimension": "internal_relation",
     "internal_relation": "DISTRUSTED", "external_relation": "BASE", "ambiguity": "CLEAR",
     "intent": 3, "question": "Use current practices as the primary source.",
     "category": "sql_databricks", "signal_kind": "semantic"},
    {"group": "ct_2", "frontier": "2_4", "dimension": "ambiguity",
     "internal_relation": "NONE", "external_relation": "UNSPECIFIED", "ambiguity": "SOURCE_AMBIGUOUS",
     "intent": 4, "question": "What would you recommend?",
     "category": "general", "signal_kind": "semantic"},
]

# ========================================================================
# OOD_SAMPLES — out-of-domain detection evaluation (Sprint 1 point 11)
# ------------------------------------------------------------------------
# Three tiers:
#   far          -> completely outside the domain (must be rejected OOD)
#   near         -> domain-related but not any of the 4 intents
#   adversarial  -> uses vocabulary of several intents but expresses a
#                   different intent (hardest). MUST be rejected.
# Each entry: question, ood_tier ("far"|"near"|"adversarial"),
#   expected_rejection -> True means the classifier should NOT commit to a
#                         confident intent (intent 4/None/clarify preferred),
#                         and the future OOD detector should flag it.
# ========================================================================

OOD_SAMPLES: list = [
    # ---- far OOD -------------------------------------------------------
    {"question": "What should I cook for dinner tonight?", "ood_tier": "far",
     "expected_rejection": True, "category": "general"},
    {"question": "Recommend a good movie to watch this weekend.", "ood_tier": "far",
     "expected_rejection": True, "category": "general"},
    {"question": "Can you tell me a joke about databases?", "ood_tier": "far",
     "expected_rejection": True, "category": "general"},

    # ---- near OOD (domain-related, no matching intent) ----------------
    {"question": "What sectors of the warehouse data are not yet ingested into the vector store?",
     "ood_tier": "near", "expected_rejection": True, "category": "ai_rag_n8n"},
    {"question": "Which internal documentation pages are missing from our knowledge base export?",
     "ood_tier": "near", "expected_rejection": True, "category": "ai_rag_n8n"},

    # ---- adversarial / misleading (vocab of several intents) ----------
    {"question": "According to our internal documentation, what does the latest external research recommend?",
     "ood_tier": "adversarial", "expected_rejection": True, "category": "sql_databricks"},
    {"question": "Use our documented process as the base, and also our documented external benchmark as the complement.",
     "ood_tier": "adversarial", "expected_rejection": True, "category": "ai_rag_n8n"},
    {"question": "Trust nothing we've recorded, and also trust nothing the field says.",
     "ood_tier": "adversarial", "expected_rejection": True, "category": "powerbi_dax"},
]
