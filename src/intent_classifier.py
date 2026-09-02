"""
intent_classifier.py — Intent Classification (Requirements Engineering Loop).

Determines, BEFORE any retrieval, what the user's intent is with respect to the
Knowledge Base (KB). This is an isolated component: it reads ONLY the raw user
prompt. Whatever the roadmap later uses as a source (RoadmapStep.source,
corpus_pct/web_pct, coverage, ...) NEVER feeds back into this classification —
roadmap generation is fully independent of the query-intent classifier.

Intents (contract): the classifier reads ONLY the raw text. Each intent is
chosen by the TEXT SIGNALS of the query — the user's expressed trust in the
internal KB — NOT by corpus coverage (coverage is decided later, after
retrieval, on the generation side).

1 = KB Only (kb_only)          -> the query points at an internal documented
                                    artifact as the ONLY source ("according to
                                    the complexity table we documented", "our
                                    internal document", "the documented case
                                    study") and requests NO external / updated
                                    information. -> KB only, no web,
                                    no confirmation.
  2 = KB + External
      (kb_plus_external)         -> the same internal anchor PLUS an explicit
                                    request to complement with web / updated /
                                    industry practice ("complement with current
                                    web practices", "add the latest web
                                    practices"). KB stays the base.
                                    -> web search ALWAYS, KB-dominant bias.
  3 = External + KB
      (external_plus_kb)         -> the user signals little trust in the internal
                                    KB for this topic: explicit distrust ("we
                                    don't trust our documentation"),
                                    "only up-to-date information", or the answer
                                    is asked as external industry best practice.
                                    -> web search ALWAYS, web-dominant bias.
  4 = Ambiguous (ambiguous)      -> contextual clarify, one of THREE subtypes:
         source_ambiguous -> a plausible technical request with NO source signal
                              -> offer intents [1, 2, 3].
         external_two_way -> explicitly asks for external / up-to-date / Web
                              info WITHOUT committing whether the KB is the base
                              or just a complement -> offer intents [2, 3].
         no_roadmap       -> greeting, personal, off-topic question unrelated to
                              any roadmap -> NO intent options, conversational.

Neutral-context rule (anti-anchor): mentioning the company name alone ("{Bamboo}")
or generic advisory phrasing ("the best way to", "recommendations for") is NOT
a source signal. A query that cites "Bamboo" about a topic with NO internal
content ("as part of what Bamboo has established, how do you deploy a data
warehouse following best practices?") is intent 3, NOT intent 1 — the
mention does not decide, the CONTENT decides.

Rules enforced here:
  - The user must NOT pick the intent manually for intents 1/2/3; the model infers it.
  - Explicitly requesting external / up-to-date / Internet info is NOT blindly
    assigned to 2 or 3: the system asks which of the two the user prefers,
    unless the phrasing clearly commits to KB-first (intent 2) or external-first
    (intent 3).
  - Mentioning the company name alone ("Bambotec"/"Bamboo") is neutral context
    and does NOT determine intent 1.
  - Documented-artifact anchors (INTERNAL_ANCHOR_TERMS) and external-standard
    phrases (EXTERNAL_ANCHOR_TERMS) are only HINTS to the LLM, never hard cutoffs.
  - A deterministic no-retrieval pre-gate (greetings / off-topic / personal,
    NO_RETRIEVAL_ENABLED) short-circuits BEFORE the LLM. The LLM can also emit
    clarify type "no_roadmap" for queries the gate does not catch.

All model-facing prompts are in ENGLISH. End users are expected to query in
English; the matching lists, dataset and examples are all English.
"""

import os
import re
import json

from src.llm_provider import get_llm

# ---------------------------------------------------------------------------
# Versioning (Sprint 2 / point 29) — reproducibility of intent decisions.
# ---------------------------------------------------------------------------
TAXONOMY_VERSION = "1.4.0"
CLASSIFIER_VERSION = "intent-2026-09"
EMBEDDING_MODEL_VERSION = "BAAI/bge-base-en-v1.5"

# ---------------------------------------------------------------------------
# Configuration (env-overridable, defaults below)
# ---------------------------------------------------------------------------
INTENT_CLASSIFIER_ENABLED = os.getenv("INTENT_CLASSIFIER_ENABLED", "true").lower().strip() in ("true", "1", "yes")
# Minimum confidence to commit to an inferred intent (1/2/3). Below this the
# classification falls back to clarify (source_ambiguous) — never assume.
INTENT_CONFIDENCE_MIN = float(os.getenv("INTENT_CONFIDENCE_MIN", "0.5"))
# CRAG gate (Interpretacion A, "gate de ausencia"): cuando el clasificador
# decide intent 1 (kb_only), el coverage de la intencion core contra la KB
# REAL se usa para detectar temas ausentes de la KB y rebajarlos a intent 4
# (clarify kb_absent). El vector_score de ChromaDB es una distancia/relevancia
# negativa para queries sin respaldo; CRAG_COVERAGE_THRESHOLD es el corte
# sobre ese score (max de los top-k) que marca "tema no cubierto".
CRAG_MODE = os.getenv("CRAG_MODE", "0").lower().strip() in ("1", "true", "yes")
CRAG_COVERAGE_THRESHOLD = float(os.getenv("CRAG_COVERAGE_THRESHOLD", "0.0"))

# Deterministic no-retrieval pre-gate. Kept as a fast/reproducible pre-filter;
# the LLM can still emit clarify type "no_roadmap" for anything missed here.
NO_RETRIEVAL_ENABLED = os.getenv("NO_RETRIEVAL_ENABLED", "true").lower().strip() in ("true", "1", "yes")
GREETING_TERMS = [
    t.strip().lower() for t in os.getenv(
        "GREETING_TERMS",
        "hi,hello,hey,good morning,good afternoon,good evening,"
        "good day,greetings"
    ).split(",") if t.strip()
]
NON_RETRIEVAL_TERMS = [
    t.strip().lower() for t in os.getenv(
        "NON_RETRIEVAL_TERMS",
        "i don't understand,i dont understand,that doesn't help,"
        "that doesnt help,not what i asked,thats not what i asked,"
        "that didn't work for me,that didnt work for me,didn't help,"
        "didnt help,explain what you just said"
    ).split(",") if t.strip()
]
# Query starts matching a personal/identity question.
_NO_RETRIEVAL_RE = re.compile(
    r"^\s*(?:what('s| is) my name|who am i|my name is)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Hints passed to the LLM prompt. Textual evidence only; they do not cut.
# ---------------------------------------------------------------------------
INTERNAL_ANCHOR_TERMS = [
    t.strip().lower() for t in os.getenv(
        "INTERNAL_ANCHOR_TERMS",
        "internal document,our internal document,"
        "the complexity table we documented,complexity table we documented,"
        "internal complexity table,internal case study,our internal case study,"
        "our documented example,documented example,our knowledge base,"
        "internal knowledge base,internal kb,our internal kb,"
        "documented internally,internally documented,"
        "our internal process,our documented process,"
        "our internal standards,our internal best practices,"
        "our internal notes,internal notes,our notes,"
        "as the base,as the anchor,as the foundation,"
        "documented foundation,what we have got internally,"
        "what we've got internally,we have got internally,we've got internally,"
        "what we have documented,our documented foundation"
    ).split(",") if t.strip()
]
EXTERNAL_ANCHOR_TERMS = [
    t.strip().lower() for t in os.getenv(
        "EXTERNAL_ANCHOR_TERMS",
        "updated information,current industry practices,"
        "current web practices,current web techniques,"
        "the latest industry practices,the latest web practices,"
        "recent industry practices,recent web practices,"
        "up-to-date,search the web,search the internet,search online,"
        "look it up online,look up online,complement with the web,"
        "complement with web,supplement with the web,supplement with web,"
        "complement it with,top it up with,pad it with,round it out with,"
        "layer on the latest,stretch it with,current external,latest external,"
        "newest external,with current best practices,with the latest best practices"
    ).split(",") if t.strip()
]
# Generic advisory/recommendation phrasing tends to be source-ambiguous.
SOFT_ADVISORY_TERMS = [
    t.strip().lower() for t in os.getenv(
        "SOFT_ADVISORY_TERMS",
        "the best way to,best way to,what is the best way,"
        "recommendations for,what would you recommend,"
        "what should i consider,improve the performance of"
    ).split(",") if t.strip()
]
COMPANY_TERMS = [t.strip().lower() for t in os.getenv("INTENT_INTERNAL_CONTEXT_TERMS", "bambotec,bamboo").split(",") if t.strip()]

# Deterministic signal detection (fed to the prompt as pre-computed flags so a
# small LLM does not have to re-derive evidence from prose).
DISTRUST_TERMS = [
    t.strip().lower() for t in os.getenv(
        "INTENT_DISTRUST_TERMS",
        "don't trust,dont trust,do not trust,doesn't trust,doesnt trust,"
        "didn't trust,didnt trust,can't rely on,cant rely on,"
        "not confident in,distrust,"
        "don't hold up,dont hold up,doesn't hold up,doesnt hold up,"
        "don't buy,dont buy,don't count on,dont count on,"
        "not reliable enough,falls short,doubtful of,i am skeptical of,"
        "i am doubtful of,i don't trust"
    ).split(",") if t.strip()
]
# Signal that the ANSWER is requested as current external industry practice.
EXTERNAL_AS_ANSWER_TERMS = [
    t.strip().lower() for t in os.getenv(
        "INTENT_EXTERNAL_AS_ANSWER_TERMS",
        "current industry best practices,"
        "following the current industry best practices,"
        "the most recent industry best practices,"
        "only up-to-date information,only external sources,"
        "only the most recent information"
    ).split(",") if t.strip()
]
# Leisure / clearly non-technical subjects -> deterministic no_roadmap
# (greetings, identity, and complaints are already caught by the pre-gate).
OFF_TOPIC_TERMS = [
    t.strip().lower() for t in os.getenv(
        "INTENT_OFF_TOPIC_TERMS",
        "movie,film,recipe,song,music,restaurant,haircut,barber,board games"
    ).split(",") if t.strip()
]

# ---------------------------------------------------------------------------
# Layer 2b — semantic paraphrase matching (embeddings + cosine similarity).
# Only reached when the literal decision table (2a) found NO signal and the
# classifier is enabled. THE LISTS ABOVE ARE THE SOURCE OF TRUTH; this layer
# only covers phrasing the literal substring match cannot see.
# ---------------------------------------------------------------------------
EMBED_ENABLED = os.getenv("EMBED_ENABLED", "true").lower().strip() in ("true", "1", "yes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2").strip() or "all-MiniLM-L6-v2"
EMBED_THRESHOLD = float(os.getenv("EMBED_THRESHOLD", "0.45"))
EMBED_MARGIN = float(os.getenv("EMBED_MARGIN", "0.03"))

# Sprint 3: number of candidate intents to retrieve (Top-K) from the semantic
# layer before reranking. The embedding narrows the space; it does NOT decide.
CANDIDATE_K = int(os.getenv("INTENT_CANDIDATE_K", "3"))

# Semantic cluster -> candidate intent (for Top-K retrieval + recall@K).
# greeting / non_retrieval -> None (no_retrieval, not a source intent).
_CLUSTER_TO_INTENT = {
    "distrust": 3,
    "as_answer": 3,
    "internal_and_external": 2,
    "internal": 1,
    "external": 4,
    "soft": 4,
    "off_topic": 4,
    "greeting": None,
    "non_retrieval": None,
}


def _parse_category_thresholds(raw: str) -> dict:
    """Parses EMBED_CATEGORY_THRESHOLDS like "distrust:0.55,greeting:0.6"
    into {category: threshold}. Unknown/invalid entries are skipped."""
    overrides = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        name, _, value = part.partition(":")
        try:
            overrides[name.strip().lower()] = float(value)
        except ValueError:
            pass
    return overrides


EMBED_CATEGORY_THRESHOLDS = _parse_category_thresholds(os.getenv("EMBED_CATEGORY_THRESHOLDS", ""))


def _term_in(text: str, term: str) -> bool:
    """Checks whether `term` occurs in `text`.

    Single-word terms shorter than 10 chars use a word boundary (\\b…\\b) to
    avoid substring false positives ("hi" inside "think/this/history/hire",
    "hey" inside "they", "song" inside "songwriter"). Multi-word terms and
    longer tokens keep plain substring matching. Text/terms are expected to be
    already lowercased.
    """
    if " " not in term and len(term) < 10:
        return re.search(r"\b" + re.escape(term) + r"\b", text) is not None
    return term in text


def _matches_any(text: str, terms: list) -> str:
    """Returns the first term found in text (word-boundary aware), or None."""
    lowered = text.lower()
    for term in terms:
        if _term_in(lowered, term):
            return term
    return None


def _detect_signals(raw_query: str) -> dict:
    """Pre-computed textual signals used by the prompt. Deterministic, literal
    substring match on the configured hint lists (the same lists the prompt
    exposes as evidence). The flags are fed to the LLM as booleans so it can
    apply the DECISION TABLE without re-deriving evidence from prose."""
    q = raw_query.strip().lower()
    internal  = _matches_any(q, INTERNAL_ANCHOR_TERMS)
    external  = _matches_any(q, EXTERNAL_ANCHOR_TERMS)
    distrust  = _matches_any(q, DISTRUST_TERMS)
    as_answer = _matches_any(q, EXTERNAL_AS_ANSWER_TERMS)
    company   = _matches_any(q, COMPANY_TERMS)
    return {
        "internal":  internal,
        "external":  external,
        "distrust":  distrust,
        "as_answer": as_answer,
        "company":   company,
    }


def _is_no_retrieval_query(raw_query: str) -> bool:
    """Deterministic pre-gate: greetings / off-topic / personal questions do not
    require retrieval and are not classified as intents 1-4."""
    q = raw_query.strip().lower()
    if len(q) < 4:
        return True
    if any(_term_in(q, t) for t in GREETING_TERMS):
        return True
    if any(_term_in(q, t) for t in NON_RETRIEVAL_TERMS):
        return True
    if _NO_RETRIEVAL_RE.match(q):
        return True
    return False


# ---------------------------------------------------------------------------
# Intent labels / clarify options / clarify questions (shown to the user)
# ---------------------------------------------------------------------------
INTENT_LABELS = {
    1: "kb_only",
    2: "kb_plus_external",
    3: "external_plus_kb",
    4: "ambiguous",
}

OPTION_1 = {"id": 1, "label": "Knowledge Base only",
            "description": "Answer using only the internal knowledge base."}
OPTION_2 = {"id": 2, "label": "Knowledge Base + external information",
            "description": "Answer from the knowledge base and complement it with external updated information."}
OPTION_3 = {"id": 3, "label": "External information + Knowledge Base",
            "description": "Answer mainly from external updated information, using the knowledge base as a complement."}

CLARIFY_PAYLOADS = {
    "source_ambiguous": {
        "question": ("I couldn't determine with enough certainty which knowledge source "
                     "you'd like to use. Please choose one of these intents:"),
        "options": [OPTION_1, OPTION_2, OPTION_3],
    },
    "external_two_way": {
        "question": ("You asked for external / up-to-date information (Web Search). "
                     "Which intent do you prefer?"),
        "options": [OPTION_2, OPTION_3],
    },
    "no_roadmap": {
        "question": ("This doesn't look like a technical knowledge request, so I can't "
                     "prepare a roadmap for it. Ask me about a specific technical process or topic."),
        "options": [],
    },
    "kb_absent": {
        "question": ("Your request points at our documentation as the source, but the topic "
                     "isn't covered in the Knowledge Base. Please choose what to do:"),
        "options": [OPTION_2, OPTION_3],
    },
}

_CLARIFY_TYPES = tuple(CLARIFY_PAYLOADS.keys())

# ---------------------------------------------------------------------------
# Layer 2b — REFERENCE_PHRASES: curated phrase clusters per semantic category.
# Each cluster is seeded from the literal lists above PLUS hand-curated
# paraphrase seeds (novel wording, no literal overlap). 9 clusters: the 8
# canonical categories plus "internal_and_external" (intent 2 phrasings, which
# combine two signals and have no dedicated list).
# ---------------------------------------------------------------------------
REFERENCE_PHRASES = {
    "greeting": GREETING_TERMS + [
        "good morning to everyone",
        "how is everyone doing today",
        "nice to see you",
        "glad you are here",
        "i hope you are having a good day",
        "hello, is anyone there",
        "greetings and salutations",
        "howdy everyone",
        "long time no talk",
    ],
    "non_retrieval": NON_RETRIEVAL_TERMS + [
        "that was not the answer i was looking for",
        "you misunderstood my request",
        "repeat what you just said",
        "i asked for something else",
        "that makes no sense to me",
        "i am lost now",
        "that went completely over my head",
        "do you remember my name",
        "what should i call myself",
    ],
    "internal": INTERNAL_ANCHOR_TERMS + [
        "according to what we have written down",
        "based on the write up we produced",
        "following the playbook we keep internally",
        "stick to what is in our shared notes",
        "as covered in our internal wiki",
        "per the reference sheet we maintain",
        "use the material we crafted ourselves",
        "going by the spec we drew up",
        "base it strictly on what we have recorded",
        "use only our internal notes, nothing external",
        "base it strictly on our internal material, not from outside",
        "only what we have documented internally",
        "stick to the notes we produced, nothing from the outside",
        "sticking strictly to what we have already written down",
        "going only by what is in our own files",
        "using nothing but what we have documented already",
        "based purely on what our team has recorded",
        "without going outside the material we already have",
        "keep it to what we have already put on paper",
        "only what we recorded in our own materials",
        "sticking strictly to what we have written down about the primary key schema",
        "going only by what is recorded in our own materials on this topic",
        "without going outside what we already have on file for the table design",
        "based purely on what we have recorded internally about the keys",
        "using nothing but what we have already documented about the setup",
    ],
    "external": EXTERNAL_ANCHOR_TERMS + [
        "pull the latest information from the web",
        "what does the industry say these days",
        "fetch fresh guidelines online",
        "get the newest findings on the topic",
        "check what is current out there",
        "bring me the state of the art",
        "what are people doing now in this area",
        "top me up with the latest developments",
        "pull together the current landscape",
        "pull the newest thinking from wherever it is out there",
        "grab whatever is freshest on the topic",
        "see what the field looks like today before answering",
        "bring back the newest take on this",
        "find the most current take on this before responding",
    ],
    "internal_and_external": [
        "based on our internal document and also the current web practices",
        "according to what we documented plus the latest industry practices",
        "stick to our internal notes and complement them with the web",
        "use our write up and top it up with the newest guidelines",
        "go by the complexity table we have and supplement with current web techniques",
        "follow our internal case study and add the most recent industry practices",
        "keep our documentation as the base and complement with updated information",
        "use the internal standards and bring in the latest web practices",
        "take our documented complexity table as the base and add the newest online guidance",
        "use our internal runbook as the foundation and layer on the latest external practices",
        "ground it in what we have got internally and stretch it with current outside thinking",
        "keep our case study as the anchor and round it out with whatever is newer out there",
        "take our internal notes as the base and top them up with current external approaches",
        "lean on our documented foundation and pad it with the latest external findings",
        "anchor the answer in our internal document and complement it with fresh web guidance",
        "build on our internal notes and bring in the newest external viewpoints",
        "root the answer in what we wrote internally and add recent industry techniques",
        "use our internal material as the starting point and fill in with current web practices",
    ],
    "distrust": DISTRUST_TERMS + [
        "i do not believe our documentation on this",
        "our internal material is not reliable enough here",
        "i dont buy what our notes say about that",
        "we cannot lean on our internal document",
        "the internal documentation does not convince me",
        "i have little faith in our own write ups",
        "our internal material falls short for this",
        "we should not count on our internal knowledge base",
        "i am not convinced by the documented approach",
        "take our documentation at face value is a mistake here",
        "i would not lean on what we have got written down",
        "our internal material does not hold up for me",
        "i am skeptical of what we have got recorded",
        "what we have got on file feels shaky on this",
        "i am doubtful of what we wrote down about this",
        "dont put much stock in our own notes on this",
    ],
    "as_answer": EXTERNAL_AS_ANSWER_TERMS + [
        "give me the industry standard answer",
        "respond with the state of the art approach",
        "reflect the current best practices in your answer",
        "make the answer match the latest practices",
        "answer me with updated best practices",
        "deliver the current industry standard solution",
        "the answer should come from the latest techniques",
        "answer based on what is the accepted practice now",
    ],
    "soft": SOFT_ADVISORY_TERMS + [
        "recommend a solid approach",
        "what approach would you suggest",
        "give me some guidance on",
        "what is the recommended path",
        "how would you advise handling",
        "what is a sensible way to",
        "suggest the best route for",
        "which tool would make the most sense",
        "i am weighing options on",
        "comparing options between",
        "weighing different ways to handle",
        "do you have any suggestions for simplifying",
        "any pointers on keeping",
        "what would be a smart approach to keep",
        "could you advise me on",
        "what is a sane way to approach this",
        "what would you look at first",
        "any advice on how to tackle this",
        "what is a sensible structure for this",
        "how do i typically approach this",
        "what kinds of things tend to go wrong with this",
        "my n8n flows feel sluggish lately what would you look at first",
        "what tends to go wrong when people set up big merges",
        "any thoughts on how to keep this from getting messy over time",
        "what is a sane way to lay out a new schema from scratch",
        "what would you look at first in a sluggish workflow",
        "how should i think about structuring this kind of thing",
    ],
    "off_topic": OFF_TOPIC_TERMS + [
        "how is the football match going",
        "tell me a joke",
        "what is the weather like today",
        "recommend a vacation spot",
        "who won the championship",
        "which tv series should i watch",
        "how is the stock market doing today",
        "what should i cook for dinner",
    ],
}

# Lazy embedder + cached reference vectors (module level, loaded once).
_embed_model = None
_ref_embeddings = None


def _get_embedder():
    """Lazy sentence-transformer model (same pattern as similarity_metrics).
    Kept private here so src/ never depends on the evaluation tree."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _get_reference_embeddings() -> dict:
    """Encodes REFERENCE_PHRASES once per process. Returns
    {category: numpy matrix (n_phrases, dim)} with L2-normalized rows."""
    global _ref_embeddings
    if _ref_embeddings is None:
        model = _get_embedder()
        _ref_embeddings = {}
        for category, phrases in REFERENCE_PHRASES.items():
            normalized = [p.strip().lower() for p in phrases if p and p.strip()]
            if not normalized:
                continue
            _ref_embeddings[category] = model.encode(
                normalized, normalize_embeddings=True
            )
    return _ref_embeddings


def _category_threshold(category: str) -> float:
    """Per-category threshold (EMBED_CATEGORY_THRESHOLDS) or the global
    EMBED_THRESHOLD for categories without an explicit override."""
    return EMBED_CATEGORY_THRESHOLDS.get(category, EMBED_THRESHOLD)


class IntentClassifier:
    """Pre-retrieval intent classifier based only on the raw user prompt."""

    def __init__(self, coverage_retriever=None):
        # Warm the semantic layer (embeddings -> torch) BEFORE building the LLM.
        # Loading sentence-transformers/torch first avoids a Windows OpenMP ABI
        # clash (access violation 0xC0000005) when langchain (ChatOllama) is
        # imported while torch is loaded afterwards. No-op if disabled or the
        # model is unavailable (the 2b layer degrades gracefully to the LLM).
        if EMBED_ENABLED:
            try:
                _get_reference_embeddings()
            except Exception:
                pass
        self.llm = get_llm(temperature=0.0, max_tokens=800)
        # CRAG gate (Interpretacion A, modo "gate de ausencia"): el coverage de
        # la intencion core contra la KB REAL rebaja un caso 1 (kb_only) a
        # caso 4 (clarify kb_absent) cuando el tema no esta en la KB. El handle
        # de coverage se puede inyectar (tests, mock) o se construye lazily
        # solo si CRAG_MODE=1 (evita cargar ChromaDB/embeddings en hot-path
        # cuando el gate esta apagado). Sin handle disponible, el gate no
        # rebaja nada (comportamiento actual intacto).
        self.coverage_retriever = coverage_retriever

    # -- Public API ---------------------------------------------------------
    def classify(self, raw_query: str) -> dict:
        """Public entry point. Runs _classify_core then attaches Sprint 2
        observability (evidence_score, candidates, constraints, versions)
        without changing any of the core contract fields."""
        result = self._classify_core(raw_query)
        signals = _detect_signals(raw_query or "")
        # CRAG gate de ausencia: se aplica sobre CUALQUIER resultado decidido
        # como intent 1 (kb_only), venga del path literal, semantico o LLM.
        # Si la intencion core no esta cubierta en la KB real -> rebajar a
        # intent 4 / clarify kb_absent (un caso 1 contradice la ausencia).
        if res := self._apply_crag_gate(raw_query or "", result):
            result = res
        self._enrich(result, signals)
        return result

    def _classify_core(self, raw_query: str) -> dict:
        """
        Returns the intent contract:

        {
          "intent": 1|2|3|4|None,
          "label": "kb_only"|"kb_plus_external"|"external_plus_kb"|"ambiguous"|"no_retrieval",
          "confidence": float,
          "rationale": str,
          "decision": "proceed"|"clarify"|"no_retrieval",
          "retrieval_necessity": bool,
          "clarify": {type, question, options} | None,
          "source_policy": {kb, external, web_search, bias_target} | None,
          "pre_gate": bool,
          "decision_source": "gate"|"literal"|"semantic"|"llm"|"disabled",
        }
        """
        q = (raw_query or "").strip()
        if not q:
            return self._no_retrieval_result("empty query")

        # 1) Deterministic no-retrieval pre-gate (never reaches the LLM).
        if NO_RETRIEVAL_ENABLED and _is_no_retrieval_query(q):
            return self._no_retrieval_result("greeting/off-topic/personal gate")

        # 1b) Semantic ROUTER as the primary decision layer: coseno against
        # curated reference clusters captures paraphrased source signal that
        # the literal substring table cannot see, with zero LLM latency. The
        # literal table only handles the residual the router cannot resolve
        # (low top1 or tight margin), so phrase-worded signals fall back to
        # deterministic matching instead of the LLM.
        semantic_result = None
        semantic_trace = None
        if EMBED_ENABLED:
            semantic_result, semantic_trace = self._semantic_match(q)
        if semantic_result is not None:
            if semantic_result.get("decision") == "no_retrieval":
                result = semantic_result
            else:
                result = self._finalize(semantic_result)
            if semantic_trace:
                result["semantic_trace"] = semantic_trace
            return result

        # 1c) Deterministic DECISION TABLE as fallback when the router was not
        # decisive: the hard source-policy decisions (1/2/3, external_two_way)
        # must not depend on a small LLM's mood. The LLM only handles the
        # signal-free residual (source_ambiguous / no_roadmap).
        signals = _detect_signals(q)
        resolved = self._deterministic_policy(q, signals)
        if resolved is not None:
            return self._finalize(resolved)

        # 2) Classifier disabled -> conservative passthrough (paraphrase layer
        # is part of the classifier, so it is skipped too).
        if not INTENT_CLASSIFIER_ENABLED:
            return self._disabled_result(q)

        # 2c) INTENT_NO_LLM: skip LLM, return unresolved with trace for
        # calibration export. Used by --no-llm evaluation mode.
        if os.getenv("INTENT_NO_LLM", "0").strip() in ("1", "true"):
            result = {"intent": None, "label": "unresolved",
                      "confidence": 0.0, "decision": "unresolved",
                      "decision_source": "unresolved",
                      "rationale": "LLM skipped (INTENT_NO_LLM)"}
            if semantic_trace:
                result["semantic_trace"] = semantic_trace
            return result

        # 3) LLM classification (residual / signal-free).
        parsed = self._call_llm(q)
        result = self._finalize(parsed)
        if semantic_trace:
            result["semantic_trace"] = semantic_trace
        return result

    # -- Result builders ----------------------------------------------------
    def _enrich(self, result: dict, signals: dict) -> None:
        """Sprint 2 observability + versioning attached in-place.

        Adds (never overwrites core fields):
          * evidence_score  -> decomposed evidence {internal, external, distrust,
                               as_answer, margin, contradiction}
          * candidates      -> constraint-derived compatible intents
          * constraint_ok   -> whether the committed intent respects constraints
          * taxonomy_version / classifier_version / embedding_model_version
        Evidence components come from the ACTUAL semantic cluster scores
        (semantic_trace["scores"]) and the literal signal flags, not from
        parsing the rationale string.
        """
        from src.intent_spec import derive_candidate_intents, intent_allowed, INTENT_SPECS

        sig = signals or {}
        trace = result.get("semantic_trace")
        scores = (trace or {}).get("scores") or {}

        # Evidence: literal flags (1.0 if the anchor term is present) combined
        # with the per-cluster semantic score when available.
        evidence = {
            "internal":  (1.0 if sig.get("internal") else float(scores.get("internal", 0.0))),
            "external":  (1.0 if sig.get("external") else float(scores.get("external", 0.0))),
            "distrust":  (1.0 if sig.get("distrust") else float(scores.get("distrust", 0.0))),
            "as_answer": (1.0 if sig.get("as_answer") else float(scores.get("as_answer", 0.0))),
            "margin":    float((trace or {}).get("margin", 0.0) or 0.0),
            "semantic_winner": (trace or {}).get("winner"),
        }
        # Contradiction proxy: internal-trust and distrust both present at once.
        internal_val = evidence["internal"]
        distrust_val = evidence["distrust"]
        if internal_val > 0.0 and distrust_val > 0.0:
            evidence["contradiction"] = -min(internal_val, distrust_val)
        else:
            evidence["contradiction"] = 0.0

        result["evidence_score"] = evidence

        # Candidate + constraint validation. The constraint engine consumes the
        # CONFIRMED literal flag (boolean) — a residual cluster score is not a
        # confirmed signal, so we only feed it the boolean signal flags from
        # _detect_signals. Evidence_score stays as continuous observability.
        constraint_flags = {
            "internal":  bool(sig.get("internal")),
            "external":  bool(sig.get("external")),
            "distrust":  bool(sig.get("distrust")),
            "as_answer": bool(sig.get("as_answer")),
        }
        candidates = derive_candidate_intents(constraint_flags)
        result["candidates"] = candidates
        intent = result.get("intent")
        if intent in (1, 2, 3, 4):
            allowed = intent_allowed(intent, constraint_flags)
            result["constraint_ok"] = bool(allowed)
            result["intent_spec"] = INTENT_SPECS.get(intent).name if intent in INTENT_SPECS else None
        else:
            result["constraint_ok"] = None  # no_retrieval / unresolved — no constraint applies

        result["taxonomy_version"] = TAXONOMY_VERSION
        result["classifier_version"] = CLASSIFIER_VERSION
        result["embedding_model_version"] = EMBEDDING_MODEL_VERSION if EMBED_ENABLED else None

    def _deterministic_policy(self, raw_query: str, sig: dict):
        """Implements the DECISION TABLE steps A-E on the precomputed signals,
        plus deterministic source_ambiguous / no_roadmap for the signal-free
        residual (small LLMs confabulate anchors; literals do not).
        Returns a parsed-like dict for _finalize, or None to fall back to the LLM."""
        q = raw_query.strip().lower()
        if sig["distrust"]:
            return {"intent": 3, "decision": "proceed", "confidence": 0.95,
                    "rationale": f"distrust of internal KB: '{sig['distrust']}'",
                    "decision_source": "literal"}
        if sig["as_answer"]:
            return {"intent": 3, "decision": "proceed", "confidence": 0.9,
                    "rationale": f"external best practice asked as the answer: '{sig['as_answer']}'",
                    "decision_source": "literal"}
        if sig["internal"] and sig["external"]:
            return {"intent": 2, "decision": "proceed", "confidence": 0.95,
                    "rationale": f"internal anchor '{sig['internal']}' + external complement "
                                 f"'{sig['external']}'",
                    "decision_source": "literal"}
        if sig["internal"]:
            return {"intent": 1, "decision": "proceed", "confidence": 0.95,
                    "rationale": f"internal artifact anchor: '{sig['internal']}', no external phrase",
                    "decision_source": "literal"}
        if sig["external"]:
            return {"intent": 4, "decision": "clarify", "clarify_type": "external_two_way",
                    "confidence": 0.9,
                    "rationale": f"external info requested without base-vs-complement "
                                 f"commitment: '{sig['external']}'",
                    "decision_source": "literal"}
        off_topic = _matches_any(q, OFF_TOPIC_TERMS)
        if off_topic:
            return {"intent": 4, "decision": "clarify", "clarify_type": "no_roadmap",
                    "confidence": 0.9,
                    "rationale": f"non-technical / leisure subject: '{off_topic}'",
                    "decision_source": "literal"}
        soft = _matches_any(q, SOFT_ADVISORY_TERMS)
        if soft:
            return {"intent": 4, "decision": "clarify", "clarify_type": "source_ambiguous",
                    "confidence": 0.85,
                    "rationale": f"advisory phrasing with no source signal: '{soft}'",
                    "decision_source": "literal"}
        return None

    def _semantic_match(self, raw_query: str):
        """Layer 2b — semantic paraphrase matching.

        Compares the raw query against every REFERENCE_PHRASES cluster with
        embeddings + cosine similarity. Decision rule (margin over priority):
          * top1 score >= threshold(category)  AND
          * top1 - top2 >= EMBED_MARGIN
        -> resolve by the top1 category; otherwise return None (let the LLM
        see the signal-free residual). Greetings / identity / complaints that
        the literal gate missed resolve here as a SEMANTIC no_retrieval.

        Returns (result_dict_or_None, semantic_trace_or_None). The trace is a
        diagnostic dict attached to the classify() result so the evaluator can
        log near-threshold cases without re-encoding the query.
        """
        q = raw_query.strip().lower()
        if not q or len(q) < 4:
            return None, None
        try:
            import numpy as np
            refs = _get_reference_embeddings()
            if not refs:
                return None, None
            query_vec = _get_embedder().encode(q, normalize_embeddings=True)
        except Exception as e:
            print(f"  [Intent Classifier] embedding layer unavailable ({e}); skipping 2b.")
            return None, None

        scores = {}
        for category, vectors in refs.items():
            sims = vectors @ query_vec
            scores[category] = float(np.max(sims))

        ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        winner, top_score = ranking[0]
        second_score = ranking[1][1] if len(ranking) > 1 else 0.0
        margin = top_score - second_score
        threshold = _category_threshold(winner)

        # Sprint 3 (Fase 3): literal-distrust guard on the semantic decision.
        # The embedder sometimes reads a distrust phrase ("don't hold up",
        # "not reliable", "falls short") as a neutral internal+external mix
        # because the *internal* and *external* anchors are also present in
        # the same sentence. But explicit distrust of the internal KB is a
        # hard, deterministic signal for intent 3 (the literal decision table
        # already resolves it so). If the router's winner is a cluster that
        # maps to intent 2 (internal_and_external) while the literal distrust
        # flag is set, force the winner to "distrust" so the router still
        # decides intent 3 without the LLM — keeping the semantic router as
        # the primary layer without regressing the 2<->3 boundary.
        if winner == "internal_and_external":
            sig = _detect_signals(raw_query)
            if sig["distrust"]:
                winner = "distrust"
                top_score = scores.get("distrust", top_score)
                threshold = _category_threshold(winner)

        # Sprint 3: candidate retrieval. The embedding now reduces to a
        # Top-K set of candidate INTENTS (deduplicated across clusters), which
        # the reranker / rejector consume in later sprints. The top-1 decision
        # below is kept as the optimistic baseline (unchanged behavior).
        cand_intents = []
        cand_clusters = []
        for cat, score in ranking:
            intent = _CLUSTER_TO_INTENT.get(cat)
            if intent is None:
                continue
            if intent not in cand_intents:
                cand_intents.append(intent)
                cand_clusters.append({"intent": intent, "cluster": cat,
                                      "score": round(score, 4)})
            if len(cand_intents) >= CANDIDATE_K:
                break

        trace = {
            "winner": winner,
            "top_score": round(top_score, 4),
            "second_score": round(second_score, 4),
            "margin": round(margin, 4),
            "threshold": threshold,
            "scores": {k: round(v, 4) for k, v in scores.items()},
            # Sprint 3: Top-K candidate intents for reranking/recall@K
            "candidate_intents": cand_intents,
            "candidate_clusters": cand_clusters,
            "candidate_k": CANDIDATE_K,
        }

        if top_score < threshold or margin < EMBED_MARGIN:
            return None, trace

        return self._semantic_decision(winner, top_score, margin), trace

    def _semantic_decision(self, category: str, top_score: float, margin: float):
        """Maps a winning semantic cluster to the same decision shapes the
        literal table (2a) produces, with decision_source="semantic"."""
        base = f"semantic '{category}' match (sim={top_score:.2f}, margin={margin:.2f})"
        if category == "distrust":
            return {"intent": 3, "decision": "proceed", "confidence": 0.82,
                    "rationale": f"distrust of internal KB: {base}",
                    "decision_source": "semantic"}
        if category == "as_answer":
            return {"intent": 3, "decision": "proceed", "confidence": 0.8,
                    "rationale": f"external best practice asked as the answer: {base}",
                    "decision_source": "semantic"}
        if category == "internal_and_external":
            return {"intent": 2, "decision": "proceed", "confidence": 0.82,
                    "rationale": f"internal anchor + external complement: {base}",
                    "decision_source": "semantic"}
        if category == "internal":
            return {"intent": 1, "decision": "proceed", "confidence": 0.8,
                    "rationale": f"internal artifact anchor: {base}",
                    "decision_source": "semantic"}
        if category == "external":
            return {"intent": 4, "decision": "clarify", "clarify_type": "external_two_way",
                    "confidence": 0.78,
                    "rationale": f"external info requested without base-vs-complement "
                                 f"commitment: {base}",
                    "decision_source": "semantic"}
        if category == "soft":
            return {"intent": 4, "decision": "clarify", "clarify_type": "source_ambiguous",
                    "confidence": 0.78,
                    "rationale": f"advisory phrasing with no source signal: {base}",
                    "decision_source": "semantic"}
        if category == "off_topic":
            return {"intent": 4, "decision": "clarify", "clarify_type": "no_roadmap",
                    "confidence": 0.78,
                    "rationale": f"non-technical / leisure subject: {base}",
                    "decision_source": "semantic"}
        if category in ("greeting", "non_retrieval"):
            return self._semantic_no_retrieval(category, top_score)
        return None

    def _semantic_no_retrieval(self, category: str, top_score: float) -> dict:
        """Greeting / identity / complaint decided by semantics: same shape as
        the gate's no_retrieval result, but decision_source="semantic" (the
        literal gate never matched; this is NOT the pre-gate path)."""
        return {
            "intent": None,
            "label": "no_retrieval",
            "confidence": 0.82,
            "rationale": f"semantic '{category}' match (sim={top_score:.2f})",
            "decision": "no_retrieval",
            "retrieval_necessity": False,
            "clarify": dict(CLARIFY_PAYLOADS["no_roadmap"], type="no_roadmap"),
            "source_policy": None,
            "pre_gate": False,
            "decision_source": "semantic",
        }

    def _no_retrieval_result(self, reason: str) -> dict:
        return {
            "intent": None,
            "label": "no_retrieval",
            "confidence": 1.0,
            "rationale": reason,
            "decision": "no_retrieval",
            "retrieval_necessity": False,
            "clarify": dict(CLARIFY_PAYLOADS["no_roadmap"], type="no_roadmap"),
            "source_policy": None,
            "pre_gate": True,
            "decision_source": "gate",
        }

    def _disabled_result(self, raw_query: str) -> dict:
        return {
            "intent": None,
            "label": "disabled",
            "confidence": 0.0,
            "rationale": "classifier disabled (INTENT_CLASSIFIER_ENABLED=false)",
            "decision": "proceed",
            "retrieval_necessity": True,
            "clarify": None,
            # Conservative passthrough: corpus first, no web search.
            "source_policy": {
                "kb": True,
                "external": True,
                "web_search": False,
                "bias_target": {"favor": "corpus", "min_pct": 70},
            },
            "pre_gate": False,
            "decision_source": "disabled",
        }

    def _finalize(self, parsed: dict) -> dict:
        confidence = float(parsed.get("confidence") or 0.0)
        intent = parsed.get("intent")
        if intent is not None:
            try:
                intent = int(intent)
            except (TypeError, ValueError):
                intent = None

        decision = str(parsed.get("decision") or "clarify").lower().strip()
        clarify_type = str(parsed.get("clarify_type") or "source_ambiguous").lower().strip()
        if clarify_type not in _CLARIFY_TYPES:
            clarify_type = "source_ambiguous"

        # Normalize decision against intent values the model may have squirmed.
        # "clarify" is a hard veto: intent is ALWAYS 4, never 1/2/3.
        if decision == "clarify":
            intent = 4
        elif intent not in (1, 2, 3):
            decision = "clarify"
            clarify_type = "source_ambiguous"
            intent = 4

        # Confidence gate: never commit to an intent below the threshold —
        # that is exactly "not enough information -> Intent 4".
        if decision == "proceed" and confidence < INTENT_CONFIDENCE_MIN:
            decision = "clarify"
            clarify_type = "source_ambiguous"
            intent = 4

        rationale = str(parsed.get("rationale") or "").strip() or "no rationale provided"
        decision_source = str(parsed.get("decision_source") or "llm").strip()

        if decision == "clarify":
            payload = dict(CLARIFY_PAYLOADS[clarify_type], type=clarify_type)
            return {
                "intent": intent,                      # 4
                "label": "ambiguous",
                "confidence": confidence,
                "rationale": rationale,
                "decision": "clarify",
                "retrieval_necessity": clarify_type != "no_roadmap",
                "clarify": payload,
                "source_policy": None,
                "pre_gate": False,
                "decision_source": decision_source,
            }

        # decision == "proceed" with a valid intent 1/2/3.
        policy = self._policy_for(intent)
        return {
            "intent": intent,
            "label": INTENT_LABELS[intent],
            "confidence": confidence,
            "rationale": rationale,
            "decision": "proceed",
            "retrieval_necessity": True,
            "clarify": None,
            "source_policy": policy,
            "pre_gate": False,
            "decision_source": decision_source,
        }

    @staticmethod
    def _policy_for(intent: int) -> dict:
        if intent == 1:
            return {"kb": True, "external": False, "web_search": False,
                    "bias_target": {"favor": "corpus", "min_pct": 100}}
        if intent == 2:
            return {"kb": True, "external": True, "web_search": True,
                    "bias_target": {"favor": "corpus", "min_pct": 70}}
        # intent == 3
        return {"kb": True, "external": True, "web_search": True,
                "bias_target": {"favor": "web", "min_pct": 60}}

    def _apply_crag_gate(self, raw_query: str, result: dict) -> dict | None:
        """CRAG gate de ausencia (Interpretacion A, modo gate de ausencia).

        Si el clasificador decidio intent 1 (kb_only) y el coverage de la
        intencion core contra la KB REAL esta por debajo de
        CRAG_COVERAGE_THRESHOLD, el tema no esta cubierto en la KB -> rebajar
        a intent 4 / clarify kb_absent (no es un caso 1 valido).

        Devuelve el resultado rebajado, o None si el gate no aplica (no es
        intent 1, CRAG_MODE apagado, sin handle de coverage, o el tema SI
        esta cubierto).
        """
        if not CRAG_MODE or result.get("intent") != 1:
            return None
        retriever = self._get_coverage_retriever()
        if retriever is None:
            return None
        from src.kb_coverage import extract_core
        core, extracted = extract_core(raw_query)
        try:
            coverage = retriever(core)
        except Exception:
            return None
        gate = {
            "core": core,
            "core_extracted": extracted,
            "coverage": coverage,
            "threshold": CRAG_COVERAGE_THRESHOLD,
            "applied": bool(coverage <= CRAG_COVERAGE_THRESHOLD),
        }
        if coverage <= CRAG_COVERAGE_THRESHOLD:
            result.update({"crag_gate": gate})
            return {
                "intent": 4,
                "label": "ambiguous",
                "confidence": result.get("confidence", 0.9),
                "rationale": (f"kb_only signal present but topic not covered in the KB "
                              f"(core='{core[:60]}', coverage={coverage:.4f}) -> clarify"),
                "decision": "clarify",
                "retrieval_necessity": True,
                "clarify": dict(CLARIFY_PAYLOADS["kb_absent"], type="kb_absent"),
                "source_policy": None,
                "pre_gate": False,
                "decision_source": result.get("decision_source", "literal"),
                "crag_gate": gate,
            }
        result["crag_gate"] = gate
        return None

    def _get_coverage_retriever(self):
        """Handle de coverage inyectable; si no se inyecto, se construye lazily
        (solo bajo CRAG_MODE) apuntando a la KB real. None si no se puede."""
        if self.coverage_retriever is not None:
            return self.coverage_retriever
        if not CRAG_MODE:
            return None
        try:
            from src.kb_coverage import make_coverage_retriever
            self.coverage_retriever = make_coverage_retriever(
                threshold=CRAG_COVERAGE_THRESHOLD
            )
        except Exception as e:
            print(f"  [Intent Classifier] coverage retriever unavailable ({e}); CRAG gate disabled.")
            self.coverage_retriever = None
        return self.coverage_retriever

    # -- LLM -----------------------------------------------------------------
    def _call_llm(self, raw_query: str) -> dict:
        prompt = self._prompt(raw_query)
        default = {"intent": 4, "confidence": 0.0, "decision": "clarify",
                   "clarify_type": "source_ambiguous",
                   "rationale": "classification failed; clarify by default"}
        try:
            raw = self.llm.invoke(prompt).content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            if not raw.startswith("{"):
                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end > start:
                    raw = raw[start:end + 1]
            data = json.loads(raw)
            return {**default, **data}
        except Exception as e:
            print(f"  [Intent Classifier] LLM call failed ({e}); clarifying by default.")
            return default

    def _prompt(self, raw_query: str) -> str:
        sig = _detect_signals(raw_query)
        return f"""\
You are an INTENT CLASSIFIER for a RAG system that generates technical roadmaps.
Your ONLY task: read the user's query exactly as written and decide which of
four intents about the use of the Knowledge Base (KB) the user has. Do NOT
evaluate the topic, do NOT judge whether the query could be answered elsewhere,
and do NOT look at any corpus or document store.

The four intents (each one is chosen by the TEXT SIGNALS of the query, never by
what the corpus actually contains — that is a retrieval-time decision):

1 = "KB Only" — The query points at a documented INTERNAL artifact ("the
    complexity table we documented", "our internal document", "the documented
    case study", "our knowledge base") as the only source, and does NOT ask for
    external / updated information. No web search, no confirmation requested.
    NOTE: an internal-artifact reference alone is enough; "Bamboo" without an
    artifact is NOT.

2 = "KB + External" — An internal-artifact anchor PLUS an explicit request to
    complement it with external knowledge (fresher/updated info, "current web
    practices", "search the web"). KB stays the base; web search runs; result
    stays KB-dominant.

3 = "External + KB" — The user trusts the internal KB little for this topic:
    explicit distrust ("we don't trust our documentation"), or the answer is
    requested as external industry best practice ("current industry best
    practices", "only up-to-date information"). Even if the query mentions the
    company, the content is external territory. Web search runs; result is
    web-dominant.

4 = "Ambiguous" — Not enough signal to pick 1/2/3 — OR the query is not a
    technical roadmap request at all — OR the user asked for external info
    without committing to 2 vs 3. When you pick this intent you must also pick
    clarify_type:

    - "source_ambiguous": the query could be a technical process/roadmap request
      but gives NO source signal. Offer all three intents (1, 2, 3).
    - "external_two_way": the user EXPLICITLY asked for external / up-to-date /
      Internet / Web Search information WITHOUT saying whether the KB is the
      base or only a complement. Do NOT silently pick 2 or 3 — ask which of the
      two they prefer. ("complement with the web" commits to 2; "we don't trust
      ours, search the web for the right thing" commits to 3; "search the
      internet for updated info and answer" does NOT commit -> two_way.)
    - "no_roadmap": greeting, personal, or off-topic question with nothing to do
      with a technical roadmap ("what's my name?", "who are you?", complaints,
      product recommendations). No intent options are offered; a conversational
      reply is used instead.

DECISION TABLE (apply mechanically, in this order, over the PRECOMPUTED SIGNALS
block below — trust those booleans, do NOT re-derive evidence from prose):
  step A: distrust = True  -> intent 3, proceed.
  step B: as_answer = True ("current industry best practices" as the ANSWER)
          -> intent 3, proceed.
  step C: internal = True AND external = True  -> intent 2, proceed. ALWAYS.
          The internal anchor overweight any later web phrase: never demote to 3.
          ALSO: an explicit internal BASE anchor ("as the base", "as the anchor",
          "as the foundation", "ground it in", "root it in", "build on our
          internal ...", "keep our ... as the base/anchor") combined with an
          external COMPLEMENT ("complement", "top it up", "pad it with", "round
          it out with", "layer on", "stretch it with", "bring in/add/latest")
          is intent 2 EVEN IF the precomputed booleans above under-detect one
          side. Treat the phrase structure as the commit: base + complement = 2,
          NOT external_two_way. Only pick external_two_way when there is NO
          internal base anchor at all.
  step D: internal = True (and external = False) -> intent 1, proceed.
  step E: external = True WITHOUT internal, and the query does not commit whether
          the KB is the base or a complement -> intent 4, external_two_way.
  step F: internal = False AND external = False, technical/process question
          -> intent 4, source_ambiguous.
  step G: greeting, personal, off-topic, complaint -> intent 4, no_roadmap.

RULES:
- Infer intents 1/2/3 from the query text; the user never picks manually.
- Company mention alone is NEUTRAL context, never intent 1.

PRECOMPUTED TEXTUAL SIGNALS OF "{raw_query}" (deterministic literal matches on
the configured hint lists; trust them and apply the DECISION TABLE steps A-G
using exactly these booleans, do not re-derive from prose):
  internal  = {sig['internal']}
  external  = {sig['external']}
  distrust  = {sig['distrust']}
  as_answer = {sig['as_answer']}
  company   = {sig['company']}

EXAMPLES (calibrate the criterion; they are not the query to classify):
- "According to the complexity table we documented, which operation is the most
  expensive?" -> intent 1, proceed.
- "How did Bamboo define, in its internal document, the example of candidate
  keys for an employees table?" -> intent 1, proceed.
- "What incremental testing sequence does Bamboo document for agent workflows
  in N8N?" -> intent 1, proceed (no external phrase anywhere).
- "According to our internal case study, what is the testing sequence for agents
  in N8N? Complement with current web practices." -> intent 2, proceed.
- "According to the complexity table we documented, how do you debug a slow
  merge when the tables exceed a million rows? Complement with current web
  techniques." -> intent 2, proceed.
- "Take our internal notes as the base and top them up with current external
  approaches." -> intent 2, proceed (internal BASE + external COMPLEMENT).
- "Ground it in what we've got internally, then stretch it with current outside
  thinking." -> intent 2, proceed (internal base + external complement -> NOT
  external_two_way; the base anchor commits the direction).
- "Keep our case study as the anchor, but round it out with whatever is newer
  out there." -> intent 2, proceed.
- "Lean on our documented foundation and pad it with the latest external
  findings." -> intent 2, proceed.
- "Following the internal complexity table, when is it better to use pivot
  instead of unpivot? Add the latest web practices." -> intent 2, proceed.
- "We don't trust our documentation for this; search the internet for the
  current industry best practices and use them, and our KB only as a
  complement." -> intent 3, proceed.
- "As part of what Bamboo has established, how do you deploy a data warehouse
  following the current industry best practices?" -> intent 3, proceed
  (Bamboo mention but the content is external territory -> NOT 1).
- "How do I optimize a data pipeline?" -> intent 4, source_ambiguous.
- "Search the internet for updated information and answer me with that."
  -> intent 4, external_two_way (asks external without committing base vs
  complement).
- "Use the Bambotec information and search the internet for updated
  information." -> intent 4, external_two_way (doesn't say if the KB is the
  base or a complement).
- "How do I do this in Bambotec?" -> intent 4, source_ambiguous.
  (company name alone is not a signal; full context is insufficient)
- "What movie do you recommend for this weekend?" -> intent 4, no_roadmap.
- "That answer you gave me before didn't help at all." -> intent 4, no_roadmap
  (complaint, no technical content).

USER QUESTION (exactly as written):
{raw_query}

Respond ONLY with valid JSON (no markdown):
{{
"intent": 1|2|3|4,
"decision": "proceed"|"clarify",
"clarify_type": "source_ambiguous"|"external_two_way"|"no_roadmap",
"confidence": <number 0-1, how certain you are>,
"rationale": "<one English sentence that QUOTES the exact passage of the query
  that motivated the decision>"
}}

When decision is "proceed", clarify_type must be null or omitted.
When decision is "clarify", intent must be 4.
"""


# Module-level convenience (process-wide cached instance).
_classifier = None


def classify_intent(raw_query: str) -> dict:
    """Convenience wrapper: builds a cached IntentClassifier on first use."""
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier.classify(raw_query)


def policy_for_intent(intent: int) -> dict:
    """Returns the source policy for an intent (1/2/3), as applied by the
    classifier. Used by the RAG orchestrator to CONTINUE an already-clarified
    Intent 4 (continuation with the user-selected intent)."""
    return IntentClassifier._policy_for(intent)