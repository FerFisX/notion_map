"""
rejector.py — Sprint 5: OOD / coverage-risk rejection.

The embedding + constraint pipeline can commit a decision even when the query
is out-of-distribution (OOD) or when the committed intent is weakly supported.
The rejector produces an OOD/coverage-risk score and a rejection verdict so the
system can send such queries back to the confidence/reject path instead of
committing to a source policy.

Design principles (from the plan):
  * The rejector is ADDITIVE observability: it never rewrites core contract
    fields on its own. It exposes `reject`/`reject_score`/`reject_reason`.
  * It rejects BEFORE the final decision when the signal is strong, but must
    NOT reorder a validly-held resolution (a strong literal/semantic winner
    with satisfied constraints stays committed).
  * It is evaluated as a metric (AUROC/AUPRC / FPR@95TPR on OOD vs in-dist)
    before any decision-path wiring, mirroring Sprints 3-4.

Reasons produced:
  * "gate"         - deterministic no-retrieval (greeting/off-topic) gate.
  * "constraint"   - committed intent violates the derived candidates.
  * "ambiguity"    - evidence spans ambiguous boundary (2<->4 / 3<->4) or
                     contradiction proxy fired.
  * "weak_evidence" - best semantic score below the per-category threshold.
"""

import os

REJECT_ENABLED = os.getenv("INTENT_REJECT_ENABLED", "true").lower().strip() in ("true", "1", "yes")
REJECT_WEAK_THRESH = float(os.getenv("INTENT_REJECT_WEAK_THRESH", "0.55"))


def _winner_evidence(evidence: dict, winner: str) -> float:
    if not evidence:
        return 0.0
    return float(evidence.get(("distrust" if winner == "distrust" else winner), 0.0))


def reject_score_for(result: dict, raw_query: str) -> dict:
    """Computes the OOD/coverage-risk score (0 low -> 1 high) and a verdict.

    Returns {"reject": bool, "score": float, "reason": str|None}.
    """
    if not REJECT_ENABLED:
        return {"reject": False, "score": 0.0, "reason": None}

    decision_source = result.get("decision_source")

    # 1) Deterministic gate already rejected it.
    if result.get("decision") == "no_retrieval" or decision_source == "gate":
        return {"reject": True, "score": 1.0, "reason": "gate"}

    # unresolved (LLM skipped / signal-free residual) -> ESCALATE to the LLM
    # judge (Sprint 6), NOT reject. Treat as neutral risk so the rejector does
    # not pre-empt the LLM. (Genuine OOD that is unresolved is still sent to
    # the LLM, which must handle it; it has not wrongly committed yet.)
    if decision_source == "unresolved":
        return {"reject": False, "score": 0.5, "reason": "unresolved"}

    intent = result.get("intent")
    if intent not in (1, 2, 3, 4):
        # no committed intent -> cannot confidently proceed
        return {"reject": True, "score": 0.8, "reason": "weak_evidence"}

    # 2) Constraint violation on the LITERAL path. NB: the constraint engine is
    # authoritative only for the literal decision table (confirmed binary
    # signals). For a SEMANTIC resolution the query legitimately commits to an
    # intent without any anchor word, so `constraint_ok=False` reflects the
    # ABSENCE of literal signals, not a contradiction — never reject there.
    if result.get("constraint_ok") is False and decision_source == "literal":
        return {"reject": True, "score": 0.85, "reason": "constraint"}

    # 3) The deterministic LITERAL table is trusted: a committed literal intent
    # with satisfied constraints proceeds. Genuine contradiction/ambiguity
    # beyond this is delegated to the LLM judge (Sprint 6), because raw
    # embedding scores are diffuse (nonzero across all clusters) and do not make
    # a reliable deterministic ambiguity signal after switching to BGE.
    if decision_source == "literal":
        return {"reject": False, "score": 0.15, "reason": None}

    # 4) Semantic path: trust the committed winner only if its evidence is
    # above the weak-evidence floor AND the winner is decisive (margin > 0).
    # Otherwise route the query to the LLM judge rather than committing.
    evidence = result.get("evidence_score") or {}
    trace = result.get("semantic_trace") or {}
    winner = trace.get("winner")
    if winner:
        w_ev = _winner_evidence(evidence, winner)
        if w_ev < REJECT_WEAK_THRESH:
            return {"reject": True, "score": 0.7, "reason": "weak_evidence"}
        if float(trace.get("margin", 0.0) or 0.0) <= 0.0:
            return {"reject": True, "score": 0.65, "reason": "weak_evidence"}

    # Default: committed with satisfied constraints and adequate evidence.
    return {"reject": False, "score": 0.2, "reason": None}


def reject(result: dict, raw_query: str) -> dict:
    """Public entry. Attaches rejection verdict to a copy of the result
    (never mutating core fields)."""
    v = reject_score_for(result, raw_query)
    out = dict(result)
    out["reject"] = v["reject"]
    out["reject_score"] = round(v["score"], 4)
    out["reject_reason"] = v["reason"]
    return out
