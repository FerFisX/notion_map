"""
intent_evaluator.py — Evaluates the ISOLATED SOURCE-INTENT CLASSIFIER
(src/intent_classifier.py), outside the rest of the pipeline (retrieval,
coverage, web search, generation).

Why separate from exclusivity_validator.py? That validation runs the full
pipeline (adapter.query) and its PASS/FAIL depends — besides the
classification — on real web search, coverage and source attribution. This
evaluator isolates the "intent classification" variable from the
"generation pipeline" variable.

Criterion (same source of truth as the dataset — TEXT signals):
  * case1_solo_knowledge_base      -> expected intent 1, decision "proceed"
  * case2_knowledge_base_con_hueco -> expected intent 2, decision "proceed"
  * case3_solo_web                 -> expected intent 3, decision "proceed"
  * case4_clarify                  -> expected intent 4, decision "clarify",
                                      and if the sample declares
                                      expected_clarify_type, that subtype
                                      (source_ambiguous / external_two_way /
                                      no_roadmap) must match the returned
                                      `clarify.type`.

The deterministic pre-gate NEGATIVES (PRE_GATE_SAMPLES) are also evaluated:
queries that NEVER reach the LLM (greetings / identity) — decision must be
"no_retrieval" and pre_gate True.

ADVERSARIAL_SAMPLES assert per-case resilience: each entry declares how the
classifier must (not) resolve, checked against the `decision_source` field
("gate" | "literal" | "semantic" | "llm").

PARAPHRASE_SAMPLES measure the Layer 2b coverage (embeddings + cosine): novel
English wording with NO literal-term match, so they can only resolve via the
semantic layer (decision_source == "semantic") or the LLM residual. They also
feed the near-threshold log (queries that fell just below the semantic
threshold/margin — real candidates to keep curating reference seeds).

v2 PARAPHRASE_SAMPLES are dicts declaring `assert_source` per sample:
  * "semantic"     -> must resolve via decision_source == "semantic"
  * "not_semantic" -> hard negative: must NOT false-fire on the tested cluster
  * "llm"          -> acceptable to fall through to the LLM by design

Usage:
  python -m evaluation.intent_evaluator [--samples N]
"""

import argparse

import sys
import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

from src.intent_classifier import (
    INTENT_CLASSIFIER_ENABLED, EMBED_ENABLED, EMBED_MARGIN,
    IntentClassifier, INTENT_LABELS, CRAG_MODE,
    _detect_signals, _matches_any,
    GREETING_TERMS, NON_RETRIEVAL_TERMS, OFF_TOPIC_TERMS, SOFT_ADVISORY_TERMS,
    _NO_RETRIEVAL_RE,
)
from evaluation.config import config
from evaluation.intent_dataset import (
    PRE_GATE_SAMPLES, ADVERSARIAL_SAMPLES, PARAPHRASE_SAMPLES,
    INTENT_SAMPLES, MINIMAL_PAIRS, OOD_SAMPLES, QueryCase,
)

REPORTS_DIR = Path(config.reports_dir) / "intent_runs"


def expected_decision(query_case: QueryCase) -> str:
    return "clarify" if query_case is QueryCase.CLARIFY else "proceed"


def has_literal_signal(query: str) -> bool:
    """Purity guard for PARAPHRASE_SAMPLES: True if the raw query would already
    be resolved by the literal layers (gate + decision table 2a). Paraphrase
    samples MUST be False so they only exercise Layer 2b / the LLM residual."""
    q = query.strip().lower()
    sig = _detect_signals(query)
    if any(sig[k] for k in ("internal", "external", "distrust", "as_answer")):
        return True
    if _NO_RETRIEVAL_RE.match(q):
        return True
    for lst in (GREETING_TERMS, NON_RETRIEVAL_TERMS, OFF_TOPIC_TERMS, SOFT_ADVISORY_TERMS):
        if _matches_any(q, lst):
            return True
    return False


def _sample_fields(sample):
    """Normalizes a dataset entry (legacy dataclass or v2 dict) into
    (question, query_case, expected_clarify_type, assert_source, cluster)."""
    if isinstance(sample, dict):
        return (sample.get("question", ""),
                sample["query_case"],
                sample.get("expected_clarify_type"),
                sample.get("assert_source"),
                sample.get("cluster"))
    return (sample.question,
            sample.query_case,
            sample.expected_clarify_type,
            getattr(sample, "assert_source", None),
            getattr(sample, "cluster", None))


def _evaluate_samples(classifier: IntentClassifier, samples, set_name: str,
                      verbose: bool = False) -> tuple:
    """Runs one sample set through the classifier and returns
    (entries, n_clarify_checked, n_clarify_type_ok). The entry carries the
    decision_source, the semantic_trace (if the Layer 2b ran) and a `set` tag.

    When a sample declares `assert_source` (the v2 PARAPHRASE_SAMPLES contract)
    the PASS criterion also depends on WHICH layer resolved the query:
      * "semantic"      -> must resolve via decision_source == "semantic"
      * "not_semantic"  -> must NOT resolve via "semantic" (hard negative: no
                           false semantic fire on the tested cluster)
      * "llm"/missing   -> any layer is acceptable
    """
    entries = []
    n_clarify_checked = 0
    n_clarify_type_ok = 0
    for sample in samples:
        question, qcase, exp_clarify_type, assert_source, cluster = _sample_fields(sample)
        exp_intent   = qcase.expected_intent
        exp_decision = expected_decision(qcase)
        result  = classifier.classify(question)

        cls_intent   = result.get("intent")
        cls_decision = result.get("decision")
        confidence   = result.get("confidence")

        intent_ok   = (cls_intent == exp_intent)
        decision_ok = (cls_decision == exp_decision)

        cls_clarify_type = (result.get("clarify") or {}).get("type")
        clarify_type_ok  = None
        if qcase is QueryCase.CLARIFY:
            intent_ok = intent_ok and decision_ok  # clarifying is NOT guessing
            if exp_clarify_type:
                n_clarify_checked += 1
                clarify_type_ok = (cls_clarify_type == exp_clarify_type)
                if clarify_type_ok:
                    n_clarify_type_ok += 1
                intent_ok = intent_ok and clarify_type_ok

        pre_gate     = result.get("pre_gate", False)
        pre_gate_hit = pre_gate and cls_decision == "no_retrieval"
        no_intent    = cls_intent is None and cls_decision not in ("clarify",)

        # For the no_roadmap subtype the goal is "don't generate a roadmap":
        # it is satisfied whether the deterministic gate cuts the query
        # (no_retrieval) or the semantics/LLM returns clarify no_roadmap.
        if exp_clarify_type == "no_roadmap" and cls_decision == "no_retrieval":
            decision_ok = True
            clarify_type_ok = True
            intent_ok = True

        correctness_ok = intent_ok and decision_ok

        ds = result.get("decision_source")
        layer_reason = None
        if assert_source == "semantic":
            layer_ok = (ds == "semantic")
            if not layer_ok:
                layer_reason = f"asserted semantic but resolved by {ds}"
        elif assert_source == "not_semantic":
            layer_ok = (ds != "semantic")
            if not layer_ok:
                winner = (result.get("semantic_trace") or {}).get("winner")
                layer_reason = f"false semantic fire (assert not_semantic, winner={winner})"
        else:
            layer_ok = True

        ok = correctness_ok and layer_ok
        entries.append({
            "set":            set_name,
            "question":       question,
            "cluster":        cluster,
            "query_case":     qcase.value,
            "expected_intent":   exp_intent,
            "classified_intent": cls_intent,
            "classified_label":  INTENT_LABELS.get(cls_intent, result.get("label", "?")),
            "confidence":      confidence,
            "expected_decision": exp_decision,
            "classified_decision": cls_decision,
            "expected_clarify_type":   exp_clarify_type,
            "classified_clarify_type": cls_clarify_type,
            "clarify_type_ok":   clarify_type_ok,
            "intent_ok":       intent_ok,
            "decision_ok":     decision_ok,
            "assert_source":   assert_source,
            "layer_ok":        layer_ok,
            "pre_gate_hit":    pre_gate_hit,
            "no_intent":       no_intent,
            "decision_source": ds,
            "semantic_trace":  result.get("semantic_trace"),
            "cluster_scores":  (result.get("semantic_trace") or {}).get("scores"),
            "verdict":         "PASS" if ok else "FAIL",
            "rationale":       result.get("rationale", ""),
        })

        status = "[PASS]" if ok else "[FAIL]"
        ct_note = ""
        if exp_clarify_type:
            ct_note = f"  clarify: expected={exp_clarify_type} classified={cls_clarify_type}"
        print(f"  {status} [{set_name}] ({qcase.value}) {question[:52]}")
        print(f"         expected intent={exp_intent}  classified={cls_intent} "
              f"({entries[-1]['classified_label']}, conf={confidence})  decision={cls_decision}"
              f"{ct_note}")
        if pre_gate_hit:
            print("         [pre-gate] the classifier did NOT call the LLM (no-retrieval query)")
        if not ok:
            reasons = []
            if not correctness_ok:
                reasons.append("wrong intent/clarify")
            if not layer_ok:
                reasons.append(layer_reason or "layer assertion failed")
            print(f"         ! {'; '.join(reasons)} :: {result.get('rationale', '')[:110]}")
    return entries, n_clarify_checked, n_clarify_type_ok


def run_pre_gate_evaluation(classifier: IntentClassifier,
                            pre_gate_samples=None, verbose: bool = False) -> list:
    """Verifies that no-retrieval queries NEVER reach the LLM."""
    entries = []
    for query in (pre_gate_samples or []):
        result  = classifier.classify(query)
        pg_hit  = result.get("pre_gate", False) and result.get("decision") == "no_retrieval"
        entries.append({
            "set":            "pre_gate",
            "query":       query,
            "decision":    result.get("decision"),
            "pre_gate":    pg_hit,
            "decision_source": result.get("decision_source"),
            "semantic_trace":  result.get("semantic_trace"),
            "verdict":     "PASS" if pg_hit else "FAIL",
        })
        status = "[PASS]" if pg_hit else "[FAIL]"
        print(f"  {status} [pre-gate] {query[:60]}")
        if not pg_hit:
            print(f"         decision={result.get('decision')} pre_gate={result.get('pre_gate')}")
    return entries


def run_adversarial_evaluation(classifier: IntentClassifier,
                               adversarial_samples=None, verbose: bool = False) -> list:
    """Checks each adversarial sample against its declared assertion, using
    the `decision_source` field of the result ("gate" | "literal" | "semantic" | "llm"):
      - "not_literal"      -> decision_source must NOT be "literal"
      - "still_gate"       -> decision_source == "gate" and decision no_retrieval
      - "literal_intent3"  -> decision_source == "literal" and intent == 3
    """
    entries = []
    for item in (adversarial_samples or []):
        query       = item.get("question", "")
        assert_kind = item.get("assert")
        result      = classifier.classify(query)
        ds          = result.get("decision_source")
        if assert_kind == "not_literal":
            ok = (ds != "literal")
        elif assert_kind == "still_gate":
            ok = (ds == "gate") and result.get("decision") == "no_retrieval"
        elif assert_kind == "literal_intent3":
            ok = (ds == "literal") and result.get("intent") == 3
        else:
            ok = False
        entries.append({
            "set":                "adversarial",
            "question":           query,
            "assert":             assert_kind,
            "decision_source":    ds,
            "classified_decision": result.get("decision"),
            "classified_intent":  result.get("intent"),
            "semantic_trace":     result.get("semantic_trace"),
            "verdict":            "PASS" if ok else "FAIL",
        })
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} [adversarial:{assert_kind}] {query[:60]}")
        if not ok:
            print(f"         decision_source={ds} decision={result.get('decision')} intent={result.get('intent')}")
    return entries


def _crag_gate_metric(intent_entries):
    """Metrica de FP del CRAG gate de ausencia sobre INTENT_SAMPLES.

    El gate rebaja intent 1 -> clarify kb_absent cuando la cobertura del core
    contra la KB esta por debajo del umbral. La metrica mide, sobre los casos
    donde el gate se evaluo y/rebajo:
      * n_rebajos:      cuantos intent 1 fueron rebajados a 4 por el gate.
      * n_mal_rebajos:  de esos, cuantos esperaban intent 1 (FP del gate:
                        rebaje innecesario; coinciden con FAIL si el tema SI
                        estaba cubierto pero el umbral fue demasiado agresivo).
      * n_extraccion:   cuantos cores no se pudieron extraer (fallos de
                        extraccion, marcados aparte — no contaminan la metrica).
      * n_evaluados:    en cuantos el gate se evaluo (cov disponible).
    Es una metrica SEPARADA de la exactitud de intencion (acierto 1/2/3/4):
    se informa aparte, no se combina.
    """
    if not intent_entries:
        return None
    rebajos = [e for e in intent_entries
               if e.get("crag_gate") and e["crag_gate"].get("applied")]
    evaluados = [e for e in intent_entries if e.get("crag_gate")]
    extracciones = [e.get("crag_gate") for e in intent_entries
                    if e.get("crag_gate") and not e["crag_gate"].get("core_extracted")]
    mal_rebajos = [e for e in rebajos if e.get("expected_intent") == 1]
    return {
        "n_rebajos":     len(rebajos),
        "n_mal_rebajos": len(mal_rebajos),
        "fp_rebajos":    len(mal_rebajos),
        "n_extraccion":  len(extracciones),
        "n_evaluados":   len(evaluados),
        "gate_flag":     bool(CRAG_MODE),
    }


def evaluate_intent_samples(classifier: IntentClassifier, intent_samples=None,
                            verbose: bool = False) -> tuple:
    """Evaluates the balanced INTENT_SAMPLES (whole pipeline, intents 1-4).

    Primary assertion is CORRECT INTENT (fills the real confusion matrix).
    When `signal_kind` == "semantic" the sample must also resolve via the
    semantic layer (decision_source "semantic" or "llm"); literal samples
    must resolve deterministically (gate/literal/semantic/llm all acceptable,
    the intent is what matters).

    Returns (entries, aggregated_intent_accuracy)."""
    entries = []
    for sample in (intent_samples or []):
        question = sample.get("question", "")
        qcase     = sample.get("query_case")
        sig_kind  = sample.get("signal_kind", "semantic")
        exp_intent   = qcase.expected_intent
        exp_decision = expected_decision(qcase)
        exp_clarify  = sample.get("expected_clarify_type")
        result       = classifier.classify(question)

        cls_intent   = result.get("intent")
        cls_decision = result.get("decision")
        cls_clarify  = (result.get("clarify") or {}).get("type")

        intent_ok   = (cls_intent == exp_intent)
        decision_ok = (cls_decision == exp_decision)
        clarify_ok  = None
        if qcase is QueryCase.CLARIFY:
            intent_ok = intent_ok and decision_ok
            if exp_clarify:
                clarify_ok = (cls_clarify == exp_clarify)
                intent_ok = intent_ok and clarify_ok

        # Layer assertion: semantic samples must go through the semantic/llm
        # path; literal samples must be deterministic (not index-free).
        ds = result.get("decision_source")
        if sig_kind == "semantic":
            layer_ok = (ds in ("semantic", "llm"))
            layer_reason = None if layer_ok else f"expected semantic path, got {ds}"
        else:
            layer_ok = (ds in ("gate", "literal", "semantic", "disabled"))
            layer_reason = None if layer_ok else f"unexpected decision_source {ds}"

        ok = intent_ok and layer_ok
        entries.append({
            "set": "intent",
            "question": question,
            "query_case": qcase.value,
            "expected_intent": exp_intent,
            "classified_intent": cls_intent,
            "classified_label": INTENT_LABELS.get(cls_intent, "?"),
            "expected_decision": exp_decision,
            "classified_decision": cls_decision,
            "expected_clarify_type": exp_clarify,
            "classified_clarify_type": cls_clarify,
            "signal_kind": sig_kind,
            "layer_ok": layer_ok,
            "intent_ok": intent_ok,
            "decision_ok": decision_ok,
            "decision_source": ds,
            "semantic_trace": result.get("semantic_trace"),
            "cluster_scores": (result.get("semantic_trace") or {}).get("scores"),
            "crag_gate": result.get("crag_gate"),
            "verdict": "PASS" if ok else "FAIL",
            "rationale": result.get("rationale", ""),
            "regression": sample.get("regression", ""),
        })
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} [intent] ({qcase.value}|{sig_kind}) {question[:52]}")
        if not ok:
            reasons = []
            if not intent_ok:
                reasons.append(f"intent expected={exp_intent} got={cls_intent}")
            if not layer_ok:
                reasons.append(layer_reason)
            print(f"         ! {'; '.join(reasons)} :: {result.get('rationale','')[:100]}")
    return entries


def evaluate_minimal_pairs(classifier: IntentClassifier, minimal_pairs=None,
                           verbose: bool = False) -> tuple:
    """Evaluates MINIMAL_PAIRS: same vocabulary, one decisive condition apart.

    Asserts the classifier picks the intent that matches the declared
    Sprint-2 dimensions. Report grouped by frontier so 2<->3, 2<->4, 3<->4
    boundaries are isolated."""
    entries = []
    for pair in (minimal_pairs or []):
        question   = pair.get("question", "")
        exp_intent = pair.get("intent")
        frontier   = pair.get("frontier", "?")
        group      = pair.get("group", "?")
        result     = classifier.classify(question)
        cls_intent = result.get("intent")
        ok         = (cls_intent == exp_intent)
        entries.append({
            "set": "minimal_pair",
            "group": group, "frontier": frontier, "dimension": pair.get("dimension"),
            "exp_internal": pair.get("internal_relation"), "exp_external": pair.get("external_relation"),
            "exp_ambiguity": pair.get("ambiguity"),
            "question": question,
            "expected_intent": exp_intent, "classified_intent": cls_intent,
            "classified_label": INTENT_LABELS.get(cls_intent, "?"),
            "decision_source": result.get("decision_source"),
            "semantic_trace": result.get("semantic_trace"),
            "cluster_scores": (result.get("semantic_trace") or {}).get("scores"),
            "verdict": "PASS" if ok else "FAIL",
        })
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} [minimal_pair {frontier} {group}] expected={exp_intent} "
              f"got={cls_intent} | {question[:42]}")
    return entries


def evaluate_ood_samples(classifier: IntentClassifier, ood_samples=None,
                         verbose: bool = False) -> tuple:
    """Evaluates OOD_SAMPLES rejection. `expected_rejection` True means the
    classifier should NOT commit to a confident intent 1/2/3 with decision
    "proceed" — a prefer intent 4/None or clarify (a future rejector should
    flag these). The near/adversarial tiers are the hard checks."""
    entries = []
    for item in (ood_samples or []):
        question   = item.get("question", "")
        tier       = item.get("ood_tier", "far")
        exp_reject = item.get("expected_rejection", True)
        result     = classifier.classify(question)
        cls_intent = result.get("intent")
        cls_decision = result.get("decision")
        # Rejection satisfied if the classifier does not commit to 1-3 proceed
        # (i.e. intent None, OR decision not "proceed", OR intent 4 clarify).
        committed = (cls_intent in (1, 2, 3)) and (cls_decision == "proceed")
        rejection_ok = (not committed) if exp_reject else committed
        ok = rejection_ok
        entries.append({
            "set": "ood",
            "ood_tier": tier,
            "question": question,
            "expected_rejection": exp_reject,
            "committed": committed,
            "classified_intent": cls_intent,
            "classified_decision": cls_decision,
            "decision_source": result.get("decision_source"),
            "semantic_trace": result.get("semantic_trace"),
            "cluster_scores": (result.get("semantic_trace") or {}).get("scores"),
            "verdict": "PASS" if ok else "FAIL",
            "rationale": result.get("rationale", ""),
        })
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} [ood:{tier}] intent={cls_intent} decision={cls_decision} "
              f"| {question[:44]}")
        if not ok:
            print(f"         ! {result.get('rationale','')[:90]}")
    return entries


def evaluate_reranker(samples, reranker=None, intent_specs=None, verbose: bool = False,
                      classifier=None) -> dict:
    """Sprint 4: measures whether the cross-encoder reranker improves the
    critical frontiers vs the embedding top-1 baseline.

    For each sample with a semantic trace (candidate_intents), compare:
      * embedding baseline: is the expected intent == the semantic WINNER intent?
      * reranker:           is the expected intent == the reranker's ranked[0]?
    Grouped by frontier so 2<->3, 2<->4, 3<->4 are isolated. This is an
    OFFLINE diagnostic (does not rewire the production decision)."""
    from src.intent_spec import INTENT_SPECS
    intent_specs = intent_specs or INTENT_SPECS
    if classifier is None:
        classifier = IntentClassifier()

    rows = []
    for e in samples:
        result = classifier.classify(e.get("question") or e.get("query"))
        trace = (result or {}).get("semantic_trace") or {}
        cands = trace.get("candidate_intents")
        if not cands:
            continue
        expected = e.get("expected_intent")
        if expected is None and e.get("intent") is not None:
            expected = e["intent"]              # minimal pairs carry canonical intent
        if expected is None and e.get("query_case") is not None:
            expected = e["query_case"].value     # intents carry QueryCase (1-4)
        query = e.get("question") or e.get("query")
        frontier = e.get("frontier")

        # Embedding baseline: the semantic winner's intent.
        winner = trace.get("winner")
        winner_intent = {
            "distrust": 3, "as_answer": 3, "internal_and_external": 2,
            "internal": 1, "external": 4, "soft": 4, "off_topic": 4,
        }.get(winner)
        emb_top1_ok = (winner_intent == expected)

        # Reranker pick.
        rr = reranker.rerank(query, cands, intent_specs)
        rr_top1 = rr["ranked"][0] if rr["ranked"] else None
        rr_top1_ok = (rr_top1 == expected)

        rows.append({
            "set": e.get("set"), "frontier": frontier,
            "question": query, "expected_intent": expected,
            "winner_intent": winner_intent, "emb_top1_ok": emb_top1_ok,
            "candidate_intents": cands,
            "reranked": rr["ranked"], "reranker_top1": rr_top1,
            "reranker_top1_ok": rr_top1_ok, "reranker_scores": rr.get("scores"),
            "reranker_source": rr.get("source"),
        })
        if verbose:
            ok = "OK " if (emb_top1_ok or rr_top1_ok) else "  "
            print(f"  [{ok}] frontier={frontier} expected={expected} "
                  f"emb_top1={winner_intent} rr_top1={rr_top1} | {query[:44]}")

    total = len(rows)
    emb_ok = sum(1 for r in rows if r["emb_top1_ok"])
    rr_ok  = sum(1 for r in rows if r["reranker_top1_ok"])
    by_frontier = {}
    for f in ("2_3", "2_4", "3_4"):
        fr = [r for r in rows if r["frontier"] == f]
        by_frontier[f] = {
            "n": len(fr),
            "emb_ok": sum(1 for r in fr if r["emb_top1_ok"]),
            "rr_ok":  sum(1 for r in fr if r["reranker_top1_ok"]),
        }
    return {
        "n_samples": total,
        "embedding_top1_ok": emb_ok,
        "reranker_top1_ok": rr_ok,
        "reranker_source": rows[0]["reranker_source"] if rows else None,
        "by_frontier": by_frontier,
        "rows": rows,
    }


def _auroc(y_true, y_score):
    """Area under ROC for binary labels + continuous score (manual, small N)."""
    preds = sorted(zip(y_score, y_true), key=lambda x: x[0])
    tp = fp = 0
    total_pos = sum(y_true)
    total_neg = len(y_true) - total_pos
    tpr_prev, fpr_prev, auc = 0.0, 0.0, 0.0
    for s, y in preds:
        if y:
            tp += 1
        else:
            fp += 1
        tpr = tp / total_pos if total_pos else 0.0
        fpr = fp / total_neg if total_neg else 0.0
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
        tpr_prev, fpr_prev = tpr, fpr
    return auc


def _auprc(y_true, y_score):
    """Area under precision-recall curve (manual)."""
    preds = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    tp = fp = 0
    total_pos = sum(y_true)
    prec_sum = 0.0
    prev_recall = 0.0
    for s, y in preds:
        if y:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / total_pos if total_pos else 0.0
        prec_sum += precision * (recall - prev_recall)
        prev_recall = recall
    return prec_sum


def _fpr_at_tpr(y_true, y_score, tpr_target=0.95):
    preds = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    tp = 0
    total_pos = sum(y_true)
    total_neg = len(y_true) - total_pos
    for s, y in preds:
        if y:
            tp += 1
        tpr = tp / total_pos if total_pos else 0.0
        if tpr >= tpr_target:
            fp = preds.index((s, y)) + 1 - tp
            fpr = fp / total_neg if total_neg else 0.0
            return max(0.0, fpr)
    return 1.0


def evaluate_rejector(ood_samples, id_samples, classifier=None, verbose: bool = False) -> dict:
    """Sprint 5: measures the rejector's OOD discriminative power.

    Positive class = OOD samples; negative = in-distribution (intents + minimal
    pairs). Reports AUROC / AUPRC / FPR@95TPR and a straight rejection accuracy
    table (reject OOD, not reject in-dist)."""
    from src.rejector import reject_score_for
    if classifier is None:
        classifier = IntentClassifier()

    y_true, y_score, meta = [], [], []
    for e in id_samples:
        q = e.get("question") or e.get("query")
        r = classifier.classify(q)
        v = reject_score_for(r, q)
        y_true.append(0)
        y_score.append(v["score"])
        meta.append(("id", v["reason"], v["reject"], e.get("query_case", "")))
    for e in ood_samples:
        q = e.get("question") or e.get("query")
        r = classifier.classify(q)
        v = reject_score_for(r, q)
        y_true.append(1)
        y_score.append(v["score"])
        meta.append(("ood", v["reason"], v["reject"], e.get("kind", "")))

    n_ood = sum(y_true)
    n_id = len(y_true) - n_ood
    auroc = _auroc(y_true, y_score)
    auprc = _auprc(y_true, y_score)
    fpr95 = _fpr_at_tpr(y_true, y_score, 0.95)

    # Straight rejection accuracy at the rejector's own verdict.
    ood_rejected = sum(1 for t, m in zip(y_true, meta) if t == 1 and m[2])
    id_not_rejected = sum(1 for t, m in zip(y_true, meta) if t == 0 and not m[2])

    reasons = {}
    for t, m in zip(y_true, meta):
        if not m[2]:
            continue  # only actual reject verdicts (skips 'unresolved' pass-through)
        reasons.setdefault(m[1], {"n": 0, "ood": 0})
        reasons[m[1]]["n"] += 1
        reasons[m[1]]["ood"] += (t == 1)

    if verbose:
        for t, m, s in zip(y_true, meta, y_score):
            print(f"  [{'OOD' if t else 'id '}] score={s:.2f} reject={m[2]} "
                  f"reason={m[1]} | {m[3]}")

    return {
        "n_ood": n_ood, "n_id": n_id,
        "auroc": round(auroc, 4), "auprc": round(auprc, 4),
        "fpr_at_95tpr": round(fpr95, 4),
        "ood_rejected_n": ood_rejected, "ood_frac": ood_rejected / max(n_ood, 1),
        "id_not_rejected_n": id_not_rejected, "id_frac": id_not_rejected / max(n_id, 1),
        "reject_reasons": {k: v for k, v in reasons.items()},
    }


def run_intent_evaluation(samples, pre_gate_samples=None, adversarial_samples=None,
                          paraphrase_samples=None, verbose: bool = False) -> dict:
    classifier = IntentClassifier()

    entries, n_clarify_checked, n_clarify_type_ok = _evaluate_samples(
        classifier, samples, "eval", verbose=verbose)

    # Paraphrase (Layer 2b coverage) — purity guard first: none may contain a
    # literal signal, otherwise the sample is not exercising the semantic layer.
    para_entries, p_clarify_checked, p_clarify_type_ok = [], 0, 0
    if paraphrase_samples:
        para_entries, p_clarify_checked, p_clarify_type_ok = _evaluate_samples(
            classifier, paraphrase_samples, "paraphrase", verbose=verbose)
    impure = [e["question"] for e in para_entries if has_literal_signal(e["question"])]
    if impure:
        print(f"  [WARN] {len(impure)} PARAPHRASE_SAMPLES contain a literal signal "
              f"(not exercising Layer 2b): {impure}")

    # Deterministic pre-gate (negatives outside the LLM)
    pg_entries = run_pre_gate_evaluation(classifier, pre_gate_samples, verbose=verbose)
    n_pg     = len(pg_entries)
    n_pg_pass = sum(1 for e in pg_entries if e["verdict"] == "PASS")

    # Adversarial (hardening regression checks)
    adv_entries = run_adversarial_evaluation(classifier, adversarial_samples, verbose=verbose)
    n_adv      = len(adv_entries)
    n_adv_pass = sum(1 for e in adv_entries if e["verdict"] == "PASS")

    # ---- Sprint 1: INTENT_SAMPLES (balreal confusion matrix) -----------
    intent_entries = evaluate_intent_samples(classifier, INTENT_SAMPLES, verbose=verbose)
    n_intent        = len(intent_entries)
    n_intent_pass   = sum(1 for e in intent_entries if e["verdict"] == "PASS")
    n_intent_sem    = sum(1 for e in intent_entries if e["signal_kind"] == "semantic")
    n_intent_sem_ok = sum(1 for e in intent_entries if e["signal_kind"] == "semantic"
                          and e["verdict"] == "PASS")

    # ---- Sprint 1: MINIMAL_PAIRS (frontier boundaries) -----------------
    mp_entries = evaluate_minimal_pairs(classifier, MINIMAL_PAIRS, verbose=verbose)
    n_mpp      = len(mp_entries)
    n_mpp_pass = sum(1 for e in mp_entries if e["verdict"] == "PASS")
    mp_by_frontier = {}
    for frontier in ("2_3", "2_4", "3_4"):
        fe = [e for e in mp_entries if e["frontier"] == frontier]
        mp_by_frontier[frontier] = {
            "n_samples": len(fe),
            "n_pass":    sum(1 for e in fe if e["verdict"] == "PASS"),
            "pass_rate": round(sum(1 for e in fe if e["verdict"] == "PASS") / len(fe) * 100, 1)
                        if fe else None,
        }

    # ---- Sprint 1: OOD_SAMPLES (rejection) -----------------------------
    ood_entries = evaluate_ood_samples(classifier, OOD_SAMPLES, verbose=verbose)
    n_ood      = len(ood_entries)
    n_ood_pass = sum(1 for e in ood_entries if e["verdict"] == "PASS")
    ood_by_tier = {}
    for tier in ("far", "near", "adversarial"):
        te = [e for e in ood_entries if e["ood_tier"] == tier]
        ood_by_tier[tier] = {
            "n_samples": len(te),
            "n_pass":    sum(1 for e in te if e["verdict"] == "PASS"),
            "pass_rate": round(sum(1 for e in te if e["verdict"] == "PASS") / len(te) * 100, 1)
                        if te else None,
        }

    n = len(entries)
    n_intent_ok   = sum(1 for e in entries if e["intent_ok"])
    n_decision_ok = sum(1 for e in entries if e["decision_ok"])
    n_pass        = sum(1 for e in entries if e["verdict"] == "PASS")

    # Confusion matrix: row = expected, column = classified
    intents = sorted({e["expected_intent"] for e in entries})
    confusion = {i: {j: 0 for j in intents + [None]} for i in intents}
    for e in entries:
        confusion[e["expected_intent"]][e["classified_intent"]] += 1

    # START sprint-1 confusion from INTENT_SAMPLES + MINIMAL_PAIRS (real 1-4).
    cm_rows = sorted({e["expected_intent"] for e in (intent_entries + mp_entries)})
    cm_cols = cm_rows + [None]
    confusion_full = {i: {j: 0 for j in cm_cols} for i in cm_rows}
    for e in (intent_entries + mp_entries):
        confusion_full[e["expected_intent"]][e["classified_intent"]] += 1
    # Critical frontiers: off-diagonal counts between pairs.
    critical_confusions = {}
    for frontier, a, b in (("2_3", 2, 3), ("2_4", 2, 4), ("3_4", 3, 4)):
        if a in cm_rows and b in cm_rows:
            critical_confusions[frontier] = {
                "a_into_b": confusion_full[a].get(b, 0),
                "b_into_a": confusion_full[b].get(a, 0),
                "total_off_diag": confusion_full[a].get(b, 0) + confusion_full[b].get(a, 0),
            }

    # ---- Sprint 3: recall@K (semantic candidate retrieval quality) ------
    # For samples that went through the semantic layer (trace has
    # candidate_intents), is the expected intent among the Top-K candidates?
    # This tells us whether a failure happened in RETRIEVAL (misrecall) or in
    # the downstream decision (correct recall but wrong pick).
    recall_entries = []
    for e in intent_entries + mp_entries:
        trace = e.get("semantic_trace") or {}
        cands = trace.get("candidate_intents")
        if not cands:
            continue
        exp = e.get("expected_intent")
        recall_ok = exp in cands
        recall_entries.append({
            "set": e.get("set"), "frontier": e.get("frontier"),
            "question": e.get("question"),
            "expected_intent": exp, "candidate_intents": cands,
            "recall_ok": recall_ok, "verdict": e.get("verdict"),
        })
    n_recall = len(recall_entries)
    n_recall_ok = sum(1 for r in recall_entries if r["recall_ok"])
    recall_at_k = {
        "K": int((recall_entries[0].get("candidate_intents") and len(recall_entries[0]["candidate_intents"])) or 0)
        if recall_entries else None,
        "n_samples":  n_recall,
        "n_recall_ok": n_recall_ok,
        "recall@K": round(n_recall_ok / n_recall * 100, 1) if n_recall else None,
        "entries": recall_entries,
    }

    by_case = {}
    for case in QueryCase:
        case_entries = [e for e in entries if e["query_case"] == case.value]
        n_case  = len(case_entries)
        n_good  = sum(1 for e in case_entries if e["verdict"] == "PASS")
        by_case[case.value] = {
            "n_samples": n_case,
            "n_pass":    n_good,
            "pass_rate": round(n_good / n_case * 100, 1) if n_case else None,
        }

    # ---- Layer 2b: paraphrase coverage + near-threshold diagnostics ---------
    n_para   = len(para_entries)
    n_para_pass = sum(1 for e in para_entries if e["verdict"] == "PASS")

    sem_pos = [e for e in para_entries if e.get("assert_source") == "semantic"]
    sem_neg = [e for e in para_entries if e.get("assert_source") == "not_semantic"]
    n_sem_covered = sum(1 for e in sem_pos if e.get("decision_source") == "semantic")
    n_false_fire  = sum(1 for e in sem_neg if e.get("decision_source") == "semantic")
    n_para_llm    = sum(1 for e in para_entries if e.get("decision_source") == "llm")

    near_threshold = []
    for e in entries + para_entries + pg_entries + adv_entries + intent_entries + mp_entries + ood_entries:
        trace = e.get("semantic_trace")
        if e["verdict"] == "PASS" and e.get("decision_source") != "llm":
            continue
        if not trace:
            continue
        score, margin, th = trace["top_score"], trace["margin"], trace["threshold"]
        below_th = th - 0.05 <= score < th
        below_margin = margin < EMBED_MARGIN + 0.02
        if below_th or below_margin:
            near_threshold.append({
                "set": e["set"], "question": e["question"],
                "cluster": e.get("cluster"),
                "winner": trace["winner"], "top_score": score,
                "margin": margin, "threshold": th,
                "decision_source": e.get("decision_source"),
            })

    all_ds = [e["decision_source"] for e in entries + para_entries + pg_entries + adv_entries + intent_entries + mp_entries + ood_entries]
    ds_counts = {}
    for ds in all_ds:
        ds_counts[ds] = ds_counts.get(ds, 0) + 1
    n_all = len(all_ds)
    n_deterministic = sum(ds_counts.get(k, 0) for k in ("gate", "literal", "semantic", "disabled"))

    para_agg = None
    if n_para:
        para_agg = {
            "n_samples": n_para,
            "pass_rate": round(n_para_pass / n_para * 100, 1) if n_para else None,
            "semantic_asserted": len(sem_pos),
            "semantic_resolved": n_sem_covered,
            "semantic_coverage": (
                round(n_sem_covered / len(sem_pos) * 100, 1) if sem_pos else None
            ),
            "semantic_false_fires": n_false_fire,
            "resolved_by_llm": n_para_llm,
            "clarify_subtype_accuracy": (
                round(p_clarify_type_ok / p_clarify_checked * 100, 1)
                if p_clarify_checked else None
            ),
            "impure_literal": len(impure),
        }

    aggregated = {
        "n_samples":           n,
        "intent_accuracy":     round(n_intent_ok / n * 100, 1) if n else 0.0,
        "decision_accuracy":   round(n_decision_ok / n * 100, 1) if n else 0.0,
        "overall_pass_rate":   round(n_pass / n * 100, 1) if n else 0.0,
        "clarify_subtype_accuracy": (
            round(n_clarify_type_ok / n_clarify_checked * 100, 1) if n_clarify_checked else None
        ),
        "pre_gate": {
            "n_samples": n_pg,
            "pass_rate": round(n_pg_pass / n_pg * 100, 1) if n_pg else None,
        },
        "adversarial": {
            "n_samples": n_adv,
            "pass_rate": round(n_adv_pass / n_adv * 100, 1) if n_adv else None,
        },
        "paraphrase":         para_agg,
        "decision_source_counts": ds_counts,
        "deterministic_coverage": (
            round(n_deterministic / n_all * 100, 1) if n_all else None
        ),
        "near_threshold":     near_threshold,
        "by_case":            by_case,
        "confusion":          confusion,
        # ---- Sprint 1: real intent + minimal pairs + OOD aggregation ---
        "intent_samples": {
            "n_samples":  n_intent,
            "n_pass":     n_intent_pass,
            "pass_rate":  round(n_intent_pass / n_intent * 100, 1) if n_intent else None,
            "semantic_resolved": n_intent_sem_ok,
            "semantic_total":    n_intent_sem,
            "crag_gate_metric": _crag_gate_metric(intent_entries),
        },
        "minimal_pairs": {
            "n_samples":  n_mpp,
            "n_pass":     n_mpp_pass,
            "pass_rate":  round(n_mpp_pass / n_mpp * 100, 1) if n_mpp else None,
            "by_frontier": mp_by_frontier,
        },
        "ood": {
            "n_samples":  n_ood,
            "n_pass":     n_ood_pass,
            "pass_rate":  round(n_ood_pass / n_ood * 100, 1) if n_ood else None,
            "by_tier":    ood_by_tier,
        },
        "confusion_full":     confusion_full,
        "critical_confusions": critical_confusions,
        "recall_at_k":        recall_at_k,
    }

    run_id = time.strftime("%Y%m%d-%H%M%S")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = REPORTS_DIR / f"{run_id}.json"
    run_path.write_text(
        json.dumps({"run_id": run_id, "entries": entries,
                    "paraphrase_entries": para_entries,
                    "pre_gate_entries": pg_entries,
                    "adversarial_entries": adv_entries,
                    "intent_entries": intent_entries,
                    "minimal_pair_entries": mp_entries,
                    "ood_entries": ood_entries,
                    "aggregated": aggregated},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Sprint 1: standalone confusion matrix export (intents 1-4 + minimal pairs)
    confusion_path = REPORTS_DIR / f"{run_id}_confusion.json"
    confusion_data = {
        "run_id": run_id,
        "confusion_full": confusion_full,
        "critical_confusions": critical_confusions,
        "intent_entries": [
            {k: e[k] for k in ("question", "expected_intent", "classified_intent",
                               "decision_source", "verdict", "signal_kind", "regression")}
            for e in intent_entries
        ],
        "minimal_pair_entries": [
            {k: e[k] for k in ("group", "frontier", "expected_intent",
                               "classified_intent", "decision_source", "verdict")}
            for e in mp_entries
        ],
        "ood_entries": [
            {k: e[k] for k in ("question", "ood_tier", "committed",
                               "classified_intent", "classified_decision", "verdict")}
            for e in ood_entries
        ],
    }
    confusion_path.write_text(
        json.dumps(confusion_data, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("  SUMMARY — Intent Classifier (isolated from the pipeline)")
    print("=" * 60)
    if n:
        print(f"  Samples:               {n}")
        print(f"  Intent accuracy:       {aggregated['intent_accuracy']}%  "
              f"({n_intent_ok}/{n})")
        print(f"  Decision accuracy:     {aggregated['decision_accuracy']}%  "
              f"({n_decision_ok}/{n})")
        print(f"  Overall pass rate:     {aggregated['overall_pass_rate']}%")
        if aggregated["clarify_subtype_accuracy"] is not None:
            print(f"  Clarify subtype acc:   {aggregated['clarify_subtype_accuracy']}%  "
                  f"({n_clarify_type_ok}/{n_clarify_checked})")
    else:
        print("  EVAL battery:          0 samples (dataset v2 is Layer 2b focused)")
    if n_pg:
        print(f"  Pre-gate (negatives):  {n_pg_pass}/{n_pg}  "
              f"({aggregated['pre_gate']['pass_rate']}%)")
    if n_adv:
        print(f"  Adversarial:           {n_adv_pass}/{n_adv}  "
              f"({aggregated['adversarial']['pass_rate']}%)")
    if para_agg is not None:
        print(f"  Paraphrase (Layer 2b): {n_para_pass}/{n_para}  "
              f"({para_agg['pass_rate']}%)  "
              f"semantic={para_agg['semantic_coverage']}% "
              f"({para_agg['semantic_resolved']}/{para_agg['semantic_asserted']} pos)  "
              f"false_fires={para_agg['semantic_false_fires']}  "
              f"llm_fallback={para_agg['resolved_by_llm']}")
    print(f"  Deterministic:         {aggregated['deterministic_coverage']}%  "
          f"decision_source={ds_counts}")
    if near_threshold:
        print(f"  Near-threshold (seeds): {len(near_threshold)}")
        for nt in near_threshold[:20]:
            print(f"    [{nt['set']}] '{nt['question'][:48]}' winner={nt['winner']} "
                  f"sim={nt['top_score']} margin={nt['margin']} th={nt['threshold']} "
                  f"-> {nt['decision_source']}")
    if n:
        for case, stats in by_case.items():
            label = QueryCase.from_id(case).label
            rate  = f"{stats['pass_rate']}%" if stats['pass_rate'] is not None else "N/A (0 samples)"
            print(f"  {label}: {stats['n_pass']}/{stats['n_samples']}  ({rate})")
        print("-" * 60)
        print("  Confusion matrix (expected -> classified):")
        print(f"  {'':8}" + "".join(f"{str(j):>8}" for j in intents + [None]))
        for i in intents:
            row = "".join(f"{confusion[i][j]:>8}" for j in intents + [None])
            print(f"  expected {i}:{row}")

    # ---- Sprint 1 beats: real intent + minimal pairs + OOD ---------------
    print("-" * 60)
    print("  SPRINT 1 — INTENT SAMPLES (real 1-4 confusion):")
    if n_intent:
        print(f"    Intent samples:        {n_intent_pass}/{n_intent}  "
              f"({aggregated['intent_samples']['pass_rate']}%)  "
              f"semantic-path={n_intent_sem_ok}/{n_intent_sem}")
        print("    Confusion (expected -> classified):")
        print(f"    {'':10}" + "".join(f"{str(j):>7}" for j in cm_cols))
        for i in cm_rows:
            row = "".join(f"{confusion_full[i][j]:>7}" for j in cm_cols)
            print(f"    expected {i}:{row}")
        print("    Critical frontiers (off-diagonal):")
        for frontier, fdata in critical_confusions.items():
            print(f"      {frontier}: a->b={fdata['a_into_b']}  b->a={fdata['b_into_a']}  "
                  f"total={fdata['total_off_diag']}")
    if n_mpp:
        print(f"    Minimal pairs:         {n_mpp_pass}/{n_mpp}  "
              f"({aggregated['minimal_pairs']['pass_rate']}%)")
        for frontier, fdata in mp_by_frontier.items():
            r = f"{fdata['pass_rate']}%" if fdata['pass_rate'] is not None else "N/A"
            print(f"      frontier {frontier}: {fdata['n_pass']}/{fdata['n_samples']} ({r})")
    if n_ood:
        print(f"    OOD rejection:         {n_ood_pass}/{n_ood}  "
              f"({aggregated['ood']['pass_rate']}%)")
        for tier, tdata in ood_by_tier.items():
            r = f"{tdata['pass_rate']}%" if tdata['pass_rate'] is not None else "N/A"
            print(f"      {tier}: {tdata['n_pass']}/{tdata['n_samples']} ({r})")
    if recall_at_k.get("recall@K") is not None:
        print(f"    Recall@K (semantic retrieval): {recall_at_k['n_recall_ok']}/{recall_at_k['n_samples']}  "
              f"({recall_at_k['recall@K']}% at K={recall_at_k['K']})")
    print("=" * 60 + "\n")
    print(f"  Run saved in: {run_path}")
    print(f"  Sprint-1 confusion saved in: {confusion_path}\n")

    # Export calibration JSON (--no-llm mode): per-sample 9-score vectors
    # for isotonic calibration. Includes ALL paraphrase entries where
    # semantic_trace is available (resolved, unresolved, or near-threshold).
    calibration_entries = []
    for e in para_entries:
        trace = e.get("semantic_trace")
        if not trace or not trace.get("scores"):
            continue
        calibration_entries.append({
            "question":          e["question"],
            "cluster":           e.get("cluster"),
            "expected_intent":   e["expected_intent"],
            "assert_source":     e.get("assert_source"),
            "cluster_scores":    trace["scores"],
            "winner":            trace["winner"],
            "top_score":         trace["top_score"],
            "margin":            trace["margin"],
            "threshold":         trace["threshold"],
            "decision_source":   e.get("decision_source"),
            "verdict":           e["verdict"],
        })

    if calibration_entries:
        scores_path = REPORTS_DIR / f"{run_id}_scores.json"
        scores_path.write_text(
            json.dumps({"run_id": run_id, "n_samples": len(calibration_entries),
                        "entries": calibration_entries}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Calibration scores exported: {scores_path} ({len(calibration_entries)} entries)\n")

    return {"entries": entries, "aggregated": aggregated, "run_id": run_id, "run_path": str(run_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluates the isolated intent classifier")
    parser.add_argument("--samples", type=int, default=None,
                        help="Limit PARAPHRASE_SAMPLES (Layer 2b set) to N samples "
                             "(default: all)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM calls (INTENT_NO_LLM=1). Unresolved samples "
                             "are logged but excluded from pass_rate. Exports a "
                             "_scores.json for isotonic calibration.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--reranker", action="store_true",
                        help="Sprint 4: run offline cross-encoder reranker "
                             "diagnostic over intent + minimal-pair fronters")
    parser.add_argument("--reject", action="store_true",
                        help="Sprint 5: measure rejector OOD discriminative power "
                             "(AUROC/AUPRC/FPR@95TPR) on OOD vs in-dist samples")
    args = parser.parse_args()

    if args.no_llm:
        os.environ["INTENT_NO_LLM"] = "1"
        print("  [MODE] INTENT_NO_LLM=1 — LLM calls skipped, unresolved samples logged.\n")

    samples = INTENT_SAMPLES
    para = PARAPHRASE_SAMPLES[: args.samples] if args.samples else PARAPHRASE_SAMPLES
    run_intent_evaluation(samples, pre_gate_samples=PRE_GATE_SAMPLES,
                          adversarial_samples=ADVERSARIAL_SAMPLES,
                          paraphrase_samples=para, verbose=args.verbose)

    if args.reranker:
        from src.reranker import get_reranker
        from evaluation.intent_dataset import MINIMAL_PAIRS, INTENT_SAMPLES as _RANK_SAMPLES
        from src.intent_spec import INTENT_SPECS
        print("\n--- SPRINT 4: RERANKER DIAGNOSTIC (offline, no decision rewired) ---")
        rr = get_reranker()
        rank_samples = list(_RANK_SAMPLES) + list(MINIMAL_PAIRS)
        res = evaluate_reranker(rank_samples, reranker=rr, intent_specs=INTENT_SPECS,
                                verbose=args.verbose)
        print(f"  reranker source      : {res['reranker_source']}")
        print(f"  samples with cands   : {res['n_samples']}")
        print(f"  embedding top-1 ok   : {res['embedding_top1_ok']} "
              f"({res['embedding_top1_ok']/max(res['n_samples'],1):.1%})")
        print(f"  reranker top-1 ok    : {res['reranker_top1_ok']} "
              f"({res['reranker_top1_ok']/max(res['n_samples'],1):.1%})")
        for f, st in res["by_frontier"].items():
            if st["n"]:
                print(f"    frontier {f}: emb={st['emb_ok']}/{st['n']}  "
                      f"rr={st['rr_ok']}/{st['n']}")

    if args.reject:
        from evaluation.intent_dataset import INTENT_SAMPLES, MINIMAL_PAIRS, OOD_SAMPLES, PARAPHRASE_SAMPLES
        print("\n--- SPRINT 5: REJECTOR OOD DISCRIMINATION (offline) ---")
        in_dist = list(INTENT_SAMPLES) + list(MINIMAL_PAIRS)
        if not in_dist:
            in_dist = [{"question": p["question"]} for p in PARAPHRASE_SAMPLES[:20]]
        rj = evaluate_rejector(OOD_SAMPLES, in_dist, verbose=args.verbose)
        print(f"  n_ood={rj['n_ood']}  n_id={rj['n_id']}")
        print(f"  AUROC       : {rj['auroc']}")
        print(f"  AUPRC       : {rj['auprc']}")
        print(f"  FPR@95TPR   : {rj['fpr_at_95tpr']}")
        print(f"  OOD rejected      : {rj['ood_rejected_n']}/{rj['n_ood']} ({rj['ood_frac']:.0%})")
        print(f"  ID not rejected   : {rj['id_not_rejected_n']}/{rj['n_id']} ({rj['id_frac']:.0%})")
        print("  reject reasons (n / ood):")
        for reason, st in rj["reject_reasons"].items():
            print(f"    {reason}: {st['n']} total, {st['ood']} ood")
