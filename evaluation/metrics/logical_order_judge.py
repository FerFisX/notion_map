"""Hybrid semantic and deterministic evaluator for roadmap logical order.

The LLM first proposes only dependencies that appear inverted in the current
sequence, with targeted recovery for omitted steps. A second evidence-backed
semantic pass audits those candidates. Deterministic code then builds a stable
valid order and measures the minimum repair scope. The LLM assigns the
human-facing score inside that computed band.
"""

from __future__ import annotations

import heapq
import json
from typing import Any

from evaluation.metrics.prompt_payloads import compact_contexts, prompt_json
from evaluation.structured_output import StructuredOutputError, invoke_json_with_retry
from src.llm_provider import get_judge_llm


_CRITERIA = (
    "prerequisite_order",
    "causal_dependency_order",
    "learning_progression",
    "validation_timing",
)
_DEPENDENCY_TYPES = {
    "prerequisite",
    "causal_output",
    "learning_progression",
    "validation_timing",
}


_DEPENDENCY_PROMPT = """\
You are a semantic dependency analyst for technical learning and execution
roadmaps. Return ONLY valid JSON, with no markdown or additional text.

USER QUESTION
{question}

CATEGORY
{category}

ORDER-RELEVANT REFERENCE CONTEXT
Use this evidence to resolve technical prerequisites and lifecycle order. Do
not score factual support or missing content here.
{contexts}

CURRENT ROADMAP ORDER
Lower positions execute before higher positions.
{ordered_steps}

TASK
Audit every current step exactly once. For each step, ask whether it requires a
concrete artifact, state, input, configuration, or knowledge that is produced
by a LATER step. Include that later dependency only when the current step
cannot validly produce its stated result without it. Return an empty
later_dependencies list for a step with no strict inversion.

Fields have fixed positional meanings: step_id is the EARLIER step being
audited; required_predecessor_step_id is the LATER step that must move before
it.

Do not report dependencies that are already satisfied. Do not report stylistic
preferences, missing content, vague instructions, unsupported claims, repeated
work, framing, closure, granularity, or schema defects.

Direction guardrails:
- defining a goal or contract normally precedes receiving runtime data;
- creation or implementation precedes its use, publication, delivery, or test;
- validation observes an existing result; the result does not require its
  later validation in order to be created;
- quality assurance before assembly is a best practice, not a strict logical
  dependency: building a page or workflow does not require prior validation
  unless that step explicitly consumes an approved status;
- error handling supports an operation but does not normally need to precede
  implementation of the core operation;
- preparation, credentials, data models, and base artifacts precede work that
  consumes them;
- identifying or defining an endpoint, destination, requirement, or target is
  planning work and does not require runtime credentials; credentials precede
  the later operation that actually accesses the protected service;
- validation precedes production activation when activation claims a tested
  result.

If a description says it publishes, validates, activates, or consumes something
before that thing exists, treat that wording as evidence of an inversion. Do
not preserve the flawed timing described by the roadmap.

EXPLICIT PREMATURITY CHECK
Pay special attention when a step says it acts "before" another artifact
exists, performs work "immediately" without its setup, or deliberately defers
required preparation until "later" or "last". Locate the later roadmap step
that creates the named requirement and report it in the audited step's
later_dependencies. Do not treat the premature wording as a valid sequence.
For every validation, test, review, publication, or activation step, identify
the roadmap step that creates the exact result being checked or released. If
that creator appears later, report the direct dependency even when another
foundation dependency has already been reported.

Allowed dependency types:
prerequisite | causal_output | learning_progression | validation_timing

Rules:
- Use exact step IDs from CURRENT ROADMAP ORDER.
- Return exactly one audit entry for every step ID, in current order.
- required_predecessor_step_id must have a numerically higher position than
  the audited step_id in every returned dependency.
- Return every independently required direct later dependency, while omitting
  only duplicate or purely transitive edges. Do not let one foundation edge
  hide a separate creation-before-use or creation-before-validation edge.
- Return no more than {max_edges} direct candidates.
- Keep explanations under 30 words and name the concrete missing prerequisite.
- The final step must normally have an empty later_dependencies list because
  no later roadmap step exists.

Expected JSON schema:
{{
  "inverted_dependency_audit": [
    {{
      "step_id": "<exact audited step ID>",
      "later_dependencies": [
        {{
          "required_predecessor_step_id": "<later exact step ID>",
          "dependency_type": "prerequisite|causal_output|learning_progression|validation_timing",
          "explanation": "<what the audited step lacks from the later step>"
        }}
      ]
    }}
  ]
}}
"""


_MISSING_DEPENDENCY_PROMPT = """\
You are completing a partial semantic dependency audit. Return ONLY valid JSON,
with no markdown or additional text.

USER QUESTION
{question}

ORDER-RELEVANT REFERENCE CONTEXT
{contexts}

COMPLETE ROADMAP ORDER
{ordered_steps}

STEPS STILL MISSING FROM THE AUDIT
{missing_step_ids}

Audit only the missing step IDs above. For each missing step, report later
roadmap steps that must move before it because they create a concrete artifact,
state, input, configuration, or knowledge that it strictly requires. Use an
empty later_dependencies list when no strict inversion exists.

Return exactly one entry for each missing ID and no entries for other steps:
{{
  "inverted_dependency_audit": [
    {{
      "step_id": "<exact missing step ID>",
      "later_dependencies": [
        {{
          "required_predecessor_step_id": "<later exact step ID>",
          "dependency_type": "prerequisite|causal_output|learning_progression|validation_timing",
          "explanation": "<concrete missing prerequisite>"
        }}
      ]
    }}
  ]
}}
"""


_EDGE_AUDIT_PROMPT = """\
You audit proposed ordering repairs for a technical roadmap. Return ONLY valid
JSON, with no markdown or additional text.

ROADMAP STEPS IN CURRENT ORDER
{steps}

ORDER-RELEVANT REFERENCE CONTEXT
Use this evidence to resolve disputed prerequisites. Do not score Grounding or
Completeness.
{contexts}

PROPOSED MOVEMENTS
{candidate_edges}

Each proposal has:
- premature_step_id: an EARLIER step that may be acting too soon;
- required_later_step_id: a LATER step that may need to move before it.

For every proposal return one categorical verdict:
- CONFIRMED_STRICT_INVERSION: the earlier step cannot validly produce its
  stated result until the later step has occurred;
- KEEP_CURRENT_ORDER: the proposal is false, reversed, merely preferable, or
  the two steps are independent.

Judge ordering only. An incomplete or incorrect required step can still need
to occur first; its quality belongs to other metrics. Creation precedes use or
validation. Setup precedes work that consumes it. Validation precedes
production activation only when activation claims a tested or approved state.
Do not let a roadmap erase an explicit lifecycle dependency by claiming that a
consumer, cloud service, visual, or automatic feature will create, repair, or
validate a prerequisite that the roadmap itself schedules later. When the
roadmap includes a later step for that named prerequisite or output, evaluate
the declared creation-before-use or creation-before-validation relationship.
Defining a goal, desired outcome, scope, or contract normally precedes runtime
data, credentials, and implementation; reject proposals that move those
operational steps before framing. Credentials are prerequisites only for steps
that actually access the protected service, not for planning or a local data
transformation unless the roadmap explicitly performs that access there.
Identifying or defining an endpoint, destination, requirement, or target is
planning work: authentication must precede the actual protected request, not
the prior identification of what will be accessed.
In a learning roadmap, applying, practicing, building with, or optimizing a
named concept strictly depends on an included later step that teaches or
introduces that concept when the application step explicitly says it requires
or depends on that knowledge. Verbatim descriptions declaring that dependency
are direct roadmap evidence even when the two step labels use different words.

EVIDENCE REQUIREMENT
Return CONFIRMED_STRICT_INVERSION only with verifiable evidence:
- roadmap evidence: copy one short verbatim quote from premature_step_id and
  one from required_later_step_id. Together they must show use before creation,
  validation before output, or another strict prerequisite relationship;
- context evidence: copy one short verbatim quote from the reference context
  that explicitly states the required order.
Do not paraphrase quotes. Merely mentioning the same topic is insufficient. If
no direct evidence exists, preserve the current relative order.

Example: if step_2 publishes before step_6 prepares the data model, return
CONFIRMED_STRICT_INVERSION because step_6 must move before step_2.

Return exactly one decision for every edge_index:
{{
  "edge_audit": [
    {{
      "edge_index": <zero-based edge_index>,
      "verdict": "CONFIRMED_STRICT_INVERSION|KEEP_CURRENT_ORDER",
      "evidence_source": "roadmap|context|none",
      "premature_step_quote": "<verbatim quote or empty string>",
      "required_step_quote": "<verbatim quote or empty string>",
      "context_quote": "<verbatim context quote or empty string>",
      "reason": "<brief case-specific reason>"
    }}
  ]
}}
"""


_FINAL_SCORING_PROMPT = """\
You are an expert evaluator of roadmap Logical Order. Return ONLY valid JSON,
with no markdown or additional text.

USER QUESTION
{question}

CATEGORY
{category}

ROADMAP STEPS
{steps}

VALIDATED DETERMINISTIC ORDER EVIDENCE
{analysis}

The accepted semantic dependencies and deterministic repair scope above are
final. Do not add, remove, or reinterpret them. Score only Logical Order and do
not penalize completeness, actionability, grounding, overlap, structure, or
schema quality.

Choose a precise decimal final score inside the mandatory {score_band} band.
Use the practical impact of the validated inversions to choose a value within
that band. Provide metric-specific criterion scores and concise rationales.

Expected JSON schema:
{{
  "logical_order_assessment": {{
    "criteria": {{
      "prerequisite_order": <0-10>,
      "causal_dependency_order": <0-10>,
      "learning_progression": <0-10>,
      "validation_timing": <0-10>
    }},
    "criteria_rationales": {{
      "prerequisite_order": "<case-specific rationale>",
      "causal_dependency_order": "<case-specific rationale>",
      "learning_progression": "<case-specific rationale>",
      "validation_timing": "<case-specific rationale>"
    }},
    "score": <number in {score_band}>,
    "reason": "<brief explanation>",
    "recommendations": ["<supported reordering action>"]
  }}
}}
"""


def _strip_markdown_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = _strip_markdown_json(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Logical Order response must be a JSON object")
    return parsed


def _normalize_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Score must be numeric") from exc
    if not 0.0 <= score <= 10.0:
        raise ValueError("Score must be between 0 and 10")
    return round(score, 2)


def _verdict(score: float) -> str:
    if score >= 8.0:
        return "PASS"
    if score >= 5.0:
        return "NEEDS_REVIEW"
    return "FAIL"


def _score_band(score: float) -> str:
    if score >= 8.0:
        return "8.0-10.0"
    if score >= 5.0:
        return "5.0-7.9"
    return "0.0-4.9"


def _band_for_scope(repair_scope: str) -> tuple[float, float, str]:
    if repair_scope == "no_repair":
        return 8.0, 10.0, "8.0-10.0"
    if repair_scope == "local_repair":
        return 5.0, 7.9, "5.0-7.9"
    return 0.0, 4.9, "0.0-4.9"


def _roadmap_steps(roadmap: dict[str, Any]) -> list[dict[str, Any]]:
    raw_steps = roadmap.get("steps", [])
    if not isinstance(raw_steps, list):
        raise ValueError("Roadmap steps must be a list")
    steps = [step for step in raw_steps if isinstance(step, dict)]
    step_ids = [str(step.get("id", "")).strip() for step in steps]
    if not steps or any(not step_id for step_id in step_ids):
        raise ValueError("Every roadmap step must have a non-empty ID")
    if len(set(step_ids)) != len(step_ids):
        raise ValueError("Roadmap step IDs must be unique for Logical Order")
    return steps


def _stable_topological_order(
    step_ids: list[str],
    edges: list[dict[str, str]],
) -> list[str]:
    positions = {step_id: index for index, step_id in enumerate(step_ids)}
    successors = {step_id: set() for step_id in step_ids}
    indegree = {step_id: 0 for step_id in step_ids}
    for edge in edges:
        before = edge["before_step_id"]
        after = edge["after_step_id"]
        if after not in successors[before]:
            successors[before].add(after)
            indegree[after] += 1

    ready = [(positions[step_id], step_id) for step_id in step_ids if indegree[step_id] == 0]
    heapq.heapify(ready)
    ordered: list[str] = []
    while ready:
        _, step_id = heapq.heappop(ready)
        ordered.append(step_id)
        for successor in sorted(successors[step_id], key=positions.get):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, (positions[successor], successor))
    if len(ordered) != len(step_ids):
        raise ValueError("Extracted semantic dependency graph contains a cycle")
    return ordered


def _minimum_relocations(current: list[str], target: list[str]) -> int:
    """Return minimum remove-and-insert moves via longest common subsequence."""
    width = len(target) + 1
    previous = [0] * width
    for current_id in current:
        row = [0] * width
        for index, target_id in enumerate(target, 1):
            if current_id == target_id:
                row[index] = previous[index - 1] + 1
            else:
                row[index] = max(previous[index], row[index - 1])
        previous = row
    return len(current) - previous[-1]


def _parse_dependency_graph(
    raw: str,
    step_ids: list[str],
    max_edges: int,
    required_step_ids: list[str] | None = None,
) -> dict[str, Any]:
    parsed = _parse_json_object(raw)
    audits = parsed.get("inverted_dependency_audit")
    if not isinstance(audits, list):
        raise ValueError("inverted_dependency_audit must be a list")

    known_ids = set(step_ids)
    required_ids = list(required_step_ids or step_ids)
    required_set = set(required_ids)
    if not required_set <= known_ids:
        raise ValueError("Required dependency-audit IDs must belong to the roadmap")
    positions = {step_id: index for index, step_id in enumerate(step_ids)}
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    audited_ids: list[str] = []
    for audit_index, audit in enumerate(audits, 1):
        if not isinstance(audit, dict):
            raise ValueError(f"Dependency audit {audit_index} must be an object")
        dependent_id = str(audit.get("step_id", "")).strip()
        if dependent_id not in known_ids:
            raise ValueError(f"Dependency audit {audit_index} contains an unknown step ID")
        if dependent_id not in required_set:
            raise ValueError(
                f"Dependency audit contains an unrequested step ID: {dependent_id}"
            )
        if dependent_id not in audited_ids:
            audited_ids.append(dependent_id)
        dependencies = audit.get("later_dependencies")
        if not isinstance(dependencies, list):
            raise ValueError(f"Dependency audit for {dependent_id} requires later_dependencies")
        for dependency_index, item in enumerate(dependencies, 1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Dependency {dependency_index} for {dependent_id} must be an object"
                )
            predecessor_id = str(
                item.get("required_predecessor_step_id", "")
            ).strip()
            dependency_type = str(item.get("dependency_type", "")).strip().lower()
            explanation = str(item.get("explanation", "")).strip()
            if predecessor_id not in known_ids:
                raise ValueError(
                    f"Dependency for {dependent_id} contains an unknown predecessor ID"
                )
            if dependency_type not in _DEPENDENCY_TYPES:
                raise ValueError(
                    f"Dependency for {dependent_id} has an invalid dependency type"
                )
            if not explanation:
                raise ValueError(f"Dependency for {dependent_id} requires an explanation")
            if positions[predecessor_id] <= positions[dependent_id]:
                continue
            key = (predecessor_id, dependent_id)
            if key in seen:
                continue
            seen.add(key)
            edges.append({
                "before_step_id": predecessor_id,
                "after_step_id": dependent_id,
                "dependency_type": dependency_type,
                "explanation": explanation,
            })

    missing = [step_id for step_id in required_ids if step_id not in audited_ids]
    return {
        "dependency_graph": {"edges": edges},
        "audited_step_ids": audited_ids,
        "missing_step_ids": missing,
        "edge_limit_exceeded": len(edges) > max_edges,
    }


def _merge_dependency_edges(*groups: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for edge in group:
            key = (edge["before_step_id"], edge["after_step_id"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(edge)
    return merged


def _normalized_evidence(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _quote_in_text(quote: str, text: Any, minimum_words: int = 3) -> bool:
    normalized_quote = _normalized_evidence(quote)
    if len(normalized_quote.split()) < minimum_words:
        return False
    haystack = _normalized_evidence(text)
    return normalized_quote in haystack


def _movement_evidence_is_supported(
    decision: dict[str, Any],
    candidate: dict[str, str],
    roadmap: dict[str, Any],
    contexts: list[str],
) -> bool:
    source = str(decision.get("evidence_source", "")).strip().lower()
    if source == "context":
        return _quote_in_text(
            str(decision.get("context_quote", "")),
            json.dumps(contexts, ensure_ascii=False),
            minimum_words=4,
        )
    if source != "roadmap":
        return False
    by_id = {
        str(step.get("id", "")).strip(): json.dumps(step, ensure_ascii=False)
        for step in roadmap.get("steps", [])
        if isinstance(step, dict)
    }
    premature_id = candidate["after_step_id"]
    required_id = candidate["before_step_id"]
    premature_quote = str(decision.get("premature_step_quote", ""))
    required_quote = str(decision.get("required_step_quote", ""))
    quotes_are_verbatim = _quote_in_text(
        premature_quote,
        by_id.get(premature_id, ""),
    ) and _quote_in_text(
        required_quote,
        by_id.get(required_id, ""),
    )
    return quotes_are_verbatim


def _analyze_order(step_ids: list[str], edges: list[dict[str, str]]) -> dict[str, Any]:
    positions = {step_id: index for index, step_id in enumerate(step_ids)}
    suggested_order = _stable_topological_order(step_ids, edges)
    inverted_edges = [
        edge for edge in edges
        if positions[edge["before_step_id"]] > positions[edge["after_step_id"]]
    ]
    movement_count = (
        _minimum_relocations(step_ids, suggested_order) if inverted_edges else 0
    )
    if movement_count == 0:
        repair_scope = "no_repair"
    elif movement_count == 1:
        repair_scope = "local_repair"
    else:
        repair_scope = "structural_reordering"

    severity = "high" if repair_scope == "structural_reordering" else "medium"
    violations = [
        {
            "dependent_step_id": edge["after_step_id"],
            "required_predecessor_step_id": edge["before_step_id"],
            "dependency_type": edge["dependency_type"],
            "severity": severity,
            "explanation": edge["explanation"],
            "suggested_fix": (
                f"Move {edge['before_step_id']} before {edge['after_step_id']}."
            ),
        }
        for edge in inverted_edges
    ]
    return {
        "current_order": step_ids,
        "suggested_order": suggested_order if inverted_edges else list(step_ids),
        "dependency_edge_count": len(edges),
        "dependency_violation_count": len(violations),
        "dependency_violations": violations,
        "movement_count": movement_count,
        "repair_scope": repair_scope,
    }


def _parse_edge_audit(
    raw: str,
    step_ids: list[str],
    candidate_edges: list[dict[str, str]],
    roadmap: dict[str, Any],
    contexts: list[str],
) -> dict[str, Any]:
    parsed = _parse_json_object(raw)
    edge_audit = parsed.get("edge_audit")
    if not isinstance(edge_audit, list):
        raise ValueError("edge_audit must be a list")
    accepted_indices: list[int] = []
    audited_indices: list[int] = []
    for decision in edge_audit:
        if not isinstance(decision, dict):
            raise ValueError("Every edge_audit decision must be an object")
        value = decision.get("edge_index")
        if isinstance(value, bool):
            raise ValueError("edge_audit edge_index must be an integer")
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("edge_audit edge_index must be an integer") from exc
        if index < 0 or index >= len(candidate_edges):
            raise ValueError(f"Unknown edge_audit index: {index}")
        if index in audited_indices:
            raise ValueError(f"Duplicate edge_audit index: {index}")
        audited_indices.append(index)
        candidate = candidate_edges[index]
        verdict = str(decision.get("verdict", "")).strip().upper()
        if not str(decision.get("reason", "")).strip():
            raise ValueError(f"edge_audit {index} requires a reason")
        if verdict not in {"CONFIRMED_STRICT_INVERSION", "KEEP_CURRENT_ORDER"}:
            # A malformed decision must not discard valid decisions for every
            # other candidate. Conservatively preserve the current order and
            # expose the rejected value for diagnostics.
            decision["original_verdict"] = decision.get("verdict")
            decision["verdict"] = "KEEP_CURRENT_ORDER"
            decision["decision"] = "keep_current_relative_order"
            decision["evidence_verified"] = False
            decision["reason"] = (
                "Rejected automatically because the auditor did not return a "
                "permitted categorical verdict."
            )
            continue
        if verdict == "CONFIRMED_STRICT_INVERSION":
            if _movement_evidence_is_supported(
                decision, candidate, roadmap, contexts
            ):
                decision["decision"] = "move_required_step_before_premature_step"
                decision["evidence_verified"] = True
                accepted_indices.append(index)
            else:
                decision["original_verdict"] = verdict
                decision["verdict"] = "KEEP_CURRENT_ORDER"
                decision["decision"] = "keep_current_relative_order"
                decision["evidence_verified"] = False
                decision["reason"] = (
                    "Rejected automatically because the proposed movement lacked "
                    "a verifiable roadmap or context quote."
                )
        else:
            decision["decision"] = "keep_current_relative_order"
            decision["evidence_verified"] = None
    if sorted(audited_indices) != list(range(len(candidate_edges))):
        raise ValueError("edge_audit must decide every candidate edge exactly once")
    accepted_edges = [candidate_edges[index] for index in accepted_indices]
    analysis = _analyze_order(step_ids, accepted_edges)
    return {
        "edge_audit": edge_audit,
        "accepted_dependency_graph": accepted_edges,
        "deterministic_analysis": analysis,
    }


def _parse_final_scoring(raw: str, repair_scope: str) -> dict[str, Any]:
    parsed = _parse_json_object(raw)
    assessment = parsed.get("logical_order_assessment")
    if not isinstance(assessment, dict):
        raise ValueError("logical_order_assessment must be an object")
    criteria_raw = assessment.get("criteria")
    rationales_raw = assessment.get("criteria_rationales")
    if not isinstance(criteria_raw, dict) or not isinstance(rationales_raw, dict):
        raise ValueError("criteria and criteria_rationales must be objects")
    criteria = {name: _normalize_score(criteria_raw.get(name)) for name in _CRITERIA}
    rationales = {name: str(rationales_raw.get(name, "")).strip() for name in _CRITERIA}
    missing = [name for name, value in rationales.items() if not value]
    if missing:
        raise ValueError(f"Missing criterion rationales: {', '.join(missing)}")
    score = _normalize_score(assessment.get("score"))
    minimum, maximum, _ = _band_for_scope(repair_scope)
    if not minimum <= score <= maximum:
        raise ValueError(
            f"Score {score} contradicts repair scope {repair_scope}; "
            f"expected {minimum}-{maximum}"
        )
    reason = str(assessment.get("reason", "")).strip()
    if not reason:
        raise ValueError("Logical Order assessment requires a reason")
    recommendations_raw = assessment.get("recommendations", [])
    if not isinstance(recommendations_raw, list):
        raise ValueError("recommendations must be a list")
    return {
        "logical_order_assessment": {
            "criteria": criteria,
            "criteria_rationales": rationales,
            "score": score,
            "reason": reason,
            "recommendations": [
                str(item).strip()
                for item in recommendations_raw
                if str(item).strip()
            ],
        }
    }


def _fallback_result(
    error: Exception,
    *,
    step_ids: list[str] | None = None,
    analysis: dict[str, Any] | None = None,
    dependency_graph: list[dict[str, str]] | None = None,
    judge_execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reason = f"Logical Order evaluation requires manual review: {type(error).__name__}."
    evidence = analysis or {
        "suggested_order": list(step_ids or []),
        "dependency_violations": [],
        "dependency_violation_count": 0,
        "movement_count": 0,
        "repair_scope": "unknown",
    }
    return {
        "logical_order": {
            "score": 5.0,
            "verdict": "NEEDS_REVIEW",
            "score_band": "5.0-7.9",
            "reason": reason,
            "criteria": {name: 5.0 for name in _CRITERIA},
            "criteria_rationales": {name: reason for name in _CRITERIA},
            "has_valid_sequence": False,
            "dependency_graph": dependency_graph or [],
            "dependency_violations": evidence["dependency_violations"],
            "dependency_violation_count": evidence["dependency_violation_count"],
            "suggested_order": evidence["suggested_order"],
            "movement_count": evidence["movement_count"],
            "repair_scope": evidence["repair_scope"],
            "issues": [
                {
                    "type": item["dependency_type"],
                    "severity": item["severity"],
                    "explanation": item["explanation"],
                }
                for item in evidence["dependency_violations"]
            ],
            "recommendations": ["Review logical order manually."],
            "normalization_notes": [str(error)],
            "manual_review_required": True,
            "judge_execution": judge_execution or {},
        }
    }


class LogicalOrderJudge:
    """Evaluate dependency order through a provider-neutral hybrid pipeline."""

    def __init__(self, llm=None):
        self.llm = llm or get_judge_llm(temperature=0.0, max_tokens=2048)

    def evaluate(
        self,
        roadmap: dict[str, Any],
        question: str = "",
        category: str = "",
        contexts: list[str] | None = None,
    ) -> dict[str, Any]:
        traces: dict[str, Any] = {}
        step_ids: list[str] = []
        graph_edges: list[dict[str, str]] = []
        analysis: dict[str, Any] | None = None
        active_stage = "dependency_extraction"
        try:
            steps = _roadmap_steps(roadmap)
            step_ids = [str(step["id"]).strip() for step in steps]
            max_edges = max(1, len(step_ids) * 2)
            ordered_steps = [
                {
                    "position": index,
                    "id": str(step["id"]).strip(),
                    "label": str(step.get("label", "")),
                    "description": str(step.get("description", "")),
                    "key_points": step.get("key_points", []),
                }
                for index, step in enumerate(steps, 1)
            ]
            dependency_prompt = _DEPENDENCY_PROMPT.format(
                question=question or "(not provided)",
                category=category or "(not provided)",
                contexts=prompt_json(compact_contexts(contexts)),
                ordered_steps=prompt_json(ordered_steps),
                max_edges=max_edges,
            )
            graph_result, graph_trace = invoke_json_with_retry(
                self.llm,
                dependency_prompt,
                operation="Logical Order dependency extraction",
                parser=lambda raw: _parse_dependency_graph(
                    raw, step_ids, max_edges
                ),
                repair_instruction=(
                    "Return inverted_dependency_audit with exactly one entry for every "
                    "roadmap step, in current order. Each later dependency must use an "
                    "exact ID positioned after the audited step, an allowed dependency "
                    f"type, and a brief explanation. Use at most {max_edges} direct edges."
                ),
            )
            traces["dependency_extraction"] = graph_trace
            graph_edges = graph_result["dependency_graph"]["edges"]
            missing_step_ids = graph_result["missing_step_ids"]
            if missing_step_ids:
                missing_prompt = _MISSING_DEPENDENCY_PROMPT.format(
                    question=question or "(not provided)",
                    contexts=prompt_json(compact_contexts(contexts)),
                    ordered_steps=prompt_json(ordered_steps),
                    missing_step_ids=json.dumps(missing_step_ids, indent=2),
                )

                def parse_missing(raw: str) -> dict[str, Any]:
                    recovered = _parse_dependency_graph(
                        raw,
                        step_ids,
                        max_edges,
                        required_step_ids=missing_step_ids,
                    )
                    if recovered["missing_step_ids"]:
                        raise ValueError(
                            "Missing-step recovery remains incomplete: "
                            + ", ".join(recovered["missing_step_ids"])
                        )
                    return recovered

                active_stage = "missing_step_recovery"
                recovered_graph, recovery_trace = invoke_json_with_retry(
                    self.llm,
                    missing_prompt,
                    operation="Logical Order missing-step recovery",
                    parser=parse_missing,
                    repair_instruction=(
                        "Return exactly one dependency-audit entry for each requested "
                        f"missing ID: {', '.join(missing_step_ids)}."
                    ),
                )
                traces["missing_step_recovery"] = recovery_trace
                graph_edges = _merge_dependency_edges(
                    graph_edges,
                    recovered_graph["dependency_graph"]["edges"],
                )
            analysis = _analyze_order(step_ids, graph_edges)
            indexed_candidates = [
                {
                    "edge_index": index,
                    "premature_step_id": edge["after_step_id"],
                    "required_later_step_id": edge["before_step_id"],
                    "dependency_type": edge["dependency_type"],
                    "explanation": edge["explanation"],
                }
                for index, edge in enumerate(graph_edges)
            ]

            if indexed_candidates:
                audit_prompt = _EDGE_AUDIT_PROMPT.format(
                    steps=prompt_json(ordered_steps),
                    contexts=prompt_json(compact_contexts(contexts)),
                    candidate_edges=json.dumps(
                        indexed_candidates, ensure_ascii=False, indent=2
                    ),
                )
                active_stage = "dependency_audit"
                audit_result, audit_trace = invoke_json_with_retry(
                    self.llm,
                    audit_prompt,
                    operation="Logical Order dependency audit",
                    parser=lambda raw: _parse_edge_audit(
                        raw,
                        step_ids,
                        graph_edges,
                        roadmap,
                        contexts or [],
                    ),
                    repair_instruction=(
                        "Return exactly one edge_audit entry for every edge_index. "
                        "For each entry, verdict must be CONFIRMED_STRICT_INVERSION or "
                        "KEEP_CURRENT_ORDER. Confirming an inversion requires exact "
                        "roadmap or context evidence quotes."
                    ),
                )
                traces["dependency_audit"] = audit_trace
                candidate_audit = audit_result["edge_audit"]
                graph_edges = audit_result["accepted_dependency_graph"]
                analysis = audit_result["deterministic_analysis"]
            else:
                candidate_audit = []
                graph_edges = []
                analysis = _analyze_order(step_ids, [])
                traces["dependency_audit"] = {
                    "attempt_count": 0,
                    "recovered": False,
                    "validation_errors": [],
                    "skipped": "no candidate inversions",
                }

            _, _, mandatory_band = _band_for_scope(analysis["repair_scope"])
            scoring_prompt = _FINAL_SCORING_PROMPT.format(
                question=question or "(not provided)",
                category=category or "(not provided)",
                steps=prompt_json(ordered_steps),
                analysis=json.dumps(analysis, ensure_ascii=False, indent=2),
                score_band=mandatory_band,
            )
            active_stage = "scoring"
            scoring_result, scoring_trace = invoke_json_with_retry(
                self.llm,
                scoring_prompt,
                operation="Logical Order scoring",
                parser=lambda raw: _parse_final_scoring(
                    raw, analysis["repair_scope"]
                ),
                repair_instruction=(
                    "Return the complete logical_order_assessment and keep "
                    f"the final score inside the mandatory {mandatory_band} band."
                ),
            )
            traces["scoring"] = scoring_trace
            assessment = scoring_result["logical_order_assessment"]
            score = assessment["score"]
            violations = analysis["dependency_violations"]
            return {
                "logical_order": {
                    "score": score,
                    "verdict": _verdict(score),
                    "score_band": _score_band(score),
                    "reason": assessment["reason"],
                    "criteria": assessment["criteria"],
                    "criteria_rationales": assessment["criteria_rationales"],
                    "has_valid_sequence": not violations,
                    "dependency_graph": graph_edges,
                    "dependency_candidate_audit": candidate_audit,
                    "dependency_violations": violations,
                    "dependency_violation_count": len(violations),
                    "suggested_order": analysis["suggested_order"],
                    "movement_count": analysis["movement_count"],
                    "repair_scope": analysis["repair_scope"],
                    "issues": [
                        {
                            "type": item["dependency_type"],
                            "severity": item["severity"],
                            "explanation": item["explanation"],
                        }
                        for item in violations
                    ],
                    "recommendations": assessment["recommendations"],
                    "normalization_notes": [],
                    "manual_review_required": False,
                    "judge_execution": traces,
                }
            }
        except StructuredOutputError as exc:
            traces[active_stage] = exc.trace
            return _fallback_result(
                exc,
                step_ids=step_ids,
                analysis=analysis,
                dependency_graph=graph_edges,
                judge_execution=traces,
            )
        except Exception as exc:
            return _fallback_result(
                exc,
                step_ids=step_ids,
                analysis=analysis,
                dependency_graph=graph_edges,
                judge_execution=traces,
            )
