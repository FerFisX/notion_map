"""Controlled smoke validation for the canonical metric orchestrator.

The suite injects hand-written roadmaps through an in-memory adapter. It never
calls retrieval or roadmap generation, so failures and timings belong only to
the canonical evaluation pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.config import config
from evaluation.dataset import EvalSample
from evaluation.metric_contract import canonical_scores, validate_canonical_metrics
from evaluation.metric_orchestrator import CanonicalMetricEvaluator, run_canonical_metrics
from evaluation.readiness import roadmap_readiness
from evaluation.reporter import save_html, save_json
from evaluation.runner import parse_metrics
from src.llm_provider import active_model_name


REPORT_DIR = Path(config.reports_dir) / "canonical_integration_validation"
SUITE_VERSION = "canonical_controlled_v2"
SEMANTIC_METRICS = (
    "grounding",
    "completeness",
    "actionability",
    "logical_order",
    "structure_quality",
    "step_distinctness",
)


def _step(
    index: int,
    label: str,
    description: str,
    points: list[str],
    step_type: str = "proceso",
) -> dict[str, Any]:
    return {
        "id": f"step_{index}",
        "label": label,
        "description": description,
        "type": step_type,
        "key_points": points,
    }


def _context(index: int, text: str) -> dict[str, str]:
    return {"id": f"context_{index}", "text": text}


@dataclass(frozen=True)
class ControlledCase:
    name: str
    sample: EvalSample
    generated: dict[str, Any]
    score_ranges: dict[str, tuple[float, float]]
    expected_readiness: str
    focus_metrics: tuple[str, ...] = ()
    purpose: str = ""


STRONG_ROADMAP = {
    "title": "Build and Validate Year-over-Year Sales in DAX",
    "steps": [
        _step(
            1,
            "Define the comparison goal",
            "Specify that the report must compare current sales with the same period in the prior year. Define a validated YoY percentage as the final output.",
            ["current versus prior year", "sales measure", "validated YoY output"],
            "inicio",
        ),
        _step(
            2,
            "Create the calendar table",
            "Create a continuous Date table, mark it as the model's date table, and relate Date[Date] to Sales[OrderDate]. This establishes the required time-intelligence foundation.",
            ["continuous dates", "mark as date table", "active date relationship"],
        ),
        _step(
            3,
            "Create the base sales measure",
            "Define [Total Sales] as SUM(Sales[SalesAmount]). Validate it against a known total before applying any time shift.",
            ["SUM", "Sales[SalesAmount]", "known-total check"],
        ),
        _step(
            4,
            "Calculate prior-year sales",
            "Define [Sales PY] with CALCULATE([Total Sales], SAMEPERIODLASTYEAR(Date[Date])). Check a period that has data in both years.",
            ["CALCULATE", "SAMEPERIODLASTYEAR", "comparable period"],
        ),
        _step(
            5,
            "Calculate the YoY percentage",
            "Define [Sales YoY %] with DIVIDE([Total Sales] - [Sales PY], [Sales PY]). Format the measure as a percentage and preserve safe blank handling.",
            ["DIVIDE", "percentage format", "blank-safe result"],
        ),
        _step(
            6,
            "Validate and reuse the pattern",
            "Place current sales, prior-year sales, and YoY percentage in a visual by month. Compare known periods, document the checks, and reuse the pattern only after the results are correct.",
            ["monthly visual", "known-period validation", "documented reusable pattern"],
            "fin",
        ),
    ],
}

STRONG_CONTEXTS = [
    _context(1, "DAX year-over-year analysis compares a current business measure with the same period in the previous year."),
    _context(2, "Time intelligence requires a continuous calendar table marked as the date table and related to the fact date column."),
    _context(3, "Total Sales can be defined as SUM(Sales[SalesAmount]) and should be validated before derived measures are created."),
    _context(4, "Prior-year sales can be calculated with CALCULATE([Total Sales], SAMEPERIODLASTYEAR(Date[Date]))."),
    _context(5, "YoY percentage can use DIVIDE([Total Sales] - [Sales PY], [Sales PY]); DIVIDE safely handles a zero denominator."),
    _context(6, "Validate time-intelligence measures in a visual across periods with known results before reusing the pattern."),
]

POOR_ROADMAP = {
    "title": "Automatic DAX Time Intelligence",
    "steps": [
        _step(
            1,
            "Define a rough reporting goal",
            "State that the report should show a year-over-year sales result, but leave the business measure, comparison period, and acceptance criteria undecided.",
            ["rough YoY goal", "undecided measure", "no acceptance criteria"],
            "inicio",
        ),
        _step(
            2,
            "Publish the report immediately",
            "Publish the final report before creating the date model or any sales measures. Assume the cloud service will automatically create and repair every missing DAX calculation.",
            ["publish before model", "automatic measure creation", "automatic repair"],
        ),
        _step(
            3,
            "Validate the missing YoY output",
            "Validate the final YoY result before the date model and measures exist. Treat a visual that renders without an error as proof that the calculation is correct.",
            ["validate before output exists", "visual rendering is sufficient"],
        ),
        _step(
            4,
            "Configure automatic YoY",
            "Use the nonexistent AUTOYOY function with default options instead of defining a Total Sales base measure, a prior-year measure, or an explicit YoY percentage measure.",
            ["AUTOYOY", "skip base measure", "skip explicit YoY measures"],
        ),
        _step(
            5,
            "Repeat the automatic calculation",
            "Configure the same nonexistent AUTOYOY function again with the same defaults, without adding a different output, validation, or responsibility.",
            ["duplicate AUTOYOY setup", "same defaults", "same output"],
        ),
        _step(
            6,
            "Add placeholder date data last",
            "After publishing, validating, and configuring YoY, add an incomplete date column. Do not create a continuous marked Date table and do not define its relationship to the sales date.",
            ["date data added last", "no marked date table", "no relationship"],
        ),
        _step(
            7,
            "Close with an unresolved visual check",
            "Render the report once, record that the required measure, comparison period, date relationship, and acceptance criteria remain unresolved, and stop without correcting them.",
            ["single visual check", "document unresolved gaps", "stop without correction"],
            "fin",
        ),
    ],
}

POOR_CONTEXTS = [
    _context(1, "Build and validate the data model before publishing a Power BI report. Publishing does not repair missing DAX calculations."),
    _context(2, "DAX has no AUTOYOY function. Prior-year analysis requires an explicit base measure and a valid date table."),
    _context(3, "Create a continuous date table and relationship before applying SAMEPERIODLASTYEAR to a validated base measure."),
    _context(4, "A visual rendering without an error is not sufficient validation; compare the measure with known period results."),
]

UNSUPPORTED_N8N_ROADMAP = {
    "title": "Build a Self-Healing Order Workflow in n8n",
    "steps": [
        _step(1, "Define the order workflow", "Define the order payload, routing result, and successful processing outcome.", ["order payload", "routing result"], "inicio"),
        _step(2, "Receive orders", "Create a Webhook trigger and accept new order payloads.", ["Webhook", "order payload"]),
        _step(3, "Enable predictive routing", "Use the built-in Quantum Router node to predict the perfect fulfillment path with no configuration or training data.", ["Quantum Router", "perfect prediction"]),
        _step(4, "Guarantee automatic recovery", "Enable n8n's Autonomous Repair mode so every failed node fixes its own credentials and payload mappings.", ["Autonomous Repair", "credential repair"]),
        _step(5, "Store the routed order", "Write the automatically corrected order to the operations database.", ["database", "corrected order"]),
        _step(6, "Confirm autonomous operation", "Run one order and approve the workflow if the execution reaches the final node.", ["single execution", "final node"], "fin"),
    ],
}

UNSUPPORTED_N8N_CONTEXTS = [
    _context(1, "An n8n Webhook node can receive an order payload and start a workflow."),
    _context(2, "Use explicit IF or Switch conditions to route orders from known payload fields."),
    _context(3, "Error workflows, retries, and logged failure details should be configured explicitly."),
    _context(4, "Database nodes require valid credentials and explicit field mappings."),
    _context(5, "n8n does not provide built-in Quantum Router or Autonomous Repair nodes; routing and recovery require explicit workflow logic."),
]

INCOMPLETE_POWER_BI_ROADMAP = {
    "title": "Start a Power BI Sales Model",
    "steps": [
        _step(1, "Define the first sales view", "Identify revenue by month as the initial report output.", ["monthly revenue"], "inicio"),
        _step(2, "Import sales data", "Load the Sales and Product tables with Power Query.", ["Sales", "Product", "Power Query"]),
        _step(3, "Clean source columns", "Set data types, remove invalid rows, and standardize product identifiers.", ["data types", "invalid rows", "product IDs"]),
        _step(4, "Create a basic relationship", "Relate Product[ProductKey] to Sales[ProductKey] and verify cardinality.", ["relationship", "cardinality"]),
        _step(5, "Create total revenue", "Define and validate a SUM-based Total Revenue measure.", ["SUM", "Total Revenue"]),
        _step(6, "Review the initial visual", "Build a monthly revenue chart and verify it against the cleaned source total.", ["monthly chart", "source total"], "fin"),
    ],
}

INCOMPLETE_POWER_BI_CONTEXTS = [
    _context(1, "Power Query can load Sales and Product tables, assign data types, remove invalid rows, and standardize product identifiers."),
    _context(2, "Relate Product[ProductKey] to Sales[ProductKey] with validated cardinality as part of a sales model."),
    _context(3, "Total Revenue can be implemented as a SUM-based explicit measure and checked in a monthly chart against a known source total."),
    _context(4, "A production Power BI model also requires defined business requirements, a complete star schema, security, refresh, performance testing, user validation, and deployment."),
]

VAGUE_ABSTRACTION_ROADMAP = {
    "title": "Apply Abstraction Levels to an API",
    "steps": [
        _step(1, "Understand the context", "Think about the API and what matters before beginning.", ["context"], "inicio"),
        _step(2, "Consider abstraction levels", "Review the conceptual, logical, and implementation levels carefully.", ["conceptual", "logical", "implementation"]),
        _step(3, "Improve the conceptual design", "Use suitable ideas to make the high-level design better.", ["high-level design"]),
        _step(4, "Improve the logical design", "Apply appropriate improvements to resources and interactions.", ["resources", "interactions"]),
        _step(5, "Handle implementation", "Complete the technical details using best practices.", ["technical details", "best practices"]),
        _step(6, "Review the result", "Check everything carefully and finish when it looks acceptable.", ["review"], "fin"),
    ],
}

VAGUE_ABSTRACTION_CONTEXTS = [
    _context(1, "Conceptual abstraction defines domain goals and capabilities without transport details."),
    _context(2, "Logical abstraction maps capabilities to resources, operations, boundaries, and contracts."),
    _context(3, "Implementation abstraction defines endpoints, schemas, authentication, errors, and deployment details."),
]

OUT_OF_ORDER_N8N_ROADMAP = {
    "title": "Automate Invoice Delivery in n8n",
    "steps": [
        _step(1, "Define the invoice outcome", "Specify the source order, generated PDF, recipient, and delivery confirmation.", ["order", "PDF", "confirmation"], "inicio"),
        _step(2, "Activate the workflow", "Activate the production workflow before its trigger, credentials, and mappings are configured.", ["activate production"]),
        _step(3, "Send the invoice email", "Send the invoice attachment before the PDF has been generated or the recipient has been validated.", ["email", "attachment"]),
        _step(4, "Generate the invoice PDF", "Create the invoice PDF from the normalized order fields.", ["PDF", "normalized fields"]),
        _step(5, "Normalize the incoming order", "Map customer, line-item, tax, and email fields from the webhook payload.", ["field mapping", "webhook payload"]),
        _step(6, "Configure the webhook and credentials", "Create the trigger and configure the PDF, email, and storage credentials.", ["Webhook", "credentials"]),
        _step(7, "Validate delivery", "Run a test order and confirm the stored PDF and delivery event.", ["test order", "delivery event"], "fin"),
    ],
}

OUT_OF_ORDER_N8N_CONTEXTS = [
    _context(1, "Configure triggers and credentials before activating an n8n workflow."),
    _context(2, "Normalize and validate order fields before generating an invoice document."),
    _context(3, "Generate the invoice PDF before attaching it to an email."),
    _context(4, "Validate the complete workflow with a controlled order before production activation."),
]

WEAK_STRUCTURE_DAX_ROADMAP = {
    "title": "DAX Running Totals",
    "steps": [
        _step(1, "Write the final running-total measure", "Create the cumulative measure immediately with CALCULATE and a date filter.", ["CALCULATE", "date filter"], "inicio"),
        _step(2, "Dates", "Create a calendar, mark it as a date table, relate it to the fact table, inspect missing dates, decide fiscal boundaries, and document ownership.", ["calendar", "relationship", "fiscal boundaries"]),
        _step(3, "Check one total", "Compare the measure with one manually calculated period.", ["manual comparison"]),
        _step(4, "Discuss filter context", "Explain row context, filter context, context transition, and unrelated iterator behavior in detail.", ["filter context", "iterators"]),
        _step(5, "Add the base measure", "Create and validate the base amount measure used by the running total.", ["base measure"]),
        _step(6, "Save the file", "Save the PBIX and continue improving it later.", ["save PBIX"], "fin"),
    ],
}

WEAK_STRUCTURE_DAX_CONTEXTS = [
    _context(1, "A running-total roadmap should frame the required result before implementation."),
    _context(2, "Create and validate the date model and base measure before the cumulative measure."),
    _context(3, "Meaningful closure validates the result across known periods and documents completion criteria."),
]

OVERLAPPING_POWER_BI_ROADMAP = {
    "title": "Create an Executive KPI Page in Power BI",
    "steps": [
        _step(1, "Define executive KPIs", "List revenue, margin, growth, and target attainment with their business definitions.", ["revenue", "margin", "growth", "targets"], "inicio"),
        _step(2, "Define KPI calculations", "Document the formulas and business definitions for revenue, margin, growth, and target attainment.", ["revenue", "margin", "growth", "targets"]),
        _step(3, "Create KPI measures", "Implement DAX measures for revenue, margin, growth, and target attainment.", ["DAX measures", "KPIs"]),
        _step(4, "Implement KPI calculations", "Build the DAX calculations for revenue, margin, growth, and target attainment and format their outputs.", ["DAX calculations", "formatting"]),
        _step(5, "Build the executive page", "Place the four KPI measures in cards and add trend and target comparisons.", ["cards", "trends", "targets"]),
        _step(6, "Review KPI correctness", "Validate each KPI against an approved source and record the result.", ["approved source", "validation"]),
        _step(7, "Validate the KPI page", "Compare every KPI with the approved source and record whether each result is correct.", ["approved source", "validation result"], "fin"),
    ],
}

OVERLAPPING_POWER_BI_CONTEXTS = [
    _context(1, "Executive KPI reports require agreed business definitions before DAX implementation."),
    _context(2, "Create explicit measures, present trends and targets, and validate results against approved sources."),
]

INVALID_SCHEMA_ROADMAP = {
    "title": "",
    "steps": [
        {
            "id": "duplicate",
            "label": "Define the automation goal",
            "description": "Define the source event and expected notification.",
            "type": "start",
            "key_points": "source event",
        },
        {
            "id": "duplicate",
            "label": "Build the workflow",
            "description": "Connect the trigger, transformation, and notification nodes.",
            "type": "proceso",
            "key_points": ["trigger", "transformation", "notification"],
        },
        {
            "id": "step_3",
            "label": "Validate the workflow",
            "description": "Run a controlled event and confirm the notification payload.",
            "type": "finish",
        },
    ],
}

INVALID_SCHEMA_CONTEXTS = [
    _context(1, "An automation roadmap can define its trigger, transformations, notification, and validation outcome."),
]

MIXED_REVIEW_N8N_ROADMAP = {
    "title": "Synchronize Qualified Leads with a CRM in n8n",
    "steps": [
        _step(1, "Define the synchronization goal", "Identify the lead source, CRM destination, qualification rule, and required CRM record.", ["lead source", "CRM", "qualification rule"], "inicio"),
        _step(2, "Receive lead events", "Configure a Webhook node and capture representative payloads for mapping.", ["Webhook", "sample payloads"]),
        _step(3, "Normalize lead fields", "Map identity, company, source, and consent fields into one internal payload.", ["field mapping", "consent"]),
        _step(4, "Check lead quality", "Use an IF node to apply the main qualification rule, but leave edge-case handling for later refinement.", ["IF node", "qualification", "edge cases"]),
        _step(5, "Create or update the CRM record", "Use the CRM node to upsert the normalized lead by email.", ["upsert", "email"]),
        _step(6, "Add basic failure handling", "Configure a retry path and notify the owner, without yet defining retry limits or escalation timing.", ["retry", "notification"]),
        _step(7, "Validate and monitor the workflow", "Test accepted and rejected leads, confirm CRM results, and review execution logs during the initial rollout.", ["test leads", "CRM results", "execution logs"], "fin"),
    ],
}

MIXED_REVIEW_N8N_CONTEXTS = [
    _context(1, "An n8n Webhook can receive lead events and representative payloads can be captured for field mapping."),
    _context(2, "Normalize identity, company, source, and consent fields before applying an explicit IF-based qualification rule."),
    _context(3, "A CRM node can create or update a normalized lead using email as a stable lookup key."),
    _context(4, "Failure handling can retry failed operations and notify an owner; production use should define retry limits and escalation timing."),
    _context(5, "Validate accepted and rejected leads, confirm CRM records, and review n8n execution logs during rollout."),
    _context(6, "Reliable lead synchronization should also define duplicate handling and representative edge cases."),
]


def _coverage_universe(
    *,
    critical: list[str],
    important: list[str] | None = None,
    supporting: list[str] | None = None,
) -> list[dict[str, str]]:
    """Create a stable, human-authored Completeness reference."""
    elements = []
    for importance, names in (
        ("critical", critical),
        ("important", important or []),
        ("supporting", supporting or []),
    ):
        for name in names:
            elements.append({
                "id": f"expected_{len(elements) + 1}",
                "name": name,
                "importance": importance,
            })
    return elements


def _controlled_case(
    *,
    name: str,
    question: str,
    ground_truth: str,
    expected_keywords: list[str],
    expected_elements: list[dict[str, str]],
    category: str,
    expected_step_order: list[str],
    roadmap: dict[str, Any],
    contexts: list[dict[str, str]],
    score_ranges: dict[str, tuple[float, float]],
    expected_readiness: str,
    focus_metrics: tuple[str, ...],
    purpose: str,
) -> ControlledCase:
    """Build a controlled case while keeping generator metadata consistent."""
    return ControlledCase(
        name=name,
        sample=EvalSample(
            question=question,
            ground_truth=ground_truth,
            expected_keywords=expected_keywords,
            category=category,
            expected_step_order=expected_step_order,
            expected_elements=expected_elements,
        ),
        generated={
            "question": question,
            "refined_question": question,
            "query_intent": {"intent": "implementation"},
            "contexts": contexts,
            "corpus_contexts": contexts,
            "web_contexts": [],
            "retrieval": {"context_strategy": "controlled", "n_contexts": len(contexts)},
            "judge_context_strategy": "controlled_same_context",
            "roadmap": roadmap,
            "answer": f"Controlled synthetic roadmap: {name}.",
        },
        score_ranges=score_ranges,
        expected_readiness=expected_readiness,
        focus_metrics=focus_metrics,
        purpose=purpose,
    )


def _case_manifest(cases: list[ControlledCase]) -> list[dict[str, Any]]:
    """Describe and fingerprint controlled inputs without exposing credentials."""
    manifest = []
    for case in cases:
        fingerprint_source = {
            "name": case.name,
            "question": case.sample.question,
            "ground_truth": case.sample.ground_truth,
            "expected_keywords": case.sample.expected_keywords,
            "expected_elements": case.sample.expected_elements,
            "expected_step_order": case.sample.expected_step_order,
            "roadmap": case.generated["roadmap"],
            "contexts": case.generated.get("contexts", []),
            "focus_metrics": case.focus_metrics,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_source,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        manifest.append({
            "name": case.name,
            "question": case.sample.question,
            "focus_metrics": list(case.focus_metrics),
            "purpose": case.purpose,
            "expected_readiness": case.expected_readiness,
            "diagnostic_score_ranges": {
                name: list(bounds) for name, bounds in case.score_ranges.items()
            },
            "fingerprint_sha256": fingerprint,
        })
    return manifest


def _execution_config() -> dict[str, Any]:
    """Capture provider-neutral settings that can explain run variation."""
    return {
        "provider": os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
        "model": active_model_name(),
        "judge_temperature": config.judge_temperature,
        "judge_reasoning_mode": os.getenv(
            "LLM_JUDGE_REASONING_MODE", "disabled"
        ).strip().lower(),
        "request_timeout_seconds": os.getenv("LLM_REQUEST_TIMEOUT", "210").strip(),
        "ollama_seed": os.getenv("OLLAMA_SEED", "").strip() or None,
        "metric_contract": "canonical_v1",
        "evaluation_source_sha256": _evaluation_source_fingerprint(),
    }


def _evaluation_source_fingerprint() -> str:
    """Fingerprint metric implementation files that affect controlled evidence."""
    evaluation_dir = Path(__file__).resolve().parents[1]
    relative_paths = (
        "metric_contract.py",
        "metric_orchestrator.py",
        "readiness.py",
        "structured_output.py",
        "metrics/actionability_judge.py",
        "metrics/completeness_judge.py",
        "metrics/grounding_judge.py",
        "metrics/logical_order_judge.py",
        "metrics/step_semantic_judge.py",
        "metrics/structure_quality_judge.py",
        "metrics/structure_validator.py",
    )
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        path = evaluation_dir / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


CASES = {
    "strong": ControlledCase(
        name="strong_supported_dax",
        sample=EvalSample(
            question="How do I build and validate year-over-year sales in DAX?",
            ground_truth=(
                "Define the comparison goal, create a continuous marked date table and relationship, "
                "validate a Total Sales base measure, calculate prior-year sales, calculate YoY percentage, "
                "and validate known periods before reuse."
            ),
            expected_keywords=["date table", "Total Sales", "SAMEPERIODLASTYEAR", "DIVIDE", "validate"],
            expected_elements=_coverage_universe(
                critical=[
                    "Continuous marked date table with an active sales-date relationship",
                    "Validated base sales measure",
                    "Explicit prior-year sales measure",
                    "Explicit year-over-year percentage measure",
                    "Validation against periods with known results",
                ],
                important=["Clear comparison goal", "Reuse only after validation"],
            ),
            category="dax_power_bi",
            expected_step_order=[
                "Define the comparison goal",
                "Create the calendar table",
                "Create the base sales measure",
                "Calculate prior-year sales",
                "Calculate the YoY percentage",
                "Validate and reuse the pattern",
            ],
        ),
        generated={
            "question": "How do I build and validate year-over-year sales in DAX?",
            "refined_question": "How can I implement and verify a reusable year-over-year sales pattern in DAX?",
            "query_intent": {"intent": "implementation"},
            "contexts": STRONG_CONTEXTS,
            "corpus_contexts": STRONG_CONTEXTS,
            "web_contexts": [],
            "retrieval": {"context_strategy": "controlled", "n_contexts": len(STRONG_CONTEXTS)},
            "judge_context_strategy": "controlled_same_context",
            "roadmap": STRONG_ROADMAP,
            "answer": "Controlled strong DAX roadmap.",
        },
        score_ranges={name: (8.0, 10.0) for name in SEMANTIC_METRICS} | {"schema_validity": (7.0, 10.0)},
        expected_readiness="READY",
        focus_metrics=SEMANTIC_METRICS,
        purpose="Positive cross-metric reference with supported, complete, executable, ordered, structured, and distinct content.",
    ),
    "poor": ControlledCase(
        name="poor_cross_metric_dax",
        sample=EvalSample(
            question="How do I build and validate year-over-year sales in DAX?",
            ground_truth=(
                "A correct solution needs a date table and relationship, a validated base measure, explicit prior-year "
                "and YoY measures, known-period validation, and publication only after validation."
            ),
            expected_keywords=["date table", "base measure", "prior year", "YoY", "validation"],
            expected_elements=_coverage_universe(
                critical=[
                    "Continuous marked date table with an active sales-date relationship",
                    "Validated base sales measure",
                    "Explicit prior-year sales measure",
                    "Explicit year-over-year percentage measure",
                    "Validation against periods with known results",
                ],
                important=["Clear comparison goal", "Publish only after validation"],
            ),
            category="dax_power_bi",
            expected_step_order=[
                "Define the comparison goal",
                "Prepare the date model",
                "Create the base measure",
                "Create prior-year and YoY measures",
                "Validate known periods",
                "Publish the report",
            ],
        ),
        generated={
            "question": "How do I build and validate year-over-year sales in DAX?",
            "refined_question": "How can I implement and verify a reusable year-over-year sales pattern in DAX?",
            "query_intent": {"intent": "implementation"},
            "contexts": POOR_CONTEXTS,
            "corpus_contexts": POOR_CONTEXTS,
            "web_contexts": [],
            "retrieval": {"context_strategy": "controlled", "n_contexts": len(POOR_CONTEXTS)},
            "judge_context_strategy": "controlled_same_context",
            "roadmap": POOR_ROADMAP,
            "answer": "Controlled poor DAX roadmap.",
        },
        score_ranges={
            "grounding": (0.0, 4.9),
            "completeness": (0.0, 4.9),
            "actionability": (5.0, 7.9),
            "logical_order": (0.0, 4.9),
            "structure_quality": (0.0, 4.9),
            "step_distinctness": (5.0, 7.9),
            "schema_validity": (7.0, 10.0),
        },
        expected_readiness="FAIL",
        focus_metrics=SEMANTIC_METRICS,
        purpose="Negative cross-metric reference containing several independent quality failures.",
    ),
    "unsupported_grounding": _controlled_case(
        name="unsupported_but_structured_n8n",
        question="How do I build a reliable order-routing workflow in n8n?",
        ground_truth="Receive orders with a webhook, validate fields, route with explicit conditions, persist the result, and configure observable error handling.",
        expected_keywords=["Webhook", "validation", "routing", "database", "error handling"],
        expected_elements=_coverage_universe(
            critical=["Webhook order intake", "Payload validation", "Explicit routing rules", "Persisted routing result"],
            important=["Observable error handling", "Controlled end-to-end validation"],
        ),
        category="n8n",
        expected_step_order=["Define the workflow", "Receive and validate orders", "Route with explicit rules", "Store results", "Handle errors", "Validate"],
        roadmap=UNSUPPORTED_N8N_ROADMAP,
        contexts=UNSUPPORTED_N8N_CONTEXTS,
        score_ranges={
            "grounding": (0.0, 4.9),
            "logical_order": (8.0, 10.0),
            "schema_validity": (7.0, 10.0),
        },
        expected_readiness="FAIL",
        focus_metrics=("grounding",),
        purpose="Verify that unsupported core claims lower Grounding while the roadmap remains technically valid.",
    ),
    "incomplete": _controlled_case(
        name="accurate_but_incomplete_power_bi",
        question="How do I build a production-ready Power BI sales model?",
        ground_truth="Define requirements, prepare data, build a star schema, create and validate measures, configure security and refresh, test performance, validate with users, and deploy.",
        expected_keywords=["star schema", "DAX", "RLS", "refresh", "performance", "deployment"],
        expected_elements=_coverage_universe(
            critical=["Defined business requirements", "Production star schema", "Validated business measures", "Security and refresh configuration", "Business validation and deployment"],
            important=["Data preparation", "Performance testing"],
        ),
        category="power_bi",
        expected_step_order=["Define requirements", "Prepare data", "Build the model", "Create measures", "Configure security and refresh", "Test performance", "Validate", "Deploy"],
        roadmap=INCOMPLETE_POWER_BI_ROADMAP,
        contexts=INCOMPLETE_POWER_BI_CONTEXTS,
        score_ranges={
            "completeness": (0.0, 4.9),
            "logical_order": (8.0, 10.0),
            "schema_validity": (7.0, 10.0),
        },
        expected_readiness="FAIL",
        focus_metrics=("completeness",),
        purpose="Verify that a coherent partial solution is penalized for missing production-critical coverage.",
    ),
    "vague_actionability": _controlled_case(
        name="vague_abstraction_actions",
        question="How do I apply abstraction levels when designing an API?",
        ground_truth="Define domain capabilities conceptually, map them to logical resources and contracts, implement concrete endpoints and schemas, and validate traceability between levels.",
        expected_keywords=["conceptual", "logical", "implementation", "resources", "endpoints", "validation"],
        expected_elements=_coverage_universe(
            critical=["Conceptual domain capabilities", "Logical resources and contracts", "Implementation endpoints and schemas", "Traceability across abstraction levels"],
            important=["Explicit design goal and boundaries"],
        ),
        category="abstraction",
        expected_step_order=["Frame the domain goal", "Model capabilities", "Define logical resources", "Specify implementation contracts", "Validate traceability"],
        roadmap=VAGUE_ABSTRACTION_ROADMAP,
        contexts=VAGUE_ABSTRACTION_CONTEXTS,
        score_ranges={
            "actionability": (5.0, 7.9),
            "logical_order": (8.0, 10.0),
            "schema_validity": (7.0, 10.0),
        },
        expected_readiness="FAIL",
        focus_metrics=("actionability",),
        purpose="Verify that generic actions without a usable method or done condition lower Actionability.",
    ),
    "out_of_order": _controlled_case(
        name="dependency_reversed_n8n",
        question="How do I automate invoice generation and delivery in n8n?",
        ground_truth="Configure the trigger and credentials, normalize and validate order data, generate the PDF, send it, validate delivery, and activate only after successful testing.",
        expected_keywords=["Webhook", "credentials", "mapping", "PDF", "email", "validation"],
        expected_elements=_coverage_universe(
            critical=["Configured trigger and credentials", "Normalized and validated order data", "Generated invoice PDF", "Invoice delivery", "End-to-end validation before activation"],
        ),
        category="n8n",
        expected_step_order=["Define the outcome", "Configure trigger and credentials", "Normalize data", "Generate PDF", "Send email", "Validate", "Activate"],
        roadmap=OUT_OF_ORDER_N8N_ROADMAP,
        contexts=OUT_OF_ORDER_N8N_CONTEXTS,
        score_ranges={"logical_order": (0.0, 4.9), "schema_validity": (7.0, 10.0)},
        expected_readiness="FAIL",
        focus_metrics=("logical_order",),
        purpose="Verify that explicit prerequisite inversions lower Logical Order.",
    ),
    "weak_structure": _controlled_case(
        name="poorly_shaped_dax_learning_path",
        question="How do I learn to create running totals in DAX?",
        ground_truth="Frame the cumulative-analysis goal, prepare the date model, validate a base measure, implement the running total, test filter behavior, and close with known-period validation.",
        expected_keywords=["date table", "base measure", "CALCULATE", "filter context", "validation"],
        expected_elements=_coverage_universe(
            critical=["Cumulative-analysis goal", "Prepared date model", "Validated base measure", "Running-total measure", "Known-period validation"],
            important=["Filter-context behavior"],
        ),
        category="dax_power_bi",
        expected_step_order=["Define the goal", "Prepare dates", "Create the base measure", "Create the running total", "Validate known periods"],
        roadmap=WEAK_STRUCTURE_DAX_ROADMAP,
        contexts=WEAK_STRUCTURE_DAX_CONTEXTS,
        score_ranges={
            "logical_order": (5.0, 7.9),
            "structure_quality": (5.0, 7.9),
            "schema_validity": (7.0, 10.0),
        },
        expected_readiness="FAIL",
        focus_metrics=("structure_quality",),
        purpose="Verify that weak framing, mixed granularity, fragmented flow, and weak closure lower Structure Quality.",
    ),
    "overlapping_steps": _controlled_case(
        name="semantically_redundant_power_bi_steps",
        question="How do I create and validate an executive KPI page in Power BI?",
        ground_truth="Agree on KPI definitions, implement each measure once, design the executive page, and validate every KPI against an approved source.",
        expected_keywords=["KPI definitions", "DAX measures", "executive page", "approved source"],
        expected_elements=_coverage_universe(
            critical=["Agreed KPI definitions", "Implemented KPI measures", "Executive KPI page", "Validation against an approved source"],
        ),
        category="power_bi",
        expected_step_order=["Define KPIs", "Create measures", "Build the page", "Validate results"],
        roadmap=OVERLAPPING_POWER_BI_ROADMAP,
        contexts=OVERLAPPING_POWER_BI_CONTEXTS,
        score_ranges={
            "step_distinctness": (5.0, 7.9),
            "schema_validity": (7.0, 10.0),
        },
        expected_readiness="NEEDS_REVIEW",
        focus_metrics=("step_distinctness",),
        purpose="Verify that repeated responsibilities are identified as semantic step overlap.",
    ),
    "invalid_schema": _controlled_case(
        name="technically_invalid_roadmap_schema",
        question="How do I build and validate a simple notification workflow?",
        ground_truth="Define the trigger and expected notification, build the transformation and notification steps, and validate the output with a controlled event.",
        expected_keywords=["trigger", "transformation", "notification", "validation"],
        expected_elements=_coverage_universe(
            critical=["Defined trigger and notification outcome", "Transformation and notification workflow", "Controlled output validation"],
        ),
        category="automation",
        expected_step_order=["Define the goal", "Build the workflow", "Validate the notification"],
        roadmap=INVALID_SCHEMA_ROADMAP,
        contexts=INVALID_SCHEMA_CONTEXTS,
        score_ranges={"schema_validity": (0.0, 6.9)},
        expected_readiness="FAIL",
        focus_metrics=("schema_validity",),
        purpose="Verify deterministic rejection of malformed roadmap schema independently of semantic quality.",
    ),
    "mixed_review": _controlled_case(
        name="usable_n8n_workflow_needing_review",
        question="How do I synchronize qualified leads with a CRM using n8n?",
        ground_truth="Define qualification and duplicate rules, receive and normalize leads, upsert CRM records, handle failures with explicit retry and escalation policy, and validate representative cases.",
        expected_keywords=["qualification", "mapping", "upsert", "duplicate handling", "retry limits", "validation"],
        expected_elements=_coverage_universe(
            critical=["Lead intake and normalized mapping", "Qualification rules", "CRM create-or-update operation", "Representative validation cases"],
            important=["Duplicate handling", "Explicit retry limits and escalation policy", "Initial monitoring"],
        ),
        category="n8n",
        expected_step_order=["Define rules", "Receive leads", "Normalize fields", "Qualify leads", "Upsert CRM", "Handle failures", "Validate and monitor"],
        roadmap=MIXED_REVIEW_N8N_ROADMAP,
        contexts=MIXED_REVIEW_N8N_CONTEXTS,
        score_ranges={
            "completeness": (5.0, 7.9),
            "actionability": (8.0, 10.0),
            "logical_order": (8.0, 10.0),
            "schema_validity": (7.0, 10.0),
        },
        expected_readiness="NEEDS_REVIEW",
        focus_metrics=("completeness", "actionability"),
        purpose="Represent a realistic usable workflow with bounded omissions that should require review rather than fail outright.",
    ),
}


class ControlledAdapter:
    """Return exact fixtures and prove that no generation engine is required."""

    def __init__(self, cases: list[ControlledCase]):
        self._generated = {case.sample.question + case.name: case.generated for case in cases}
        self._case_names = [case.name for case in cases]
        self.query_count = 0

    def query(self, question: str) -> dict[str, Any]:
        case_name = self._case_names[self.query_count]
        self.query_count += 1
        return deepcopy(self._generated[question + case_name])


def _attach_expectations(results: dict[str, Any], cases: list[ControlledCase]) -> dict[str, Any]:
    matched = 0
    total = 0
    for sample_result, case in zip(results["per_sample"], cases):
        checks = {}
        for name, (minimum, maximum) in case.score_ranges.items():
            if name not in sample_result:
                continue
            score = (
                float(sample_result[name].get("support_score", 0))
                if name == "grounding"
                else float(sample_result[name].get("score", 0))
            )
            ok = minimum <= score <= maximum
            checks[name] = {
                "score": score,
                "expected_range": [minimum, maximum],
                "matched": ok,
            }
            matched += int(ok)
            total += 1
        actual_readiness = sample_result["roadmap_readiness"]["status"]
        if actual_readiness != "NOT_EVALUATED":
            readiness_ok = actual_readiness == case.expected_readiness
            checks["readiness"] = {
                "actual": actual_readiness,
                "expected": case.expected_readiness,
                "matched": readiness_ok,
            }
            matched += int(readiness_ok)
            total += 1
        sample_result["controlled_case"] = case.name
        sample_result["expected_check"] = checks

    results["controlled_validation"] = {
        "matched_checks": matched,
        "total_checks": total,
        "match_rate": round(matched / total, 4) if total else 0.0,
        "interpretation": "Observed outputs are preserved even when they differ from expected ranges.",
    }
    return results


def run(case_selection: str, metrics: str | None = None) -> dict[str, Any]:
    selected = list(CASES.values()) if case_selection == "all" else [CASES[case_selection]]
    enabled_metrics = parse_metrics(metrics)
    adapter = ControlledAdapter(selected)
    results = run_canonical_metrics(
        adapter,
        [case.sample for case in selected],
        verbose=True,
        enabled_metrics=enabled_metrics,
    )
    if adapter.query_count != len(selected):
        raise AssertionError("The orchestrator did not consume each controlled roadmap exactly once.")
    results = _attach_expectations(results, selected)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metric_suffix = "full" if not metrics else metrics.replace(",", "-")
    stem = f"canonical_integration_{case_selection}_{metric_suffix}"
    json_path = REPORT_DIR / f"{stem}.json"
    html_path = REPORT_DIR / f"{stem}.html"
    payload = {
        "suite": "canonical_controlled_integration",
        "suite_version": SUITE_VERSION,
        "case_selection": case_selection,
        "case_count": len(selected),
        "case_manifest": _case_manifest(selected),
        "metrics": metrics or "canonical_default",
        "model": active_model_name(),
        "execution_config": _execution_config(),
        "generator_calls": 0,
        "controlled_adapter_calls": adapter.query_count,
        "judge": results,
    }
    save_json(payload, str(json_path))
    save_html(None, results, str(html_path))

    validation = results["controlled_validation"]
    print("\nCONTROLLED INTEGRATION RESULT")
    print(f"  Generator calls: 0")
    print(f"  Controlled roadmaps: {adapter.query_count}")
    print(f"  Expected checks: {validation['matched_checks']}/{validation['total_checks']}")
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")
    return payload


def _load_sample(filename: str) -> dict[str, Any]:
    path = REPORT_DIR / filename
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload["judge"]["per_sample"][0]


def _merge_samples(base: dict[str, Any], additions: list[dict[str, Any]]) -> dict[str, Any]:
    merged = deepcopy(base)
    result_keys = (*SEMANTIC_METRICS, "step_overlap", "schema_validity")
    for addition in additions:
        for name in result_keys:
            if name in addition:
                merged[name] = deepcopy(addition[name])
        merged.setdefault("metric_timings", {}).update(addition.get("metric_timings", {}))

    canonical = {name: merged[name] for name in SEMANTIC_METRICS}
    merged["metric_scores"] = canonical_scores(canonical)
    merged["metric_contract_errors"] = validate_canonical_metrics(canonical)
    readiness_inputs = {name: merged[name] for name in result_keys if name in merged}
    merged["roadmap_readiness"] = roadmap_readiness(
        readiness_inputs,
        merged["metric_contract_errors"],
    )
    merged["response_time"] = 0.0
    return merged


def combine_existing() -> dict[str, Any]:
    """Combine completed partial runs without invoking any LLM or generator."""
    strong = _merge_samples(
        _load_sample("canonical_integration_strong.json"),
        [_load_sample("canonical_integration_strong_actionability.json")],
    )
    poor = _merge_samples(
        _load_sample("canonical_integration_poor_grounding.json"),
        [
            _load_sample("canonical_integration_poor_completeness.json"),
            _load_sample("canonical_integration_poor_actionability.json"),
            _load_sample(
                "canonical_integration_poor_logical_order-structure_quality-"
                "step_distinctness-schema_validity.json"
            ),
        ],
    )
    results = {
        "per_sample": [strong, poor],
        "sample_count": 2,
        "failure_count": 0,
        "failures": [],
        "metric_contract": "canonical_v1",
    }
    results["aggregated"] = CanonicalMetricEvaluator._aggregate(results["per_sample"])
    results = _attach_expectations(results, [CASES["strong"], CASES["poor"]])

    json_path = REPORT_DIR / "canonical_integration_controlled_final.json"
    html_path = REPORT_DIR / "canonical_integration_controlled_final.html"
    payload = {
        "suite": "canonical_controlled_integration",
        "suite_version": SUITE_VERSION,
        "case_selection": "strong+poor",
        "case_count": 2,
        "case_manifest": _case_manifest([CASES["strong"], CASES["poor"]]),
        "metrics": "canonical_default",
        "model": active_model_name(),
        "execution_config": _execution_config(),
        "generator_calls": 0,
        "source": "completed controlled partial runs",
        "judge": results,
    }
    save_json(payload, str(json_path))
    save_html(None, results, str(html_path))
    validation = results["controlled_validation"]
    print("\nCONSOLIDATED CONTROLLED RESULT")
    print("  Generator calls: 0")
    print(f"  Cases: {results['sample_count']}")
    print(f"  Expected checks: {validation['matched_checks']}/{validation['total_checks']}")
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonical controlled integration validation")
    parser.add_argument("--case", choices=(*CASES.keys(), "all"), default="strong")
    parser.add_argument(
        "--metrics",
        default=None,
        help="Optional comma-separated canonical metric selection.",
    )
    parser.add_argument(
        "--combine-existing",
        action="store_true",
        help="Combine completed partial controlled runs without calling the LLM.",
    )
    args = parser.parse_args()
    if args.combine_existing:
        combine_existing()
    else:
        run(args.case, args.metrics)
