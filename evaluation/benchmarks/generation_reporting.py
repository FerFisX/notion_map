"""Lightweight result helpers for roadmap generation benchmarks."""

from __future__ import annotations

from typing import Any


def schema_not_evaluated(reason: str) -> dict[str, Any]:
    return {
        "score": None,
        "verdict": "NOT_EVALUATED",
        "passed": 0,
        "total_checks": 0,
        "pass_rate": None,
        "checks": [],
        "violations": [reason],
    }


def schema_display(schema: dict[str, Any]) -> str:
    score = schema.get("score")
    verdict = str(schema.get("verdict", "UNKNOWN"))
    return verdict if score is None else f"{float(score):.2f}/10 {verdict}"


def recovery_status(trace: dict[str, Any]) -> str:
    attempts = trace.get("generation_attempts", [])
    if not attempts:
        return "UNKNOWN"
    if attempts[0].get("contract_valid") is True:
        return "NOT_NEEDED"
    return "RECOVERED" if trace.get("generation_recovered") else "FAILED"


def _initial_contract_failed(result: dict[str, Any]) -> bool:
    attempts = result.get("generation_trace", {}).get("generation_attempts", [])
    return bool(attempts) and attempts[0].get("contract_valid") is False


def _first_contract_valid(result: dict[str, Any]) -> bool:
    attempts = result.get("generation_trace", {}).get("generation_attempts", [])
    return bool(attempts) and attempts[0].get("contract_valid") is True


def aggregate_generation_results(
    results: list[dict[str, Any]],
    initialization_s: float,
    target_seconds: float = 120.0,
) -> dict[str, Any]:
    stage_names = sorted({
        name
        for result in results
        for name in result.get("generation_trace", {}).get("timings", {})
    })
    stages = {}
    for name in stage_names:
        values = [
            float(result["generation_trace"]["timings"][name])
            for result in results
            if name in result.get("generation_trace", {}).get("timings", {})
        ]
        stages[name] = {
            "mean_s": round(sum(values) / len(values), 4),
            "min_s": round(min(values), 4),
            "max_s": round(max(values), 4),
            "total_s": round(sum(values), 4),
        }

    totals = [float(result.get("observed_total_s", 0)) for result in results]
    successes = [result for result in results if result.get("generation_succeeded")]
    schema_valid = [
        result
        for result in results
        if result.get("schema_validity", {}).get("verdict") == "PASS"
    ]
    schema_evaluated = [
        result
        for result in results
        if result.get("schema_validity", {}).get("verdict") != "NOT_EVALUATED"
    ]
    initial_contract_failures = [
        result for result in results if _initial_contract_failed(result)
    ]
    recovered = [
        result
        for result in initial_contract_failures
        if result.get("generation_trace", {}).get("generation_recovered")
    ]
    first_attempt_valid_count = sum(
        1 for result in results if _first_contract_valid(result)
    )
    recovery_failure_count = len(initial_contract_failures) - len(recovered)
    web_required = [result for result in results if result.get("web_required")]
    web_valid = [result for result in web_required if result.get("web_requirement_passed")]
    within_target = [
        result for result in results if result["observed_total_s"] <= target_seconds
    ]
    return {
        "initialization_s": round(initialization_s, 4),
        "sample_count": len(results),
        "generation_success_count": len(successes),
        "generation_failure_count": len(results) - len(successes),
        "generation_success_rate": round(len(successes) / len(results), 4),
        "schema_validity_pass_count": len(schema_valid),
        "schema_validity_evaluated_count": len(schema_evaluated),
        "schema_validity_not_evaluated_count": len(results) - len(schema_evaluated),
        "schema_validity_pass_rate": (
            round(len(schema_valid) / len(schema_evaluated), 4)
            if schema_evaluated else None
        ),
        "generation_initial_contract_failure_count": len(initial_contract_failures),
        "generation_first_attempt_valid_count": first_attempt_valid_count,
        "generation_first_attempt_valid_rate": round(
            first_attempt_valid_count / len(results), 4
        ),
        "generation_retry_activation_rate": round(
            len(initial_contract_failures) / len(results), 4
        ),
        "generation_recovered_count": len(recovered),
        "generation_recovery_rate": (
            round(len(recovered) / len(initial_contract_failures), 4)
            if initial_contract_failures else None
        ),
        "generation_recovery_failure_count": recovery_failure_count,
        "generation_recovery_failure_rate": (
            round(recovery_failure_count / len(initial_contract_failures), 4)
            if initial_contract_failures else None
        ),
        "web_required_count": len(web_required),
        "web_requirement_pass_count": len(web_valid),
        "web_requirement_pass_rate": (
            round(len(web_valid) / len(web_required), 4) if web_required else 1.0
        ),
        "target_seconds": target_seconds,
        "within_target_count": len(within_target),
        "within_target_rate": round(len(within_target) / len(results), 4),
        "total_generation": {
            "mean_s": round(sum(totals) / len(totals), 4),
            "min_s": round(min(totals), 4),
            "max_s": round(max(totals), 4),
            "total_s": round(sum(totals), 4),
        },
        "stage_timings": stages,
    }
