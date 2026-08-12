"""Provider-neutral recovery for judges that must return structured JSON."""

from __future__ import annotations

import json
from typing import Any, Callable

from src.llm_provider import invoke_llm_text


class StructuredOutputError(ValueError):
    """Raised after all structured-output attempts fail validation."""

    def __init__(self, operation: str, trace: dict[str, Any], last_error: Exception):
        self.operation = operation
        self.trace = trace
        self.last_error = last_error
        self.last_error_type = type(last_error).__name__
        super().__init__(
            f"{operation} returned invalid structured output after "
            f"{trace['attempt_count']} attempts: {self.last_error_type}"
        )


def _error_record(attempt: int, error: Exception, response: str) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "error_type": type(error).__name__,
        "message": str(error)[:300],
        "response_characters": len(response),
    }


def _corrective_prompt(
    original_prompt: str,
    error: Exception,
    response_length: int,
    repair_instruction: str,
) -> str:
    return (
        f"{original_prompt}\n\n"
        "CORRECTIVE STRUCTURED-OUTPUT RETRY\n"
        f"The previous response could not be validated ({type(error).__name__}: "
        f"{str(error)[:240]}). Its length was {response_length} characters.\n"
        "Regenerate the complete answer from the original inputs. Do not explain "
        "the error and do not return a fragment of the previous answer.\n"
        f"{repair_instruction}\n"
    )


def invoke_json_with_retry(
    llm: Any,
    prompt: str,
    *,
    operation: str,
    parser: Callable[[str], dict[str, Any]],
    repair_instruction: str,
    max_attempts: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Invoke, parse, and retry invalid JSON once with auditable diagnostics."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    errors: list[dict[str, Any]] = []
    current_prompt = prompt
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        raw = invoke_llm_text(
            llm,
            current_prompt,
            operation=operation,
            attempt=attempt,
        )
        try:
            parsed = parser(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Structured response must be a JSON object")
            return parsed, {
                "attempt_count": attempt,
                "recovered": attempt > 1,
                "validation_errors": errors,
            }
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            errors.append(_error_record(attempt, exc, raw))
            if attempt < max_attempts:
                current_prompt = _corrective_prompt(
                    prompt,
                    exc,
                    len(raw),
                    repair_instruction,
                )

    trace = {
        "attempt_count": max_attempts,
        "recovered": False,
        "validation_errors": errors,
    }
    raise StructuredOutputError(operation, trace, last_error or ValueError("Unknown error"))
