"""Source-selection contracts for roadmap generation.

The selected mode is request-scoped.  It must never be implemented by
mutating process-wide environment variables because concurrent users may
choose different source policies.
"""

from __future__ import annotations

import os
from enum import Enum


class SourceMode(str, Enum):
    """Allowed evidence policies for one roadmap request."""

    CORPUS = "corpus"
    WEB = "web"
    AUTO = "auto"


DEFAULT_SOURCE_MODE = SourceMode.AUTO


def _positive_timeout(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


SOURCE_MODE_TIMEOUTS = {
    SourceMode.CORPUS: _positive_timeout("CORPUS_MODE_TIMEOUT", 120.0),
    SourceMode.WEB: _positive_timeout("WEB_MODE_TIMEOUT", 300.0),
    SourceMode.AUTO: _positive_timeout("AUTO_MODE_TIMEOUT", 210.0),
}


def normalize_source_mode(value: SourceMode | str | None) -> SourceMode:
    """Return a validated source mode, defaulting to automatic selection."""

    if value is None or not str(value).strip():
        return DEFAULT_SOURCE_MODE
    if isinstance(value, SourceMode):
        return value
    try:
        return SourceMode(str(value).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in SourceMode)
        raise ValueError(
            f"Invalid source_mode '{value}'. Expected one of: {allowed}"
        ) from exc


def timeout_for_mode(mode: SourceMode | str | None) -> float:
    """Return the server-side end-to-end deadline for a source mode."""

    return SOURCE_MODE_TIMEOUTS[normalize_source_mode(mode)]


class InsufficientEvidenceError(RuntimeError):
    """Raised when the selected source policy cannot support a roadmap."""


class PipelineDeadlineExceeded(TimeoutError):
    """Raised when the end-to-end source-mode deadline is exhausted."""


class GroundingValidationError(RuntimeError):
    """Raised when evidence validation cannot safely accept a roadmap."""
