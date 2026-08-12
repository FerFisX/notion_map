"""Developer-facing run directories, progress output, timing, and logs."""

from __future__ import annotations

import re
import json
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return normalized.lower() or "evaluation"


def create_run_dir(reports_dir: str, run_name: str) -> Path:
    """Create a timestamped directory without overwriting an existing run."""
    parent = Path(reports_dir) / "runs"
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = parent / f"{_slug(run_name)}-{timestamp}"
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{_slug(run_name)}-{timestamp}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class RunProgress:
    """Print concise progress and mirror it to a persistent run log."""

    def __init__(self, run_dir: Path, total_units: int):
        self.run_dir = run_dir
        self.log_path = run_dir / "run.log"
        self.total_units = max(0, total_units)
        self.completed_units = 0
        self.completed_durations: list[float] = []
        self.started_at = time.perf_counter()
        self._write(f"Run directory: {run_dir}")

    def _write(self, message: str = "") -> None:
        print(message, flush=True)
        timestamp = datetime.now().isoformat(timespec="seconds")
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {message}\n")

    def sample_started(self, index: int, total: int, question: str) -> None:
        self._write(f"\n[{index}/{total}] {question[:80]}")

    def stage_started(self, label: str) -> None:
        self._write(f"  -> {label}...")

    def stage_completed(
        self,
        label: str,
        elapsed_seconds: float,
        score: float | None = None,
        verdict: str | None = None,
    ) -> None:
        self.completed_units += 1
        self.completed_durations.append(float(elapsed_seconds))
        score_text = f"  {score:.1f}/10" if score is not None else ""
        verdict_text = f"  {verdict}" if verdict else ""
        self._write(
            f"  OK {label:<30}{score_text}{verdict_text}  "
            f"[{format_duration(elapsed_seconds)}]"
        )
        remaining = max(0, self.total_units - self.completed_units)
        if remaining and self.completed_durations:
            estimate = sum(self.completed_durations) / len(self.completed_durations) * remaining
            self._write(
                f"     elapsed {format_duration(time.perf_counter() - self.started_at)}"
                f" | estimated remaining ~{format_duration(estimate)}"
            )

    def stage_failed(self, label: str, error: BaseException) -> None:
        self._write(f"  ERROR {label}: {type(error).__name__}: {error}")

    def message(self, value: str) -> None:
        self._write(value)

    def finish(self, status: str = "COMPLETED") -> float:
        elapsed = self.elapsed_seconds()
        self._write(f"\nRun status: {status} | total time {format_duration(elapsed)}")
        return elapsed

    def elapsed_seconds(self) -> float:
        return round(time.perf_counter() - self.started_at, 2)


def selected_metric_count(enabled_metrics: list[str] | None) -> int:
    """Count execution units; Step Semantics is one call and Schema is local."""
    return len(enabled_metrics) if enabled_metrics is not None else 7


def metric_result_summary(name: str, result: dict[str, Any]) -> tuple[float | None, str | None]:
    """Extract the user-facing score and verdict for progress output."""
    if name == "grounding":
        value = result.get("grounding", {})
        return float(value.get("support_score", 0)), str(value.get("verdict", ""))
    if name == "step_semantics":
        value = result.get("step_distinctness", {})
        return float(value.get("score", 0)), str(value.get("verdict", ""))
    value = result.get(name, {})
    if not value:
        return None, None
    return float(value.get("score", 0)), str(value.get("verdict", ""))


class RunCheckpoint:
    """Atomically persist generated roadmaps and completed metric outputs."""

    VERSION = 1

    def __init__(self, run_dir: Path, state: dict[str, Any]):
        self.run_dir = run_dir
        self.path = run_dir / "checkpoint.json"
        self.state = state

    @classmethod
    def create(
        cls,
        run_dir: Path,
        run_name: str,
        mode: str,
        selected_metrics: list[str],
        samples: list[Any],
        metadata: dict[str, Any] | None = None,
    ) -> "RunCheckpoint":
        entries = []
        for sample in samples:
            if is_dataclass(sample):
                sample_data = asdict(sample)
            else:
                sample_data = {
                    "question": str(getattr(sample, "question", "")),
                    "ground_truth": str(getattr(sample, "ground_truth", "")),
                    "expected_keywords": list(getattr(sample, "expected_keywords", [])),
                    "category": str(getattr(sample, "category", "")),
                    "expected_step_order": list(getattr(sample, "expected_step_order", [])),
                }
            entries.append({
                "sample": sample_data,
                "generated": None,
                "generation_seconds": None,
                "result": None,
                "completed_metrics": [],
            })
        checkpoint = cls(run_dir, {
            "version": cls.VERSION,
            "status": "RUNNING",
            "run_name": run_name,
            "mode": mode,
            "selected_metrics": list(selected_metrics),
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "samples": entries,
        })
        checkpoint.save()
        return checkpoint

    @classmethod
    def load(cls, value: str | Path) -> "RunCheckpoint":
        path = Path(value)
        if path.is_dir():
            path = path / "checkpoint.json"
        with path.open("r", encoding="utf-8") as file:
            state = json.load(file)
        if state.get("version") != cls.VERSION:
            raise ValueError(f"Unsupported checkpoint version: {state.get('version')}")
        return cls(path.parent, state)

    def save(self) -> None:
        self.state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        temporary = self.path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(self.state, file, ensure_ascii=False, indent=2)
        temporary.replace(self.path)

    def entry(self, index: int) -> dict[str, Any]:
        return self.state["samples"][index]

    def record_generation(
        self,
        index: int,
        generated: dict[str, Any],
        elapsed_seconds: float,
    ) -> None:
        entry = self.entry(index)
        entry["generated"] = generated
        entry["generation_seconds"] = elapsed_seconds
        self.save()

    def record_result(self, index: int, metric: str, result: dict[str, Any]) -> None:
        entry = self.entry(index)
        entry["result"] = result
        if metric not in entry["completed_metrics"]:
            entry["completed_metrics"].append(metric)
        self.save()

    def mark(self, status: str) -> None:
        self.state["status"] = status
        self.save()

    def remaining_units(self) -> int:
        total = 0
        selected = self.state.get("selected_metrics", [])
        for entry in self.state.get("samples", []):
            if not entry.get("generated"):
                total += 1
            completed = set(entry.get("completed_metrics", []))
            total += sum(name not in completed for name in selected)
        return total

    def completed_results(self) -> list[dict[str, Any]]:
        return [
            entry["result"]
            for entry in self.state.get("samples", [])
            if isinstance(entry.get("result"), dict)
        ]
