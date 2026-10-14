"""Run records and client/adapter interfaces; no workers or registry are started."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .problems import RuntimeProblem
from .specs import ExperimentSpec, VersionedSpec, _freeze_mapping, _name


class RunStatus(str, Enum):
    """Truthful execution states, including indeterminate worker liveness."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    STALE = "stale"


@dataclass(frozen=True)
class RunRecord(VersionedSpec):
    """Immutable registry snapshot, distinct from the underlying statistical run."""

    execution_id: str
    run_id: str
    data_id: str
    status: RunStatus
    created_at: str
    resolved_spec_path: str
    run_root: str
    progress: Mapping[str, object] = field(default_factory=dict)
    worker: Mapping[str, object] = field(default_factory=dict)
    logs: Mapping[str, object] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        """Validate a snapshot without inferring process liveness from a PID."""
        super().__post_init__()
        for name in ("execution_id", "run_id", "data_id", "resolved_spec_path", "run_root"):
            _name(name, getattr(self, name))
        object.__setattr__(self, "status", RunStatus(self.status))
        times = {}
        for name in ("created_at", "started_at", "finished_at"):
            value = getattr(self, name)
            if value is not None:
                _name(name, value)
                parsed = datetime.fromisoformat(value)
                if parsed.tzinfo is None:
                    raise ValueError(f"{name} must include a timezone")
                times[name] = parsed
        if "created_at" not in times:
            raise ValueError("created_at is required")
        if self.status == RunStatus.RUNNING and self.started_at is None:
            raise ValueError("A running execution requires started_at")
        terminal = (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)
        if self.status in terminal and self.finished_at is None:
            raise ValueError("A terminal execution requires finished_at")
        if self.finished_at is not None and self.status not in terminal:
            raise ValueError("Only terminal executions have finished_at")
        for later in ("started_at", "finished_at"):
            if later in times and times[later] < times["created_at"]:
                raise ValueError(f"{later} precedes created_at")
        if self.started_at is not None and self.finished_at is not None:
            if times["finished_at"] < times["started_at"]:
                raise ValueError("finished_at precedes started_at")
        for name in ("progress", "worker", "logs"):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name)))
        if self.error is not None:
            _name("error", self.error)


@runtime_checkable
class ExecutionClient(Protocol):
    """Configured local API boundary shared by CLI and UI clients."""

    def validate(self, spec: ExperimentSpec) -> None:
        """Resolve provider/sampler support and reject invalid specifications."""
        ...

    def dry_run(self, spec: ExperimentSpec) -> Mapping[str, object]:
        """Describe work without launching workers or writing sample artifacts."""
        ...

    def submit(self, spec: ExperimentSpec) -> RunRecord:
        """Persist a submission before launching its independent worker."""
        ...

    def status(self, execution_id: str) -> RunRecord:
        """Return a reconciled snapshot, including stale or failed executions."""
        ...

    def cancel(self, execution_id: str) -> RunRecord:
        """Request cancellation; do not promise an immediate cancelled state."""
        ...


@runtime_checkable
class SamplerAdapter(Protocol):
    """Platform-owned wrapper around a kernel, not a problem provider operation."""

    def run(
        self, problem: RuntimeProblem, spec: ExperimentSpec,
        sampler_index: int, chain_index: int,
    ) -> Mapping[str, Any]:
        """Produce one chain's arrays and cost counters using its recorded seed path."""
        ...