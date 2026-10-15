"""SQLite run registry and immutable execution directories; standard library only."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .execution import RunRecord, RunStatus
from .specs import ExperimentSpec

REGISTRY_FILENAME = "registry.sqlite"
TERMINAL = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED})
_ALLOWED = {
    RunStatus.PENDING: {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLING,
                        RunStatus.CANCELLED, RunStatus.STALE},
    RunStatus.RUNNING: {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLING,
                        RunStatus.CANCELLED, RunStatus.STALE},
    RunStatus.CANCELLING: {RunStatus.CANCELLED, RunStatus.COMPLETED, RunStatus.FAILED,
                           RunStatus.STALE},
    RunStatus.STALE: {RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.FAILED,
                      RunStatus.CANCELLED},
    **{status: set() for status in TERMINAL},
}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    execution_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, data_id TEXT NOT NULL,
    status TEXT NOT NULL, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
    heartbeat_at TEXT, execution_dir TEXT NOT NULL, spec_sha256 TEXT NOT NULL,
    progress TEXT NOT NULL DEFAULT '{}', worker TEXT NOT NULL DEFAULT '{}', error TEXT
)"""


class TransitionError(ValueError):
    """A requested status change is not a legal registry transition."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hostname() -> str:
    return socket.gethostname()


def process_stat(pid: int) -> tuple[str, int] | None:
    """Return (state, start ticks) from /proc, or None when unavailable."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = text[text.rindex(")") + 2:].split()
    return fields[0], int(fields[19])


def process_alive(pid: int, start_ticks: int | None = None) -> bool:
    """A PID is live only if it exists, is not a zombie, and keeps its start time."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    stat = process_stat(pid)
    if stat is None:
        return True
    state, ticks = stat
    return state != "Z" and (start_ticks is None or ticks == start_ticks)


class Registry:
    """Platform-owned catalog of executions rooted at a configured run directory."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / REGISTRY_FILENAME
        with self._connect() as connection:
            connection.execute(_SCHEMA)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self):
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")

    def execution_dir(self, execution_id: str) -> Path:
        return self.root / self._row(execution_id)["execution_dir"]

    def _row(self, execution_id: str) -> sqlite3.Row:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE execution_id = ?", (execution_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown execution: {execution_id}")
        return row

    def register(self, spec: ExperimentSpec) -> RunRecord:
        """Persist the resolved spec and a pending row before any worker is launched."""
        execution_id = "exec_" + uuid.uuid4().hex
        relative = f"{spec.run_id}/{execution_id}"
        directory = self.root / relative
        (directory / "logs").mkdir(parents=True)
        (directory / "artifacts").mkdir()
        text = spec.to_json()
        with (directory / "spec.json").open("x", encoding="utf-8") as stream:
            stream.write(text)
        (directory / "spec.json").chmod(0o444)
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO runs (execution_id, run_id, data_id, status, created_at, "
                "execution_dir, spec_sha256) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (execution_id, spec.run_id, spec.data_id, RunStatus.PENDING.value, utc_now(),
                 relative, hashlib.sha256(text.encode("utf-8")).hexdigest()),
            )
        return self.snapshot(execution_id)

    def spec_text(self, execution_id: str) -> str:
        """Return the persisted spec bytes after checking them against the registered digest."""
        row = self._row(execution_id)
        text = (self.root / row["execution_dir"] / "spec.json").read_text(encoding="utf-8")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != row["spec_sha256"]:
            raise ValueError(f"Persisted spec does not match its registered checksum: {execution_id}")
        return text

    def snapshot(self, execution_id: str) -> RunRecord:
        row = self._row(execution_id)
        directory = self.root / row["execution_dir"]
        return RunRecord(
            execution_id=row["execution_id"], run_id=row["run_id"], data_id=row["data_id"],
            status=RunStatus(row["status"]), created_at=row["created_at"],
            resolved_spec_path=str(directory / "spec.json"), run_root=str(directory),
            progress=json.loads(row["progress"]), worker=json.loads(row["worker"]),
            logs={"stdout": str(directory / "logs" / "stdout.log"),
                  "stderr": str(directory / "logs" / "stderr.log")},
            started_at=row["started_at"], finished_at=row["finished_at"], error=row["error"],
        )

    def list(self, status: RunStatus | None = None) -> list[RunRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT execution_id FROM runs ORDER BY created_at, execution_id"
            ).fetchall()
        records = [self.snapshot(row["execution_id"]) for row in rows]
        return [record for record in records if status is None or record.status == status]

    def transition(self, execution_id: str, new: RunStatus, *, error: str | None = None,
                   finished_at: str | None = None) -> RunRecord:
        """Apply one legal status change atomically."""
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT status FROM runs WHERE execution_id = ?", (execution_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown execution: {execution_id}")
            current = RunStatus(row["status"])
            if new not in _ALLOWED[current]:
                raise TransitionError(f"Illegal transition {current.value} -> {new.value}")
            now = utc_now()
            connection.execute(
                "UPDATE runs SET status = ?, finished_at = ?, error = COALESCE(?, error), "
                "started_at = CASE WHEN ? = 'running' AND started_at IS NULL THEN ? "
                "ELSE started_at END WHERE execution_id = ?",
                (new.value, (finished_at or now) if new in TERMINAL else None, error,
                 new.value, now, execution_id),
            )
        return self.snapshot(execution_id)

    def begin_running(self, execution_id: str, worker: dict[str, object]) -> bool:
        """Move pending -> running for this worker; False if cancelled or already claimed."""
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT status FROM runs WHERE execution_id = ?", (execution_id,)
            ).fetchone()
            if row is None or RunStatus(row["status"]) != RunStatus.PENDING:
                return False
            now = utc_now()
            connection.execute(
                "UPDATE runs SET status = 'running', started_at = ?, heartbeat_at = ?, "
                "worker = ? WHERE execution_id = ?",
                (now, now, json.dumps(worker), execution_id),
            )
        return True

    def set_worker(self, execution_id: str, **fields: object) -> None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT worker FROM runs WHERE execution_id = ?", (execution_id,)
            ).fetchone()
            merged = {**json.loads(row["worker"]), **fields}
            connection.execute("UPDATE runs SET worker = ? WHERE execution_id = ?",
                               (json.dumps(merged), execution_id))

    def heartbeat(self, execution_id: str, progress: dict[str, object] | None = None) -> None:
        with self._transaction() as connection:
            if progress is None:
                connection.execute("UPDATE runs SET heartbeat_at = ? WHERE execution_id = ?",
                                   (utc_now(), execution_id))
            else:
                connection.execute(
                    "UPDATE runs SET heartbeat_at = ?, progress = ? WHERE execution_id = ?",
                    (utc_now(), json.dumps(progress), execution_id),
                )

    def reconcile(self, execution_id: str, *, stale_after_seconds: float = 60.0) -> RunRecord:
        """Align an active row with manifest, process, and heartbeat evidence."""
        row = self._row(execution_id)
        record = self.snapshot(execution_id)
        if record.status in TERMINAL:
            return record
        manifest = self.root / row["execution_dir"] / "manifest.json"
        if manifest.exists():
            try:
                done = json.loads(manifest.read_text(encoding="utf-8"))["record"]
            except (ValueError, KeyError):
                done = {}
            if done.get("status") == "completed":
                return self.transition(execution_id, RunStatus.COMPLETED,
                                       finished_at=done.get("finished_at"))
        worker = record.worker
        reference = datetime.fromisoformat(row["heartbeat_at"] or row["created_at"])
        young = (datetime.now(timezone.utc) - reference).total_seconds() < stale_after_seconds
        if worker.get("pid") is not None and worker.get("host") == hostname():
            alive = process_alive(int(worker["pid"]), worker.get("start_ticks"))
        else:
            alive = young
        if alive:
            if record.status == RunStatus.STALE:
                return self.transition(execution_id, RunStatus.RUNNING)
            return record
        if record.status == RunStatus.CANCELLING:
            return self.transition(execution_id, RunStatus.CANCELLED)
        if record.status in (RunStatus.PENDING, RunStatus.RUNNING):
            return self.transition(
                execution_id, RunStatus.STALE,
                error="Worker is not alive and left no completion manifest",
            )
        return record
