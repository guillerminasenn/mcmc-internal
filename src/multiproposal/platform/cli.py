"""`mcmc-platform` command line; every command is a thin wrapper over importable functions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from .analysis import compare_executions, report_data
from .diagnostics import diagnose_execution
from .execution import RunRecord, RunStatus
from .registry import TERMINAL, Registry, hostname, process_alive
from .runroot import resolve_run_root
from .specs import ExperimentSpec, canonical_json
from .validation import discover_providers, dry_run, validate_experiment


def load_spec(path: str) -> ExperimentSpec:
    return ExperimentSpec.from_json(Path(path).read_text(encoding="utf-8"))


def launch_worker(registry: Registry, execution_id: str) -> int:
    """Start a session-detached worker whose output goes to the execution logs."""
    logs = registry.execution_dir(execution_id) / "logs"
    with (logs / "stdout.log").open("ab") as out, (logs / "stderr.log").open("ab") as err:
        process = subprocess.Popen(
            [sys.executable, "-m", "multiproposal.platform.worker",
             "--run-root", str(registry.root), "--execution-id", execution_id],
            stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True,
            close_fds=True,
        )
    return process.pid


def submit(registry: Registry, spec: ExperimentSpec) -> RunRecord:
    validate_experiment(spec)
    record = registry.register(spec)
    try:
        launch_worker(registry, record.execution_id)
    except OSError as exc:
        return registry.transition(record.execution_id, RunStatus.FAILED,
                                   error=f"Worker launch failed: {exc}")
    return registry.snapshot(record.execution_id)


def cancel(registry: Registry, execution_id: str) -> RunRecord:
    record = registry.reconcile(execution_id)
    if record.status in TERMINAL:
        return record
    worker = record.worker
    if record.status == RunStatus.PENDING and not worker:
        return registry.transition(execution_id, RunStatus.CANCELLED)
    record = registry.transition(execution_id, RunStatus.CANCELLING)
    if worker.get("host") == hostname() and process_alive(
            int(worker["pid"]), worker.get("start_ticks")):
        os.kill(int(worker["pid"]), signal.SIGTERM)
    return record


def _emit(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _record_dict(record: RunRecord) -> dict[str, object]:
    return {**record.to_dict()}


def register_legacy_manifest(registry: Registry, manifest_path: str) -> RunRecord:
    """Register a completed historical manifest under the configured run root."""
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "experiment" not in payload or "record" not in payload:
        raise ValueError("legacy manifest must contain experiment and record")

    spec = ExperimentSpec.from_json(canonical_json(payload["experiment"]))
    record_payload = payload["record"]
    status = RunStatus(record_payload["status"])
    if status not in TERMINAL:
        raise ValueError("legacy manifest record must be terminal")

    run_directory = path.parent
    try:
        relative = run_directory.relative_to(registry.root)
    except ValueError as exc:
        raise ValueError("legacy manifest must be inside the configured run root") from exc

    spec_path = run_directory / "spec.json"
    if not spec_path.exists():
        raise ValueError("legacy execution directory is missing spec.json")
    digest = hashlib.sha256(spec_path.read_bytes()).hexdigest()

    return registry.register_legacy(
        execution_id=record_payload["execution_id"],
        run_id=spec.run_id,
        data_id=spec.data_id,
        status=status,
        created_at=record_payload["created_at"],
        started_at=record_payload.get("started_at"),
        finished_at=record_payload.get("finished_at"),
        heartbeat_at=record_payload.get("finished_at") or record_payload.get("started_at")
        or record_payload["created_at"],
        execution_dir=str(relative),
        spec_sha256=digest,
        progress=record_payload.get("progress") or {},
        worker=record_payload.get("worker") or {},
        error=record_payload.get("error"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mcmc-platform")
    parser.add_argument("--run-root", help="registry and run artifact directory (or set "
                                           "MCMC_PLATFORM_RUN_ROOT)")
    parser.add_argument("--allow-repository-run-root", action="store_true",
                        help="permit a run root inside an unignored repository (development tests)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("providers", help="list installed problem providers")
    for name in ("validate", "dry-run", "submit"):
        sub = commands.add_parser(name)
        sub.add_argument("spec", help="resolved experiment JSON")
    for name in ("status", "cancel"):
        sub = commands.add_parser(name)
        sub.add_argument("execution_id")
    sub = commands.add_parser("diagnose")
    sub.add_argument("execution_id")
    sub = commands.add_parser("report-data")
    sub.add_argument("execution_id")
    sub = commands.add_parser("compare")
    sub.add_argument("execution_ids", nargs="+")
    sub = commands.add_parser("register-legacy")
    sub.add_argument("manifest", help="path to a completed manifest.json")
    commands.add_parser("list", help="list registered executions")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "providers":
            providers, failures = discover_providers()
            _emit({"available": {name: p.describe()["problem_id"] for name, p in providers.items()},
                   "failed": failures})
            return 0
        if args.command == "validate":
            validate_experiment(load_spec(args.spec))
            print("valid")
            return 0
        if args.command == "dry-run":
            _emit(dry_run(load_spec(args.spec)))
            return 0
        registry = Registry(resolve_run_root(
            args.run_root, allow_repository=args.allow_repository_run_root))
        if args.command == "submit":
            _emit(_record_dict(submit(registry, load_spec(args.spec))))
        elif args.command == "status":
            _emit(_record_dict(registry.reconcile(args.execution_id)))
        elif args.command == "cancel":
            _emit(_record_dict(cancel(registry, args.execution_id)))
        elif args.command == "diagnose":
            _emit(diagnose_execution(registry, args.execution_id))
        elif args.command == "report-data":
            _emit(report_data(registry, args.execution_id))
        elif args.command == "compare":
            _emit(compare_executions(registry, args.execution_ids))
        elif args.command == "register-legacy":
            _emit(_record_dict(register_legacy_manifest(registry, args.manifest)))
        elif args.command == "list":
            _emit([_record_dict(registry.reconcile(r.execution_id)) for r in registry.list()])
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
