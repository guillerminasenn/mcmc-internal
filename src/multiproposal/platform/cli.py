"""`mcmc-platform` command line; every command is a thin wrapper over importable functions."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from .diagnostics import diagnose_execution
from .execution import RunRecord, RunStatus
from .registry import TERMINAL, Registry, hostname, process_alive
from .runroot import resolve_run_root
from .specs import ExperimentSpec
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
        elif args.command == "list":
            _emit([_record_dict(registry.reconcile(r.execution_id)) for r in registry.list()])
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
