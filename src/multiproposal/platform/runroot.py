"""Explicit run-root resolution; run artifacts must not land in a tracked repository by accident."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ENV_RUN_ROOT = "MCMC_PLATFORM_RUN_ROOT"


class RunRootError(ValueError):
    """The configured run root is missing or would place artifacts in source control."""


def _repository_top(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def resolve_run_root(explicit: str | os.PathLike[str] | None = None, *,
                     allow_repository: bool = False) -> Path:
    """Return the configured run root; there is deliberately no default location."""
    value = explicit or os.environ.get(ENV_RUN_ROOT)
    if not value:
        raise RunRootError(f"A run root is required: pass --run-root or set {ENV_RUN_ROOT}")
    root = Path(value).expanduser().resolve()
    if not allow_repository:
        top = _repository_top(root)
        if top is not None:
            try:
                ignored = subprocess.run(
                    ["git", "-C", str(top), "check-ignore", "-q", "--", str(root) + "/"],
                    capture_output=True, check=False,
                ).returncode == 0
            except FileNotFoundError as exc:
                raise RunRootError("git is required to verify the run root is not tracked") from exc
            if not ignored:
                raise RunRootError(
                    f"{root} is inside the Git repository {top} and is not ignored; choose an "
                    "ignored or external location, or pass allow_repository for development tests"
                )
    return root
