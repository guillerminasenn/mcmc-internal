"""Artifact, labeled posterior, and provenance contracts for read-only analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from .execution import RunRecord, RunStatus
from .problems import VariableSpec
from .specs import (
    ExperimentSpec, VersionedSpec, _freeze_mapping, _integer, _name, _names,
    content_hash,
)


@dataclass(frozen=True)
class ArtifactRecord(VersionedSpec):
    """Checksum-bearing artifact path relative to one execution directory."""

    path: str
    sha256: str
    size_bytes: int
    kind: str

    def __post_init__(self) -> None:
        """Reject ambiguous paths and malformed content checksums."""
        super().__post_init__()
        _name("path", self.path)
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.path or ":" in self.path:
            raise ValueError("Artifact paths must be relative POSIX paths without traversal")
        if str(path) == "." or str(path) != self.path:
            raise ValueError("Artifact path must be normalized and identify a file")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("sha256 must be a lowercase 64-character hex digest")
        _integer("size_bytes", self.size_bytes)
        _name("kind", self.kind)


@dataclass(frozen=True)
class ResultSchema(VersionedSpec):
    """Posterior arrays always start with chain/draw, never an unlabeled transpose."""

    variables: tuple[VariableSpec, ...]
    warmup_draws: int = 0
    warmup_included: bool = False

    def __post_init__(self) -> None:
        """Validate shared sampling axes and explicit warmup interpretation."""
        super().__post_init__()
        if not isinstance(self.variables, (tuple, list)) or not self.variables or any(
            not isinstance(variable, VariableSpec) for variable in self.variables
        ):
            raise ValueError("A result requires VariableSpec entries")
        _names(tuple(variable.name for variable in self.variables))
        _integer("warmup_draws", self.warmup_draws)
        if type(self.warmup_included) is not bool:
            raise ValueError("warmup_included must be a boolean")
        for variable in self.variables:
            if variable.dims[:2] != ("chain", "draw"):
                raise ValueError("Posterior dimensions must start with ('chain', 'draw')")
            if variable.shape[:2] != self.variables[0].shape[:2]:
                raise ValueError("Posterior variables must share chain/draw sizes")
            if self.warmup_included and self.warmup_draws >= variable.shape[1]:
                raise ValueError("Warmup would leave no posterior draws")
        object.__setattr__(self, "variables", tuple(self.variables))

    def validate_arrays(self, arrays: Mapping[str, Any]) -> None:
        """Check loaded shapes without modifying arrays or reading sample values."""
        if set(arrays) != {variable.name for variable in self.variables}:
            raise ValueError("Loaded variables do not match the result schema")
        for variable in self.variables:
            if tuple(getattr(arrays[variable.name], "shape", ())) != variable.shape:
                raise ValueError(f"Array shape does not match schema: {variable.name}")


@dataclass(frozen=True)
class RunManifest(VersionedSpec):
    """New-run provenance joins a resolved spec, execution snapshot, and artifacts."""

    experiment: ExperimentSpec
    record: RunRecord
    artifacts: tuple[ArtifactRecord, ...] = ()
    result_schemas: Mapping[str, ResultSchema] = field(default_factory=dict)
    code_revisions: Mapping[str, object] = field(default_factory=dict)
    environment: Mapping[str, object] = field(default_factory=dict)
    resolved_spec_sha256: str = field(init=False)
    environment_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Check identity joins and detach provenance from mutable caller state."""
        super().__post_init__()
        if not isinstance(self.experiment, ExperimentSpec) or not isinstance(self.record, RunRecord):
            raise ValueError("Manifest requires resolved experiment and execution record")
        if self.record.run_id != self.experiment.run_id or self.record.data_id != self.experiment.data_id:
            raise ValueError("Manifest record does not match experiment identity")
        if not isinstance(self.artifacts, (tuple, list)) or any(
            not isinstance(artifact, ArtifactRecord) for artifact in self.artifacts
        ):
            raise ValueError("artifacts must contain ArtifactRecord values")
        _names(tuple(artifact.path for artifact in self.artifacts))
        if not isinstance(self.result_schemas, Mapping) or any(
            not isinstance(value, ResultSchema) for value in self.result_schemas.values()
        ):
            raise ValueError("result_schemas must map result names to ResultSchema values")
        _names(tuple(self.result_schemas))
        expected_results = {f"sampler_{index}" for index in range(len(self.experiment.samplers))}
        if set(self.result_schemas) - expected_results:
            raise ValueError("Result names must identify sampler indices from the experiment")
        for schema in self.result_schemas.values():
            expected_draws = self.experiment.draws + (
                self.experiment.warmup if schema.warmup_included else 0
            )
            if schema.variables[0].shape[:2] != (self.experiment.chains, expected_draws):
                raise ValueError("Result chain/draw sizes do not match the experiment")
            if schema.warmup_draws != self.experiment.warmup:
                raise ValueError("Result warmup metadata does not match the experiment")
        if self.record.status == RunStatus.COMPLETED:
            if not self.artifacts or set(self.result_schemas) != expected_results:
                raise ValueError("Completed runs require artifacts and every sampler's labeled results")
            if not self.code_revisions or not self.environment:
                raise ValueError("Completed runs require code and environment provenance")
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "result_schemas", MappingProxyType(dict(self.result_schemas)))
        for name in ("code_revisions", "environment"):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name)))
        object.__setattr__(self, "resolved_spec_sha256", content_hash(self.experiment))
        object.__setattr__(self, "environment_sha256", content_hash(self.environment))


@runtime_checkable
class ResultReader(Protocol):
    """Optional installed read-only reader; a provider failure must not block it."""

    def read(
        self, run_root: str, artifacts: tuple[ArtifactRecord, ...], schema: ResultSchema,
    ) -> Mapping[str, Any]:
        """Verify artifacts and return read-only arrays matching the declared schema."""
        ...