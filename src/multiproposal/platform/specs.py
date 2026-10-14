"""Versioned, immutable specifications; no numerical or UI dependencies."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType


SCHEMA_VERSION = 1
HASH_VERSION = "canonical-json-sha256-v1"


def _json_value(value: object) -> object:
    """Convert supported contract values to strict JSON-compatible objects."""
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"Unsupported JSON value: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Encode deterministic UTF-8 JSON with sorted keys and finite numbers."""
    return json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def content_hash(value: object) -> str:
    """Return the SHA-256 digest of the canonical JSON encoding."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON fields rather than silently accepting the last one."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _freeze(value: object) -> object:
    """Detach and recursively freeze validated configuration values."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    """Validate a JSON object and preserve an immutable snapshot."""
    if not isinstance(value, Mapping):
        raise ValueError("Configuration must be a JSON object")
    canonical_json(value)
    return MappingProxyType({key: _freeze(item) for key, item in value.items()})


def _integer(name: str, value: int, minimum: int = 0) -> None:
    """Validate an integer count or seed without accepting booleans."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _name(name: str, value: str) -> None:
    """Require nonempty textual identifiers."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _names(values: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize a set of named capabilities or observables."""
    if not isinstance(values, (tuple, list)):
        raise ValueError("Names must be a sequence of strings")
    for value in values:
        _name("name", value)
    if len(set(values)) != len(values):
        raise ValueError("Names must be unique")
    return tuple(sorted(values))


@dataclass(frozen=True, kw_only=True)
class VersionedSpec:
    """Base encoding for V1 contracts; unknown versions are rejected."""

    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject unknown or malformed contract versions."""
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {self.schema_version!r}")

    def to_dict(self) -> dict[str, object]:
        """Return detached, mutable JSON data for clients and persistence."""
        return json.loads(canonical_json(self))


@dataclass(frozen=True)
class ProblemSpec(VersionedSpec):
    """Resolved provider identity, model/data settings, and capabilities."""

    problem_id: str
    provider: str
    provider_version: str
    model: Mapping[str, object] = field(default_factory=dict)
    data: Mapping[str, object] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate identity and freeze model/data configuration."""
        super().__post_init__()
        for name in ("problem_id", "provider", "provider_version"):
            _name(name, getattr(self, name))
        for name in ("model", "data"):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name)))
        object.__setattr__(self, "capabilities", _names(self.capabilities))

    @property
    def data_id(self) -> str:
        """Identify model/data and provider revision independently of sampling."""
        payload = self.to_dict()
        payload.pop("capabilities")
        return "data_sha256_" + content_hash(payload)


@dataclass(frozen=True)
class SamplerSpec(VersionedSpec):
    """Sampler identity, statistical settings, and required problem operations."""

    sampler_id: str
    config: Mapping[str, object] = field(default_factory=dict)
    required_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate a sampler configuration and its capability requirements."""
        super().__post_init__()
        _name("sampler_id", self.sampler_id)
        object.__setattr__(self, "config", _freeze_mapping(self.config))
        object.__setattr__(self, "required_capabilities", _names(self.required_capabilities))


@dataclass(frozen=True)
class SeedSpec(VersionedSpec):
    """Separate data and experiment entropy; shard placement owns no science RNG."""

    data_seed: int
    experiment_seed: int

    def __post_init__(self) -> None:
        """Require explicit nonnegative seed entropy."""
        super().__post_init__()
        _integer("data_seed", self.data_seed)
        _integer("experiment_seed", self.experiment_seed)

    def sampler_spawn_key(self, chain_index: int, sampler_index: int) -> tuple[int, ...]:
        """Describe a stable SeedSequence path independent of worker ordering."""
        _integer("chain_index", chain_index)
        _integer("sampler_index", sampler_index)
        return (0, chain_index, sampler_index)


@dataclass(frozen=True)
class ExperimentSpec(VersionedSpec):
    """Resolved statistical experiment plus separately recorded execution options."""

    problem: ProblemSpec
    samplers: tuple[SamplerSpec, ...]
    seeds: SeedSpec
    chains: int
    draws: int
    warmup: int = 0
    observables: tuple[str, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    comparison: Mapping[str, object] = field(default_factory=dict)
    execution: Mapping[str, object] = field(default_factory=dict)
    hash_version: str = HASH_VERSION

    def __post_init__(self) -> None:
        """Validate nested specs, counts, capabilities, and immutable settings."""
        super().__post_init__()
        if self.hash_version != HASH_VERSION:
            raise ValueError(f"Unsupported hash_version: {self.hash_version!r}")
        if not isinstance(self.problem, ProblemSpec) or not isinstance(self.seeds, SeedSpec):
            raise ValueError("problem and seeds must be resolved specifications")
        if not isinstance(self.samplers, (tuple, list)) or not self.samplers:
            raise ValueError("At least one sampler is required")
        for sampler in self.samplers:
            if not isinstance(sampler, SamplerSpec):
                raise ValueError("samplers must contain SamplerSpec values")
            missing = set(sampler.required_capabilities) - set(self.problem.capabilities)
            if missing:
                raise ValueError(f"{sampler.sampler_id}: missing capabilities {sorted(missing)}")
        object.__setattr__(self, "samplers", tuple(self.samplers))
        _integer("chains", self.chains, 1)
        _integer("draws", self.draws, 1)
        _integer("warmup", self.warmup)
        object.__setattr__(self, "observables", _names(self.observables))
        for name in ("diagnostics", "comparison", "execution"):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name)))

    @property
    def data_id(self) -> str:
        """Include the data seed in model/data identity, but not chain settings."""
        return "data_sha256_" + content_hash({
            "problem_data_id": self.problem.data_id, "data_seed": self.seeds.data_seed,
            "schema_version": self.schema_version, "hash_version": self.hash_version,
        })

    @property
    def run_id(self) -> str:
        """Identify statistical settings without execution-only scheduling choices."""
        payload = self.to_dict()
        payload.pop("execution")
        return "run_sha256_" + content_hash(payload)

    def to_json(self) -> str:
        """Return the canonical resolved spec shared by CLI and UI."""
        return canonical_json(self)

    @classmethod
    def from_json(cls, text: str) -> ExperimentSpec:
        """Load and validate a resolved V1 experiment; never silently migrate it."""
        try:
            payload = json.loads(text, object_pairs_hook=_unique_object)
            if "schema_version" not in payload or "hash_version" not in payload:
                raise ValueError("Resolved specs require schema_version and hash_version")
            nested_specs = [payload["problem"], payload["seeds"], *payload["samplers"]]
            if any("schema_version" not in item for item in nested_specs):
                raise ValueError("Nested specs require schema_version")
            payload["problem"] = ProblemSpec(**payload["problem"])
            payload["samplers"] = tuple(SamplerSpec(**item) for item in payload["samplers"])
            payload["seeds"] = SeedSpec(**payload["seeds"])
            return cls(**payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid experiment specification: {exc}") from exc