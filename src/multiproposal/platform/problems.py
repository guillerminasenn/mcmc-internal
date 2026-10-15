"""Structural provider contracts and a compatibility adapter for existing problems."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .specs import VersionedSpec, _freeze_mapping, _integer, _name, _names


PROBLEM_PROVIDER_GROUP = "bayesian.problem_providers"


@dataclass(frozen=True)
class VariableSpec(VersionedSpec):
    """Named array axes and coordinates; scalar variables have no intrinsic axes."""

    name: str
    dims: tuple[str, ...] = ()
    shape: tuple[int, ...] = ()
    coordinates: Mapping[str, object] = field(default_factory=dict)
    units: str | None = None

    def __post_init__(self) -> None:
        """Validate axis sizes and preserve detached coordinate labels."""
        super().__post_init__()
        _name("name", self.name)
        _names(self.dims)
        if not isinstance(self.shape, (list, tuple)) or len(self.dims) != len(self.shape):
            raise ValueError("dims and shape must have equal length")
        for size in self.shape:
            _integer("axis size", size, 1)
        coordinates = _freeze_mapping(self.coordinates)
        for dimension, labels in coordinates.items():
            if dimension not in self.dims:
                raise ValueError(f"Unknown coordinate dimension: {dimension}")
            if not isinstance(labels, (list, tuple)) or len(labels) != self.shape[self.dims.index(dimension)]:
                raise ValueError(f"Coordinate length does not match dimension: {dimension}")
            if any(not isinstance(label, (str, int, float, bool)) for label in labels):
                raise ValueError("Coordinates must contain scalar labels")
        if self.units is not None:
            _name("units", self.units)
        object.__setattr__(self, "dims", tuple(self.dims))
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "coordinates", coordinates)


@dataclass(frozen=True)
class ProviderMetadata(VersionedSpec):
    """Validated view of the plain dictionary returned by a provider's describe."""

    problem_id: str
    label: str
    provider_version: str
    parameters: tuple[VariableSpec, ...]
    observations: tuple[VariableSpec, ...] = ()
    capabilities: tuple[str, ...] = ()
    model_schema: Mapping[str, object] = field(default_factory=dict)
    data_schema: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate descriptions without importing provider or numerical packages."""
        super().__post_init__()
        for name in ("problem_id", "label", "provider_version"):
            _name(name, getattr(self, name))
        if not self.parameters:
            raise ValueError("At least one parameter variable is required")
        for name in ("parameters", "observations"):
            variables = getattr(self, name)
            if not isinstance(variables, (list, tuple)) or any(
                not isinstance(variable, VariableSpec) for variable in variables
            ):
                raise ValueError(f"{name} must contain VariableSpec values")
            _names(tuple(variable.name for variable in variables))
            object.__setattr__(self, name, tuple(variables))
        object.__setattr__(self, "capabilities", _names(self.capabilities))
        for name in ("model_schema", "data_schema"):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name)))

    @classmethod
    def from_dict(cls, description: Mapping[str, object]) -> ProviderMetadata:
        """Validate a provider dictionary without requiring it to import this class."""
        try:
            payload = dict(description)
            if "schema_version" not in payload:
                raise ValueError("Provider metadata requires schema_version")
            for name in ("parameters", "observations"):
                payload[name] = tuple(VariableSpec(**item) for item in payload.get(name, ()))
            return cls(**payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid provider metadata: {exc}") from exc


@runtime_checkable
class ProblemProvider(Protocol):
    """Duck-typed entry point; implementing packages need no platform/UI imports."""

    def describe(self) -> Mapping[str, object]:
        """Return JSON metadata matching ProviderMetadata.to_dict()."""
        ...

    def build(self, model: Mapping[str, object], data: Mapping[str, object]) -> Any:
        """Build a problem from resolved settings; never start a sampler."""
        ...


@runtime_checkable
class SimulationProvider(Protocol):
    """Optional synthetic observation generation, separate from inference."""

    def simulate(self, model: Mapping[str, object], rng: Any) -> Mapping[str, object]:
        """Generate data using only the explicitly supplied RNG."""
        ...


@runtime_checkable
class PriorProvider(Protocol):
    """Optional prior sampling that does not require observed data or a sampler."""

    def sample_prior(self, model: Mapping[str, object], rng: Any) -> Any:
        """Draw one parameter value using only the supplied random generator."""
        ...


@runtime_checkable
class PriorPredictiveProvider(Protocol):
    """Optional prior-predictive exploration for headless inspection."""

    def prior_predictive(
        self, model: Mapping[str, object], rng: Any, samples: int,
    ) -> Mapping[str, Any]:
        """Return prior parameter and simulated observation arrays."""
        ...


@runtime_checkable
class ReferenceProvider(Protocol):
    """Optional reference calculation; no sampler or GUI is required."""

    def reference(
        self, model: Mapping[str, object], data: Mapping[str, object],
        config: Mapping[str, object],
    ) -> Mapping[str, Any]:
        """Return reference arrays and accuracy metadata for explicit settings."""
        ...


@runtime_checkable
class PlotDataProvider(Protocol):
    """Optional domain computation, independent of Matplotlib and Shiny."""

    def plot_data(
        self, view: str, arrays: Mapping[str, Any], selection: Mapping[str, object],
    ) -> Mapping[str, Any]:
        """Build plot-ready data without changing source arrays or selection state."""
        ...


@runtime_checkable
class RuntimeProblem(Protocol):
    """Minimal density target; prior sampling and gradients are optional."""

    @property
    def dim(self) -> int:
        """Return the flattened parameter dimension."""
        ...

    def log_posterior(self, parameters: Any) -> float:
        """Evaluate an unnormalized target log density."""
        ...


@runtime_checkable
class GaussianPriorRuntime(RuntimeProblem, Protocol):
    """Operations used by the existing Gaussian-prior samplers."""

    def log_prior(self, parameters: Any) -> float:
        """Evaluate the unnormalized Gaussian prior density."""
        ...

    def log_likelihood(self, parameters: Any) -> float:
        """Evaluate the observation log likelihood."""
        ...

    def prior_mean(self) -> Any:
        """Return the Gaussian prior mean."""
        ...

    @property
    def L(self) -> Any:
        """Return the prior covariance Cholesky factor used by Gaussian kernels."""
        ...

    def sample_prior(self, rng: Any) -> Any:
        """Sample the prior using the supplied generator."""
        ...


@runtime_checkable
class ForwardProblem(Protocol):
    """Optional forward operation for prior-predictive and observation views."""

    def forward(self, parameters: Any) -> Any:
        """Map parameters to noise-free observable values."""
        ...


def validate_runtime(problem: object, capabilities: tuple[str, ...]) -> None:
    """Check known operation declarations, not their mathematical correctness."""
    _integer("problem dimension", getattr(problem, "dim", None), 1)
    _names(capabilities)
    operations = {
        "log_density": ("log_posterior",),
        "log_prior": ("log_prior",),
        "log_likelihood": ("log_likelihood",),
        "sample_prior": ("sample_prior",),
        "gaussian_prior": ("prior_mean", "sample_prior", "log_prior"),
        "gradient": ("grad_log_posterior",),
        "forward": ("forward",),
    }
    for capability in capabilities:
        for operation in operations.get(capability, ()):
            if not callable(getattr(problem, operation, None)):
                raise ValueError(f"Capability {capability!r} requires {operation}()")
        if capability == "gaussian_prior" and not hasattr(problem, "L"):
            raise ValueError("Capability 'gaussian_prior' requires L")


@dataclass(frozen=True)
class GaussianPriorAdapter:
    """Delegate existing GaussianPriorProblem operations without changing kernels."""

    problem: Any

    def __post_init__(self) -> None:
        """Require the legacy Gaussian-prior operation surface."""
        validate_runtime(self.problem, self.capabilities)

    @property
    def dim(self) -> int:
        """Return the unchanged legacy parameter dimension."""
        return self.problem.dim

    @property
    def L(self) -> Any:
        """Expose the Cholesky factor required by existing Gaussian-prior samplers."""
        return self.problem.L

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Declare only operations supplied by the existing base-class contract."""
        return ("gaussian_prior", "log_density", "log_likelihood", "log_prior", "sample_prior")

    def log_posterior(self, parameters: Any) -> float:
        """Delegate target evaluation to the existing problem."""
        return self.problem.log_posterior(parameters)

    def log_likelihood(self, parameters: Any) -> float:
        """Delegate likelihood evaluation to the existing problem."""
        return self.problem.log_likelihood(parameters)

    def log_prior(self, parameters: Any) -> float:
        """Delegate prior evaluation to the existing problem."""
        return self.problem.log_prior(parameters)

    def prior_mean(self) -> Any:
        """Delegate the Gaussian mean without copying or altering it."""
        return self.problem.prior_mean()

    def sample_prior(self, rng: Any) -> Any:
        """Delegate sampling with the caller's unchanged RNG stream."""
        return self.problem.sample_prior(rng)