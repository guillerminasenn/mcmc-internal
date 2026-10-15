"""Platform-owned wrappers around existing kernels; the kernels themselves are unchanged."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from .specs import ExperimentSpec


class CountingProblem:
    """Delegate to a runtime problem while counting likelihood evaluations."""

    def __init__(self, problem: Any):
        self._problem = problem
        self.likelihood_evaluations = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)

    def log_likelihood(self, parameters: Any) -> float:
        self.likelihood_evaluations += 1
        return self._problem.log_likelihood(parameters)


class PCNSampler:
    """One preconditioned Crank-Nicolson chain started from a prior draw."""

    sampler_id = "pcn"
    requires = frozenset({"gaussian_prior", "log_likelihood"})
    config_keys = frozenset({"rho"})

    @staticmethod
    def validate_config(config: Mapping[str, object]) -> None:
        unknown = set(config) - PCNSampler.config_keys
        if unknown:
            raise ValueError(f"Unknown pcn settings: {sorted(unknown)}")
        rho = config.get("rho")
        if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not 0 <= rho < 1:
            raise ValueError("pcn requires rho in [0, 1)")

    @staticmethod
    def planned_likelihood_evaluations(iterations: int) -> int:
        return 2 * iterations

    def run(self, problem: Any, spec: ExperimentSpec, sampler_index: int,
            chain_index: int) -> Mapping[str, Any]:
        import numpy as np
        from multiproposal.algorithms.pcn import pcn_chain

        sampler = spec.samplers[sampler_index]
        self.validate_config(sampler.config)
        sequence = np.random.SeedSequence(
            entropy=spec.seeds.experiment_seed,
            spawn_key=spec.seeds.sampler_spawn_key(chain_index, sampler_index),
        )
        rng = np.random.default_rng(sequence)
        counted = CountingProblem(problem)
        iterations = spec.warmup + spec.draws
        start = time.perf_counter()
        x0 = counted.sample_prior(rng)
        chain, acceptance = pcn_chain(
            x0, counted, rng, iterations, rho=float(sampler.config["rho"]),
            return_acceptance=True,
        )
        return {
            "arrays": {"parameters": chain[1:]},
            "costs": {
                "log_likelihood_evaluations": counted.likelihood_evaluations,
                "wall_seconds": time.perf_counter() - start,
                "acceptance_rate": float(acceptance),
            },
        }


SAMPLERS = {PCNSampler.sampler_id: PCNSampler}
