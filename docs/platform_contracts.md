# Platform Contracts V1

Status: Phase 3, step 13 implemented. These contracts do not launch workers,
discover providers, load historical arrays, compute diagnostics, or start a UI.
Those implementations remain gated by the coordination plan. Scientific kernels
and historical entry points are unchanged.

## Ownership

| Boundary | Implementation |
|---|---|
| Resolved problem, sampler, seed, and experiment specs | `multiproposal.platform.specs` |
| Provider metadata, runtime protocols, legacy adapter | `multiproposal.platform.problems` |
| Execution snapshots and client/sampler protocols | `multiproposal.platform.execution` |
| Artifact records, posterior schemas, manifests, reader protocol | `multiproposal.platform.results` |

All contracts are also exported by `multiproposal.platform`. Importing that
package uses only the Python standard library. BIP packages implement ordinary
classes returning dictionaries; they need not import this package or BayesLab.
Generic execution and diagnostics belong to the platform. Domain calculations
belong to problem packages. Headless Matplotlib builders and UI state belong to
BayesLab, with no Shiny imports in reusable rendering code.

## Provider Boundary

The shared Python entry-point group is **`bayesian.problem_providers`**. Each entry
point will load an object with these operations, not a sampler or a factory:

- `describe()` returns a plain dictionary accepted by
  `ProviderMetadata.from_dict()`, including `schema_version: 1`, `problem_id`,
  `label`, `provider_version`, parameter variables, optional observation variables,
  capabilities, and model/data configuration schemas.
- `build(model, data)` accepts resolved settings and returns a numerical problem.
  It must not launch a sampler, generate unrecorded randomness, or discover data
  through the current working directory.

Illustrative registration only; no provider is registered in this stage:

```toml
[project.entry-points."bayesian.problem_providers"]
example = "example_bip.providers:provider"
```

`VariableSpec` records `name`, ordered `dims`, `shape`, optional scalar coordinate
labels, and units. Provider parameter shapes describe intrinsic axes; posterior
result shapes additionally include leading chain/draw axes. A matrix parameter
is not assumed to be a two-element vector. Metadata normalizes capability order
but preserves dimension and variable order. Configuration schemas are stored as
JSON data; semantic model validation belongs to the provider/resolver, not a
new general-purpose schema framework in this stage.

Optional operations are independent structural protocols:

- `SimulationProvider.simulate(model, rng)` produces observations/truth using
  the supplied generator.
- `ForwardProblem.forward(parameters)` evaluates noise-free observables.
- `ReferenceProvider.reference(model, data, config)` returns reference arrays
  and accuracy metadata for explicit numerical settings.
- `PlotDataProvider.plot_data(view, arrays, selection)` computes domain plot data
  without rendering, writing samples, or mutating selection state.

The minimal inference target, `RuntimeProblem`, exposes `dim` and
`log_posterior(parameters)`. Gaussian priors, gradients, and generative models
are not universally required. Known numerical capability names are
`log_density`, `log_prior`, `log_likelihood`, `sample_prior`, `gaussian_prior`,
`gradient` (requiring `grad_log_posterior`), and `forward`.
`validate_runtime()` checks declared known operations; it cannot prove Gaussianity,
correct derivatives, or mathematical invariance. Extension capability names
require their sampler adapter's validation.

`GaussianPriorAdapter` delegates the existing base-class methods and RNG directly.
It does not rewrite algorithms or infer Gaussian observation noise. A provider's
availability is distinct from its scientific identity: polar twist, multiwell,
solute transport, and BSBD are all problems; actual provider integration begins
in Phase 4, with BSBD's external package integration in Phase 8.

## Resolved Experiments

The following is a contract fixture, not an executable sampler registration:

```python
from multiproposal.platform import (
    ExperimentSpec, ProblemSpec, SamplerSpec, SeedSpec,
)

spec = ExperimentSpec(
    problem=ProblemSpec(
        problem_id="fixture", provider="fixture-provider", provider_version="1",
        model={"prior_mean": [0.0, 0.0]},
        data={"observations": [1.0, 2.0]},
        capabilities=("log_likelihood", "sample_prior", "gaussian_prior"),
    ),
    samplers=(SamplerSpec(
        sampler_id="fixture-sampler", config={"M": 4},
        required_capabilities=("log_likelihood", "sample_prior", "gaussian_prior"),
    ),),
    seeds=SeedSpec(data_seed=202, experiment_seed=0),
    chains=4, draws=20, warmup=5,
    execution={"workers": 2},
)
restored = ExperimentSpec.from_json(spec.to_json())
assert restored.to_json() == spec.to_json()
assert restored.run_id == spec.run_id
```

Specs are frozen and detach nested configuration from caller-owned containers.
`to_dict()` returns a mutable JSON copy. Serialization rejects nonfinite numbers,
non-string object keys, unsupported values, duplicate JSON fields, missing version
markers, unknown fields, and unsupported versions.

`canonical-json-sha256-v1` uses UTF-8, sorted object keys, compact JSON separators,
and finite JSON numbers. It is a local versioned convention, not a claim to
implement RFC 8785; for example, integer and floating representations are distinct.
The canonical bytes persisted by CLI and UI must be identical.

Use `ExperimentSpec.data_id` and `ExperimentSpec.run_id` for new experiments:

- `data_sha256_...` includes model/data settings, provider identity/version, and
  data entropy. `ProblemSpec.data_id` is the model/data component before the
  experiment's separate data seed is included.
- `run_sha256_...` additionally includes samplers, chain/draw/warmup counts,
  seeds, observables, and diagnostics/comparison policies. Only `execution`
  is excluded from this scientific ID.
- Worker placement, logs, output roots, and scheduling belong in `execution`.
  Any parallel mode that changes numerical output belongs in the sampler config,
  not in execution-only options.
- The exact resolved-spec checksum includes execution options. A separate
  `execution_id` identifies each submission/retry of the same scientific run.

New chain counts participate in identity because new runs are immutable. This is
deliberately distinct from historical appendable independent-chain indexes.
Legacy SHA-1/SHA-256 IDs, paths, and artifact bytes must not be recomputed or renamed.
The new manifest contract does not silently migrate an unversioned historical run.

## Seed Ownership

`SeedSpec` records independent data and experiment entropy. Data generation uses
the data seed. A sampler/chain stream uses the experiment seed and the tuple from
`sampler_spawn_key(chain_index, sampler_index)`: `(0, chain_index, sampler_index)`.
The worker implementation will pass that tuple as NumPy `SeedSequence.spawn_key`.
Worker IDs, shard count, launch order, and retry placement are not seed inputs.
No SeedSequence tree, resumable algorithm state, or worker is executed by the
contracts themselves. Record NumPy/bit-generator versions when execution is added.

## Results And Provenance

`ResultSchema` describes posterior arrays with leading dimensions `chain`,
`draw`, followed by named intrinsic axes. All variables share chain/draw sizes.
Warmup metadata is explicit: `ExperimentSpec.draws` counts retained posterior
draws; `warmup` counts additional adaptation/discarded draws. If a stored array
includes warmup, its draw axis is `warmup + draws`; otherwise it is `draws`.
`ResultSchema.validate_arrays()` checks variable names and shapes without changing
sample values. It does not calculate convergence diagnostics.

`ArtifactRecord` stores normalized relative POSIX paths, SHA-256, byte size,
and kind. It validates descriptor syntax, not file contents or symlink safety.
Phase 4 readers must resolve paths safely under the configured run root, verify
file checksums, expose read-only arrays, and never rewrite source artifacts.
`ResultReader.read(run_root, artifacts, schema)` is the boundary for that work;
NumPy/xarray/ArviZ adapters and actual disk readers are not implemented here.

`RunManifest` joins an experiment to its execution record, artifacts, schemas,
code revisions, and environment. Result keys are `sampler_0`, `sampler_1`, etc.,
matching the resolved sampler ordering. Completed manifests require artifacts,
all samplers' result schemas, matching counts/warmup, and nonempty code/environment
provenance. Exact code commits/tree state and package/lock information must be
filled by the worker; this stage does not collect them. Canonical spec and
environment SHA-256 values are included in the serialized manifest.

## Execution Boundary

`ExecutionClient` defines `validate`, `dry_run`, `submit`, `status`, and `cancel`.
It is a protocol, not a working service or CLI. `SamplerAdapter.run()` is the
platform-owned kernel wrapper for a specified sampler/chain index; it will return
that chain's arrays and cost counters for worker assembly into labeled results.

`RunRecord` snapshots pending/running/completed/failed/cancelling/cancelled/stale
status, execution/scientific IDs, configured paths, timezone-aware timestamps,
progress, worker identity, logs, and errors. Terminal states need an end time;
stale is not a synonym for completed or failed. A PID alone proves no liveness.
Heartbeat reconciliation, legal transitions, durable submission, cancellation,
SQLite persistence, and detached workers remain Phase 5 responsibilities.

## Verification

From the `mcmc-internal` repository root, with Python 3.12:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -p test_platform_contracts.py -v
```

The standard-library tests need no packages; the real Gaussian adapter test
requires NumPy/SciPy and explicitly skips if they are unavailable. All 24 tests
passed without skips in an isolated Python 3.12 environment with NumPy 2.5.4 and
SciPy 1.18.1. This does not certify the LP transition suite or a posterior run.