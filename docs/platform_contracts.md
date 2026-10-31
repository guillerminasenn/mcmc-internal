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

The first installed provider is registered by `mess-internal`:

```toml
[project.entry-points."bayesian.problem_providers"]
polar-twist = "mess.providers.polar_twist:provider"
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
- `PriorProvider.sample_prior(model, rng)` and
  `PriorPredictiveProvider.prior_predictive(model, rng, samples)` expose optional
  model exploration without requiring observed data or a sampler.
- `ForwardProblem.forward(parameters)` evaluates noise-free observables.
- `ReferenceProvider.reference(model, data, config)` returns reference arrays
  and accuracy metadata for explicit numerical settings.
- `PlotDataProvider.plot_data(view, arrays, selection)` computes domain plot data
  without rendering, writing samples, or mutating selection state.

Phase 4 registers only polar twist. Its implementation wraps the existing
`mess.problems.polar_twist` scientific code and imports neither BayesLab nor this
platform package. Provider-level prior sampling and prior-predictive generation
do not require observed data. Other catalog problems remain unimplemented until
the polar-twist execution and visualization path is validated.

The minimal inference target, `RuntimeProblem`, exposes `dim` and
`log_posterior(parameters)`. Gaussian priors, gradients, and generative models
are not universally required. Known numerical capability names are
`log_density`, `log_prior`, `log_likelihood`, `sample_prior`, `gaussian_prior`
(requiring `prior_mean`, `sample_prior`, `log_prior`, and the Cholesky factor
`L` used by current Gaussian kernels), `gradient` (requiring
`grad_log_posterior`), and `forward`.
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
## Control Plane (Phase 5, first increment)

`mcmc-platform` (`multiproposal.platform.cli`) provides `providers`, `validate`,
`dry-run`, `submit`, `status`, `cancel`, and `list`. `validate` and `dry-run` write
nothing. There is no default run root: pass `--run-root` or set
`MCMC_PLATFORM_RUN_ROOT`. A root inside a Git repository is rejected unless it is
ignored (or `--allow-repository-run-root` is given for platform development tests).
Experiment specs belong to the problem owner; generated runs belong outside
source control.

### Workbench storage amendment (2026-10-19; verified correction)

The local BayesLab workbench uses an explicitly configured
`bayeslab/experiments/<bip>/runs/` as its per-problem registry/artifact root,
with populated inputs in sibling `configurations/`, optional observations in
`data/`, and separate exports in `reports/`. The entire BayesLab `/experiments/`
tree must be ignored before runtime writes. Scientific providers/recipes remain
BIPs-owned; execution, SQLite state, bundle layout, and NPZ chain storage remain
platform-owned. No global registry or new chain store is introduced.

Focused tests verify the existing repository-output guard accepts ignored
workspace storage and continues to reject unignored repository paths. No guard
change or development override was needed. The platform's explicit external-root
CLI support remains unchanged; the BayesLab UI confines its implemented path
operations to its launch-configured experiment root, including pasted paths,
symlinks, artifacts, and submission destinations. Browser acceptance covers this
correction; scientific/durability gates and later UI stages remain open.

`analysis.load_bundle_manifest(path)` accepts a completed execution folder or
manifest without a registry, validates its resolved configuration identity/hash,
artifact paths/sizes/checksums, and schema/array agreement, and uses its actual
directory rather than archived absolute paths. `bundle_report_data(path)` and
`compare_bundles(paths)` reuse platform report calculations without registration.
`registry.read_registry_records(root)` reads an existing SQLite registry without
initialization, reconciliation, or writes. It reports recorded snapshots only,
not live-worker certification; standalone bundles have no live worker evidence.

Existing BIPs smoke and historical bundles stay unchanged at their original
paths. Any separately requested copy into the workspace must verify source and
destination checksums, reject conflicts, load relative to the copied directory,
and neither rewrite sealed manifests nor silently register copies. Active runs
stay at their original roots. BayesLab's explicit `init-workspace` creates empty
storage; `import-bundle` performs an optional verified copy and rejects conflicts
and symbolic links. Only synthetic temporary bundles were imported in tests.
No historical move/import, submission enablement, or sampling was performed.

`submit` validates, persists `spec.json` (read-only, checksummed in the registry),
inserts a `pending` row, and only then launches `multiproposal.platform.worker` in a
new session, so closing the submitting client does not stop it. The worker records
PID, host, and process start time, heartbeats, writes read-only `.npz` arrays plus
`costs.json`, writes `manifest.json` atomically, and then marks `completed`.
`status`/`list` reconcile: a completion manifest wins; a dead or recycled PID (or a
silent remote worker) becomes `stale`, never `running`; `cancelling` becomes
`cancelled` once the worker is gone. `cancel` sends SIGTERM only after verifying
PID, host, and start time. Resume is not supported.

Results are `artifacts/sampler_<i>.npz` with `parameters` shaped
`(chain, draw, *parameter_shape)`; draws include warmup, as declared in the manifest
`ResultSchema` (`warmup_included=True`). The only registered sampler is `pcn`; its
chain `c` uses `SeedSequence(experiment_seed, spawn_key=(0, c, sampler_index))`.
`execution.chain_delay_seconds` is a testing hook that slows a worker.
Not yet implemented: diagnostics, comparison, report data, legacy registration,
other samplers, multi-axis parameters.

## Applied Analysis Selection (2026-10-31)

This amendment supersedes historical "not implemented" statements above for
analysis/reporting; it does not certify sampling, reference accuracy, or durability.

`analysis.select_chain_array(values, cutoff=None, warmup_draws=0,
warmup_included=False, chains=None)` returns a frozen `DrawSelection` with a
read-only array. Cutoff `B` is the first included stored draw: retain
`values[:, B:, ...]`, once, with integer `0 <= B < N`. Booleans, nonintegers,
negative and out-of-range cutoffs are rejected. The default is recorded warmup
when that prefix is stored, otherwise zero. Missing warmup cannot be recovered.
Intrinsic axes and repeated states are preserved; selected chain IDs remain
the original archive indices, including when reordered.

Selection metadata records applied cutoff, stored/retained draw counts, retained
sample count, original chain IDs, recorded warmup, and whether warmup was stored.
It is analysis metadata, not a replacement experiment spec or manifest. The pCN
warmup prefix is part of its fixed-rho trajectory, not a separate tuning chain;
zero refers to the first stored state, not an omitted initial state.

`analysis.summarize_selection(selection)` computes mean/std, component and total
MSJD, and component diagnostics from the same retained values.
`bundle_report_data` and `report_data` accept optional keyword `cutoff`, `chains`,
`sampler`, and `variable`; defaults preserve recorded-warmup behavior and existing
report keys. `diagnostics.diagnose_execution` accepts cutoff/chains and reuses the
validated report path. No registration or artifact writes are added.

Means/stds, MSJD, and ArviZ metrics are finite scalars or `None`, with additive
availability reasons. `summarize_chain_array` retains its five metric keys and
coarse `status`/`warnings`: `unavailable` means at least one metric is unavailable,
not necessarily all. Its `availability` mapping identifies each reason. R-hat
requires two chains; ESS/MCSE support one chain with at least four draws. Short,
nonfinite, or within-chain constant inputs are conservatively unavailable for
ArviZ metrics, even when ArviZ returns a nominal constant-chain ESS. Individual
nonfinite results do not suppress other supported metrics. Comparisons propagate
unavailable moments as unavailable deltas.

`diagnostics.summarize_msjd` averages squared successive retained-state distances
within chains, including zero/rejection repeats and never joining chains. Total
MSJD sums over intrinsic coordinates and is in squared-coordinate units. Fewer
than two states makes it unavailable; constant trajectories can have zero MSJD.
Neither MSJD nor a finite diagnostic certifies convergence.

BayesLab's read-only reader and inspection orchestration reuse this selection.
Posterior density, summaries, diagnostics, autocorrelation and future rank inputs
share retained draws; traces keep all stored draws and use an independent display
window. Caller-owned previous inspection payloads can reuse prior, observations
and reference data when resolved-spec checksum and rendering parameters match.
There is no global cache. Editable recomputation UI, persistence and rank plots
remain later stages.

### Derived source provenance amendment (2026-10-31)

`analysis.bundle_analysis_provenance(path)` reuses the validated standalone-bundle
loader and returns source manifest/resolved-spec/artifact digests plus the original
manifest provenance. BayesLab captures this snapshot for inspection and checks it
again during computation and saving, rejecting changed sources rather than
attaching old metrics to new artifact digests. Derived generation storage and
rendering remain BayesLab-owned; no source manifest, chain or registry is rewritten.
