# Run Output Policy

Goal: outputs are stored under run-specific folders that uniquely identify dataset generation and algorithm configuration. This enables cache reuse and prevents accidental reuse when hyperparameters change.

## Required practice
- Always build a run-specific directory using a deterministic hash of:
  - dataset name
  - data generation parameters (noise, priors, seeds, etc.)
  - algorithm configuration (method, P, rho grid, M, n_iters, burn_in, etc.)
  - sweep configuration (if applicable)
- All new functions should include a minimal, sufficient docstring that explains purpose and key inputs.
- Never write outputs under the notebooks folder. All outputs must live under the repo root
  in estimations/ and reports/ (or a user-specified root via MULTIPROPOSAL_RUN_ROOT).
- Store samples, metrics, diagnostics, and derived tables under the run-specific estimations directory.
- Store figures under the matching run-specific reports directory.
- Write the full configuration to config.json in the estimations directory.
- Reuse cached outputs when present: load existing samples/metrics from the run directory and skip re-running chains.

## Job scripts
- Store long-running sweep scripts under jobs/<dataset>/<run_id>_data_<data_id>/run.py.
- Keep notebooks focused on loading cached outputs and plotting, not running chains.
- Each job script should mirror the run_id/data_id used for estimations and reports.

## How to run jobs (commands)
- From the repo root, launch independent-chain workers with:
  - bash jobs/<dataset>/<run_id>_data_<data_id>/launch_independent_workers.sh
- Optional arguments:
  - bash jobs/<dataset>/<run_id>_data_<data_id>/launch_independent_workers.sh <grid_count> <pcn_count> <mpcn_count>
- Example (12 workers, 100 pCN, 100 mPCN):
  - bash jobs/solute_transport/mpcn_pcn_convergence_h57eaaa0da6e8_data_h4afe80f670cc/launch_independent_workers.sh 12 100 100
- Check logs while running:
  - tail -f jobs/<dataset>/<run_id>_data_<data_id>/logs/independent_worker_0.log

## Grid sharding (P, rho sweeps)
- Prefer launching multiple job scripts with disjoint (P, rho) subsets over nested process pools.
- Use deterministic grid slicing (for example, grid_index/grid_count) so reruns are reproducible.
- Keep inner mPCN parallelization enabled; avoid additional per-grid process pools to prevent
  over-subscription unless explicitly managed.

Plain-language guidance:
- Run several copies of the sweep script, each responsible for a different slice of (P, rho).
- Each copy runs chains one-at-a-time (serial likelihood), but the copies run in parallel on
  different CPU cores.
- This is easier to manage and to stop than a single job that tries to use all cores internally.

## Directory structure
Use a two-level identifier split:
- data_id: hash of a stable, minimal dataset-generation config (data hyperparameters,
  observation config, and data seeds). This should be identical across notebooks
  that use the same data, even if their sweep dimensions differ.
- run_id: hash of the algorithm/MCMC configuration (including sweep settings when applicable).

Required layout:
- estimations/<dataset>/<data_id>/<sweep|fixed>/<run_id>/...
- reports/<dataset>/<data_id>/<sweep|fixed>/<run_id>/...

## Figures
- Apply the publication style from reports/figure_style.py for all plots.

## Naming conventions
- Save chain outputs as <method>_P{P}_rho{rho}_seed{seed}.npz (or <method>_rho{rho}_seed{seed}.npz for pCN).
- Save metrics next to the chain file as <method>_P{P}_rho{rho}_seed{seed}_metrics.json.
- Store algorithm-specific diagnostics (e.g., mPCN diagnostics) alongside the chain files.

## Independent chain sweeps
- For independent-chain variants that reuse a base run_id, store outputs under a subfolder of
  the main chains directory (for example, estimations/.../chains/independent_chains).
- Use a stable naming scheme for each chain such as
  pcn_independent_rho{rho}_seed{seed}_chain{idx}.npz, where {idx} is the chain index.
- Store an index file per rho (for example, pcn_independent_rho{rho}_seed{seed}_index.json)
  that lists the chain files and metadata; notebooks should discover chains via the index.
- Treat P as the number of independent chains to aggregate, not as part of the filename.
- Do not include the number of chains requested in the data_id or run_id hash.
- When loading an index, validate only the algorithmic metadata; allow chain counts to change
  so additional chains can be appended to an existing run folder.

## Variants and sub-experiments
- For secondary experiments that share the same base run configuration (e.g., random-start sweeps), save under a subfolder of the main run directory (for example, estimations_dir/random_start).

## Parallelization policy
- Parallelization is an execution detail, not a statistical configuration. Do not include it in the run_id hash unless it changes the algorithmic output.
- Record parallel settings in config.json under a separate execution section when helpful (for example, n_jobs, backend, per-proposal parallelization).
- Prefer deterministic ordering of worker outputs (executor.map preserves input order) so results are stable across runs.
- Use threads when the problem object is not picklable. Use processes only if the problem and inputs can be serialized.
- Proposal-level parallelization (parallelize_props=True) changes the RNG stream relative to serial execution, because each proposal is generated from a worker-local SeedSequence. Treat this as an algorithmic variant when comparing mixing.

## Checkpoints and progress
- For long sweeps, write partial chains every fixed number of iterations (for example, every 10,000).
- Store partial chains alongside the final chain with a _partial suffix.
- Write a progress JSON file with completed_iters, total_iters, and timestamp so status can be read
  without opening the chain file.
- Keep checkpoints as execution metadata; do not include them in the run_id hash.
- Independent-chain jobs should write per-chain progress JSON files next to each chain file
  (for example, <stem>.progress.json) and update them at the same checkpoint interval.
- Independent-chain jobs should write per-chain partial chains in the same folder as the final chain
  (for example, <stem>_partial.npz).
- Store per-chain metrics/diagnostics in a diagnostics subfolder under the run directory
  (for example, estimations/.../diagnostics/independent_chains/<stem>_metrics.json).

## Parallelization implementation pattern
- Keep the vectorized path as the default and add an opt-in switch (for example, parallelize_props) to enable proposal-level parallelism.
- Use SeedSequence to spawn per-proposal RNGs and avoid sharing RNG state across workers.
- Structure worker tasks to return both proposal and log-likelihood so the caller can assemble candidates and weights without extra passes.
- Example pattern:
  - Main thread: draw a master seed, spawn n_props SeedSequence objects.
  - Worker: sample z, build proposal, evaluate log-likelihood.
  - Main thread: collect props + log_likelihoods, then proceed with weighting and selection.

## Current recommended parallel setup (mPCN)
- Default to process-based log-likelihood evaluation with chunking.
- Keep proposal generation in the main process (vectorized).
- Use chunks sized at roughly ceil(P / n_jobs), or override explicitly.
- Disable proposal-level parallelization when using chunked log-likelihoods.
- Record the execution settings (backend, n_jobs, llh_chunk_size) in config.json and, where helpful, in chain metadata.

## Suggested helper
When building paths, resolve repo_root from MULTIPROPOSAL_RUN_ROOT or by walking up to
the repo's pyproject.toml. Do not assume Path.cwd() points at the repo root when
executing notebooks.
