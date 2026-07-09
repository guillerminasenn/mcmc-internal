# Solute Transport Convergence Experiment (Config 2)

This document is a precise replication spec for the convergence workflow implemented in:
- notebooks/05_toy_solute_transport_mpcn_convergence.ipynb
- jobs/solute_transport/mpcn_pcn_convergence_h57eaaa0da6e8_data_h4afe80f670cc/run.py
- jobs/solute_transport/mpcn_pcn_convergence_h57eaaa0da6e8_data_h4afe80f670cc/launch_independent_workers.sh

It also includes the requested next variant (MESS/ESS) as an implementation target.

## 1. Objective and experiment definition

The notebook compares warm-up/convergence behavior for solute transport posterior sampling in d=40 (config 2), using running trace diagnostics and running-MSE diagnostics.

Two estimator types are compared:
1) Multiproposal chain estimator (mPCN in current notebook).
2) Independent-chain estimator built from single-proposal chains (pCN in current notebook), including an EP-style grouped estimator.

In this context, one "replicate" means:
- either one multiproposal chain run,
- or one group of M independent single-proposal chains that are aggregated together at each iteration (EP-style).

## 2. Data-generation configuration (must match exactly)

Use these exact data settings:
- seed_data = 0
- config_id = 2
- d = 40
- obs_highest_freq = 12
- obs_bandwidth = 7
- kappa = 0.02
- sigma = 0.5
- alpha = 3.0
- gamma = 2.0
- tau2 = 2.0
- a_mode = "nearest_neighbor"
- use_prior_A = true
- shared_draws_seed = 0
- obs_config = "central_modes"

Important implementation detail:
- Because use_prior_A = true, shared draws are built with a_mode = "prior".

Observation indices are:
- start = max(0, obs_highest_freq - obs_bandwidth + 1) = 6
- obs_indices = [6, 7, 8, 9, 10, 11, 12]

Parameter dimension is:
- m = d * (d - 1) / 2 = 780

## 3. Deterministic IDs and paths

IDs are deterministic hashes of JSON payloads.

### 3.1 data_id payload
- seed_data, kappa, sigma, alpha, gamma, tau2, a_mode, use_prior_A, shared_draws_seed,
  obs_highest_freq, obs_bandwidth, obs_config, d

For this experiment:
- data_id = data_h4afe80f670cc

### 3.2 run_id payloads used by current workflow

Run A (main notebook run config):
- n_iters = 100000
- rho = 0.9
- P = 100
- seed_mcmc = 202
- burn_in = 0
- config_id = 2
- run_id = mpcn_pcn_convergence_h57eaaa0da6e8

Run B (EP support run used by MSE sections):
- n_iters = 20000
- rho = 0.9
- P = 100
- seed_mcmc = 202
- burn_in = 0
- config_id = 2
- run_id = mpcn_pcn_convergence_hc6f2f62899e7

Directory layout:
- estimations/solute_transport/<data_id>/fixed/<run_id>/...
- reports/solute_transport/<data_id>/fixed/<run_id>/...

## 4. Chain-generation protocol in the job script

The script at jobs/solute_transport/mpcn_pcn_convergence_h57eaaa0da6e8_data_h4afe80f670cc/run.py currently computes run_id from n_iters=20000, which resolves to Run B (hc6f2f62899e7). The folder name contains h57, but the output run_id is determined by the payload inside run.py.

### 4.1 Shared starting points
- max_num_chains = max(pcn_count, mpcn_count)
- rng_starts = np.random.default_rng(seed_mcmc)
- all_start_points = sample_prior_points(rng_starts, prior_diag, max_num_chains)
- pcn_start_points = all_start_points[:pcn_count]
- mpcn_start_points = all_start_points[:mpcn_count]

### 4.2 pCN chain files
- Base folder: chains/independent_chains
- Path pattern (standard): pcn_independent_rho<rho_tag>_seed<seed_mcmc>_chain<idx>.npz
- Path pattern (replicate mode in this script): replicate_<idx>/pcn_independent_rho<rho_tag>_seed<seed_mcmc>_chain<idx>.npz
- Per-chain RNG seed: seed_mcmc + 2000 + idx
- Checkpoints:
  - <stem>.progress.json
  - <stem>_partial.npz

### 4.3 mPCN chain files
- Base folder: chains/mpcn_independent
- Path pattern: mpcn_P<P>_rho<rho_tag>_seed<seed>_chain<idx>.npz
- Per-chain RNG seed: seed_mcmc + 5000 + idx
- Checkpoints:
  - <stem>.progress.json
  - <stem>_partial.npz

### 4.4 Indices and metrics
- pCN index: chains/independent_chains/pcn_independent_rho<rho_tag>_seed<seed_mcmc>_index.json
- mPCN index: chains/mpcn_independent/mpcn_P<P>_rho<rho_tag>_seed<seed_mcmc>_index.json
- Per-chain metrics JSON:
  - diagnostics/independent_chains/<stem>_metrics.json

## 5. Notebook workflow (exact behavior)

The notebook mostly loads cached outputs and plots diagnostics.

### 5.1 Data and run setup
- Rebuilds shared-draws problem deterministically from config_id = 2.
- Creates/uses:
  - estimations/solute_transport/data_h4afe80f670cc/fixed/mpcn_pcn_convergence_h57eaaa0da6e8
  - reports/solute_transport/data_h4afe80f670cc/fixed/mpcn_pcn_convergence_h57eaaa0da6e8

### 5.2 Chain loading for diagnostics
- Standard pCN chains loaded from Run A:
  - .../h57.../chains/independent_chains
- Standard mPCN chains loaded from Run A:
  - .../h57.../chains/mpcn_independent
- EP pCN replicate chains loaded from Run B:
  - .../hc6.../chains/independent_chains/replicate_*/...
- For MSE convenience, the notebook trims/copies up to 50 mPCN chains from Run A into Run B:
  - source: .../h57.../chains/mpcn_independent
  - target: .../hc6.../chains/mpcn_independent
  - n_keep = 20000

### 5.3 Running-MSE setup
- nr_replicates = 50
- max_iter_mse = 1000
- effective_P = min(P, len(pcn_chains_for_ep))
- EP groups are contiguous blocks of size effective_P from pcn_chains_for_ep.
- Observable targets are cached under Run A metrics, from post-burnin mPCN means.
- Running-MSE curves are cached under Run A metrics/mse.

## 6. Commands to reproduce script-based generation

From repo root:

1) Activate environment:
- source .multip-env/bin/activate

2) Launch workers:
- bash jobs/solute_transport/mpcn_pcn_convergence_h57eaaa0da6e8_data_h4afe80f670cc/launch_independent_workers.sh
- Optional explicit args:
  - bash jobs/solute_transport/mpcn_pcn_convergence_h57eaaa0da6e8_data_h4afe80f670cc/launch_independent_workers.sh <grid_count> <pcn_count> <mpcn_count>

3) Monitor logs:
- tail -f jobs/solute_transport/mpcn_pcn_convergence_h57eaaa0da6e8_data_h4afe80f670cc/logs/independent_worker_0.log

## 7. Portability rules for reproducing outside this machine

To reproduce in another repo clone or machine without ambiguity:
- Resolve repo root via MULTIPROPOSAL_RUN_ROOT or pyproject.toml walk-up.
- Avoid hard-coded absolute roots such as /home/senng/multiproposal-internal in notebook cells.
- Keep the same seeds and payload fields for data_id/run_id determinism.
- Keep the same file naming conventions and index JSON structure.
- Keep checkpoint sidecar files and metrics JSON format.

## 8. Requested next variant (to implement after approval)

Target replacement requested by user:
- Replace mPCN with MESS using M=10 proposals.
- Replace pCN with MESS using M=1 (equivalent to ESS).
- Use 20 replicates.
- Preserve the logic/structure of the current convergence workflow.

Precise replicate semantics for this variant:
- Multiproposal side: 20 independent MESS(M=10) chains.
- Independent-chain side (EP-style): for each replicate r, run M=10 independent ESS chains and aggregate them as one EP group.
- Total ESS chains needed for EP groups: 20 * 10 = 200 chains.

Keep unchanged unless explicitly required:
- Data-generation config in Section 2.
- Directory conventions and cache layout style.
- Diagnostics/observable definitions and running-MSE formulas.

Note on implementation API:
- src/multiproposal/algorithms/mess.py provides mess_step(M=...).
- src/multiproposal/algorithms/ess.py defines ess_step as mess_step with M=1.
- There is no existing mess_chain helper in this repo; chain loops must be implemented analogously to pcn_chain logic.

## 9. Implemented artifacts in this repo

The following artifacts now exist and follow this spec:

- Notebook copy for analysis:
  - notebooks/05_toy_solute_transport_mess_ess_convergence.ipynb
- Job folder for background execution:
  - jobs/solute_transport/mess_ess_convergence_h1b486aada8cc_data_h4afe80f670cc/
  - jobs/solute_transport/mess_ess_convergence_h1b486aada8cc_data_h4afe80f670cc/run.py
  - jobs/solute_transport/mess_ess_convergence_h1b486aada8cc_data_h4afe80f670cc/launch_independent_workers.sh

Expected run/data IDs for this variant:
- data_id: data_h4afe80f670cc
- run_id: mess_ess_convergence_h1b486aada8cc

Launch command used:
- bash jobs/solute_transport/mess_ess_convergence_h1b486aada8cc_data_h4afe80f670cc/launch_independent_workers.sh 8 20 10

## 10. Current production setting for convergence runs

Use the fast-convergence setting requested by the user:
- n_iters = 500
- nr_replicates = 20
- M = 10

This produces:
- 20 independent MESS chains (M=10)
- 200 ESS chains grouped into 20 EP groups of size 10

Recommended launch command:
- bash jobs/solute_transport/mess_ess_convergence_h1b486aada8cc_data_h4afe80f670cc/launch_independent_workers.sh 8 20 10 500

The runner default is now n_iters=500, so omitting the fourth argument also uses 500 iterations.
