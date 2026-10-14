# README.md

# Multiproposal algorithms based on pCN-style proposal 

This repository implements multiproposal MCMC algorithms based on the pCN proposal that optionally allow for self-tuning of the proposal aggressiveness  using Elliptical Slice Sampling (ESS). The algorithms are used for Bayesian inference in models with Gaussian priors. Related papers are:
* pMCMC by Glatt-Holtz et al. (2024)
* pre-print "Multproposal Elliptical Slice Sampling" by Senn et al. (2026).
* Carigi et al. (2026, to appear)
---

## Key concepts
- All algorithms require: sampling from a Gaussian prior and evaluating a log-likelihood.

### New algorithms
- MPCN
- MTPCN
- MMESS (under depvelopment)

### MESS
- MESS proposes M candidate angles per subiteration and accepts one uniformly or based on a transition matrix.
- ESS is recovered by setting M = 1.


---

## Design principles

1. Algorithms are model-agnostic
   - Samplers only need a prior sampler and a log-likelihood.

2. Each problem constructs its own prior
   - Prior mean and covariance are derived in the problem class.
   - Problems considered: examples 1 (GP) and 2 (LR) in Murray et. al (2010), Semi-Blind Deconvolution from Senn et al. (2025, 2026), toy model for solute transport from Glatt-Holtz et al. (2024).

3. Gaussian priors only
   - All problems inherit from GaussianPriorProblem.
   - Gaussian sampling is performed via Cholesky decomposition.

---

## Repository structure

### Platform contracts (development)

The first versioned problem-provider, experiment, result, and execution contracts
are available under `multiproposal.platform`. See
[Platform Contracts V1](docs/platform_contracts.md) for ownership, serialization,
provider entry points, and focused verification. These are interfaces and data
contracts, not yet a worker service, registry, problem catalog, or visualization app.

### src/multiproposal/algorithms

- ess.py: ESS (equivalent to MESS with M = 1)
- mess.py: MESS with optional LP-based transition matrices
- utils.py: angle sampling, bracket logic, and helper routines

### src/multiproposal/problems

All problems implement log_likelihood(x) and provide prior construction.

- gp_regression.py: Gaussian process regression
- logistic_regression.py: Bayesian logistic regression
- sbd.py: semi-blind deconvolution inverse problem
- advection_diffusion.py: toy solute transport inverse problem

### src/multiproposal/kernels

- stationary.py: RBF and exponential kernels for GP priors

### src/multiproposal/data

- Synthetic data generators used by the notebooks and tests

---

## Notebooks

The notebooks/ directory reproduces the experiments and figures used in the paper.

- gp_regression.ipynb: baseline ESS/MESS comparison on GP regression.
- logistic_regression_ess_mess.ipynb: Bayesian logistic regression with multiple distance metrics.
- sbd_ess_mess.ipynb: semi-blind deconvolution.
- solute_transport_d10_mess_ellipse_iter10000.ipynb: solute transport toy problem at fixed dimension.
- solute_transport_dim_sweep_shared_draws.ipynb: solute transport dimension sweep with shared draws.
- solute_transport_dim_sweep_shared_draws_lp_compare.ipynb: LP-based transition comparison for the dimension sweep.

---

## Typical usage

```python
import numpy as np
from mess.data.gp_regression import generate_gp_regression_data
from mess.problems.gp_regression import GaussianProcessRegression
from mess.algorithms.ess import ess_step

data = generate_gp_regression_data(seed=0)
problem = GaussianProcessRegression(
    X=data["X"],
    y=data["y"],
    length_scale=1.0,
    noise_variance=0.09,
)

x = data["f_init"]
rng = np.random.default_rng(0)
for _ in range(1000):
    x, _, _ = ess_step(x, problem, rng)
```
