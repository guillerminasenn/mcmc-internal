"""Adaptive multi-proposal pCN (aMPCN).

This variant uses the same mPCN transition as `mpcn_step`, but adapts `rho`
during a warmup window to target a desired acceptance rate. After warmup,
`rho` is frozen and the chain proceeds as standard mPCN.
"""

import numpy as np

from .mpcn import mpcn_step


def _to_unconstrained(rho, rho_min, rho_max):
    scaled = (rho - rho_min) / (rho_max - rho_min)
    scaled = np.clip(scaled, 1e-12, 1.0 - 1e-12)
    return np.log(scaled) - np.log1p(-scaled)


def _from_unconstrained(z, rho_min, rho_max):
    sig = 1.0 / (1.0 + np.exp(-z))
    return rho_min + (rho_max - rho_min) * sig


def ampcn_chain(
    x0,
    problem,
    rng,
    n_iters,
    rho_init=0.1,
    n_props=10,
    adapt_iters=1000,
    target_accept=0.35,
    step_scale=0.1,
    rho_bounds=(1e-3, 0.999),
    return_indices=False,
):
    """Run adaptive mPCN with Robbins-Monro adaptation for `rho`.

    Parameters
    ----------
    x0 : ndarray
        Initial state.
    problem : object
        Problem with `dim`, `prior_mean`, `L`, and `log_likelihood` interface.
    rng : np.random.Generator
        Random generator.
    n_iters : int
        Number of MCMC iterations.
    rho_init : float
        Initial mPCN correlation parameter.
    n_props : int
        Number of proposals per mPCN step.
    adapt_iters : int
        Number of initial iterations with adaptation enabled.
    target_accept : float
        Target acceptance for indicator `(accepted_index != 0)`.
    step_scale : float
        Global learning-rate scale for Robbins-Monro updates.
    rho_bounds : tuple[float, float]
        Lower and upper bounds used in the constrained transform.
    return_indices : bool
        If True, return accepted candidate indices.
    """
    rho_min, rho_max = rho_bounds
    if not (0.0 < rho_min < rho_max < 1.0):
        raise ValueError("rho_bounds must satisfy 0 < rho_min < rho_max < 1")

    rho = float(np.clip(rho_init, rho_min, rho_max))
    z = _to_unconstrained(rho, rho_min, rho_max)

    chain = np.zeros((n_iters + 1, problem.dim), dtype=float)
    chain[0] = x0
    x = x0
    accepted_index = np.zeros(n_iters, dtype=int) if return_indices else None
    rho_history = np.zeros(n_iters, dtype=float)

    for t in range(n_iters):
        x_new, idx = mpcn_step(
            x,
            problem,
            rng,
            rho=rho,
            n_props=n_props,
            return_idx=True,
        )

        if return_indices:
            accepted_index[t] = int(idx)

        # Adapt only during warmup and then freeze rho.
        if t < int(adapt_iters):
            accepted = 1.0 if int(idx) != 0 else 0.0
            gamma_t = step_scale / np.sqrt(t + 1.0)
            z = z - gamma_t * (accepted - float(target_accept))
            rho = float(np.clip(_from_unconstrained(z, rho_min, rho_max), rho_min, rho_max))

        rho_history[t] = rho
        x = x_new
        chain[t + 1] = x

    if return_indices:
        return chain, accepted_index, rho_history
    return chain, rho_history
