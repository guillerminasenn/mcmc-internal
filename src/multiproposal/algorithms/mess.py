# MESS (M >= 1 proposals)

# algorithms/mess.py
import numpy as np
from .utils import solve_transition_assignment, solve_transition_lp

def mess_step(
    x,
    problem,
    rng,
    M=1,
    use_lp=False,
    distance_metric='angular',
    lam=0.1,
    P0=None,
    return_diagnostics=False,
    lp_solver=None,
):
    """Perform a MESS step.
    Parameters
    ----------
    x : np.ndarray
        Current state.
    problem : multiproposal.problems.base.Problem
        Problem instance.
    rng : np.random.Generator
        Random number generator.
    M : int
        Number of proposals to generate.
    use_lp : bool
        If True, compute the entries of the transition matrix using
        linear programming. If False, split probability evenly across
        candidate proposals.
    distance_metric: string
        Specifies the distance metric used to compute the distance
        between candidate proposals. Use 'angular' for great-circle 
        angular distance between angles and 'euclidean' for Euclidean 
        distance between the corresponding proposals.
    lam : float
        Weight parameter in the regularization term of the objective
        function in lp.
    P0 : np.ndarray or None
        Initial doubly stochastic matrix for the transition probabilities.
        If None, a uniform matrix with zero diagonal is used. Only used
        by the 'highs' solver; never modified in place.
    lp_solver : {'assignment', 'highs'} or None
        Solver for the transition matrix when use_lp=True. 'assignment'
        is exact for lam == 0 (max-weight derangement). 'highs' solves
        the regularised LP with cvxpy. None selects 'assignment' if
        lam == 0 and 'highs' otherwise.
    return_diagnostics : bool
        If True, return per-iteration diagnostics including the accepted
        proposal index and distances to all candidates.

    Returns
    -------
    x_new : np.ndarray
        New state after the MESS step.
    nr_intervals : int
        Number of shrinking steps performed.
    P1 : np.ndarray
        Transition matrix used to sample the new state (only if lp=True).
    diagnostics : list
        Per-iteration diagnostic data (only if return_diagnostics=True).
    """
    if use_lp:
        if lp_solver is None:
            lp_solver = 'assignment' if lam == 0 else 'highs'
        if lp_solver not in ('assignment', 'highs'):
            raise ValueError(f"Unknown lp_solver: {lp_solver!r}")
        if lp_solver == 'assignment' and lam != 0:
            raise ValueError("lp_solver='assignment' requires lam == 0")

    # Initialize transition matrix to None
    P1 = None
    accepted_sorted_pos = None
    diagnostics = [] if return_diagnostics else None

    # Center the current state and the auxiliary sample from the prior
    x_centered = x - problem.prior_mean()
    nu_centered = problem.sample_prior(rng) - problem.prior_mean() 

    # Sample the likelihood threshold
    logy = problem.log_likelihood(x) + np.log(rng.uniform())

    # Sample alpha, the angle corresponding to the current state
    alpha = rng.uniform(0, 2*np.pi)

    # Initialize the angle interval
    phi_min = 0
    phi_max = 2 * np.pi

    # Shrink interval until acceptance
    nr_intervals = 0
    while True:
        # Sample M angles
        phi_vector = rng.uniform(phi_min, phi_max, size=M)

        # Compute the proposals
        x_prop_vector = (
            problem.prior_mean()[:, np.newaxis]
            + np.cos(phi_vector - alpha) * x_centered[:, np.newaxis]
            + np.sin(phi_vector - alpha) * nu_centered[:, np.newaxis]
        )

        # Evaluate the likelihood for all proposals
        log_likelihoods = np.array([
            problem.log_likelihood(x_prop_vector[:, i])
            for i in range(M)
        ])

        # Compute A_i, the set of indexes of the candidate proposals
        A = np.where(log_likelihoods > logy)[0]

        diag_entry = None
        if return_diagnostics:
            abs_angular_dist = np.abs(phi_vector - alpha)
            angular_distances = np.minimum(abs_angular_dist, 2 * np.pi - abs_angular_dist)
            diff = x_prop_vector - x[:, np.newaxis]
            euclidean_distances = np.linalg.norm(diff, axis=0)
            diag_entry = {
                'phi_min': float(phi_min),
                'phi_max': float(phi_max),
                'alpha': float(alpha),
                'phi_vector': phi_vector.copy(),
                'log_likelihoods': log_likelihoods.copy(),
                'valid_indices': A.copy(),
                'angular_distances': angular_distances,
                'euclidean_distances': euclidean_distances,
                'accepted_index': None,
            }
            diagnostics.append(diag_entry)

        # If there are valid proposals, select one and return
        if len(A) > 0:

            # Sample the proposal using a transition matrix computed with lp
            if use_lp:
                psi = np.concatenate([phi_vector[A], np.array([alpha])])
                order = np.argsort(psi)  # order[k] = index in psi of k-th smallest angle
                psi_sorted = psi[order]
                
                # Compute the distance matrix
                if distance_metric== 'angular':
                    abs_angular_dist = np.abs(psi_sorted[:, None] - psi_sorted[None, :])
                    D = np.minimum(abs_angular_dist, 2 * np.pi - abs_angular_dist)

                elif distance_metric== 'euclidean' or distance_metric == 'euclidean_squared':
                    # Compute the proposals corresponding to the sorted angles (NOTE: inefficient, because I already have them above)
                    x_psi_sorted = (
                        problem.prior_mean()[:, np.newaxis]
                        + np.cos(psi_sorted - alpha) * x_centered[:, np.newaxis]
                        + np.sin(psi_sorted - alpha) * nu_centered[:, np.newaxis]
                    )

                    # Compute the Euclidean distance matrix between these proposals
                    diff = x_psi_sorted[:, :, np.newaxis] - x_psi_sorted[:, np.newaxis, :]  # (d, n, n)

                    if distance_metric == 'euclidean':
                        D = np.linalg.norm(diff, axis=0)
                    elif distance_metric == 'euclidean_squared':
                        D = np.sum(diff**2, axis=0)

                # Compute the transition matrix

                # If not specified, use the initial doubly stochastic matrix (uniform, zero diagonal)
                if lp_solver == 'highs':
                    if P0 is None:
                        P0_used = np.ones((len(A) + 1, len(A) + 1)) / len(A)
                    else:
                        P0_used = np.array(P0, dtype=float)
                    np.fill_diagonal(P0_used, 0)

                # Solve
                if lp_solver == 'assignment':
                    P1 = solve_transition_assignment(D)
                else:
                    P1 = solve_transition_lp(D, P0_used, lam=lam, verbose=False)
                
                # Sample i according to the row of P1 corresponding to the current state
                # alpha was appended last to psi, so it sits at index len(A)
                current_index = int(np.where(order == len(A))[0][0])
                row_P1 = P1[current_index, :]

                # Remove index corresponding to the current state
                row_P1 = np.delete(row_P1, current_index)
                row_P1 = np.clip(row_P1, 0.0, None)
                row_P1 = row_P1 / row_P1.sum()

                # Map sorted positions back to psi indices (h^{-1}), then to A
                labels = np.delete(order, current_index)
                k = rng.choice(len(row_P1), p=row_P1)
                i = A[labels[k]]
                accepted_sorted_pos = int(k if k < current_index else k + 1)

            # Sample uniformly among the valid proposals
            else:
                i = rng.choice(A)
            if return_diagnostics and diag_entry is not None:
                diag_entry['accepted_index'] = int(i)
                if distance_metric == 'euclidean':
                    dist_all = diag_entry['euclidean_distances']
                elif distance_metric == 'euclidean_squared':
                    dist_all = diag_entry['euclidean_distances'] ** 2
                else:
                    dist_all = diag_entry['angular_distances']
                others = A[A != i]
                diag_entry['accepted_distance'] = float(dist_all[i])
                diag_entry['mean_other_valid_distance'] = (
                    float(dist_all[others].mean()) if others.size else float('nan')
                )
                diag_entry['accepted_sorted_position'] = accepted_sorted_pos
                diag_entry['lp_solver'] = lp_solver if use_lp else None

            if return_diagnostics:
                return x_prop_vector[:, i], nr_intervals, P1, diagnostics
            return x_prop_vector[:, i], nr_intervals, P1

        # Otherwise, shrink the angle interval
        phi_min = np.max(np.concatenate([np.array([phi_min]), phi_vector[np.where(phi_vector < alpha)]]))
        phi_max = np.min(np.concatenate([np.array([phi_max]), phi_vector[np.where(phi_vector >= alpha)]]))

        # Count the number of shrinking steps
        nr_intervals += 1