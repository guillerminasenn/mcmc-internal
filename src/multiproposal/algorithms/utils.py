# Small algorithm helpers (angles, brackets)
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve_transition_assignment(D):
    """Exact solution of the lam=0 transition problem.

    max <D, P> over doubly stochastic P with zero diagonal is attained at a
    vertex of that face of the Birkhoff polytope, i.e. a permutation matrix
    without fixed points. Returns the maximum-weight such permutation matrix.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if D.ndim != 2 or D.shape[1] != n or n < 2:
        raise ValueError("D must be a square matrix of size >= 2")
    cost = -D
    np.fill_diagonal(cost, np.inf)
    rows, cols = linear_sum_assignment(cost)
    P = np.zeros((n, n))
    P[rows, cols] = 1.0
    return P


def solve_transition_lp(
    D,
    P_prev,
    lam=1.0,
    solver="HIGHs",
    warm_start=True,
    verbose=False
):
    """
    Solve:
        max <D, P> - lam * ||P - P_prev||_1
    subject to:
        P >= 0
        row sums = 1
        col sums = 1
        diag(P) = 0
    """
    import cvxpy as cp

    d = D.shape[0]

    # Decision variables
    P = cp.Variable((d, d))
    Z = cp.Variable((d, d))

    constraints = []

    # Non-negativity
    constraints += [P >= 0]

    # Row and column stochasticity
    constraints += [cp.sum(P, axis=1) == 1]
    constraints += [cp.sum(P, axis=0) == 1]

    # Zero diagonal
    constraints += [cp.diag(P) == 0]

    # L1 regularization constraints
    constraints += [Z >= P - P_prev]
    constraints += [Z >= -(P - P_prev)]

    # Objective
    objective = cp.Maximize(cp.sum(cp.multiply(D, P)) - lam * cp.sum(Z))

    problem = cp.Problem(objective, constraints)

    problem.solve(
        solver=cp.HIGHS,
        warm_start=warm_start,
        verbose=verbose,
    )

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"LP did not solve: {problem.status}")

    return P.value

