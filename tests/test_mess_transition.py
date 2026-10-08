"""Tests for the LP-based acceptance step of mess_step (labelling fix + assignment solver)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import multiproposal.algorithms.mess as mess_module
from multiproposal.algorithms.mess import mess_step
from multiproposal.algorithms.utils import solve_transition_assignment, solve_transition_lp
from multiproposal.problems.base import GaussianPriorProblem


class ConstantLikelihood(GaussianPriorProblem):
    """Every proposal is valid."""

    def log_likelihood(self, x):
        return 0.0


class GaussianLikelihood(GaussianPriorProblem):
    """N(0, I) prior x N(y, s2 I) likelihood: the posterior is Gaussian."""

    def __init__(self, mu, cov, y, s2):
        super().__init__(mu, cov)
        self.y = y
        self.s2 = s2

    def log_likelihood(self, x):
        return -0.5 * np.sum((x - self.y) ** 2) / self.s2


class ScriptedRng:
    """Feeds scripted uniform draws; choice is the argmax of p."""

    def __init__(self, uniforms):
        self.queue = list(uniforms)

    def standard_normal(self, n):
        return np.ones(n)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self.queue.pop(0)

    def choice(self, a, p=None):
        return int(np.argmax(p))


def _doubly_stochastic_zero_diag(P):
    n = P.shape[0]
    assert np.all(P >= -1e-9)
    np.testing.assert_allclose(P.sum(axis=0), np.ones(n), atol=1e-7)
    np.testing.assert_allclose(P.sum(axis=1), np.ones(n), atol=1e-7)
    np.testing.assert_allclose(np.diag(P), np.zeros(n), atol=1e-9)


def test_label_mapping_follows_sorted_order(monkeypatch):
    # Sorted angles (0.4, 1.0, 3.0=alpha, 3.2, 5.5); draw order is scrambled.
    phi = np.array([5.5, 0.4, 3.2, 1.0])
    # Permutation sending the current state (sorted position 2) to sorted position 4.
    P = np.zeros((5, 5))
    for r, c in {0: 1, 1: 0, 2: 4, 4: 3, 3: 2}.items():
        P[r, c] = 1.0
    monkeypatch.setattr(mess_module, "solve_transition_assignment", lambda D: P)

    problem = ConstantLikelihood(np.zeros(2), np.eye(2))
    rng = ScriptedRng([0.5, 3.0, phi])
    _, _, _, diag = mess_step(
        np.zeros(2), problem, rng, M=4, use_lp=True, lam=0.0, return_diagnostics=True
    )

    entry = diag[-1]
    assert entry["accepted_sorted_position"] == 4
    assert entry["phi_vector"][entry["accepted_index"]] == 5.5


@pytest.mark.parametrize("metric", ["angular", "euclidean"])
def test_assignment_matches_highs(metric):
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = int(rng.integers(2, 9))
        theta = np.sort(rng.uniform(0, 2 * np.pi, size=n))
        if metric == "angular":
            ad = np.abs(theta[:, None] - theta[None, :])
            D = np.minimum(ad, 2 * np.pi - ad)
        else:
            pts = np.stack([2.0 * np.cos(theta), 0.5 * np.sin(theta)])
            D = np.linalg.norm(pts[:, :, None] - pts[:, None, :], axis=0)

        P_assign = solve_transition_assignment(D)
        P_highs = solve_transition_lp(D, np.ones((n, n)) / (n - 1), lam=0.0)

        _doubly_stochastic_zero_diag(P_assign)
        _doubly_stochastic_zero_diag(P_highs)
        assert set(np.unique(P_assign)) <= {0.0, 1.0}
        assert np.sum(D * P_assign) == pytest.approx(np.sum(D * P_highs), abs=1e-6)


def test_uniform_branch_matches_pre_fix_values():
    # Recorded from the unmodified code at tag pre-fix-lp-2026-09-30.
    expected_xs = np.array([
        [1.1871177717861279, 1.0552705785467855, 0.7039285902376101],
        [1.1267164350183898, 1.0801080865058525, 0.7155078044121821],
        [1.201488349885337, 1.2385706668141825, 0.8044005733698953],
        [1.183501121443998, 1.2160791438057985, 0.8677940822514209],
        [1.1218153717790826, 1.1526359630190375, 0.830914452013134],
        [1.10156448029278, 1.177809735920827, 0.9036767695317844],
        [1.1023363574891278, 1.1767959825484873, 0.901631392168344],
        [1.1536378110343282, 0.9815235619437758, 0.9294870961941575],
    ])
    expected_ns = [1, 2, 1, 0, 0, 0, 2, 0]

    problem = GaussianLikelihood(np.zeros(3), np.eye(3), np.ones(3), 0.05)
    rng = np.random.default_rng(123)
    x = np.ones(3)
    xs, ns = [], []
    for _ in range(8):
        x, n, P1 = mess_step(x, problem, rng, M=5, use_lp=False)
        assert P1 is None
        xs.append(x)
        ns.append(n)

    np.testing.assert_allclose(np.array(xs), expected_xs, rtol=1e-12, atol=0)
    assert ns == expected_ns


def test_solver_argument_validation():
    problem = ConstantLikelihood(np.zeros(2), np.eye(2))
    with pytest.raises(ValueError):
        mess_step(np.zeros(2), problem, np.random.default_rng(0), M=3, use_lp=True,
                  lam=0.1, lp_solver="assignment")
    with pytest.raises(ValueError):
        mess_step(np.zeros(2), problem, np.random.default_rng(0), M=3, use_lp=True,
                  lam=0.0, lp_solver="bogus")


def test_highs_path_does_not_mutate_P0():
    problem = ConstantLikelihood(np.zeros(2), np.eye(2))
    P0 = np.ones((4, 4)) / 3
    P0_before = P0.copy()
    mess_step(np.zeros(2), problem, np.random.default_rng(1), M=3, use_lp=True,
              lam=0.1, P0=P0, lp_solver="highs")
    np.testing.assert_array_equal(P0, P0_before)


@pytest.mark.parametrize("use_lp,metric", [(False, "angular"), (True, "angular"), (True, "euclidean")])
def test_invariance_on_gaussian_posterior(use_lp, metric):
    # Start from exact posterior draws; one MESS step must leave the law unchanged.
    y = np.array([1.0, -0.5])
    s2 = 0.25
    post_var = s2 / (1 + s2)
    post_mean = y / (1 + s2)
    problem = GaussianLikelihood(np.zeros(2), np.eye(2), y, s2)

    rng = np.random.default_rng(2024)
    n = 2000
    x0 = post_mean + np.sqrt(post_var) * rng.standard_normal((n, 2))
    x1 = np.empty_like(x0)
    for t in range(n):
        x1[t], _, _ = mess_step(
            x0[t], problem, rng, M=4, use_lp=use_lp, distance_metric=metric, lam=0.0
        )

    z = (x1 - post_mean) / np.sqrt(post_var)
    assert stats.kstest(z[:, 0], "norm").pvalue > 1e-3
    assert stats.kstest(z[:, 1], "norm").pvalue > 1e-3
    assert stats.kstest(np.sum(z**2, axis=1), "chi2", args=(2,)).pvalue > 1e-3
