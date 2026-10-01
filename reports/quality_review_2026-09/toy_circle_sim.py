"""Angle-space toy simulation of ESS / MESS on a fixed slice of the ellipse.

No likelihood is needed: the slice S is a union of arcs in rotation
coordinates theta in (-pi, pi], with the current state at theta = 0.

Algorithms
----------
murray : ESS of Murray, Adams & MacKay (2010): initial bracket anchored at
         the first proposal, i.e. the "cut" of the circle is a rejected point.
paper  : MESS as in the arXiv paper / mess.py: alpha ~ U(0, 2pi] places an
         independent uniform cut of the circle at rotation -alpha; M i.i.d.
         uniform proposals per round; uniform selection among valid ones.
         M = 1 is the ESS implemented in ess.py (not Murray's).
circ   : MESS with the shrinkage done on the circle (bracket = arc between the
         two rejected angles nearest to alpha on either side); no artificial cut.
         M = 1 reproduces Murray's ESS.

Run:  PYTHONPATH= /path/to/mess-env/bin/python toy_circle_sim.py
"""
import numpy as np

TWO_PI = 2 * np.pi


def wrap(theta):
    """Wrap to (-pi, pi]."""
    return (theta + np.pi) % TWO_PI - np.pi


def in_slice(theta, arcs):
    t = wrap(theta)
    ok = np.zeros_like(t, dtype=bool)
    for lo, hi in arcs:  # arcs given in (-pi, pi] coordinates, lo < hi
        ok |= (t > lo) & (t <= hi)
    return ok


def murray_ess(arcs, rng):
    theta = rng.uniform(0, TWO_PI)
    lo, hi = theta - TWO_PI, theta
    n = 0
    while True:
        n += 1
        if in_slice(np.array([theta]), arcs)[0]:
            return wrap(theta), n, n
        if theta < 0:
            lo = theta
        else:
            hi = theta
        theta = rng.uniform(lo, hi)


def paper_mess(arcs, rng, M):
    alpha = rng.uniform(0, TWO_PI)
    lo, hi = 0.0, TWO_PI
    rounds = 0
    while True:
        rounds += 1
        phi = rng.uniform(lo, hi, size=M)
        valid = in_slice(phi - alpha, arcs)
        if valid.any():
            j = rng.choice(np.flatnonzero(valid))
            return wrap(phi[j] - alpha), rounds, rounds * M
        below = phi[phi < alpha]
        above = phi[phi >= alpha]
        if below.size:
            lo = max(lo, below.max())
        if above.size:
            hi = min(hi, above.min())


def circ_mess(arcs, rng, M):
    """Shrinkage on the circle: rotation coordinate, current at 0.
    bracket = (lo, hi) with lo <= 0 <= hi, initially the full circle (-2pi.., ..)
    represented as lo = -inf sentinel until a rejected angle is found on that side."""
    lo, hi = None, None  # None = no rejected point yet on that side
    rounds = 0
    while True:
        rounds += 1
        if lo is None and hi is None:
            theta = rng.uniform(-np.pi, np.pi, size=M)
        elif lo is None:      # rejected only above: bracket is (hi - 2pi, hi)
            theta = rng.uniform(hi - TWO_PI, hi, size=M)
        elif hi is None:      # rejected only below: bracket is (lo, lo + 2pi)
            theta = rng.uniform(lo, lo + TWO_PI, size=M)
        else:
            theta = rng.uniform(lo, hi, size=M)
        valid = in_slice(theta, arcs)
        if valid.any():
            j = rng.choice(np.flatnonzero(valid))
            return wrap(theta[j]), rounds, rounds * M
        # shrink: nearest rejected angles on each side of 0 (on the circle)
        t = theta.copy()
        if lo is None and hi is None:
            t = wrap(t)
        neg = t[t < 0]
        pos = t[t >= 0]
        if neg.size:
            lo = neg.max() if lo is None else max(lo, neg.max())
        if pos.size:
            hi = pos.min() if hi is None else min(hi, pos.min())
        # a single rejected point on one side is enough to define the bracket
        # on the circle (Murray): handled by the None branches above.


def connected_experiment(p1, n_iter, rng):
    """Connected slice of fraction p1; current at a uniform position in it.
    Returns dict alg -> (KS distance of new position from U(0,1),
                         mean rounds, mean evaluations)."""
    algs = {"murray": lambda a: murray_ess(a, rng)}
    for M in (1, 2, 4, 8, 16, 32):
        algs[f"paper_M{M}"] = (lambda a, M=M: paper_mess(a, rng, M))
        algs[f"circ_M{M}"] = (lambda a, M=M: circ_mess(a, rng, M))
    out = {}
    for name, step in algs.items():
        pos = np.empty(n_iter)
        rounds = np.empty(n_iter)
        evals = np.empty(n_iter)
        for t in range(n_iter):
            u = rng.uniform()
            a = TWO_PI * p1 * u
            b = TWO_PI * p1 - a
            arcs = [(-a, b)]
            th, r, e = step(arcs)
            pos[t] = (th + a) / (a + b)
            rounds[t], evals[t] = r, e
        pos.sort()
        ks = np.max(np.abs(pos - (np.arange(1, n_iter + 1) / n_iter)))
        out[name] = (ks, rounds.mean(), evals.mean())
    return out


def two_arc_experiment(q, n_iter, rng):
    """Two equal arcs (fraction q each), antipodal: S1 around 0, S2 = S1 + pi.
    Returns dict alg -> (P(switch), mean rounds, mean evaluations)."""
    algs = {"murray": lambda a: murray_ess(a, rng)}
    for M in (1, 2, 4, 8, 16, 32, 64):
        algs[f"paper_M{M}"] = (lambda a, M=M: paper_mess(a, rng, M))
        algs[f"circ_M{M}"] = (lambda a, M=M: circ_mess(a, rng, M))
    out = {}
    for name, step in algs.items():
        sw = np.empty(n_iter)
        rounds = np.empty(n_iter)
        evals = np.empty(n_iter)
        for t in range(n_iter):
            u = rng.uniform()
            w = TWO_PI * q
            a = w * u
            b = w - a
            arcs = [(-a, b), (-a + np.pi, b + np.pi)]
            # second arc may cross pi: split it
            lo2, hi2 = -a + np.pi, b + np.pi
            arcs = [(-a, b)]
            if hi2 > np.pi:
                arcs += [(lo2, np.pi), (-np.pi, hi2 - TWO_PI)]
            else:
                arcs += [(lo2, hi2)]
            th, r, e = step(arcs)
            sw[t] = not ((th > -a) & (th <= b))
            rounds[t], evals[t] = r, e
        out[name] = (sw.mean(), rounds.mean(), evals.mean())
    return out


def autocorr_experiment(p1, n_iter, rng):
    """Run each algorithm as a Markov chain on a FIXED connected arc (0, w),
    w = 2*pi*p1, and report the lag-1 autocorrelation of the absolute
    position. Murray's ESS draws independently from the arc (rho_1 = 0);
    the paper's alpha-cut version is reversible w.r.t. uniform on the arc
    but not an independent draw."""
    w = TWO_PI * p1
    algs = {"murray": lambda a: murray_ess(a, rng)}
    for M in (1, 2, 4):
        algs[f"paper_M{M}"] = (lambda a, M=M: paper_mess(a, rng, M))
        algs[f"circ_M{M}"] = (lambda a, M=M: circ_mess(a, rng, M))
    out = {}
    for name, step in algs.items():
        s = w * rng.uniform()
        traj = np.empty(n_iter)
        for t in range(n_iter):
            th, _, _ = step([(-s, w - s)])
            s = s + th
            traj[t] = s
        z = traj - traj.mean()
        rho1 = np.dot(z[:-1], z[1:]) / np.dot(z, z)
        out[name] = rho1
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n = 40000
    print("=== fixed connected arc: lag-1 autocorrelation of position ===")
    for p1 in (0.3, 0.1):
        res = autocorr_experiment(p1, n, rng)
        print(f"--- p1 = {p1}  (noise level ~ {1/np.sqrt(n):.4f})")
        for k, r in res.items():
            print(f"{k:12s} rho1={r:+.4f}")
    print()
    print("=== connected slice: KS vs uniform-on-slice, rounds, evaluations ===")
    for p1 in (0.3, 0.1, 0.03):
        print(f"--- p1 = {p1}  (KS 95% noise level ~ {1.36/np.sqrt(n):.4f})")
        res = connected_experiment(p1, n, rng)
        for k, (ks, r, e) in res.items():
            print(f"{k:12s} KS={ks:.4f}  rounds={r:6.2f}  evals={e:7.2f}")
    print()
    print("=== two antipodal arcs (fraction q each): P(switch), rounds, evals ===")
    for q in (0.05, 0.01):
        print(f"--- q = {q}")
        res = two_arc_experiment(q, n, rng)
        base = res["murray"]
        for k, (s, r, e) in res.items():
            # per-evaluation and per-round switch rates relative to Murray ESS
            per_eval = s / e
            per_round = s / r
            print(f"{k:12s} P(sw)={s:.4f}  rounds={r:6.2f} evals={e:7.2f}  "
                  f"sw/eval={per_eval:.5f} ({per_eval/(base[0]/base[2]):.2f}x ESS)  "
                  f"sw/round={per_round:.4f} ({per_round/(base[0]/base[1]):.2f}x ESS)")
