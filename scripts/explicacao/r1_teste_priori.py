"""Exact correctness test: truth drawn from the prior, so a correct sampler covers 90% on average."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.special import expit
import r1

def task(args):
    k, d, t1, t2, censura = args
    rng = np.random.default_rng([9, k, d])
    X, pairs = r1.design(d)
    v0 = r1.prior_diag(d, t1, t2)
    s2 = 1.0 / rng.gamma(r1.A0, 1.0 / r1.B0)
    theta = np.sqrt(s2 * v0) * rng.standard_normal(len(v0))
    if censura:
        theta[0] += 6.0  # shift the intercept so that most coalitions sit in the censored tail
    y = X @ theta + np.sqrt(s2) * rng.standard_normal(len(X))
    ph = r1.round2(expit(y))
    lo, hi = r1.intervals(ph)
    v0_fit = v0.copy()
    draws = r1.gibbs(X, lo, hi, v0_fit, rng)
    i = int(rng.integers(d))
    w_true = theta[1 + i] + sum(theta[1 + d + q] for q, (a, c) in enumerate(pairs) if i in (a, c))
    w, _ = r1.wi_ei(draws, d, pairs, i)
    q = np.percentile(w, [5, 95])
    return bool(q[0] <= w_true <= q[1]), float((ph >= 1).mean())

if __name__ == "__main__":
    for t1, t2 in [(2.0, 1.0)]:
        for censura in (False, True):
            with ProcessPoolExecutor(8) as ex:
                res = list(ex.map(task, [(k, d, t1, t2, censura) for k in range(150) for d in (4, 6)], chunksize=4))
            print(f"verdade da priori, deslocamento de censura={censura}: cobertura {np.mean([r[0] for r in res]):.3f} "
                  f"em {len(res)}; fração de coalizões em 1,00: {np.mean([r[1] for r in res]):.2f}")
