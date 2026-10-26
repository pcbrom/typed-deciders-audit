"""Correctness check of the R1 sampler on synthetic order-two games built from calibration items only."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.special import expit
import r1, games as G

def task(args):
    item, d, probs, t1, t2, sigma = args
    rng = np.random.default_rng([7, item])
    j = int(np.argmax(probs[(1 << d) - 1]))
    X, pairs = r1.design(d)
    v = G.value(probs, j)
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)          # order-two truth from the calibration game
    y = X @ beta + sigma * rng.standard_normal(len(v))
    ph = r1.round2(expit(y))
    lo, hi = r1.intervals(ph)
    draws = r1.gibbs(X, lo, hi, r1.prior_diag(d, t1, t2), rng)
    i = int(rng.integers(d))
    w_true = beta[1 + i] + sum(beta[1 + d + q] for q, (a, c) in enumerate(pairs) if i in (a, c))
    w, _ = r1.wi_ei(draws, d, pairs, i)
    q = np.percentile(w, [5, 95])
    return bool(q[0] <= w_true <= q[1]), bool(ph[(1 << d) - 1] >= 1.0)

if __name__ == "__main__":
    (t1, t2), _ = r1.choose_tau("en_multilingual")
    games = [(it, d, p) for it, d, p in r1.games_of("en_multilingual", "_calib") if d >= 2][:300]
    for sigma in (0.1, 0.5):
        with ProcessPoolExecutor(8) as ex:
            res = list(ex.map(task, [(it, d, p, t1, t2, sigma) for it, d, p in games], chunksize=4))
        ok = [r[0] for r in res]; cens = [r[0] for r in res if r[1]]
        print(f"sigma={sigma}: cobertura de W (ordem 2 verdadeira) {np.mean(ok):.3f} em {len(ok)} itens; nos itens com p0 = 1,00: {np.mean(cens):.3f} em {len(cens)}")
