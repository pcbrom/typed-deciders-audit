#!/usr/bin/env python3
"""R2 of the explanation front: Bayesian order-two model of the coalition game fitted on a sample of
coalitions, for utterances of 9 to 12 words with one unit per word (gate G2, R2 part).

Exact Laya values, no rounding, closed-form normal-inverse-gamma posterior with the R1 prior. Events:
the posterior predictive of a held-out coalition and the exact W_i, whose coalition N without i is
never fitted (preregistration/implementation_notes.md, R2).

Usage: r2.py --case r2_pt [--n-fit 256]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import betaincinv
from scipy.stats import t as student

import games as G
from r1 import GRID, A0, B0, beta_summary, design, games_of, log_marginal, prior_diag

DER = Path(__file__).resolve().parents[2] / "data/derived/explicacao"
CASE_ID = {"r2_pt": 0, "r2_en": 1}


def fit_mask(d, item, cid, tag, n_fit):
    sizes = np.array([bin(m).count("1") for m in range(1 << d)])
    elig = np.flatnonzero((sizes != 1) & (sizes != d - 1))
    rng = np.random.default_rng(np.random.SeedSequence([20261021, tag, cid, item]))
    return rng.choice(elig, min(256, len(elig)), replace=False)[:n_fit]


def posterior(X, y, v0):
    A = np.diag(1.0 / v0) + X.T @ X
    Ainv = np.linalg.inv(A)
    mn = Ainv @ (X.T @ y)
    an = A0 + len(y) / 2
    bn = B0 + 0.5 * (y @ y - mn @ A @ mn)
    return mn, Ainv, an, bn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASE_ID))
    ap.add_argument("--n-fit", type=int, default=256)
    a = ap.parse_args()
    cid = CASE_ID[a.case]
    tot = {}
    for item, d, probs in games_of(a.case, "_calib"):
        j = int(np.argmax(probs[(1 << d) - 1]))
        v = G.value(probs, j)
        X, _ = design(d)
        fm = fit_mask(d, item, cid, 73, a.n_fit)
        for t1 in GRID:
            for t2 in GRID:
                tot[(t1, t2)] = tot.get((t1, t2), 0.0) + log_marginal(X[fm], v[fm], prior_diag(d, t1, t2))
    t1, t2 = max(tot, key=tot.get)
    rows = []
    for item, d, probs in games_of(a.case):
        j = int(np.argmax(probs[(1 << d) - 1]))
        v = G.value(probs, j)
        X, pairs = design(d)
        fm = fit_mask(d, item, cid, 70, a.n_fit)
        mn, Ainv, an, bn = posterior(X[fm], v[fm], prior_diag(d, t1, t2))
        rest = np.setdiff1d(np.arange(1 << d), fm)
        s = int(np.random.default_rng(np.random.SeedSequence([20261021, 71, cid, item])).choice(rest))
        scale = np.sqrt(bn / an * (1 + X[s] @ Ainv @ X[s]))
        lo, hi = student.ppf([0.05, 0.95], 2 * an, loc=X[s] @ mn, scale=scale)
        rng = np.random.default_rng(np.random.SeedSequence([20261021, 72, cid, item]))
        i = int(rng.integers(d))
        s2 = 1.0 / rng.gamma(an, 1.0 / bn, 2000)
        Lc = np.linalg.cholesky(Ainv)
        th = mn + np.sqrt(s2)[:, None] * (rng.standard_normal((2000, len(mn))) @ Lc.T)
        w = th[:, 1 + i] + sum(th[:, 1 + d + q] for q, (u, c) in enumerate(pairs) if i in (u, c))
        qw = np.percentile(w, [5, 95])
        W = v[(1 << d) - 1] - v[((1 << d) - 1) & ~(1 << i)]
        rows.append({"item": item, "d": d, "s_size": bin(s).count("1"), "v_ok": bool(lo <= v[s] <= hi), "W_ok": bool(qw[0] <= W <= qw[1]),
                     "v_width": float(hi - lo), "W_width": float(qw[1] - qw[0])})
    V, Wt = beta_summary([r["v_ok"] for r in rows]), beta_summary([r["W_ok"] for r in rows])
    por_tamanho = {}
    for r in rows:
        por_tamanho.setdefault(r["s_size"], []).append(r["v_ok"])
    out = {"case": a.case, "n_fit": a.n_fit, "tau1": t1, "tau2": t2, "n_items": len(rows), "v_retida": V, "W": Wt,
           "passa_G2_R2": V["L"] >= 0.85 and Wt["L"] >= 0.85,
           "largura_mediana": {"v": round(float(np.median([r["v_width"] for r in rows])), 3), "W": round(float(np.median([r["W_width"] for r in rows])), 3)},
           "v_por_tamanho": {str(k): f"{sum(v)}/{len(v)}" for k, v in sorted(por_tamanho.items())}}
    (DER / f"r2_{a.case}_nfit{a.n_fit}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
