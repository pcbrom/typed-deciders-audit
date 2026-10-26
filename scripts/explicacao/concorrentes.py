#!/usr/bin/env python3
"""Competitors of R1 in gate G2: KernelSHAP with the paired-sampling variance of Covert and Lee (2021)
and BayesSHAP of Slack et al. (2021), on the same rounded Laya games and coverage event (exact E_i)
(preregistration/implementation_notes.md, competitors in G2).

Usage: concorrentes.py --case pt_multilingual_ft
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
from math import comb  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import expit  # noqa: E402
from scipy.stats import norm, t as student  # noqa: E402

import games as G  # noqa: E402
from r1 import CASE_ID, beta_summary, games_of, round2  # noqa: E402

DER = Path(__file__).resolve().parents[2] / "data/derived/explicacao"
PAIRS = 256


def kernel_samples(d, rng):
    """Paired masks drawn from the Shapley kernel over sizes 1..d-1."""
    sizes = np.arange(1, d)
    w = (d - 1) / (sizes * (d - sizes))
    s = rng.choice(sizes, PAIRS, p=w / w.sum())
    masks = np.zeros((PAIRS, d), bool)
    for r, k in enumerate(s):
        masks[r, rng.choice(d, k, replace=False)] = True
    return masks


def as_index(z):
    return (z.astype(int) << np.arange(z.shape[1])).sum(1)


def exact_A(d):
    """E[z z'] under the Shapley kernel over sizes 1..d-1, in closed form (Covert and Lee, 2021)."""
    sizes = np.arange(1, d)
    w = (d - 1) / (sizes * (d - sizes))
    p = w / w.sum()
    off = (p * sizes * (sizes - 1) / (d * (d - 1))).sum() if d > 1 else 0.0
    A = np.full((d, d), off)
    np.fill_diagonal(A, (p * sizes / d).sum())
    return A


def kernelshap_cl(game, d, masks):
    """Unbiased KernelSHAP of Covert and Lee: exact A, b from paired samples, efficiency constraint,
    and the CLT variance of the paired b-terms."""
    v0, v1 = game[0], game[(1 << d) - 1]
    Z = np.concatenate([masks, ~masks]).astype(float)
    y = game[as_index(Z.astype(bool))] - v0
    n = len(masks)
    zy = 0.5 * (Z[:n] * y[:n, None] + Z[n:] * y[n:, None])
    b = zy.mean(0)
    Ainv = np.linalg.inv(exact_A(d))
    one = np.ones(d)
    P = Ainv - np.outer(Ainv @ one, one @ Ainv) / (one @ Ainv @ one)
    phi = P @ b + Ainv @ one * (v1 - v0) / (one @ Ainv @ one)
    C = np.cov(zy, rowvar=False) / n
    var = np.diag(P @ C @ P.T)
    return phi, np.sqrt(np.maximum(var, 0))


def bayesshap(game, d, masks):
    """Slack et al.: weighted Bayesian linear regression with kernel weights and a flat prior."""
    Z = np.concatenate([masks, ~masks]).astype(float)
    y = game[as_index(Z.astype(bool))]
    k = Z.sum(1).astype(int)
    w = (d - 1) / (np.array([comb(d, x) for x in k]) * k * (d - k))
    X = np.column_stack([np.ones(len(Z)), Z])
    XtW = X.T * w
    V = np.linalg.pinv(XtW @ X)
    beta = V @ (XtW @ y)
    res = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = (w * res ** 2).sum() / dof
    return beta[1:], np.sqrt(np.maximum(np.diag(V)[1:] * s2, 0)), dof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASE_ID))
    a = ap.parse_args()
    cid = CASE_ID[a.case]
    ev = {"kernelshap_cl": [], "bayesshap_prob": [], "bayesshap_logit": []}
    items = []
    for item, d, probs in games_of(a.case):
        if d < 2:
            continue
        ph = round2(probs)
        j = int(np.argmax(ph[(1 << d) - 1]))
        exact_logit = G.shapley(G.mobius(G.value(probs, j)), d)
        exact_prob = G.shapley(G.mobius(np.asarray(probs, float)[:, j]), d)
        game_logit = G.value(ph, j)
        game_prob = ph[:, j]
        i = int(np.random.default_rng(np.random.SeedSequence([20261021, 61, cid, item])).integers(d))
        masks = kernel_samples(d, np.random.default_rng(np.random.SeedSequence([20261021, 63, cid, item])))
        phi, se = kernelshap_cl(game_logit, d, masks)
        z = norm.ppf(0.95)
        ev["kernelshap_cl"].append(bool(abs(exact_logit[i] - phi[i]) <= z * se[i]))
        for scale, game, exact in (("prob", game_prob, exact_prob), ("logit", game_logit, exact_logit)):
            b, s, dof = bayesshap(game, d, masks)
            q = student.ppf(0.95, dof)
            ev[f"bayesshap_{scale}"].append(bool(abs(exact[i] - b[i]) <= q * s[i]))
        items.append({"item": item, "unit": i, **{k: v[-1] for k, v in ev.items()}})
    out = {"case": a.case, **{k: beta_summary(v) for k, v in ev.items()}}
    (DER / f"concorrentes_g2_{a.case}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    with open(DER / f"concorrentes_g2_{a.case}_itens.jsonl", "w") as f:
        for r in items:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
