#!/usr/bin/env python3
"""R1 of the explanation front: Bayesian order-two model of the coalition game under rounding and
censoring, validated on Laya with probabilities rounded to two decimals (gate G2, R1 part).

Model per item: y(S) = b0 + sum_{i in S} b_i + sum_{i<k in S} g_ik + e, e ~ N(0, s2); b0 ~ N(0, 100 s2),
b_i ~ N(0, t1^2 s2), g_ik ~ N(0, t2^2 s2), s2 ~ InvGamma(2, 1). Each coalition is only known to lie in
the logit interval of its rounded probability (preregistration/implementation_notes.md, R1). Gibbs
sampling with truncated-normal augmentation. Coverage of the exact W_i and E_i by the central 90%
posterior interval, with the author's Beta rule on the coverage events.

Usage: r1.py --case pt_multilingual_ft [--workers 8] [--max-items N]
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import betaincinv, gammaln, logit, ndtr, ndtri  # noqa: E402

import games as G  # noqa: E402

DER = Path(__file__).resolve().parents[2] / "data/derived/explicacao"
CASE_ID = {"pt_multilingual_ft": 0, "en_multilingual": 1}
GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
A0, B0, V_INT = 2.0, 1.0, 100.0
ITERS, BURN = 3000, 1000
LO_TAIL, HI_TAIL = float(logit(0.005)), float(logit(0.995))


def round2(p):
    return np.floor(np.asarray(p, float) * 100 + 0.5) / 100


def design(d):
    idx = np.arange(1 << d)
    z = ((idx[:, None] >> np.arange(d)) & 1).astype(float)
    pairs = list(itertools.combinations(range(d), 2))
    cols = [np.ones(len(idx))] + [z[:, i] for i in range(d)] + [z[:, i] * z[:, k] for i, k in pairs]
    return np.stack(cols, 1), pairs


def prior_diag(d, t1, t2):
    return np.array([V_INT] + [t1 ** 2] * d + [t2 ** 2] * (d * (d - 1) // 2))


def intervals(ph):
    lo = np.where(ph <= 0.0, -np.inf, logit(np.clip(ph - 0.005, 1e-12, 1)))
    hi = np.where(ph >= 1.0, np.inf, logit(np.clip(ph + 0.005, 0, 1 - 1e-12)))
    lo = np.where(ph >= 1.0, HI_TAIL, lo)
    hi = np.where(ph <= 0.0, LO_TAIL, hi)
    return lo, hi


def log_marginal(X, y, v0):
    """Closed-form normal-inverse-gamma log marginal likelihood."""
    n = len(y)
    prec0 = np.diag(1.0 / v0)
    A = prec0 + X.T @ X
    L = np.linalg.cholesky(A)
    mn = np.linalg.solve(A, X.T @ y)
    an = A0 + n / 2
    bn = B0 + 0.5 * (y @ y - mn @ A @ mn)
    logdetA = 2 * np.log(np.diag(L)).sum()
    return (gammaln(an) - gammaln(A0) + A0 * np.log(B0) - an * np.log(bn) - 0.5 * logdetA - 0.5 * np.log(v0).sum()
            - n / 2 * np.log(2 * np.pi))


def games_of(case, half=""):
    z = np.load(DER / f"laya_games_{case}{half}.npz")
    for k in range(len(z["d"])):
        yield int(z["item"][k]), int(z["d"][k]), z["probs"][z["offsets"][k]:z["offsets"][k + 1]]


def choose_tau(case):
    tot = {}
    for _, d, probs in games_of(case, "_calib"):
        ph = round2(probs)
        j = int(np.argmax(ph[(1 << d) - 1]))
        keep = (ph[:, j] > 0) & (ph[:, j] < 1)
        if keep.sum() < 2:
            continue
        X, _ = design(d)
        y = logit(ph[keep, j])
        for t1 in GRID:
            for t2 in GRID:
                tot[(t1, t2)] = tot.get((t1, t2), 0.0) + log_marginal(X[keep], y, prior_diag(d, t1, t2))
    best = max(tot, key=tot.get)
    return best, {f"{a},{b}": round(v, 1) for (a, b), v in tot.items()}


def tn_draw(rng, mu, sd, lo, hi):
    a, b = ndtr((lo - mu) / sd), ndtr((hi - mu) / sd)
    u = a + rng.random(len(mu)) * (b - a)
    y = mu + sd * ndtri(np.clip(u, 1e-300, 1 - 1e-16))
    tiny = (b - a) < 1e-12
    y[tiny] = np.where(np.abs(lo[tiny] - mu[tiny]) < np.abs(hi[tiny] - mu[tiny]), lo[tiny], hi[tiny])
    y[~np.isfinite(y)] = np.where(np.isfinite(lo[~np.isfinite(y)]), lo[~np.isfinite(y)], hi[~np.isfinite(y)])
    return y


def gibbs(X, lo, hi, v0, rng):
    n, p = X.shape
    prec0 = np.diag(1.0 / v0)
    A = prec0 + X.T @ X
    Ainv = np.linalg.inv(A)
    Lc = np.linalg.cholesky(Ainv)
    y = np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2, np.where(np.isfinite(lo), lo, np.where(np.isfinite(hi), hi, 0.0)))
    draws = np.empty((ITERS - BURN, p))
    for it in range(ITERS):
        mn = Ainv @ (X.T @ y)
        bn = B0 + 0.5 * (y @ y - mn @ A @ mn)
        s2 = 1.0 / rng.gamma(A0 + n / 2, 1.0 / max(bn, 1e-12))
        theta = mn + np.sqrt(s2) * (Lc @ rng.standard_normal(p))
        y = tn_draw(rng, X @ theta, np.sqrt(s2), lo, hi)
        if it >= BURN:
            draws[it - BURN] = theta
    return draws


def wi_ei(draws, d, pairs, i):
    b = draws[:, 1 + i]
    g = sum(draws[:, 1 + d + q] for q, (a, c) in enumerate(pairs) if i in (a, c)) if d > 1 else 0.0
    return b + g, b + 0.5 * g


def item_task(args):
    case, item, d, probs, t1, t2 = args
    cid = CASE_ID[case]
    ph = round2(probs)
    j = int(np.argmax(ph[(1 << d) - 1]))
    full = (1 << d) - 1
    X, pairs = design(d)
    lo, hi = intervals(ph[:, j])
    draws = gibbs(X, lo, hi, prior_diag(d, t1, t2), np.random.default_rng(np.random.SeedSequence([20261021, 60, cid, item])))
    v = G.value(probs, j)
    m = G.mobius(v)
    _, W = G.first_last(v, d)
    E = G.shapley(m, d)

    def event(i):
        w, e = wi_ei(draws, d, pairs, i)
        qw, qe = np.percentile(w, [5, 95]), np.percentile(e, [5, 95])
        return {"unit": int(i), "W_ok": bool(qw[0] <= W[i] <= qw[1]), "E_ok": bool(qe[0] <= E[i] <= qe[1]),
                "W_width": float(qw[1] - qw[0]), "E_width": float(qe[1] - qe[0])}

    i = int(np.random.default_rng(np.random.SeedSequence([20261021, 61, cid, item])).integers(d))
    out = {"item": item, "d": d, "p0": float(ph[full, j]), "event": event(i)}
    dc = [u for u in range(d) if ph[full, j] >= 1.0 and ph[full & ~(1 << u), j] >= 1.0]
    if dc:
        u = dc[int(np.random.default_rng(np.random.SeedSequence([20261021, 62, cid, item])).integers(len(dc)))]
        out["event_dc"] = event(u)
    return out


def beta_summary(oks):
    n, k = len(oks), int(sum(oks))
    return {"n": n, "k": k, "c_hat": round(k / n, 4) if n else None,
            "L": round(float(betaincinv(k + 1, n - k + 1, 0.05)), 4) if n else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASE_ID))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-items", type=int, default=0)
    a = ap.parse_args()
    (t1, t2), grid = choose_tau(a.case)
    print(f"tau1={t1} tau2={t2}", flush=True)
    tasks = [(a.case, it, d, probs, t1, t2) for it, d, probs in games_of(a.case)][: a.max_items or None]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(item_task, tasks, chunksize=4))
    ev = [r["event"] for r in res]
    dc = [r["event_dc"] for r in res if "event_dc" in r]
    cal_p0 = [round2(p)[(1 << d) - 1].max() for _, d, p in games_of(a.case, "_calib")]
    edges = np.quantile(cal_p0, [0.2, 0.4, 0.6, 0.8])
    faixas = {}
    for qi in range(5):
        sel = [r["event"]["W_ok"] for r in res if np.searchsorted(edges, r["p0"], side="right") == qi]
        faixas[str(qi)] = {"n": len(sel), "laplace_W": round((sum(sel) + 1) / (len(sel) + 2), 4)}
    W, E = beta_summary([e["W_ok"] for e in ev]), beta_summary([e["E_ok"] for e in ev])
    Wd, Ed = beta_summary([e["W_ok"] for e in dc]), beta_summary([e["E_ok"] for e in dc])
    passa = all(s["L"] is not None and s["L"] >= 0.85 for s in (W, E, Wd, Ed)) and W["c_hat"] <= 0.97 and E["c_hat"] <= 0.97
    out = {"case": a.case, "tau1": t1, "tau2": t2, "n_items": len(res), "W": W, "E": E, "duplamente_censurado": {"W": Wd, "E": Ed},
           "largura_mediana": {"W": round(float(np.median([e["W_width"] for e in ev])), 3), "E": round(float(np.median([e["E_width"] for e in ev])), 3)},
           "por_faixa_p0": {"limites": [round(float(x), 3) for x in edges], "faixas": faixas}, "passa_G2_R1": passa,
           "grade_log_marginal": grid}
    (DER / f"r1_{a.case}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    with open(DER / f"r1_{a.case}_itens.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "grade_log_marginal"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
