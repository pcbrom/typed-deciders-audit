#!/usr/bin/env python3
"""Gate G3, measure on the redundant type: macro F1 per item of the labels único, redundante, sinérgico
and nenhum over units and pairs, for our Gibbs, the Shapley interaction index, the sign of H_i and HEDGE,
all on outputs rounded to two decimals, with the exact Möbius ceiling (implementation_notes.md, G3).

Usage: g3_f1.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import itertools  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import games as G  # noqa: E402
from r1 import design, gibbs, intervals, prior_diag, round2  # noqa: E402

DER = Path(__file__).resolve().parents[2] / "data/derived/explicacao"
LABELS = ("unico", "redundante", "sinergico", "nenhum")
METHODS = ("teto_mobius", "gibbs", "interacao_shapley", "sinal_H", "hedge")


def mask(units):
    return sum(1 << u for u in units)


def label_by_thresholds(d, main, inter):
    """Signature thresholds applied to main effects a_i and pairwise interactions c_ik."""
    lab = {}
    for i, k in itertools.combinations(range(d), 2):
        a, b, c = main[i], main[k], inter[(i, k)]
        if a >= 1 and b >= 1 and c <= -0.5 * min(a, b):
            lab[(i, k)] = "redundante"
        elif abs(a) <= 0.5 and abs(b) <= 0.5 and c >= 1:
            lab[(i, k)] = "sinergico"
        else:
            lab[(i, k)] = "nenhum"
    for i in range(d):
        s = sum(inter[tuple(sorted((i, k)))] for k in range(d) if k != i)
        lab[i] = "unico" if main[i] >= 1 and abs(s) <= 0.5 * main[i] else "nenhum"
    return lab


def banzhaf(v, d, i):
    others = [u for u in range(d) if u != i]
    return np.mean([v[mask(c) | (1 << i)] - v[mask(c)] for r in range(len(others) + 1) for c in itertools.combinations(others, r)])


def span_interaction(v, left, right, others):
    vals = []
    for r in range(len(others) + 1):
        for cs in itertools.combinations(others, r):
            C = mask([u for s in cs for u in s])
            Lm, Rm = mask(left), mask(right)
            vals.append(v[Lm | Rm | C] - v[Lm | C] - v[Rm | C] + v[C])
    return float(np.mean(vals))


def hedge(v, d):
    spans, sep = [list(range(d))], {}
    while any(len(s) > 1 for s in spans):
        best = None
        for si, s in enumerate(spans):
            if len(s) < 2:
                continue
            others = [o for oi, o in enumerate(spans) if oi != si]
            for t in range(1, len(s)):
                val = span_interaction(v, s[:t], s[t:], others)
                if best is None or abs(val) < abs(best[0]):
                    best = (val, si, t)
        val, si, t = best
        s = spans.pop(si)
        left, right = s[:t], s[t:]
        for i in left:
            for k in right:
                sep[tuple(sorted((i, k)))] = val
        spans[si:si] = [left, right]
    phi = [banzhaf(v, d, i) for i in range(d)]
    lab = {}
    for (i, k), c in sep.items():
        a, b = phi[i], phi[k]
        lab[(i, k)] = ("redundante" if a >= 1 and b >= 1 and c <= -0.5 * min(a, b) else
                       "sinergico" if abs(a) <= 0.5 and abs(b) <= 0.5 and c >= 1 else "nenhum")
    for i in range(d):
        lab[i] = "unico" if phi[i] >= 1 and all(lab[p] == "nenhum" for p in lab if isinstance(p, tuple) and i in p) else "nenhum"
    return lab


def gibbs_labels(ph_t, d, seed, item):
    X, pairs = design(d)
    lo, hi = intervals(ph_t)
    dr = gibbs(X, lo, hi, prior_diag(d, 2.0, 1.0), np.random.default_rng(np.random.SeedSequence([20261021, 85, seed, item])))
    beta = dr[:, 1:1 + d]
    q = lambda x, p: np.percentile(x, p)
    lab, cred = {}, {}
    for qi, (i, k) in enumerate(pairs):
        g = dr[:, 1 + d + qi]
        cred[(i, k)] = q(g, 5) > 0 or q(g, 95) < 0
        if q(beta[:, i], 5) > 0 and q(beta[:, k], 5) > 0 and q(g, 95) < 0:
            lab[(i, k)] = "redundante"
        elif q(g, 5) > 0:
            lab[(i, k)] = "sinergico"
        else:
            lab[(i, k)] = "nenhum"
    for i in range(d):
        lab[i] = "unico" if q(beta[:, i], 5) > 0 and not any(c for p, c in cred.items() if i in p) else "nenhum"
    return lab


def macro_f1(truth, pred):
    f1s = []
    for L in LABELS:
        tp = sum(1 for e in truth if truth[e] == L and pred[e] == L)
        fp = sum(1 for e in truth if truth[e] != L and pred[e] == L)
        fn = sum(1 for e in truth if truth[e] == L and pred[e] != L)
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def score(args):
    seed, n, probs, d, target, pos, item = args
    a1, a2 = sorted((pos["A1"], pos["A2"]))
    truth = {i: "nenhum" for i in range(d)}
    truth.update({p: "nenhum" for p in itertools.combinations(range(d), 2)})
    truth[(a1, a2)] = "redundante"
    m_exact = G.mobius(G.value(probs, target))
    ph = round2(probs)
    v = G.value(ph, target)
    m = G.mobius(v)
    pairs = list(itertools.combinations(range(d), 2))
    out = {}
    out["teto_mobius"] = label_by_thresholds(d, [m_exact[1 << i] for i in range(d)], {p: m_exact[mask(p)] for p in pairs})
    Iik = G.interaction(m, d)
    out["interacao_shapley"] = label_by_thresholds(d, [m[1 << i] for i in range(d)], {(i, k): Iik[i, k] for i, k in pairs})
    F, W = G.first_last(v, d)
    H = W - F
    lab = {}
    for i, k in pairs:
        lab[(i, k)] = ("redundante" if F[i] >= 1 and F[k] >= 1 and H[i] < 0 and H[k] < 0 else
                       "sinergico" if abs(F[i]) <= 0.5 and abs(F[k]) <= 0.5 and H[i] > 0 and H[k] > 0 else "nenhum")
    for i in range(d):
        lab[i] = "unico" if F[i] >= 1 and abs(H[i]) <= 0.5 * F[i] else "nenhum"
    out["sinal_H"] = lab
    out["hedge"] = hedge(v, d)
    out["gibbs"] = gibbs_labels(ph[:, target], d, seed, item)
    return {"seed": seed, "item": item, **{k: macro_f1(truth, out[k]) for k in METHODS}}


def main():
    tasks = []
    for seed in (0, 1, 2):
        z = np.load(DER / f"plantio_s{seed}.npz")
        meta = json.loads(str(z["meta"]))["redundante"]
        for n, x in enumerate(meta):
            d = int(z["redundante_d"][n])
            P = z["redundante_probs"][z["redundante_offsets"][n]:z["redundante_offsets"][n + 1]]
            tasks.append((seed, n, P, d, x["target"], x["pos"], x["item"]))
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(score, tasks, chunksize=4))
    res = {"por_semente": {}, "agrupado": {}}
    for s in (0, 1, 2):
        rs = [r for r in rows if r["seed"] == s]
        res["por_semente"][str(s)] = {k: round(float(np.mean([r[k] for r in rs])), 4) for k in METHODS}
    A = {k: np.array([r[k] for r in rows]) for k in METHODS}
    res["agrupado"] = {k: round(float(A[k].mean()), 4) for k in METHODS}
    comp = max(("interacao_shapley", "sinal_H", "hedge"), key=lambda k: A[k].mean())
    rng = np.random.default_rng(np.random.SeedSequence([20261021, 84]))
    diff = A["gibbs"] - A[comp]
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    g = A["gibbs"].mean()
    res["melhor_concorrente"] = comp
    res["diferenca_gibbs_menos_concorrente"] = {"media": round(float(diff.mean()), 4), "IC95": [round(float(lo), 4), round(float(hi), 4)]}
    res["leitura"] = ("passa" if g >= 0.80 and diff.mean() >= 0.10 and lo > 0 else
                      "equivalente ao estado da arte sob censura" if g >= 0.80 else "resultado negativo")
    res["n_itens"] = len(rows)
    (DER / "g3_f1_redundante.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
