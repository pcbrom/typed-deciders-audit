#!/usr/bin/env python3
"""Validation 4 of the explanation front on Laya: does the explanation depend on the caller's options?

For each item, W^{jk*}_i = y_jk(N) - y_jk(N without i) with y_jk = log p_j - log p_k* on the original
options and on each alternative configuration (the three options other than j and k* replaced). The
maximal set A gathers the units within 0.01 logit of the maximum; there is a change when the sets of the
two configurations do not intersect. An item leaves a configuration when its full-utterance decision is
no longer j or its runner-up is no longer k*. D = change rate against the alternative minus change rate
against the placebo (original options permuted), with a 95% item bootstrap
(preregistration/explicacao_portoes.md, Validação 4).

Usage: val4.py --case pt_multilingual
"""

import argparse
import json
from pathlib import Path

import numpy as np

import games as G

DER = Path(__file__).resolve().parents[2] / "data/derived/explicacao"


def wjk(probs, d, j, k):
    v = G.value(probs, j, k)
    full = (1 << d) - 1
    return np.array([v[full] - v[full & ~(1 << i)] for i in range(d)])


def max_set(w):
    return set(np.flatnonzero(w >= w.max() - 0.01))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=("pt_multilingual", "en_multilingual"))
    a = ap.parse_args()
    z = np.load(DER / f"laya_validations_{a.case}.npz")
    g = np.load(DER / f"laya_games_{a.case}.npz")
    orig = {int(g["item"][k]): g["probs"][g["offsets"][k]:g["offsets"][k + 1]] for k in range(len(g["d"]))}
    rows = []
    for n, item in enumerate(z["item"]):
        item, d, j, k = int(item), int(z["d"][n]), int(z["j"][n]), int(z["k_star"][n])
        a_orig = max_set(wjk(orig[item], d, j, k))
        rec = {"item": item, "d": d}
        for kind in ("alt1", "alt2", "placebo"):
            p = z[f"{kind}_probs"][z[f"{kind}_offsets"][n]:z[f"{kind}_offsets"][n + 1]]
            if kind == "placebo":
                orig_opts = list(z["operator_options"][n])  # the original options
                opts = list(z["placebo_options"][n])
                jj, kk = opts.index(orig_opts[j]), opts.index(orig_opts[k])
            else:
                jj, kk = j, k
            order = np.argsort(-p[(1 << d) - 1], kind="stable")
            valid = int(order[0]) == jj and int(order[1]) == kk
            rec[kind] = {"valid": valid, "change": (len(a_orig & max_set(wjk(p, d, jj, kk))) == 0) if valid else None}
        rows.append(rec)
    rng = np.random.default_rng(np.random.SeedSequence([20261021, 56]))
    res = {"case": a.case, "n_items": len(rows)}
    for alt in ("alt1", "alt2"):
        ok = [r for r in rows if r[alt]["valid"] and r["placebo"]["valid"]]
        ca = np.array([r[alt]["change"] for r in ok], float)
        cp = np.array([r["placebo"]["change"] for r in ok], float)
        D = float(ca.mean() - cp.mean())
        boot = [float(ca[s].mean() - cp[s].mean()) for s in (rng.integers(0, len(ok), len(ok)) for _ in range(10000))]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        cls = ("resultado central" if D >= 0.10 and lo > 0 else "dependência detectável" if D >= 0.02 and lo > 0 else "invariância aproximada")
        res[alt] = {"n_validos": len(ok), "excluidos_frac": round(1 - len(ok) / len(rows), 4), "mudanca_alt": round(float(ca.mean()), 4),
                    "mudanca_placebo": round(float(cp.mean()), 4), "D": round(D, 4), "IC95": [round(float(lo), 4), round(float(hi), 4)], "leitura": cls}
    (DER / f"val4_laya_{a.case}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
