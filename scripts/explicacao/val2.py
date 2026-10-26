#!/usr/bin/env python3
"""Validation 2 of the explanation front on Laya: deletion and insertion percentiles by method and
operator, with the Kendall agreement of the method rankings across operators (implementation_notes.md).

Usage: val2.py --case pt_multilingual
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

import concorrentes as C  # noqa: E402
import games as G  # noqa: E402
from g3_f1 import banzhaf  # noqa: E402
from g5 import deletion_percentile, orders  # noqa: E402

DER = Path(__file__).resolve().parents[2] / "data/derived/explicacao"
CID = {"pt_multilingual": 0, "en_multilingual": 1}
METHODS = ("shapley", "W", "F", "kernelshap", "bayesshap", "hedge")


def insertion_percentile(v, d, order):
    _, prefix = orders(d)
    areas = (v[prefix] - v[0]).sum(1)
    own = (v[np.cumsum([1 << int(u) for u in order])] - v[0]).sum()
    return float(((areas < own - 1e-12).sum() + 0.5 * (np.abs(areas - own) <= 1e-12).sum()) / len(areas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CID))
    a = ap.parse_args()
    z = np.load(DER / f"laya_validations_{a.case}.npz")
    g = np.load(DER / f"laya_games_{a.case}.npz")
    orig = {int(g["item"][k]): g["probs"][g["offsets"][k]:g["offsets"][k + 1]] for k in range(len(g["d"]))}
    acc = {(m, op, cv): [] for m in METHODS for op in ("remocao", "troca") for cv in ("morf", "insercao")}
    for n, item in enumerate(z["item"]):
        item, d, j = int(item), int(z["d"][n]), int(z["j"][n])
        if d < 2:
            continue
        v = G.value(orig[item], j)
        vop = G.value(z["operator_probs"][z["operator_offsets"][n]:z["operator_offsets"][n + 1]], j)
        F, W = G.first_last(v, d)
        masks = C.kernel_samples(d, np.random.default_rng(np.random.SeedSequence([20261021, 86, CID[a.case], item])))
        att = {"shapley": G.shapley(G.mobius(v), d), "W": W, "F": F,
               "kernelshap": C.kernelshap_cl(v, d, masks)[0],
               "bayesshap": C.bayesshap(np.asarray(orig[item], float)[:, j], d, masks)[0],
               "hedge": np.array([banzhaf(v, d, i) for i in range(d)])}
        for m, s in att.items():
            order = np.argsort(-np.asarray(s), kind="stable")
            for op, game in (("remocao", v), ("troca", vop)):
                acc[(m, op, "morf")].append(deletion_percentile(game, d, order))
                acc[(m, op, "insercao")].append(insertion_percentile(game, d, order))
    np.savez_compressed(DER / f"val2_laya_{a.case}_itens.npz", **{"|".join(k): np.array(v) for k, v in acc.items()})
    res = {"case": a.case, "n_itens": len(acc[("shapley", "remocao", "morf")]), "media_u": {}, "kendall": {}}
    for cv in ("morf", "insercao"):
        means = {op: {m: float(np.mean(acc[(m, op, cv)])) for m in METHODS} for op in ("remocao", "troca")}
        res["media_u"][cv] = {op: {m: round(x, 4) for m, x in mm.items()} for op, mm in means.items()}
        tau = kendalltau([means["remocao"][m] for m in METHODS], [means["troca"][m] for m in METHODS])[0]
        res["kendall"][cv] = {"tau": round(float(tau), 4), "ordena": bool(tau >= 0.6),
                              "ranking_remocao": sorted(METHODS, key=lambda m: -means["remocao"][m])}
    (DER / f"val2_laya_{a.case}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
