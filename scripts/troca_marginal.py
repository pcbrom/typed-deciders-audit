#!/usr/bin/env python3
"""Accuracy among accepted items and reviews per avoided error along coverage, report half (items
1000 to 1999), Jev as returned and the fine-tuned multilingual Laya after temperature scaling, pt-PT.
Lower bound: 5% quantile of the Beta(c + 1, n - c + 1) posterior. Descriptive.

Usage: troca_marginal.py
"""

import json

import numpy as np
from scipy.special import betaincinv

from common import DATA, fit_temperature, load_items, load_systems, log_probs, softmax_t

GRID = (1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65)


def main():
    _, gold = load_items("pt")
    systems = load_systems("pt")
    out = {}
    for key in ("jev", "ft-multilingual"):
        P = systems[key]
        if key != "jev":
            lp = log_probs(P)
            P = softmax_t(lp, fit_temperature(lp[:1000], gold[:1000]))
        conf, ok = P[1000:].max(1), P[1000:].argmax(1) == gold[1000:]
        okc = ok[np.argsort(-conf, kind="stable")]
        rows, prev = [], None
        for cov in GRID:
            k = int(round(cov * len(okc)))
            err = int((~okc[:k]).sum())
            row = {"cobertura": cov, "aceitos": k, "erros": err, "acc": round(float(okc[:k].mean()), 4),
                   "L95": round(float(betaincinv(k - err + 1, err + 1, 0.05)), 4)}
            if prev:
                row["revisoes_a_mais"], row["erros_evitados"] = prev[0] - k, prev[1] - err
            rows.append(row)
            prev = (k, err)
        out[key] = rows
    (DATA / "derived" / "troca_marginal.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
