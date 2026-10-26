#!/usr/bin/env python3
"""Preregistered sensitivity of the drift main run (preregistration/drift.md, section 6): each retrain
costs 50 reviews. Author's loop minus periodic arm, paired by drift, pattern and replicate, and the
mean cost of the arms the article cites.

Usage: sensibilidade_retreino.py [--file main.jsonl] [--out analysis_main_retreino50.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

DER = Path(__file__).resolve().parents[2] / "data/derived/drift"
RETRAIN_COST = 50
ARMS = ("author", "periodic", "recal", "ltt_window", "ddm")


def cost(r):
    return r["cost_per_item"] + RETRAIN_COST * r["retrains"] / r["length"]


def paired(a, b, seed=20261022, draws=10000):
    d = np.asarray(a) - np.asarray(b)
    n, m, s = len(d), float(d.mean()), float(d.std(ddof=1))
    t = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    boot = np.random.default_rng(seed).choice(d, (draws, n)).mean(1)
    return {"n": n, "media": round(m, 5), "IC95_t": [round(m - t, 5), round(m + t, 5)],
            "IC95_boot": [round(float(x), 5) for x in np.percentile(boot, [2.5, 97.5])], "dp": round(s, 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="main.jsonl")
    ap.add_argument("--out", default="analysis_main_retreino50.json")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(DER / a.file)]
    by = {(r["arm"], r["drift"], r["pattern"], r["replicate"]): r for r in rows}
    out = {"sensibilidade": "custo de retreino 50 (pre-registro, secao 6)"}
    for drift in ("D1", "D2", "D3", "control"):
        keys = sorted(k[2:] for k in by if k[0] == "author" and k[1] == drift and ("periodic", drift) + k[2:] in by)
        out[drift] = {"autor_menos_periodico": paired([cost(by[("author", drift) + k]) for k in keys],
                                                      [cost(by[("periodic", drift) + k]) for k in keys]),
                      "medias": {arm: round(float(np.mean([cost(r) for r in rows if r["arm"] == arm and r["drift"] == drift])), 4)
                                 for arm in ARMS}}
    (DER / a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({d: (out[d]["autor_menos_periodico"]["media"], out[d]["autor_menos_periodico"]["IC95_t"]) for d in ("D1", "D2", "D3", "control")}))


if __name__ == "__main__":
    main()
