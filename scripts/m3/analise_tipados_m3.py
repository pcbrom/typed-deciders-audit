#!/usr/bin/env python3
"""Typed deciders against ETS on the M3 sample (implementation_notes.md, gate on time series): 150 series at the
last 12 origins. Accuracy and macro-F1 with 95% intervals (bootstrap over series), share of "higher" decisions,
paired accuracy difference against ETS on the same decisions, and the selective ceiling (accuracy of the most
confident decisions). Jev and Laya are read on this sample only.

Usage: analise_tipados_m3.py
Output: data/derived/m3/analise_tipados_m3.json
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from common import DATA  # noqa: E402

D = DATA / "derived" / "m3"
DRAWS, SEED = 2000, 20261026


def macro_f1(pred_higher, higher):
    f = []
    for cls in (True, False):
        tp = np.sum((pred_higher == cls) & (higher == cls))
        fp = np.sum((pred_higher == cls) & (higher != cls))
        fn = np.sum((pred_higher != cls) & (higher == cls))
        f.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f))


def boot(series, fn, rng):
    """Bootstrap over series: resample series with replacement, keep all their origins."""
    uniq = np.unique(series)
    idx = {s: np.flatnonzero(series == s) for s in uniq}
    vals = []
    for _ in range(DRAWS):
        take = np.concatenate([idx[s] for s in rng.choice(uniq, len(uniq))])
        vals.append(fn(take))
    return [round(float(x), 4) for x in np.percentile(vals, [2.5, 97.5])]


def deciders():
    L = np.load(D / "laya.npz")
    series, origin, gold = L["series"], L["origin"], L["gold"]
    higher = gold == 0
    E = np.load(D / "ets.npz")
    p = {"ETS": E["p_higher"][series, origin]}
    assert np.array_equal(E["higher"][series, origin].astype(bool), higher)
    jev = {(r["series"], r["origin"]): r["probs"][0] for r in map(json.loads, open(D / "jev.jsonl"))}
    p["Jev"] = np.array([jev[(s, o)] for s, o in zip(series, origin)])
    p["Laya released"] = L["released"][:, 0]
    for k in range(3):
        p[f"Laya fine-tuned, seed {k}"] = L[f"ft{k}"][:, 0]
    p["Laya fine-tuned, mean of seeds"] = np.mean([L[f"ft{k}"][:, 0] for k in range(3)], axis=0)
    return series, higher, p


def main():
    series, higher, P = deciders()
    rng = np.random.default_rng(SEED)
    ok = {name: (p >= 0.5) == higher for name, p in P.items()}
    out = {"n_decisions": int(len(series)), "n_series": int(len(np.unique(series))), "base_rate_higher": round(float(higher.mean()), 4),
           "constant_not_higher_accuracy": round(float(1 - higher.mean()), 4), "deciders": {}}
    for name, p in P.items():
        pred = p >= 0.5
        conf = np.maximum(p, 1 - p)
        order = np.argsort(-conf, kind="stable")
        d = ok[name].astype(float) - ok["ETS"].astype(float)
        out["deciders"][name] = {
            "accuracy": round(float(ok[name].mean()), 4), "accuracy_IC95": boot(series, lambda t: ok[name][t].mean(), rng),
            "macro_f1": round(macro_f1(pred, higher), 4), "macro_f1_IC95": boot(series, lambda t: macro_f1(pred[t], higher[t]), rng),
            "share_higher": round(float(pred.mean()), 4),
            "minus_ets": None if name == "ETS" else {"mean": round(float(d.mean()), 4), "IC95": boot(series, lambda t: d[t].mean(), rng)},
            "ceiling": {str(k): round(float(ok[name][order[:k]].mean()), 4) for k in (50, 100, 200, 500)}}
        r = out["deciders"][name]
        print(f"{name:32s} acc {r['accuracy']} {r['accuracy_IC95']} F1 {r['macro_f1']} higher {r['share_higher']}",
              "vs ETS", r["minus_ets"], "top100", r["ceiling"]["100"])
    (D / "analise_tipados_m3.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
