#!/usr/bin/env python3
"""Summaries of data/derived/ablation_gate.json: ranks of the core rules on every gate measure,
the ladder that adds one ingredient at a time from the point rule to Learn then Test, Pareto layers,
epsilon-constraint winners by family, and the effect of calibration-set size and of calibration.

Main setting: n_cal = 1000, temperature for Laya, Jev as returned.

Usage: ablation_report.py   (writes data/derived/ablation_summary.json and prints the tables)
"""

import json

import numpy as np
from scipy.stats import rankdata

from common import DATA

USER = "beta q=0.05 uniform min 20 ascending"
CORE = ["point", USER, "beta q=0.05 uniform min 20 sequence_testable", "ltt delta=0.05", "sgr delta=0.05",
        "bayes C_e/C_r=10", "bayes C_e/C_r=100", "per-class point", "per-class beta"]
SHORT = {"point": "point", USER: "Beta (ascending)", "beta q=0.05 uniform min 20 sequence_testable": "Beta, fixed sequence",
         "ltt delta=0.05": "LTT", "sgr delta=0.05": "SGR", "bayes C_e/C_r=10": "Bayes 10:1", "bayes C_e/C_r=100": "Bayes 100:1",
         "per-class point": "per-class point", "per-class beta": "per-class Beta"}
LADDER = ["point", USER, "beta q=0.05 uniform min 20 sequence_testable", "beta q=0.05 cp min 20 sequence_testable", "ltt delta=0.05"]
# measure: True if higher is better
MEASURES = {"broken_per_split": False, "clearly_broken_per_split": False, "decisions_per_1000": True, "type1_per_1000": False,
            "type2_per_1000": False, "gate_f1": True, "cost_ratio10": False, "cost_ratio100": False, "reviews_per_avoided_error": False}
SYSTEMS = [("pt", "ft-multilingual"), ("pt", "ft-typed-decisions"), ("pt", "jev"), ("en", "jev"), ("en", "typed-decisions"), ("en", "root")]


def main_key(loc, key, target, n_cal=1000, calib=None):
    calib = calib or ("as_returned" if key == "jev" else "temperature")
    return f"{loc}/{key}/{calib}/ncal{n_cal}/{target}"


def family(label):
    if label.startswith("beta"):
        parts = label.split()
        return f"beta {parts[2]} {parts[-1]}"
    return label.split()[0] if not label.startswith("per-class") else label


def pareto_layers(points):
    """Non-dominated sorting for (minimise a, maximise b); returns layer index per point (0 = front)."""
    layer = np.full(len(points), -1)
    remaining, k = set(range(len(points))), 0
    while remaining:
        front = {i for i in remaining if not any(points[j][0] <= points[i][0] and points[j][1] >= points[i][1]
                                                  and points[j] != points[i] for j in remaining)}
        for i in front:
            layer[i] = k
        remaining -= front
        k += 1
    return layer


def main():
    d = json.loads((DATA / "derived" / "ablation_gate.json").read_text())
    R = d["results"]
    out = {"ranks": {}, "mean_rank": {}, "ladder": {}, "pareto": {}, "epsilon": {}, "n_cal": {}, "calibration": {}}
    cases = [(loc, key, t) for loc, key in SYSTEMS for t in (0.95, 0.99)]
    # ranks of the core rules per measure and case; ties share the mean rank
    for m, higher in MEASURES.items():
        per_rule = {r: [] for r in CORE}
        for loc, key, t in cases:
            rows = {x["label"]: x for x in R[main_key(loc, key, t)]}
            vals = np.array([rows[r][m] if rows[r][m] is not None else np.nan for r in CORE], float)
            ok = ~np.isnan(vals)
            if ok.sum() < 2:
                continue
            rk = np.full(len(CORE), np.nan)
            rk[ok] = rankdata(-vals[ok] if higher else vals[ok])
            for r, v in zip(CORE, rk):
                per_rule[r].append(v)
        out["mean_rank"][m] = {SHORT[r]: round(float(np.nanmean(v)), 2) for r, v in per_rule.items()}
    # constrained rank: most decisions among rules whose broken rate is at most 0.10 in that case
    cons = {r: [] for r in CORE}
    for loc, key, t in cases:
        rows = {x["label"]: x for x in R[main_key(loc, key, t)]}
        elig = [r for r in CORE if rows[r]["broken_per_split"] <= 0.10]
        rk = rankdata([-rows[r]["decisions_per_1000"] for r in elig])
        for r in CORE:
            cons[r].append(float(rk[elig.index(r)]) if r in elig else np.nan)
    out["mean_rank"]["decisions, broken <= 0.10"] = {SHORT[r]: (round(float(np.nanmean(v)), 2) if not np.all(np.isnan(v)) else None)
                                                      for r, v in cons.items()}
    out["mean_rank"]["eligible cases, broken <= 0.10"] = {SHORT[r]: int(np.sum(~np.isnan(v))) for r, v in cons.items()}
    # ladder and Pareto layers per case
    for loc, key, t in cases:
        k = main_key(loc, key, t)
        rows = R[k]
        by = {x["label"]: x for x in rows}
        out["ladder"][k] = [{m: by[r][m] for m in ("label", "broken_per_split", "clearly_broken_per_split", "decisions_per_1000",
                                                   "type1_per_1000", "optimism", "qualified_splits")} for r in LADDER]
        cfg_rows = [x for x in rows if x["label"] != "oracle"]
        lay = pareto_layers([(x["broken_per_split"], x["decisions_per_1000"]) for x in cfg_rows])
        lay2 = pareto_layers([(x["type1_per_1000"], -x["type2_per_1000"]) for x in cfg_rows])
        out["pareto"][k] = {SHORT.get(x["label"], x["label"]): [int(a), int(b)] for x, a, b in zip(cfg_rows, lay, lay2)
                            if x["label"] in CORE}
        out["pareto"][k]["oracle_decisions"] = by["oracle"]["decisions_per_1000"]
        eps = {}
        for e in (0.05, 0.10):
            best = {}
            for x in cfg_rows:
                f = family(x["label"])
                if x["broken_per_split"] <= e and x["decisions_per_1000"] > best.get(f, (-1, ""))[0]:
                    best[f] = (x["decisions_per_1000"], x["label"])
            eps[str(e)] = {f: v for f, v in sorted(best.items(), key=lambda z: -z[1][0])}
        out["epsilon"][k] = eps
    # calibration-set size and calibration on/off for the core rules
    for loc, key in SYSTEMS:
        for t in (0.95, 0.99):
            out["n_cal"][f"{loc}/{key}/{t}"] = {n: {SHORT[x["label"]]: [x["broken_per_split"], x["decisions_per_1000"]]
                                                     for x in R[main_key(loc, key, t, n)] if x["label"] in CORE}
                                                for n in (1000, 500, 250)}
            if key != "jev":
                out["calibration"][f"{loc}/{key}/{t}"] = {c: {SHORT[x["label"]]: [x["broken_per_split"], x["decisions_per_1000"]]
                                                              for x in R[main_key(loc, key, t, 1000, c)] if x["label"] in CORE}
                                                          for c in ("temperature", "as_returned")}
    (DATA / "derived" / "ablation_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    names = [SHORT[r] for r in CORE]
    print("mean rank over 12 cases (1 = best)")
    print(f"{'measure':34s}" + "".join(f"{n[:13]:>14s}" for n in names))
    for m, v in out["mean_rank"].items():
        print(f"{m[:34]:34s}" + "".join(f"{(v[n] if v[n] is not None else float('nan')):>14.2f}" for n in names))


if __name__ == "__main__":
    main()
