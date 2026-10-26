#!/usr/bin/env python3
"""Divergence of the ascending and fixed-sequence scans of the Beta rule on the plug-in population
(implementation_notes.md, ascending against fixed-sequence scan): per draw, the thresholds of the
author's rule (ascending), the same rule with the fixed-sequence scan over testable thresholds, and
Learn then Test; the event D that the first two differ; violations; coverage of the ascending threshold.

Usage: population_gate_divergence.py [--draws 10000] [--workers 12]
"""

import argparse
import json
from multiprocessing import Pool

import numpy as np

import population_gate as PG
from ablation_gate import beta_cfg, threshold
from common import DATA, fit_temperature, load_items, load_systems, log_probs, softmax_t

RULES = {"asc": beta_cfg(0.05, "uniform", "ascending"), "fix": beta_cfg(0.05, "uniform", "sequence_testable"),
         "ltt": {"rule": "ltt", "delta": 0.05, "min_n": 20}}


def job(args):
    case_idx, n_cal, m = args
    loc, key = PG.CASES[case_idx]
    _, gold = load_items(loc)
    P = load_systems(loc)[key]
    rng = np.random.default_rng(np.random.SeedSequence([PG.SEED, case_idx, n_cal, 0]))
    lp = log_probs(P)
    out = {t: {"D": 0, "V_asc": 0, "V_fix": 0, "V_ltt": 0, "V_asc_D": 0, "V_asc_notfix": 0, "cov_D": [], "cov_noD": []} for t in PG.TARGETS}
    for _ in range(m):
        cal = rng.integers(0, len(gold), n_cal)
        Q = P if key == "jev" else softmax_t(lp, fit_temperature(lp[cal], gold[cal]))
        conf, ok = Q.max(axis=1), Q.argmax(axis=1) == gold
        for t in PG.TARGETS:
            tau = {r: threshold(cfg, conf[cal], ok[cal], t) for r, cfg in RULES.items()}
            viol = {r: (tau[r] is not None and ok[conf >= tau[r]].mean() < t) for r in RULES}
            d = tau["asc"] != tau["fix"]
            o = out[t]
            o["D"] += d
            o["V_asc"] += viol["asc"]; o["V_fix"] += viol["fix"]; o["V_ltt"] += viol["ltt"]
            o["V_asc_D"] += viol["asc"] and d
            o["V_asc_notfix"] += viol["asc"] and not viol["fix"]
            if tau["asc"] is not None:
                (o["cov_D"] if d else o["cov_noD"]).append(float((conf >= tau["asc"]).mean()))
    res = {}
    for t, o in out.items():
        res[str(t)] = {k: round(o[k] / m, 4) for k in ("D", "V_asc", "V_fix", "V_ltt", "V_asc_D", "V_asc_notfix")}
        res[str(t)]["inclusao_ok"] = bool(o["V_asc_notfix"] <= o["D"])
        for k in ("cov_D", "cov_noD"):
            res[str(t)][k + "_mediana"] = round(float(np.median(o[k])), 4) if o[k] else None
    return (case_idx, n_cal), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    jobs = [(c, n, a.draws) for c in range(len(PG.CASES)) for n in PG.N_CAL]
    with Pool(a.workers) as pool:
        res = dict(pool.map(job, jobs, chunksize=1))
    out = {"seed": PG.SEED, "draws": a.draws, "rules": RULES, "results": {}}
    for (c, n), r in res.items():
        loc, key = PG.CASES[c]
        for t, v in r.items():
            out["results"][f"{loc}/{key}/ncal{n}/{t}"] = v
    (DATA / "derived" / "population_gate_divergence.json").write_text(json.dumps(out, indent=1))
    for k, v in sorted(out["results"].items()):
        print(k, v)


if __name__ == "__main__":
    main()
