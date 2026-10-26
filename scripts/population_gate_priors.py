#!/usr/bin/env python3
"""Priors of the Beta rule at equal promise (implementation_notes.md, measurements added under the
comparison protocol, part d): the plug-in and nested population levels of population_gate.py with the
Beta rule under the uniform prior of the author's rule (reference), the Beta(0,1) prior and the Jeffreys prior,
same seeds, differences taken draw by draw against the uniform prior.

Usage: population_gate_priors.py [--plugin-draws 10000] [--draws 1000] [--populations 200] [--workers 12]
"""

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
from scipy.special import betaincinv

import population_gate as PG
from ablation_gate import beta_cfg
from common import DATA

PG.RULES = {"uniform": beta_cfg(0.05, "uniform", "ascending"), "cp": beta_cfg(0.05, "cp", "ascending"),
            "jeffreys": beta_cfg(0.05, "jeffreys", "ascending")}


def job(args):
    case_idx, n_cal, b, m = args
    viol, dec = PG.draws(case_idx, n_cal, b, m)
    dv = viol.astype(float) - viol[:, [0], :]
    dd = dec - dec[:, [0], :]
    return (case_idx, n_cal, b), {"viol": viol.mean(axis=2), "dec": dec.mean(axis=2), "dviol": dv.mean(axis=2), "dviol_var": dv.var(axis=2, ddof=1),
                                  "ddec": dd.mean(axis=2), "ddec_var": dd.var(axis=2, ddof=1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin-draws", type=int, default=10000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--populations", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    m, B, mp = a.draws, a.populations, a.plugin_draws
    jobs = [(c, n, b, mp if b == 0 else m) for c in range(len(PG.CASES)) for n in PG.N_CAL for b in range(B + 1)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = dict(pool.map(job, jobs, chunksize=1))
    out = {"seed": PG.SEED, "reference": "uniform", "rules": PG.RULES, "seconds": round(time.time() - t0, 1), "results": {}}
    for c, (loc, key) in enumerate(PG.CASES):
        for n in PG.N_CAL:
            for ti, t in enumerate(PG.TARGETS):
                row = {}
                for ri, r in enumerate(PG.RULES):
                    p = res[(c, n, 0)]
                    k = int(round(p["viol"][ti, ri] * mp))
                    nest = [res[(c, n, b)] for b in range(1, B + 1)]
                    row[r] = {"violation": p["viol"][ti, ri], "violation_upper95": 1.0 if k == mp else float(betaincinv(k + 1, mp - k, 0.95)),
                              "decisions": p["dec"][ti, ri], "minus_uniform_decisions": p["ddec"][ti, ri],
                              "minus_uniform_violation": p["dviol"][ti, ri],
                              "nested_minus_uniform_decisions": PG.split([x["ddec"][ti, ri] for x in nest], [x["ddec_var"][ti, ri] for x in nest], m),
                              "nested_minus_uniform_violation": PG.split([x["dviol"][ti, ri] for x in nest], [x["dviol_var"][ti, ri] for x in nest], m)}
                out["results"][f"{loc}/{key}/ncal{n}/{t}"] = row
    (DATA / "derived" / "population_gate_priors.json").write_text(json.dumps(out, indent=1))
    print("seconds", out["seconds"])


if __name__ == "__main__":
    main()
