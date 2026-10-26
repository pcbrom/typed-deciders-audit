#!/usr/bin/env python3
"""Gate rules against a plug-in population, the primary measure of the article.

Under the exchangeability already assumed by every rule, the 2,000 items of a locale are taken as
the population. A calibration sample of n_cal items is drawn from it with replacement, the
temperature (Laya) and the threshold are fitted on the sample, and the true selective accuracy of
the chosen threshold is computed on the whole population. A violation is a true selective accuracy
below the target. This checks rules against the empirical distribution; it is not a statement about
the unknown population behind the sample.

Two levels:
  plug-in  population = the 2,000 observed items, M draws
  nested   B bootstrap resamples of the 2,000 items, each taken as a population, M draws each

From the nested level, the between-population variance of a rate is split into a Monte Carlo part
and an item part:  w = mean_b p_b (1 - p_b) / (M - 1),  s2_item = max(0, var_b(p_b) - w).
The same split is applied to decisions (within-population variance of decisions / M) and to the
paired differences between rules, computed draw by draw.

Seeds: numpy SeedSequence([20261020, case, n_cal, b]), b = 0 for the plug-in population and
b = 1..B for the nested ones. The plug-in level uses --plugin-draws draws (default 10,000), the nested
level --draws draws per population.

Usage: population_gate.py [--plugin-draws 10000] [--draws 1000] [--populations 200] [--workers 20]
"""

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
from scipy.special import betaincinv

from ablation_gate import beta_cfg, threshold
from common import DATA, fit_temperature, load_items, load_systems, log_probs, softmax_t

SEED = 20261020
RULES = {"point": {"rule": "point", "min_n": 20},
         "beta_ascending": beta_cfg(0.05, "uniform", "ascending"),
         "beta_fixed_sequence": beta_cfg(0.05, "uniform", "sequence_testable"),
         "ltt": {"rule": "ltt", "delta": 0.05, "min_n": 20},
         "sgr": {"rule": "sgr", "delta": 0.05, "min_n": 20}}
CASES = [("pt", "jev"), ("pt", "ft-multilingual"), ("pt", "ft-typed-decisions"),
         ("en", "jev"), ("en", "typed-decisions"), ("en", "root")]
TARGETS = (0.95, 0.99)
N_CAL = (1000, 500)


def draws(case_idx, n_cal, b, m):
    """Violation and decisions per draw, per target and rule, for population b of a case."""
    loc, key = CASES[case_idx]
    _, gold0 = load_items(loc)
    P0 = load_systems(loc)[key]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, case_idx, n_cal, b]))
    pop = np.arange(len(gold0)) if b == 0 else rng.integers(0, len(gold0), len(gold0))
    P, gold = P0[pop], gold0[pop]
    lp = log_probs(P)
    viol = np.zeros((len(TARGETS), len(RULES), m), bool)
    dec = np.zeros((len(TARGETS), len(RULES), m))
    for i in range(m):
        cal = rng.integers(0, len(gold), n_cal)
        Q = P if key == "jev" else softmax_t(lp, fit_temperature(lp[cal], gold[cal]))
        conf, ok = Q.max(axis=1), Q.argmax(axis=1) == gold
        for ti, t in enumerate(TARGETS):
            for ri, cfg in enumerate(RULES.values()):
                tau = threshold(cfg, conf[cal], ok[cal], t)
                if tau is None:
                    continue
                acc = conf >= tau
                viol[ti, ri, i] = ok[acc].mean() < t
                dec[ti, ri, i] = acc.mean() * 1000
    return viol, dec


def job(args):
    case_idx, n_cal, b, m = args
    viol, dec = draws(case_idx, n_cal, b, m)
    li = list(RULES).index("ltt")
    return (case_idx, n_cal, b), {
        "viol": viol.mean(axis=2), "dec": dec.mean(axis=2), "dec_var": dec.var(axis=2, ddof=1),
        "dviol": (viol.astype(float) - viol[:, [li], :]).mean(axis=2),
        "dviol_var": (viol.astype(float) - viol[:, [li], :]).var(axis=2, ddof=1),
        "ddec": (dec - dec[:, [li], :]).mean(axis=2), "ddec_var": (dec - dec[:, [li], :]).var(axis=2, ddof=1),
        "only_rule": (viol & ~viol[:, [li], :]).mean(axis=2), "only_ltt": (~viol & viol[:, [li], :]).mean(axis=2)}


def split(means, within_var, m):
    """Mean, Monte Carlo SD and item SD of a per-population mean over B populations."""
    means = np.asarray(means)
    w = float(np.mean(within_var)) / m
    total = float(np.var(means, ddof=1))
    return {"mean": float(np.mean(means)), "sd_total": total ** 0.5, "sd_mc": w ** 0.5, "sd_item": max(0.0, total - w) ** 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin-draws", type=int, default=10000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--populations", type=int, default=200)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    m, B, mp = args.draws, args.populations, args.plugin_draws
    jobs = [(c, n, b, mp if b == 0 else m) for c in range(len(CASES)) for n in N_CAL for b in range(B + 1)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        res = dict(pool.map(job, jobs, chunksize=1))
    out = {"seed": SEED, "plugin_draws": mp, "draws": m, "populations": B, "rules": RULES, "targets": TARGETS, "n_cal": N_CAL,
           "seconds": round(time.time() - t0, 1), "results": {}}
    for c, (loc, key) in enumerate(CASES):
        for n in N_CAL:
            for ti, t in enumerate(TARGETS):
                row = {}
                for ri, r in enumerate(RULES):
                    p = res[(c, n, 0)]
                    k = int(round(p["viol"][ti, ri] * mp))
                    nest = [res[(c, n, b)] for b in range(1, B + 1)]
                    pv = [x["viol"][ti, ri] for x in nest]
                    row[r] = {
                        "plugin": {"violation": p["viol"][ti, ri], "violations": k,
                                   "violation_upper95": 1.0 if k == mp else float(betaincinv(k + 1, mp - k, 0.95)),
                                   "decisions": p["dec"][ti, ri], "minus_ltt_decisions": p["ddec"][ti, ri],
                                   "minus_ltt_decisions_se": (p["ddec_var"][ti, ri] / mp) ** 0.5,
                                   "minus_ltt_violation": p["dviol"][ti, ri],
                                   "violates_only_this": p["only_rule"][ti, ri], "violates_only_ltt": p["only_ltt"][ti, ri]},
                        "nested": {"violation": split(pv, [x * (1 - x) * m / (m - 1) for x in pv], m),
                                   "decisions": split([x["dec"][ti, ri] for x in nest], [x["dec_var"][ti, ri] for x in nest], m),
                                   "minus_ltt_violation": split([x["dviol"][ti, ri] for x in nest], [x["dviol_var"][ti, ri] for x in nest], m),
                                   "minus_ltt_decisions": split([x["ddec"][ti, ri] for x in nest], [x["ddec_var"][ti, ri] for x in nest], m)}}
                out["results"][f"{loc}/{key}/ncal{n}/{t}"] = row
    (DATA / "derived" / "population_gate.json").write_text(json.dumps(out, indent=1))
    print("seconds", out["seconds"])


if __name__ == "__main__":
    main()
