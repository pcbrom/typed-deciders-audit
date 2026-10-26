#!/usr/bin/env python3
"""Ablation of the Beta lower-bound gate rule against every gate rule discussed in the article, on
the same random calibration/report splits, scored with every gate measure of the article.

Rules
  point    accept from the lowest threshold whose calibration accuracy c/n reaches the target
  beta     the author's rule: lower q-quantile of Beta(c + a0, n - c + b0) reaches the target;
           priors uniform (1,1), Jeffreys (1/2,1/2) or the improper Beta(0,1), whose lower bound
           is the Clopper-Pearson bound; scans
             ascending          lowest threshold that qualifies anywhere (the author's rule)
             sequence           from the top down, stop at the first threshold that fails
             sequence_testable  as sequence, skipping thresholds where even c = n cannot qualify
           Beta(0,1) with sequence_testable and q = delta is exactly Learn then Test.
  ltt      Learn then Test, binomial p-value, fixed sequence over testable thresholds
  sgr      selection with guaranteed risk, Clopper-Pearson bound at delta / ceil(log2 m)
  bayes    tau* = 1 - C_r / C_e on the confidence as given (cost ratios 10, 100, 1000)
  class    one threshold per predicted intent, with the point or the Beta rule inside each class
  oracle   most decisions whose accuracy on the report half reaches the target (not attainable)

Factors: target (full grid at 0.95 and 0.99, core rules at 0.995), calibration on/off (temperature for Laya; Jev is always used as returned), and
calibration-set size n_cal (250, 500, 1000; the report half is always the other 1,000 items). The
non-fine-tuned checkpoints of EXTRA are run with the core rules only, calibrated, n_cal = 1000.

Measures per configuration, averaged over splits (counts per 1,000 report items)
  broken           accuracy among accepted on the report half below the target (no acceptance:
                   not broken); clearly broken: the one-sided 95% Clopper-Pearson upper bound of
                   that accuracy below the target
  kept, theta      promise kept and selective accuracy among splits that accept something
  optimism         calibration-half minus report-half accuracy among accepted
  decisions, type I (accepted wrong), type II (escalated correct), gate F1 (positive = correct
  answer accepted), cost per item for C_e/C_r = 10 and 100 (one review costs 1), reviews per
  avoided error (escalations over errors avoided against accepting everything), fit time per split

Usage: ablation_gate.py [--splits 1000]
"""

import argparse
import json
import time

import numpy as np
from scipy.special import betaincinv
from scipy.stats import binom

from common import DATA, fit_temperature, load_items, load_systems, log_probs, softmax_t

CASES = [("pt", "ft-multilingual"), ("pt", "ft-typed-decisions"), ("pt", "jev"),
         ("en", "jev"), ("en", "typed-decisions"), ("en", "root")]
EXTRA = [("pt", "multilingual"), ("pt", "root"), ("pt", "typed-decisions"), ("en", "multilingual")]
TARGETS = [0.95, 0.99, 0.995]
FULL_TARGETS = (0.95, 0.99)
N_CAL = [1000, 500, 250]
PRIORS = {"uniform": (1.0, 1.0), "jeffreys": (0.5, 0.5), "cp": (0.0, 1.0)}


def candidates(conf, ok, min_n):
    order = np.argsort(-conf, kind="stable")
    cs, oks = conf[order], ok[order]
    n = np.arange(1, len(conf) + 1)
    c = np.cumsum(oks)
    last = np.r_[cs[1:] != cs[:-1], True]
    thr, n, c = cs[last], n[last], c[last]
    keep = n >= min_n
    return thr[keep], n[keep], c[keep]


def pick(thr, passes, scan, testable=None):
    if scan == "ascending":
        idx = np.nonzero(passes)[0]
        return float(thr[idx.max()]) if len(idx) else None
    if scan == "sequence_testable":
        thr, passes = thr[testable], passes[testable]
    chosen = None
    for j in range(len(thr)):
        if passes[j]:
            chosen = thr[j]
        else:
            break
    return None if chosen is None else float(chosen)


def beta_cfg(q, prior, scan, min_n=20):
    a0, b0 = PRIORS[prior]
    return {"rule": "beta", "q": q, "prior": prior, "a0": a0, "b0": b0, "min_n": min_n, "scan": scan,
            "label": f"beta q={q} {prior} min {min_n} {scan}"}


USER_RULE = beta_cfg(0.05, "uniform", "ascending")


def configurations():
    cfgs = [{"rule": "point", "min_n": 20, "label": "point"}]
    for q in (0.01, 0.025, 0.05, 0.10, 0.20):
        for prior in PRIORS:
            for scan in ("ascending", "sequence", "sequence_testable"):
                cfgs.append(beta_cfg(q, prior, scan))
    cfgs.append(beta_cfg(0.05, "uniform", "ascending", min_n=100))
    for d in (0.01, 0.05, 0.10, 0.20):
        cfgs.append({"rule": "ltt", "delta": d, "min_n": 20, "label": f"ltt delta={d}"})
        cfgs.append({"rule": "sgr", "delta": d, "min_n": 20, "label": f"sgr delta={d}"})
    for r in (10, 100, 1000):
        cfgs.append({"rule": "bayes", "ratio": r, "label": f"bayes C_e/C_r={r}"})
    cfgs.append({"rule": "class", "inner": {"rule": "point", "min_n": 20}, "label": "per-class point"})
    cfgs.append({"rule": "class", "inner": USER_RULE, "label": "per-class beta"})
    return cfgs


CORE = {"point", USER_RULE["label"], "beta q=0.05 uniform min 20 sequence_testable", "ltt delta=0.05",
        "sgr delta=0.05", "bayes C_e/C_r=10", "bayes C_e/C_r=100", "per-class point", "per-class beta"}


def threshold(cfg, conf, ok, target):
    if cfg["rule"] == "bayes":
        return 1.0 - 1.0 / cfg["ratio"]
    thr, n, c = candidates(conf, ok, cfg["min_n"])
    if len(thr) == 0:
        return None
    if cfg["rule"] == "point":
        return pick(thr, c / n >= target, "ascending")
    if cfg["rule"] == "beta":
        a, b = c + cfg["a0"], n - c + cfg["b0"]
        lower = np.where(a > 0, betaincinv(np.maximum(a, 1e-12), b, cfg["q"]), 0.0)
        testable = betaincinv(n + cfg["a0"], cfg["b0"], cfg["q"]) >= target
        return pick(thr, lower >= target, cfg["scan"], testable)
    r_star, e = 1 - target, n - c
    if cfg["rule"] == "ltt":
        testable = (1 - r_star) ** n <= cfg["delta"]
        p = binom.cdf(e, n, r_star)
        thr_t, pass_t = thr[testable], (p <= cfg["delta"])[testable]
        return pick(thr_t, pass_t, "sequence") if len(thr_t) else None
    m = len(thr)
    k = max(1, int(np.ceil(np.log2(m))))
    upper = np.where(e >= n, 1.0, betaincinv(e + 1, np.maximum(n - e, 1e-12), 1 - cfg["delta"] / k))
    lo, hi, best = 0, m - 1, None
    for _ in range(k + 1):
        if lo > hi:
            break
        mid = (lo + hi) // 2
        if upper[mid] < r_star:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return None if best is None else float(thr[best])


def accepted(cfg, cal, ev, target):
    """Masks of accepted report items and accepted calibration items (None if nothing accepted)."""
    cc, kc, gc = cal
    ce, ge = ev[0], ev[2]
    if cfg["rule"] != "class":
        tau = threshold(cfg, cc, kc, target)
        if tau is None:
            return np.zeros(len(ce), bool), None
        return ce >= tau, cc >= tau
    acc, acc_cal = np.zeros(len(ce), bool), np.zeros(len(cc), bool)
    for g in np.unique(gc):
        tau = threshold(cfg["inner"], cc[gc == g], kc[gc == g], target)
        if tau is not None:
            acc |= (ge == g) & (ce >= tau)
            acc_cal |= (gc == g) & (cc >= tau)
    return acc, (acc_cal if acc_cal.any() else None)


def score(cfg, per_split, target):
    s = {k: [] for k in ("dec", "e1", "e2", "f1", "cost10", "cost100", "rev", "theta", "opt", "secs")}
    broken = clear = 0
    for cal, ev in per_split:
        ke = ev[1]
        t0 = time.perf_counter()
        acc, acc_cal = accepted(cfg, cal, ev, target)
        s["secs"].append(time.perf_counter() - t0)
        n_rep, wrong_all = len(ke), int((~ke).sum())
        m, k = int(acc.sum()), int((acc & ke).sum())
        e1, e2 = m - k, int((~acc & ke).sum())
        s["dec"].append(m * 1000 / n_rep)
        s["e1"].append(e1 * 1000 / n_rep)
        s["e2"].append(e2 * 1000 / n_rep)
        s["f1"].append(2 * k / (m + int(ke.sum())))
        esc = n_rep - m
        s["cost10"].append((esc + 10 * e1) / n_rep)
        s["cost100"].append((esc + 100 * e1) / n_rep)
        if wrong_all - e1 > 0:
            s["rev"].append(esc / (wrong_all - e1))
        if m:
            a = k / m
            s["theta"].append(a)
            broken += a < target
            clear += (1.0 if k == m else betaincinv(k + 1, m - k, 0.95)) < target
            if acc_cal is not None and acc_cal.any():
                s["opt"].append(float(cal[1][acc_cal].mean()) - a)
    mean = lambda v, d=4: round(float(np.mean(v)), d) if v else None
    return {"label": cfg["label"], "broken_per_split": round(broken / len(per_split), 4),
            "clearly_broken_per_split": round(clear / len(per_split), 4),
            "qualified_splits": len(s["theta"]), "promise_kept": mean([t >= target for t in s["theta"]], 3),
            "selective_accuracy": mean(s["theta"]), "optimism": mean(s["opt"]),
            "decisions_per_1000": mean(s["dec"], 1), "type1_per_1000": mean(s["e1"], 2), "type2_per_1000": mean(s["e2"], 1),
            "gate_f1": mean(s["f1"]), "cost_ratio10": mean(s["cost10"]), "cost_ratio100": mean(s["cost100"]),
            "reviews_per_avoided_error": mean(s["rev"], 2), "fit_ms": mean([x * 1000 for x in s["secs"]], 3)}


def oracle(per_split, target):
    dec = []
    for _, (ce, ke, _) in per_split:
        thr, n, c = candidates(ce, ke, 20)
        ok_t = c / n >= target
        dec.append(int(n[np.nonzero(ok_t)[0].max()]) * 1000 / len(ce) if ok_t.any() else 0)
    return {"label": "oracle", "decisions_per_1000": round(float(np.mean(dec)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(20261020)
    splits = [rng.permutation(2000) for _ in range(args.splits)]
    cfgs = configurations()
    core = [c for c in cfgs if c["label"] in CORE]
    out = {"splits": args.splits, "seed": 20261020, "configurations": cfgs, "core": sorted(CORE),
           "user_rule": USER_RULE["label"], "results": {}, "temperature_seconds": {}}
    for loc, key in CASES + EXTRA:
        extra = (loc, key) in EXTRA
        items, gold = load_items(loc)
        vocab = {}
        P = load_systems(loc)[key]
        lp = log_probs(P)
        for calibrate in ((False,) if key == "jev" else (True,) if extra else (True, False)):
            for n_cal in ([1000] if extra else N_CAL if calibrate or key == "jev" else [1000]):
                per_split, t_secs = [], []
                for idx in splits:
                    cal, ev = idx[:n_cal], idx[1000:]
                    if calibrate:
                        t0 = time.perf_counter()
                        Q = softmax_t(lp, fit_temperature(lp[cal], gold[cal]))
                        t_secs.append(time.perf_counter() - t0)
                    else:
                        Q = P
                    top = Q.argmax(axis=1)
                    conf, ok = Q.max(axis=1), top == gold
                    g = np.array([vocab.setdefault(it["options"][j], len(vocab)) for it, j in zip(items, top)])
                    per_split.append(((conf[cal], ok[cal], g[cal]), (conf[ev], ok[ev], g[ev])))
                tag = f"{loc}/{key}/{'temperature' if calibrate else 'as_returned'}/ncal{n_cal}"
                if t_secs:
                    out["temperature_seconds"][tag] = round(float(np.mean(t_secs)), 5)
                for target in TARGETS:
                    use = cfgs if n_cal == 1000 and not extra and target in FULL_TARGETS else core
                    out["results"][f"{tag}/{target}"] = [oracle(per_split, target)] + [score(c, per_split, target) for c in use]
                    print(f"{tag}/{target}: done", flush=True)
    with open(DATA / "derived" / "ablation_gate.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
