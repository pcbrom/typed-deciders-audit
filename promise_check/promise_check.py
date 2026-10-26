#!/usr/bin/env python3
"""Check whether a selective-accuracy promise survives a change of population.

A typed decider returns, per item, a probability for each option. A gate accepts the decision when
the top probability reaches a threshold fitted on a labelled calibration sample, promising that the
accuracy among accepted decisions reaches a target. promise_check measures how often that promise
breaks, by the plug-in population method of the article:

  the items of the reference group are the calibration population; each draw takes n_cal of them
  with replacement, fits the temperature (optional) and the threshold on that sample, and applies
  both to every item of each evaluated group. A violation is an accuracy among accepted items of the
  evaluated group below the target. Evaluating the reference group itself is the static gate check;
  evaluating another group (a language, a noise condition, a model version, a later drift window)
  measures how much of the promise survives the change.

Input: JSON Lines, one item per line, with
  group  the population the item belongs to (string)
  probs  the probabilities of the options, in a fixed order
  gold   the index of the correct option
or, when only the decision is available, with
  confidence  the probability of the decided option
  correct     whether the decision is right (true or false)
Items without a gold label cannot be used. --temperature needs probs.

Rules (all with at least min_n = 20 calibration items at or above the threshold)
  point                 lowest threshold whose calibration accuracy reaches the target
  beta                  lowest threshold whose lower 0.05 quantile of Beta(c + 1, n - c + 1) reaches
                        the target, ascending scan (the author's rule)
  beta_fixed_sequence   the same bound scanned from the top down over testable thresholds
  ltt                   Learn then Test, binomial p-value at delta = 0.05, fixed sequence
  sgr                   selection with guaranteed risk, delta = 0.05

Output: JSON with, per evaluated group, target and rule, the violation rate, its one-sided 95%
Clopper-Pearson upper bound, the mean accuracy among accepted items and the mean number of
decisions per 1,000 items.

Usage: promise_check.py ITEMS.jsonl --reference GROUP [--evaluate G1 G2 ...] [--targets 0.95 0.99]
       [--n-cal 1000] [--draws 10000] [--temperature] [--seed 20261020] [--out result.json]
"""

import argparse
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.special import betaincinv
from scipy.stats import binom

RULES = {"point": {"rule": "point"},
         "beta": {"rule": "beta", "q": 0.05, "a0": 1.0, "b0": 1.0, "scan": "ascending"},
         "beta_fixed_sequence": {"rule": "beta", "q": 0.05, "a0": 1.0, "b0": 1.0, "scan": "sequence_testable"},
         "ltt": {"rule": "ltt", "delta": 0.05},
         "sgr": {"rule": "sgr", "delta": 0.05}}
MIN_N = 20


def candidates(conf, ok, min_n=MIN_N):
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


def threshold(cfg, conf, ok, target):
    """Lowest accepted confidence chosen by a rule on a calibration sample, or None."""
    thr, n, c = candidates(conf, ok)
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


def log_probs(p):
    return np.log(np.clip(p, 1e-6, 1.0))


def fit_temperature(logp, gold, iters=60):
    """Newton's method on beta = 1/T for the mean negative log-likelihood (convex in beta)."""
    b = 1.0
    for _ in range(iters):
        z = b * logp
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        m = (p * logp).sum(axis=1)
        hess = np.mean((p * logp ** 2).sum(axis=1) - m ** 2)
        if not np.isfinite(hess) or hess <= 1e-12:
            break
        step = np.mean(m - logp[np.arange(len(gold)), gold]) / hess
        b = float(np.clip(b - step, 0.05, 20.0))
        if abs(step) < 1e-10:
            break
    return 1.0 / b


def softmax_t(logp, t):
    z = logp / t
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def load(path):
    """Groups as (probs, gold) when every item has probs, else as {"conf", "ok"}."""
    groups = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                groups[str(r["group"])].append(r)
    out = {}
    for g, rows in groups.items():
        if all("probs" in r for r in rows):
            out[g] = (np.array([r["probs"] for r in rows], dtype=float), np.array([int(r["gold"]) for r in rows]))
        else:
            out[g] = {"conf": np.array([r["confidence"] for r in rows], dtype=float), "ok": np.array([bool(r["correct"]) for r in rows])}
    return out


def conf_ok(group):
    if isinstance(group, dict):
        return group["conf"], group["ok"]
    P, y = group
    return P.max(axis=1), P.argmax(axis=1) == y


def check(groups, reference, evaluate, targets=(0.95, 0.99), n_cal=1000, draws=10000, temperature=False, seed=20261020, rules=RULES):
    """Violation, accuracy among accepted and decisions per 1,000, per evaluated group, target and rule."""
    if temperature and any(isinstance(groups[g], dict) for g in [reference, *evaluate]):
        raise ValueError("temperature scaling needs the probabilities of every option")
    if temperature:
        P_ref, y_ref = groups[reference]
        lp_ref = log_probs(P_ref)
        ev = {g: (log_probs(groups[g][0]), groups[g][1]) for g in evaluate}
    else:
        conf_ref, ok_ref = conf_ok(groups[reference])
        ev = {g: conf_ok(groups[g]) for g in evaluate}
    n_ref = len(lp_ref) if temperature else len(conf_ref)
    shape = (len(evaluate), len(targets), len(rules), draws)
    viol, acc, dec = np.zeros(shape, bool), np.full(shape, np.nan), np.zeros(shape)
    rng = np.random.default_rng(seed)
    for i in range(draws):
        cal = rng.integers(0, n_ref, n_cal)
        if temperature:
            t = fit_temperature(lp_ref[cal], y_ref[cal])
            Q = softmax_t(lp_ref[cal], t)
            conf_c, ok_c = Q.max(axis=1), Q.argmax(axis=1) == y_ref[cal]
        else:
            conf_c, ok_c = conf_ref[cal], ok_ref[cal]
        taus = [[threshold(cfg, conf_c, ok_c, tg) for cfg in rules.values()] for tg in targets]
        for gi, g in enumerate(evaluate):
            if temperature:
                Qg = softmax_t(ev[g][0], t)
                conf, ok = Qg.max(axis=1), Qg.argmax(axis=1) == ev[g][1]
            else:
                conf, ok = ev[g]
            for ti, tg in enumerate(targets):
                for ri, tau in enumerate(taus[ti]):
                    if tau is None:
                        continue
                    a = conf >= tau
                    dec[gi, ti, ri, i] = a.mean() * 1000
                    if a.any():
                        acc[gi, ti, ri, i] = ok[a].mean()
                        viol[gi, ti, ri, i] = acc[gi, ti, ri, i] < tg
    out = {"reference": reference, "n_cal": n_cal, "draws": draws, "temperature": temperature, "seed": seed,
           "targets": list(targets), "rules": rules, "results": {}}
    for gi, g in enumerate(evaluate):
        for ti, tg in enumerate(targets):
            row = {}
            for ri, r in enumerate(rules):
                k = int(viol[gi, ti, ri].sum())
                row[r] = {"violation": k / draws, "violations": k,
                          "violation_upper95": 1.0 if k == draws else float(betaincinv(k + 1, draws - k, 0.95)),
                          "accuracy_accepted": None if np.isnan(acc[gi, ti, ri]).all() else float(np.nanmean(acc[gi, ti, ri])),
                          "decisions": float(dec[gi, ti, ri].mean()), "decisions_sd": float(dec[gi, ti, ri].std(ddof=1))}
            out["results"][f"{g}/{tg}"] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("items")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--evaluate", nargs="*")
    ap.add_argument("--targets", nargs="+", type=float, default=[0.95, 0.99])
    ap.add_argument("--n-cal", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--temperature", action="store_true")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--out")
    a = ap.parse_args()
    groups = load(a.items)
    evaluate = a.evaluate or sorted(groups)
    missing = [g for g in [a.reference, *evaluate] if g not in groups]
    if missing:
        sys.exit(f"groups not in {a.items}: {missing}")
    out = check(groups, a.reference, evaluate, tuple(a.targets), a.n_cal, a.draws, a.temperature, a.seed)
    text = json.dumps(out, indent=1)
    if a.out:
        open(a.out, "w").write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
