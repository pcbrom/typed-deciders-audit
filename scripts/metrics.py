#!/usr/bin/env python3
"""Every number reported in the article, for MASSIVE pt-PT and en-US (5 options, 2,000 utterances).

Protocol: the first 1,000 utterances calibrate (Laya temperature by Newton; thresholds), the last
1,000 report. The confidence gate is also evaluated over S random calibration/report splits, with
the Laya temperature refitted on each calibration half. Jev is always used as returned.

Per system: accuracy with Wilson 95% interval, macro-F1 over intents, exact McNemar test against Jev,
temperature, ECE (15 equal bins) as returned and after temperature, Brier score, negative
log-likelihood, AUROC of confidence as a detector of a correct answer (bootstrap 95% interval),
accuracy at the top confidence level (Clopper-Pearson interval), errors at confidence >= 0.97, ECE per
MASSIVE scenario, Bayes-threshold cost with and without calibration against the threshold that
minimises cost on the calibration half, the F1-maximising threshold, and the point and Beta gate
rules, Learn then Test and selection with guaranteed risk (promise kept, splits with a qualifying threshold, broken promises per split, decisions and
errors per 1,000, human reviews per avoided error). Curves for the figures are stored as well.

Usage: metrics.py [--splits 1000] [--bootstrap 2000]
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.special import betaincinv
from scipy.stats import beta as beta_dist
from scipy.stats import binom, binomtest

from common import CAL, DATA, LOCALES, NAMES, REP, auroc, ece, fit_temperature, load_items, load_systems, log_probs, softmax_t, wilson

TARGETS = [0.95, 0.99, 0.995]
COST_RATIOS = [10.0, 100.0]
MIN_ACCEPTED = 20


def mcnemar(ok_ref, ok_new):
    b = int((ok_ref & ~ok_new).sum())
    c = int((~ok_ref & ok_new).sum())
    n = len(ok_ref)
    d = (c - b) / n
    se = np.sqrt(((b + c) / n - d * d) / n)
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"ref_right_new_wrong": b, "new_right_ref_wrong": c, "diff_pp": round(100 * d, 2),
            "ci95_pp": [round(100 * (d - 1.96 * se), 2), round(100 * (d + 1.96 * se), 2)], "p": float(p)}


def macro_f1(pred, gold):
    f1 = []
    for c in sorted(set(gold)):
        tp = sum(1 for p, g in zip(pred, gold) if p == c and g == c)
        fp = sum(1 for p, g in zip(pred, gold) if p == c and g != c)
        fn = sum(1 for p, g in zip(pred, gold) if p != c and g == c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1))


def _candidates(conf, ok):
    """Distinct thresholds from the highest down, with accepted counts n and correct counts c."""
    order = np.argsort(-conf, kind="stable")
    cs, oks = conf[order], ok[order]
    n = np.arange(1, len(conf) + 1)
    c = np.cumsum(oks)
    last = np.r_[cs[1:] != cs[:-1], True]
    thr, n, c = cs[last], n[last], c[last]
    keep = n >= MIN_ACCEPTED
    return thr[keep], n[keep], c[keep]


def gate_threshold(conf, ok, target, rule, delta=0.05):
    """Threshold chosen on a calibration sample for the promise accuracy(accepted) >= target.

    point: lowest threshold with c/n >= target (ascending scan of the author's rule).
    beta:  lowest threshold whose Beta(c+1, n-c+1) 5% quantile >= target (author's rule).
    ltt:   Learn then Test, fixed sequence from the highest testable threshold down (testable:
           (1-r*)^n <= delta, so that zero errors could reject), binomial p-value
           P(Bin(n, 1-target) <= n-c) for H0: risk among accepted > 1-target; stops at the first
           non-rejection and keeps the last rejected threshold.
    sgr:   selection with guaranteed risk, binary search over accepted counts with the
           Clopper-Pearson upper bound on risk at level delta / ceil(log2 m).
    """
    thr, n, c = _candidates(conf, ok)
    if len(thr) == 0:
        return None
    if rule in ("point", "beta"):
        est = betaincinv(c + 1, n - c + 1, delta) if rule == "beta" else c / n
        idx = np.nonzero(est >= target)[0]
        return float(thr[idx.max()]) if len(idx) else None
    r_star = 1 - target
    e = n - c
    if rule == "ltt":
        # thresholds where even zero errors cannot reject are skipped: this depends on the
        # accepted counts (confidences only), not on the labels, so the p-values stay valid
        testable = (1 - r_star) ** n <= delta
        pvals = binom.cdf(e, n, r_star)
        chosen = None
        for j in np.nonzero(testable)[0]:
            if pvals[j] <= delta:
                chosen = thr[j]
            else:
                break
        return None if chosen is None else float(chosen)
    if rule == "sgr":
        m = len(thr)
        k = max(1, int(np.ceil(np.log2(m))))
        d = delta / k
        upper = np.where(e >= n, 1.0, betaincinv(e + 1, np.maximum(n - e, 1e-12), 1 - d))
        lo, hi, best = 0, m - 1, None
        for _ in range(k + 1):
            if lo > hi:
                break
            mid = (lo + hi) // 2
            if upper[mid] < r_star:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return None if best is None else float(thr[best])
    raise ValueError(rule)


def cost(conf, ok, tau, ratio):
    """Cost per item: a wrong automatic action costs `ratio`, each escalation costs 1."""
    acc = conf >= tau
    return float((ratio * (acc & ~ok).sum() + (~acc).sum()) / len(ok))


def analyse(locale, n_splits, n_boot, rng):
    items, gold = load_items(locale)
    options = [it["options"] for it in items]
    labels = [it["intent"] for it in items]
    raw = load_systems(locale)
    ok_jev = raw["jev"].argmax(axis=1) == gold
    scenario = defaultdict(list)
    for i in REP:
        scenario[labels[i].split("_")[0]].append(i)
    splits = [rng.permutation(2000) for _ in range(n_splits)]

    out = {}
    for key, P in raw.items():
        laya = key != "jev"
        lp = log_probs(P)
        t = fit_temperature(lp[CAL], gold[CAL]) if laya else None
        Q = softmax_t(lp, t) if laya else P
        conf, pred = Q.max(axis=1), Q.argmax(axis=1)
        ok = pred == gold
        conf_raw, ok_raw = P.max(axis=1), P.argmax(axis=1) == gold
        cr, kr = conf[REP], ok[REP]
        r = {"name": NAMES[key], "temperature": round(t, 4) if laya else None}
        k = int(kr.sum())
        r["accuracy"], r["accuracy_ci95"] = round(k / 1000, 4), wilson(k, 1000)
        r["macro_f1"] = round(macro_f1([options[i][pred[i]] for i in REP], [labels[i] for i in REP]), 4)
        r["vs_jev"] = None if key == "jev" else mcnemar(ok_jev[REP], kr)
        r["ece_as_returned"] = round(ece(conf_raw[REP], ok_raw[REP]), 4)
        r["ece_after_temperature"] = round(ece(cr, kr), 4) if laya else None
        onehot = np.eye(5)[gold[REP]]
        r["brier"] = round(float(np.mean(((Q[REP] - onehot) ** 2).sum(axis=1))), 4)
        r["nll"] = round(float(-np.mean(np.log(np.clip(Q[REP][np.arange(1000), gold[REP]], 1e-12, 1)))), 4)
        boots = []
        for _ in range(n_boot):
            b = rng.integers(0, 1000, 1000)
            if kr[b].all() or (~kr[b]).all():
                continue
            boots.append(auroc(cr[b], kr[b]))
        r["auroc"] = round(auroc(cr, kr), 4)
        r["auroc_ci95"] = [round(float(np.quantile(boots, 0.025)), 4), round(float(np.quantile(boots, 0.975)), 4)]
        level = 0.995 if not laya else 0.9995    # Jev returns 2 decimals: 1.00 means p >= 0.995
        top = cr >= level
        kk, nn = int(kr[top].sum()), int(top.sum())
        r["top_confidence"] = {"threshold": level, "n": nn, "correct": kk,
                               "ci95_exact": None if nn == 0 else
                               [round(float(beta_dist.ppf(0.025, kk, nn - kk + 1)), 4) if kk else 0.0,
                                round(float(beta_dist.ppf(0.975, kk + 1, nn - kk)), 4) if kk < nn else 1.0]}
        hi = cr >= 0.97
        r["errors_at_conf_097"] = {"errors": int((hi & ~kr).sum()), "n": int(hi.sum())}
        eces = {s: round(ece(conf[idx], ok[idx]), 4) for s, idx in scenario.items() if len(idx) >= 30}
        worst = max(eces.items(), key=lambda kv: kv[1])
        r["ece_by_scenario"] = {"n_scenarios": len(eces), "median": round(float(np.median(list(eces.values()))), 4),
                                "worst": {"scenario": worst[0], "ece": worst[1]}, "all": eces}
        r["cost_per_item"] = {}
        for ratio in COST_RATIOS:
            tb = 1 - 1 / ratio
            best = min(np.unique(conf[CAL]), key=lambda tt: cost(conf[CAL], ok[CAL], tt, ratio))
            r["cost_per_item"][f"{int(ratio)}:1"] = {
                "bayes_threshold": round(tb, 4), "bayes_calibrated": round(cost(cr, kr, tb, ratio), 4),
                "bayes_as_returned": round(cost(conf_raw[REP], ok_raw[REP], tb, ratio), 4),
                "empirical_threshold": round(float(best), 4), "empirical": round(cost(cr, kr, best, ratio), 4)}

        def f1_gate(cc, kk_, tt):
            acc = cc >= tt
            tp, fp, fn = (acc & kk_).sum(), (acc & ~kk_).sum(), (~acc & kk_).sum()
            return 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        tf1 = max(np.unique(conf[CAL]), key=lambda tt: f1_gate(conf[CAL], ok[CAL], tt))
        acc = cr >= tf1
        r["max_f1_threshold"] = {"tau": round(float(tf1), 4), "coverage": round(float(acc.mean()), 4),
                                 "type_I": int((acc & ~kr).sum()), "type_II": int((~acc & kr).sum())}

        thr = np.unique(cr)[::-1]
        roc_x, roc_y = [0.0], [0.0]
        for tt in thr:
            sel = cr >= tt
            roc_y.append(round(float((sel & kr).sum() / kr.sum()), 5))
            roc_x.append(round(float((sel & ~kr).sum() / (~kr).sum()), 5))
        edges = np.linspace(0, 1, 11)
        bins = []
        for i, (lo, hi_) in enumerate(zip(edges[:-1], edges[1:])):
            m = ((cr >= lo) if i == 0 else (cr > lo)) & (cr <= hi_)
            if m.sum() >= 5:
                bins.append([round(float(cr[m].mean()), 4), round(float(kr[m].mean()), 4), int(m.sum())])
        grid = np.round(np.linspace(0, 1, 201), 3)
        r["curves"] = {"roc": [roc_x, roc_y], "reliability": bins,
                       "cost_10": [[float(g), round(cost(cr, kr, g, 10.0), 4)] for g in grid],
                       "cost_10_as_returned": [[float(g), round(cost(conf_raw[REP], ok_raw[REP], g, 10.0), 4)] for g in grid],
                       "accuracy_coverage": [[round(float((cr >= tt).mean()), 4), round(float(kr[cr >= tt].mean()), 4)]
                                             for tt in thr]}

        gate = {}
        for target in TARGETS:
            agg = {}
            for rule in ("point", "beta", "ltt", "sgr"):
                kept, dec, err, pairs = [], [], [], []
                for idx in splits:
                    c_, e_ = idx[:1000], idx[1000:]
                    if laya:
                        qs = softmax_t(lp, fit_temperature(lp[c_], gold[c_]))
                        cs, ks = qs.max(axis=1), qs.argmax(axis=1) == gold
                    else:
                        cs, ks = conf, ok
                    tau = gate_threshold(cs[c_], ks[c_], target, rule)
                    accepted = cs[e_] >= tau if tau is not None else np.zeros(1000, bool)
                    a_ = float(ks[e_][accepted].mean()) if accepted.any() else None
                    dec.append(int(accepted.sum()))
                    err.append(int((accepted & ~ks[e_]).sum()))
                    pairs.append([round(accepted.mean(), 4), None if a_ is None else round(a_, 4)])
                    if a_ is not None:
                        kept.append(a_ >= target)
                agg[rule] = {"promise_kept": round(float(np.mean(kept)), 3) if kept else None,
                             "qualified_splits": len(kept), "no_threshold_splits": n_splits - len(kept),
                             "broken_per_split": round((len(kept) - int(np.sum(kept))) / n_splits, 3),
                             "decisions_per_1000": round(float(np.mean(dec)), 1),
                             "errors_per_1000": round(float(np.mean(err)), 2)}
                if target == 0.99:
                    agg[rule]["splits_coverage_accuracy"] = pairs
            dd = agg["point"]["decisions_per_1000"] - agg["beta"]["decisions_per_1000"]
            de = agg["point"]["errors_per_1000"] - agg["beta"]["errors_per_1000"]
            agg["reviews_per_avoided_error"] = round(dd / de, 1) if de > 0 else None
            for rule in ("ltt", "sgr"):
                dd = agg["point"]["decisions_per_1000"] - agg[rule]["decisions_per_1000"]
                de = agg["point"]["errors_per_1000"] - agg[rule]["errors_per_1000"]
                agg[f"reviews_per_avoided_error_{rule}"] = round(dd / de, 1) if de > 0 else None
            gate[str(target)] = agg
        r["gate"] = gate
        out[key] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=1000)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(20261020)
    result = {"protocol": {"calibration": "items 0-999", "report": "items 1000-1999", "splits": args.splits,
                           "bootstrap": args.bootstrap, "seed": 20261020, "ece_bins": 15, "min_accepted": MIN_ACCEPTED},
              "results": {loc: analyse(loc, args.splits, args.bootstrap, rng) for loc in LOCALES}}
    with open(DATA / "derived" / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    for loc, res in result["results"].items():
        print(f"== {loc}")
        for key, r in res.items():
            g = r["gate"]["0.99"]
            print(f"  {r['name']:34s} acc {r['accuracy']:.3f} ECE {r['ece_as_returned']:.3f}->{r['ece_after_temperature']} "
                  f"AUROC {r['auroc']:.3f} | gate 0.99 broken point {g['point']['broken_per_split']} "
                  f"beta {g['beta']['broken_per_split']} ({g['beta']['qualified_splits']}) "
                  f"ltt {g['ltt']['broken_per_split']} ({g['ltt']['qualified_splits']}) "
                  f"sgr {g['sgr']['broken_per_split']} ({g['sgr']['qualified_splits']})")


if __name__ == "__main__":
    main()
