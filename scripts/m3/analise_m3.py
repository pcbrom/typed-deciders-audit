#!/usr/bin/env python3
"""Gates on the direction decisions of M3 monthly (implementation_notes.md, gate on time series).

Origins are the time axis (36 windows, one decision per series and origin). Sliding gates are calibrated on the 12
(or 6) preceding origins and tested on the next, sliding by one. A gate calibrated once on origins 1 to 12 is a contrast. Per-bin
gates (forgetting Beta and the dynamic adapter with a seasonal pair of period 12 origins) are warmed up on origins
1 to 12 and then gate each origin before its outcomes arrive, every outcome being observed afterwards.
Measures per gate and target: accuracy among acted decisions per origin, share of origins below the target,
decisions per 1,000, for all series and by category. Output: data/derived/m3/analise_m3.json.

Usage: analise_m3.py [--decider ets] [--delta 0.9]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "drift"))
sys.path.insert(0, str(HERE.parents[2] / "promise_check"))
import arms as A  # noqa: E402
import promise_check as pc  # noqa: E402
from common import DATA  # noqa: E402

D = DATA / "derived" / "m3"
TARGETS = (0.60, 0.70, 0.80)
RULES = {r: pc.RULES[r] for r in ("point", "beta", "ltt")}
WARM, ROLLS = 12, (12, 6)


def evaluate(conf, ok, accept):
    """accept: origins x series boolean for origins WARM..end; per-origin accuracy among acted and coverage."""
    out = []
    for k in range(WARM, conf.shape[1]):
        a = accept[k - WARM]
        out.append((float(ok[a, k].mean()) if a.any() else None, float(a.mean())))
    return out


def summary(per_origin, target):
    accs = [x for x, _ in per_origin]
    return {"origins_below": float(np.mean([x is not None and x < target for x in accs])),
            "origins_deciding": int(sum(x is not None for x in accs)),
            "accuracy_acted": float(np.nanmean([np.nan if x is None else x for x in accs])) if any(x is not None for x in accs) else None,
            "decisions_per_1000": float(1000 * np.mean([c for _, c in per_origin]))}


def gates(conf, ok, target, delta):
    n_s, n_o = conf.shape
    res = {}
    cal = (conf[:, :WARM].ravel(), ok[:, :WARM].ravel())
    for r, cfg in RULES.items():
        tau = pc.threshold(cfg, cal[0], cal[1], target)
        acc = np.array([conf[:, k] >= (tau if tau is not None else 2.0) for k in range(WARM, n_o)])
        res[f"static_{r}"] = evaluate(conf, ok, acc)
        for w in ROLLS:
            rows = []
            for k in range(WARM, n_o):
                c, o = conf[:, k - w:k].ravel(), ok[:, k - w:k].ravel()
                t = pc.threshold(cfg, c, o, target)
                rows.append(conf[:, k] >= (t if t is not None else 2.0))
            res[f"sliding{w}_{r}"] = evaluate(conf, ok, np.array(rows))
    A.ALPHA = target
    for name, arm in (("recal", A.Recalibration(bins=20, lam=0.9)), ("dynamic", A.Dynamic(bins=20, delta=delta, period=12)),
                      ("dynamic_lb", A.Dynamic(bins=20, delta=delta, period=12, criterion="lower"))):
        arm.initial(conf[:, 0], ok[:, 0])
        rows = []
        everything = np.ones(n_s, bool)
        for k in range(1, n_o):
            if hasattr(arm, "prepare"):
                arm.prepare(np.zeros(2))
            if k >= WARM:
                rows.append(arm.accept(conf[:, k]))
            arm.observe(conf[:, k], ok[:, k], everything, everything, everything)
        res[name] = evaluate(conf, ok, np.array(rows))
    A.ALPHA = 0.95
    return res


def ceiling(conf, ok):
    c, o = conf[:, WARM:].ravel(), ok[:, WARM:].ravel()
    order = np.argsort(-c, kind="stable")
    return {str(k): float(o[order[:k]].mean()) for k in (100, 500, 1000, 5000, len(c))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decider", default="ets")
    ap.add_argument("--delta", type=float, default=0.9)
    args = ap.parse_args()
    Z = np.load(D / f"{args.decider}.npz")
    p, y = Z["p_higher"], Z["higher"]
    conf, ok = np.maximum(p, 1 - p), (p >= 0.5) == y
    types = np.array([s["type"] for s in json.loads((D / "series.json").read_text())])
    out = {"decider": args.decider, "delta": args.delta, "accuracy_all": float(ok.mean()), "base_rate_higher": float(y.mean()),
           "by_group": {}}
    for group in ["ALL", *sorted(set(types))]:
        m = np.ones(len(types), bool) if group == "ALL" else types == group
        g = {"series": int(m.sum()), "accuracy": float(ok[m].mean()), "ceiling": ceiling(conf[m], ok[m]), "targets": {}}
        for t in TARGETS:
            res = gates(conf[m], ok[m], t, args.delta)
            g["targets"][str(t)] = {name: summary(v, t) for name, v in res.items()}
        out["by_group"][group] = g
        print(group, g["series"], "acc", round(g["accuracy"], 3), "ceiling", {k: round(v, 3) for k, v in g["ceiling"].items()}, flush=True)
    (D / f"analise_m3_{args.decider}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
