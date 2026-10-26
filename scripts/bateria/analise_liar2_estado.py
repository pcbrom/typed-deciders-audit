#!/usr/bin/env python3
"""LIAR2 with a structured state against the statement alone (implementation_notes.md, half-split note), and the
exploratory ceiling measures of the LIAR2 note: accuracy on the report items, paired difference against the
statement-only answers of the same decider, accuracy among the most confident items, the battery gate, the mean
of Jev and Laya, two coarse levels and split conformal sets. Output: data/derived/bateria/liar2_teto.json.

Usage: analise_liar2_estado.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "promise_check"))
import promise_check as pc  # noqa: E402
from common import CAL, DATA, REP, calibrated  # noqa: E402

D = DATA / "derived" / "bateria"
RULES = {r: pc.RULES[r] for r in ("point", "beta", "ltt")}
TOP = (20, 50, 100, 200, 500)


def jev(path, cond=None):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if cond is None or (r.get("domain") == "liar2" and r.get("condition") == cond):
            rows[r["item"]] = r["probs"]
    return np.array([rows[i] for i in range(2000)], dtype=float)


def describe(P, gold, temperature):
    Q = calibrated(P, CAL, gold)[0] if temperature else P
    c, ok = Q[REP].max(1), Q[REP].argmax(1) == gold[REP]
    o = np.argsort(-c, kind="stable")
    out = {"accuracy": float(ok.mean()), "top": {str(k): float(ok[o[:k]].mean()) for k in TOP}}
    g = pc.check({"all": (P, gold)} if temperature else {"all": {"conf": P.max(1), "ok": P.argmax(1) == gold}},
                 "all", ["all"], (0.90, 0.95), 1000, 2000, temperature, 20261023, RULES)
    out["gate"] = {k: {r: {m: v[r][m] for m in ("violation", "violation_upper95", "decisions")} for r in RULES} for k, v in g["results"].items()}
    return out, Q, ok


def paired(a, b):
    d = a.astype(float) - b
    rng = np.random.default_rng(np.random.SeedSequence([20261023, 35]))
    m = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]}


def conformal(P, gold, alpha):
    s = 1 - P[CAL, gold[CAL]]
    n = len(s)
    q = np.quantile(s, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")
    S = (1 - P[REP]) <= q
    return {"coverage": float(S[np.arange(len(REP)), gold[REP]].mean()), "mean_set_size": float(S.sum(1).mean())}


def main():
    items = json.loads((DATA / "items" / "bateria" / "liar2.json").read_text())["items"]
    gold = np.array([it["gold"] for it in items])
    Zb, Zs = np.load(D / "laya_liar2.npz"), np.load(D / "laya_liar2_estado_metade.npz")
    models = {"jev/statement": (jev(D / "jev.jsonl", "clean"), False), "jev/structured": (jev(D / "jev_liar2_estado_metade.jsonl"), False),
              "laya_released/statement": (Zb["released|0|0|clean"], True), "laya_released/structured": (Zs["released"], True)}
    for s in (0, 1, 2):
        models[f"laya_ft{s}/statement"] = (Zb[f"ft|8000|{s}|clean"], True)
        models[f"laya_ft{s}/structured"] = (Zs[f"ft{s}"], True)
    out, cal, oks = {}, {}, {}
    for name, (P, t) in models.items():
        out[name], cal[name], oks[name] = describe(P, gold, t)
    out["structured_minus_statement"] = {m: paired(oks[f"{m}/structured"], oks[f"{m}/statement"]) for m in ("jev", "laya_released", "laya_ft0", "laya_ft1", "laya_ft2")}
    ceiling = {}
    for state in ("statement", "structured"):
        F = np.mean([cal[f"laya_ft{s}/{state}"] for s in (0, 1, 2)], axis=0)
        J = cal[f"jev/{state}"]
        for name, P in (("jev", J), ("laya_ft_mean", F), ("jev_laya_mean", (J + F) / 2)):
            c, ok = P[REP].max(1), P[REP].argmax(1) == gold[REP]
            o = np.argsort(-c, kind="stable")
            yb = (gold >= 3).astype(int)
            B = np.stack([P[:, :3].sum(1), P[:, 3:].sum(1)], 1)
            cb, okb = B[REP].max(1), B[REP].argmax(1) == yb[REP]
            ob = np.argsort(-cb, kind="stable")
            ceiling[f"{state}/{name}"] = {"accuracy": float(ok.mean()), "top": {str(k): float(ok[o[:k]].mean()) for k in TOP},
                                          "two_levels": {"accuracy": float(okb.mean()), "top": {str(k): float(okb[ob[:k]].mean()) for k in TOP}},
                                          "conformal": {str(1 - a): conformal(P, gold, a) for a in (0.10, 0.05)}}
    out["ceiling"] = ceiling
    out["two_levels_majority"] = float(max((gold[REP] >= 3).mean(), (gold[REP] < 3).mean()))
    out["majority_six_levels"] = float(np.bincount(gold[REP]).max() / len(REP))
    (D / "liar2_teto.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if "/" in k:
            print(k, round(v["accuracy"], 3), {t: round(x, 2) for t, x in v["top"].items()},
                  {g: {r: (round(x["violation"], 3), round(x["decisions"])) for r, x in v["gate"][g].items()} for g in v["gate"]})
    print(json.dumps(out["structured_minus_statement"], indent=0))


if __name__ == "__main__":
    main()
