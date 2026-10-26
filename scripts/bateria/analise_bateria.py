#!/usr/bin/env python3
"""Analysis of the multi-domain battery (implementation_notes.md, multi-domain battery): per domain, accuracy,
labels for non-inferiority and superiority to Jev, ECE, the gate against the plug-in population, the promise
under noise, the empty state and cost. The noise accuracy interval uses 10,000 item bootstrap resamples with
SeedSequence([20261023, 33, d]), a seed fixed here because the note does not name one.
Output: data/derived/bateria/analise_bateria.json.

Usage: analise_bateria.py [--domains news finance liar2 fake_pt] [--workers 20]
"""

import argparse
import collections
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "promise_check"))
import promise_check as pc  # noqa: E402
from common import CAL, DATA, REP, calibrated, ece, wilson  # noqa: E402
from dominios import DOMAINS  # noqa: E402

D = DATA / "derived" / "bateria"
PRICE, MARGIN, TARGETS, DRAWS, SEED = 0.042e-6, 0.02, (0.90, 0.95), 2000, 20261023
RULES = {r: pc.RULES[r] for r in ("point", "beta", "ltt")}
CONDITIONS = ("typo_0.10", "misto_0.10")


def boot(d, seed, n=10000):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return [round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)]


def gate_job(args):
    key, groups, reference, evaluate, temperature = args
    out = pc.check(groups, reference, evaluate, TARGETS, 1000, DRAWS, temperature, SEED, RULES)
    return key, {f"{g}/{t}": {r: {k: v[k] for k in ("violation", "violation_upper95", "decisions")} for r, v in out["results"][f"{g}/{t}"].items()}
                 for g in evaluate for t in TARGETS}


def load_jev(domain, n_report_items):
    rows = collections.defaultdict(dict)
    tokens = []
    # MASSIVE pt reuses earlier Jev answers (implementation_notes.md, reference row), kept in their own file
    for line in open(D / ("jev_massive_pt.jsonl" if domain == "massive_pt" else "jev.jsonl")):
        r = json.loads(line)
        if r["domain"] == domain:
            rows[r["condition"]][r["item"]] = np.array(r["probs"], dtype=float)
            if r["input_tokens"] is not None:
                tokens.append(r["input_tokens"])
    return rows, tokens


def analyse(domain, pool):
    d = DOMAINS.index(domain)
    meta_items = json.loads((DATA / "items" / "bateria" / f"{domain}.json").read_text())
    items = meta_items["items"]
    gold = np.array([it["gold"] for it in items])
    g_rep = gold[REP]
    k = len(items[0]["options"])
    jev, tokens = load_jev(domain, len(REP))
    J = np.array([jev["clean"][i] for i in range(len(items))])
    Z = dict(np.load(D / f"laya_{domain}.npz"))
    meta = json.loads((D / f"laya_{domain}.json").read_text())
    sizes = sorted(int(s) for s in meta["sizes"])
    top = sizes[-1]
    seeds = sorted({int(key.split("|")[2]) for key in Z if key.startswith(f"ft|{top}|")})
    ft = {s: Z[f"ft|{top}|{s}|clean"] for s in seeds}
    rel = Z["released|0|0|clean"]
    out = {"domain": domain, "locale": meta_items["locale"], "options": k, "chance": 1 / k, "n_pool": meta["n_pool"]}

    # 1 accuracy and 2 labels
    jev_ok = J[REP].argmax(1) == g_rep
    acc = {"jev": {"acc": float(jev_ok.mean()), "wilson95": wilson(int(jev_ok.sum()), len(jev_ok))},
           "released": {"acc": float((rel[REP].argmax(1) == g_rep).mean())}}
    acc["released"]["wilson95"] = wilson(int(round(acc["released"]["acc"] * len(REP))), len(REP))
    lab = {}
    for s in sizes:
        oks = np.array([Z[f"ft|{s}|{sd}|clean"][REP].argmax(1) == g_rep for sd in seeds], float)
        diff = (oks - jev_ok[None, :]).mean(0)
        lo, hi = boot(diff, [SEED, 32, d])
        lab[str(s)] = {"acc_per_seed": [round(float(o.mean()), 4) for o in oks], "acc_mean": round(float(oks.mean()), 4),
                       "minus_jev": round(float(diff.mean()), 4), "ci95": [lo, hi], "non_inferior": lo > -MARGIN, "superior": lo > 0}
    acc["ft"] = {s: lab[s]["acc_mean"] for s in lab}
    out["accuracy"] = acc
    out["labels"] = {"sizes": lab,
                     "non_inferior_from": next((s for s in sizes if all(lab[str(t)]["non_inferior"] for t in sizes if t >= s)), None),
                     "superior_from": next((s for s in sizes if all(lab[str(t)]["superior"] for t in sizes if t >= s)), None)}

    # 3 ECE
    def ece_laya(P):
        Q, _ = calibrated(P, CAL, gold)
        return ece(Q[REP].max(1), Q[REP].argmax(1) == g_rep)
    out["ece"] = {"jev": ece(J[REP].max(1), jev_ok), "released": ece_laya(rel), "ft_per_seed": [ece_laya(ft[s]) for s in seeds]}

    # 4 gate and 5 noise promise
    jobs = [("gate|jev", {"all": {"conf": J.max(1), "ok": J.argmax(1) == gold}}, "all", ["all"], False),
            ("gate|released", {"all": (rel, gold)}, "all", ["all"], True)]
    jobs += [(f"gate|ft{s}", {"all": (ft[s], gold)}, "all", ["all"], True) for s in seeds]
    noisy = {}
    for c in CONDITIONS:
        noisy[("jev", c)] = np.array([jev[c][i] for i in REP])
        noisy[("released", c)] = Z[f"released|0|0|{c}"]
        for s in seeds:
            noisy[(f"ft{s}", c)] = Z[f"ft|{top}|{s}|{c}"]
    jobs.append(("noise|jev", {"cal": {"conf": J[CAL].max(1), "ok": J[CAL].argmax(1) == gold[CAL]},
                               **{c: {"conf": noisy[("jev", c)].max(1), "ok": noisy[("jev", c)].argmax(1) == g_rep} for c in CONDITIONS}},
                 "cal", list(CONDITIONS), False))
    for m, P in [("released", rel)] + [(f"ft{s}", ft[s]) for s in seeds]:
        jobs.append((f"noise|{m}", {"cal": (P[CAL], gold[CAL]), **{c: (noisy[(m, c)], g_rep) for c in CONDITIONS}}, "cal", list(CONDITIONS), True))
    res = dict(pool.map(gate_job, jobs, chunksize=1))
    out["gate"] = {key.split("|")[1]: v for key, v in res.items() if key.startswith("gate|")}
    out["noise_promise"] = {key.split("|")[1]: v for key, v in res.items() if key.startswith("noise|")}
    loss = {}
    for c in CONDITIONS:
        for m, P in [("jev", J[REP]), ("released", rel[REP])] + [(f"ft{s}", ft[s][REP]) for s in seeds]:
            dd = (noisy[(m, c)].argmax(1) == g_rep).astype(float) - (P.argmax(1) == g_rep)
            loss[f"{m}/{c}"] = {"mean": round(float(dd.mean()), 4), "ci95": boot(dd, [SEED, 33, d])}
    out["noise_accuracy"] = loss

    # 6 empty state
    if len(jev["empty"]) == 1:
        dec = int(next(iter(jev["empty"].values())).argmax())
        jev_empty = float((g_rep == dec).mean())
    else:
        E = np.array([jev["empty"][i] for i in REP])
        jev_empty = float((E.argmax(1) == g_rep).mean())
    emp = {"jev": jev_empty, "released": float((Z["released|0|0|empty"].argmax(1) == g_rep).mean())}
    shares = []
    for s in seeds:
        E = Z[f"ft|{top}|{s}|empty"].argmax(1)
        counts = next(x["label_counts"] for x in meta["sizes"][str(top)] if x["seed"] == s)
        freq = np.array([int(np.argmax([counts.get(o, 0) for o in items[i]["options"]])) for i in REP])
        emp[f"ft{s}"] = float((E == g_rep).mean())
        shares.append(float((E == freq).mean()))
    emp["ft_most_frequent_share_per_seed"] = shares
    out["empty"] = emp

    # 7 cost
    out["cost"] = {"jev_tokens_mean": float(np.mean(tokens)), "jev_usd_per_1000": float(np.mean(tokens)) * PRICE * 1000, "jev_calls": len(tokens),
                   "ft_seconds_largest": [x["seconds"] for x in meta["sizes"][str(top)]], "ft_size_largest": top}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS))
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    path = D / "analise_bateria.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    with Pool(args.workers) as pool:
        for domain in args.domains:
            result[domain] = analyse(domain, pool)
            r = result[domain]
            print(domain, "acc jev", round(r["accuracy"]["jev"]["acc"], 3), "released", round(r["accuracy"]["released"]["acc"], 3),
                  "ni from", r["labels"]["non_inferior_from"], "sup from", r["labels"]["superior_from"], flush=True)
    path.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
