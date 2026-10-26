#!/usr/bin/env python3
"""Audit of the claim about the Beta rule (implementation_notes.md, audit of the claim about the Beta rule).

1. Inventory: every evaluation of a rule already computed, with a two-sided 95% Clopper-Pearson interval on its
   violation rate and the three categories kept, undetermined and broken.
2. Re-runs with 10,000 draws on the plug-in population, recording per draw the violation and decisions of the point
   rule, the Beta rule with the ascending and the fixed-sequence scans and Learn then Test, and the divergence of the
   two scans: battery cases undetermined or broken with 2,000 draws, the released pt-PT checkpoints left out of the
   static gate, and Laya calibrated in each MASSIVE locale.
Output: data/derived/auditoria_beta.json.

Usage: auditoria_beta.py [--workers 20]
"""

import argparse
import collections
import json
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import beta as B

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "promise_check"))
import promise_check as pc  # noqa: E402
from common import DATA, fit_temperature, load_items, load_systems, log_probs, softmax_t  # noqa: E402

DRAWS, SEED = 10000, 20261024
RULES = {"point": pc.RULES["point"], "beta": pc.RULES["beta"], "beta_fixed": pc.RULES["beta_fixed_sequence"], "ltt": pc.RULES["ltt"]}


def cp(k, n):
    lo = 0.0 if k == 0 else float(B.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(B.ppf(0.975, k + 1, n - k))
    return lo, hi


def category(k, n):
    lo, hi = cp(k, n)
    return "kept" if hi <= 0.05 else "broken" if lo > 0.05 else "undetermined"


def inventory():
    rows = []
    g = json.loads((DATA / "derived" / "population_gate.json").read_text())
    n = g["plugin_draws"]
    for case, v in g["results"].items():
        for r, name in (("point", "point"), ("beta_ascending", "beta"), ("beta_fixed_sequence", "beta_fixed"), ("ltt", "ltt")):
            rows.append({"setting": "static gate, MASSIVE", "kind": "exchangeable", "case": case, "rule": name,
                         "violations": v[r]["plugin"]["violations"], "draws": n, "decisions": v[r]["plugin"]["decisions"]})
    b = json.loads((DATA / "derived" / "bateria" / "analise_bateria.json").read_text())
    for d, r in b.items():
        for m, tv in r["gate"].items():
            for t, rv in tv.items():
                for rule, x in rv.items():
                    rows.append({"setting": "battery gate", "kind": "exchangeable", "case": f"{d}/{m}/{t.split('/')[-1]}", "rule": rule,
                                 "violations": int(round(x["violation"] * 2000)), "draws": 2000, "decisions": x["decisions"]})
        for m, tv in r["noise_promise"].items():
            for t, rv in tv.items():
                for rule, x in rv.items():
                    rows.append({"setting": "battery under noise", "kind": "shifted", "case": f"{d}/{m}/{t.replace('/', '@')}", "rule": rule,
                                 "violations": int(round(x["violation"] * 2000)), "draws": 2000, "decisions": x["decisions"]})
    ml = json.loads((DATA / "derived" / "multilingue" / "analise_multilingue.json").read_text())
    for row in ml["locales"]:
        for m in ("jev", "laya_multilingual", "laya_ft"):
            for rule, x in row[m]["promise_en"].items():
                rows.append({"setting": "calibrated in en-US, checked per locale", "kind": "shifted", "case": f"{row['locale']}/{m}", "rule": rule,
                             "violations": x["violations"], "draws": ml["draws"], "decisions": x["decisions"]})
    for r in rows:
        r["category"] = category(r["violations"], r["draws"])
        r["ci95"] = cp(r["violations"], r["draws"])
    return rows


def run_case(args):
    """Plug-in population: draws with replacement, rules fitted on the draw, evaluated on the whole population."""
    key, conf_ok, P, gold, temperature, n_cal, target = args
    rng = np.random.default_rng(np.random.SeedSequence([SEED, zlib.crc32(key.encode())]))
    if temperature:
        lp = log_probs(P)
    else:
        conf_all, ok_all = conf_ok
    n = len(gold)
    viol = {r: np.zeros(DRAWS, bool) for r in RULES}
    dec = {r: np.zeros(DRAWS) for r in RULES}
    div = np.zeros(DRAWS, bool)
    for i in range(DRAWS):
        cal = rng.integers(0, n, n_cal)
        if temperature:
            Q = softmax_t(lp, fit_temperature(lp[cal], gold[cal]))
            conf_all, ok_all = Q.max(1), Q.argmax(1) == gold
        taus = {r: pc.threshold(cfg, conf_all[cal], ok_all[cal], target) for r, cfg in RULES.items()}
        div[i] = taus["beta"] != taus["beta_fixed"]
        for r, tau in taus.items():
            if tau is None:
                continue
            a = conf_all >= tau
            dec[r][i] = a.mean() * 1000
            viol[r][i] = a.any() and ok_all[a].mean() < target
    out = {"draws": DRAWS, "n_cal": n_cal, "target": target, "rules": {}}
    for r in RULES:
        k = int(viol[r].sum())
        out["rules"][r] = {"violations": k, "violation": k / DRAWS, "ci95": cp(k, DRAWS), "category": category(k, DRAWS), "decisions": float(dec[r].mean())}
    d = dec["beta"] - dec["ltt"]
    out["beta_minus_ltt_decisions"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(DRAWS))}
    out["divergence"] = float(div.mean())
    out["beta_breaks_fixed_keeps"] = float((viol["beta"] & ~viol["beta_fixed"]).mean())
    out["bound_check"] = bool(((viol["beta"] & ~viol["beta_fixed"]) <= div).all())
    return key, out


def jobs_from(inv):
    jobs = []
    # battery cases undetermined or broken for beta or ltt
    bat = json.loads((DATA / "derived" / "bateria" / "analise_bateria.json").read_text())
    need = {(r["case"]) for r in inv if r["setting"] == "battery gate" and r["rule"] in ("beta", "ltt") and r["category"] != "kept"}
    cache = {}
    for case in sorted(need):
        d, m, t = case.split("/")
        if d not in cache:
            items = json.loads((DATA / "items" / "bateria" / f"{d}.json").read_text())["items"]
            gold = np.array([it["gold"] for it in items])
            jev = {}
            for line in open(DATA / "derived" / "bateria" / ("jev_massive_pt.jsonl" if d == "massive_pt" else "jev.jsonl")):
                r = json.loads(line)
                if r["domain"] == d and r["condition"] == "clean":
                    jev[r["item"]] = r["probs"]
            J = np.array([jev[i] for i in range(len(items))], dtype=float)
            Z = np.load(DATA / "derived" / "bateria" / f"laya_{d}.npz")
            top = bat[d]["cost"]["ft_size_largest"]
            cache[d] = (gold, J, {"released": Z["released|0|0|clean"], **{f"ft{s}": Z[f"ft|{top}|{s}|clean"] for s in (0, 1, 2)}})
        gold, J, L = cache[d]
        if m == "jev":
            jobs.append((f"battery/{case}", (J.max(1), J.argmax(1) == gold), None, gold, False, 1000, float(t)))
        else:
            jobs.append((f"battery/{case}", None, L[m].astype(float), gold, True, 1000, float(t)))
    # released pt-PT checkpoints with the static gate design
    _, gold = load_items("pt")
    S = load_systems("pt")
    for key in ("multilingual", "root", "typed-decisions"):
        for n_cal in (1000, 500):
            for t in (0.95, 0.99):
                jobs.append((f"static/pt/{key}/ncal{n_cal}/{t}", None, S[key], gold, True, n_cal, t))
    # Laya calibrated in each locale, plug-in on the locale's 2,000 items
    D = DATA / "derived" / "multilingue"
    items = json.loads((DATA / "items" / "multilingue" / "massive_en-US_k5.json").read_text())["items"]
    gold = np.array([it["options"].index(it["intent"]) for it in items])
    for m in ("multilingual", "ft"):
        Z = np.load(D / f"laya_{m}.npz")
        for loc in Z.files:
            jobs.append((f"own/{m}/{loc}", None, Z[loc].astype(float), gold, True, 1000, 0.95))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    inv = inventory()
    jobs = jobs_from(inv)
    print("inventory", len(inv), "re-runs", len(jobs), flush=True)
    with Pool(args.workers) as pool:
        reruns = dict(pool.imap_unordered(run_case, jobs, chunksize=1))
    summary = collections.defaultdict(lambda: collections.Counter())
    for r in inv:
        summary[f"{r['setting']} | {r['rule']}"][r["category"]] += 1
    out = {"inventory": inv, "inventory_summary": {k: dict(v) for k, v in summary.items()}, "reruns": reruns}
    (DATA / "derived" / "auditoria_beta.json").write_text(json.dumps(out, indent=1))
    for k, v in sorted(out["inventory_summary"].items()):
        print(k, v)


if __name__ == "__main__":
    main()
