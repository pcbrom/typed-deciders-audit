#!/usr/bin/env python3
"""Analysis of languages without fine-tuning (implementation_notes.md, languages without fine-tuning,
and promise across languages): per locale and model, accuracy on the report items with a 95% Wilson
interval, paired difference against en-US with a 95% item bootstrap, non-inferiority to en-US with the
2-point margin, ECE, majority script, fine-tuning transfer, and the 0.95 promise calibrated in en-US and
checked in each locale with promise_check. The bootstrap seed is SeedSequence([20261023, 20, k]) with k
the index of the locale in alphabetical order. Locales without Jev answers are listed and left out.
Output: data/derived/multilingue/analise_multilingue.json and tabela_linguas.csv.

Usage: analise_multilingue.py [--workers 20]
"""

import argparse
import collections
import csv
import json
import sys
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "promise_check"))
import promise_check as pc  # noqa: E402
from common import CAL, DATA, REP, ece, fit_temperature, log_probs, softmax_t, wilson  # noqa: E402

D = DATA / "derived" / "multilingue"
ITEMS = DATA / "items" / "multilingue"
MARGIN, TARGET, DRAWS, SEED = 0.02, 0.95, 2000, 20261023
RULES = {r: pc.RULES[r] for r in ("point", "beta", "ltt")}
REUSED = {"pt-PT": "pt", "en-US": "en"}


def script_of(texts):
    c = collections.Counter(unicodedata.name(ch, "UNKNOWN").split()[0] for t in texts for ch in t if ch.isalpha())
    s = c.most_common(1)[0][0]
    return "Han" if s == "CJK" else s.title()


def load():
    locales = sorted(p.stem.split("_")[1] for p in ITEMS.glob("massive_*_k5.json"))
    items = {loc: json.loads((ITEMS / f"massive_{loc}_k5.json").read_text())["items"] for loc in locales}
    gold = np.array([it["options"].index(it["intent"]) for it in items["en-US"]])
    laya = {m: dict(np.load(D / f"laya_{m}.npz")) for m in ("multilingual", "ft")}
    jev = collections.defaultdict(dict)
    for line in open(D / "jev.jsonl"):
        r = json.loads(line)
        jev[r["locale"]][r["item"]] = r["probs"]
    for line in open(DATA / "derived" / "ruido" / "jev.jsonl"):
        r = json.loads(line)
        if r["condition"] == "limpo":
            jev[{v: k for k, v in REUSED.items()}[r["loc"]]][r["item"]] = r["probs"]
    full = {loc: np.array([v[i] for i in REP]) for loc, v in jev.items() if all(i in v for i in REP)}
    jev_cal_en = np.array(json.loads((DATA / "raw" / "jev_en.json").read_text())["probs"], dtype=float)[CAL]
    return locales, items, gold, laya, full, jev_cal_en


def boot(d, k, n=10000):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 20, k]))
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def promise_job(args):
    name, ref, evaluate, temperature = args
    return name, pc.check({**ref, **evaluate}, "ref", sorted(evaluate), (TARGET,), 1000, DRAWS, temperature, SEED, RULES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    locales, items, gold, laya, jev, jev_cal_en = load()
    g_rep, g_cal = gold[REP], gold[CAL]
    models = {"jev": {loc: jev[loc] for loc in locales if loc in jev},
              "laya_multilingual": {loc: laya["multilingual"][loc][REP] for loc in locales},
              "laya_ft": {loc: laya["ft"][loc][REP] for loc in locales}}
    ok = {m: {loc: P.argmax(axis=1) == g_rep for loc, P in v.items()} for m, v in models.items()}
    jobs = [("jev/en-US", {"ref": (jev_cal_en, g_cal)}, {loc: (P, g_rep) for loc, P in models["jev"].items()}, False)]
    for m in ("multilingual", "ft"):
        jobs.append((f"laya_{m}/en-US", {"ref": (laya[m]["en-US"][CAL], g_cal)}, {loc: (laya[m][loc][REP], g_rep) for loc in locales}, True))
        jobs += [(f"own:laya_{m}/{loc}", {"ref": (laya[m][loc][CAL], g_cal)}, {loc: (laya[m][loc][REP], g_rep)}, True) for loc in locales]
    with Pool(args.workers) as pool:
        promise = dict(pool.map(promise_job, jobs, chunksize=1))
    rows = []
    for k, loc in enumerate(locales):
        row = {"locale": loc, "script": script_of(it["utterance"] for it in items[loc])}
        for m, v in models.items():
            if loc not in v:
                continue
            o = ok[m][loc]
            d = o.astype(float) - ok[m]["en-US"]
            lo, hi = boot(d, k)
            if m == "jev":
                conf, okc = v[loc].max(axis=1), o
            else:
                full = laya[m.split("_")[1]][loc]
                lp = log_probs(full)
                Q = softmax_t(lp[REP], fit_temperature(lp[CAL], g_cal))
                conf, okc = Q.max(axis=1), Q.argmax(axis=1) == g_rep
            row[m] = {"accuracy": float(o.mean()), "wilson95": wilson(int(o.sum()), len(o)), "minus_en": float(d.mean()), "minus_en_ci95": [lo, hi],
                      "holds": lo > -MARGIN, "ece": ece(conf, okc), "ece_calibration": "as returned" if m == "jev" else "temperature on the locale calibration items",
                      "promise_en": {r: promise[f"{m}/en-US"]["results"][f"{loc}/{TARGET}"][r] for r in RULES}}
            if m != "jev":
                row[m]["promise_own"] = {r: promise[f"own:{m}/{loc}"]["results"][f"{loc}/{TARGET}"][r] for r in RULES}
        f, b = ok["laya_ft"][loc].astype(float), ok["laya_multilingual"][loc]
        row["ft_minus_released"] = {"mean": float((f - b).mean()), "ci95": boot(f - b, 1000 + k)}
        rows.append(row)
    out = {"target": TARGET, "margin": MARGIN, "draws": DRAWS, "seed": SEED, "rules": list(RULES), "n_report": len(REP),
           "jev_missing": [loc for loc in locales if loc not in jev], "locales": rows}
    (D / "analise_multilingue.json").write_text(json.dumps(out, indent=1))
    with open(D / "tabela_linguas.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["locale", "script"]
        for m in models:
            head += [f"{m}_accuracy", f"{m}_ci_low", f"{m}_ci_high", f"{m}_minus_en", f"{m}_holds", f"{m}_ece", f"{m}_beta_violation_en", f"{m}_beta_violation_upper95_en", f"{m}_beta_decisions_en"]
        w.writerow(head)
        for r in rows:
            line = [r["locale"], r["script"]]
            for m in models:
                x = r.get(m)
                line += ([round(x["accuracy"], 4), *x["wilson95"], round(x["minus_en"], 4), x["holds"], round(x["ece"], 4),
                          x["promise_en"]["beta"]["violation"], round(x["promise_en"]["beta"]["violation_upper95"], 4), round(x["promise_en"]["beta"]["decisions"], 1)]
                         if x else [""] * 9)
            w.writerow(line)
    print("locales", len(rows), "jev missing", out["jev_missing"])


if __name__ == "__main__":
    main()
