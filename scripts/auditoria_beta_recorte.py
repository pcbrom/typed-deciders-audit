#!/usr/bin/env python3
"""Breakdown of the audit of the Beta rule by kind of decider (Jev, Laya as released, Laya fine-tuned) over every
exchangeable evaluation, each case taken once with its tightest measurement (the 10,000-draw re-run when there is one).
Per kind and rule: cases kept, undetermined and broken, mean violation rate and mean decisions per 1,000, and, per case,
the Beta rule against the point rule (violations avoided, decisions given up) and against Learn then Test.
Output: data/derived/auditoria_beta_recorte.json.

Usage: auditoria_beta_recorte.py
"""

import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from common import DATA  # noqa: E402

RULES = ("point", "beta", "ltt")
RELEASED_STATIC = {"typed-decisions", "root", "multilingual"}


def kind_of(source, case):
    if source == "static":
        system = case.split("/")[1]
        return "jev" if system == "jev" else "laya_released" if system in RELEASED_STATIC else "laya_finetuned"
    if source == "battery":
        m = case.split("/")[1]
        return "jev" if m == "jev" else "laya_released" if m == "released" else "laya_finetuned"
    return "laya_released" if case.split("/")[0] == "multilingual" else "laya_finetuned"


def main():
    a = json.loads((DATA / "derived" / "auditoria_beta.json").read_text())
    cases = {}
    for r in a["inventory"]:
        if r["kind"] != "exchangeable" or r["rule"] not in RULES:
            continue
        src = "static" if r["setting"].startswith("static") else "battery"
        key = f"{src}/{r['case']}"
        cases.setdefault(key, {"source": src, "case": r["case"], "rules": {}})["rules"][r["rule"]] = {
            "violation": r["violations"] / r["draws"], "category": r["category"], "decisions": r["decisions"], "draws": r["draws"]}
    for key, v in a["reruns"].items():
        src = key.split("/")[0]
        case = key.split("/", 1)[1]
        if src == "static":
            case = case.replace("pt/", "pt/", 1)
        cases[f"{src}/{case}"] = {"source": "own" if src == "own" else src, "case": case,
                                  "rules": {r: {"violation": v["rules"][r]["violation"], "category": v["rules"][r]["category"],
                                                "decisions": v["rules"][r]["decisions"], "draws": v["draws"]} for r in RULES}}
    by_kind = collections.defaultdict(list)
    for key, c in cases.items():
        by_kind[kind_of(c["source"], c["case"])].append(c)
    out = {}
    for kind, cs in sorted(by_kind.items()):
        row = {"cases": len(cs)}
        for r in RULES:
            cat = collections.Counter(c["rules"][r]["category"] for c in cs)
            row[r] = {"kept": cat["kept"], "undetermined": cat["undetermined"], "broken": cat["broken"],
                      "mean_violation": float(np.mean([c["rules"][r]["violation"] for c in cs])),
                      "max_violation": float(np.max([c["rules"][r]["violation"] for c in cs])),
                      "mean_decisions": float(np.mean([c["rules"][r]["decisions"] for c in cs]))}
        dp = np.array([c["rules"]["beta"]["decisions"] - c["rules"]["point"]["decisions"] for c in cs])
        dl = np.array([c["rules"]["beta"]["decisions"] - c["rules"]["ltt"]["decisions"] for c in cs])
        vp = np.array([c["rules"]["point"]["violation"] - c["rules"]["beta"]["violation"] for c in cs])
        row["beta_vs_point"] = {"violation_avoided_mean": float(vp.mean()), "decisions_given_up_mean": float(-dp.mean()),
                                "cases_point_broken_beta_not_broken": int(sum(c["rules"]["point"]["category"] == "broken" and c["rules"]["beta"]["category"] != "broken" for c in cs))}
        row["beta_vs_ltt"] = {"decisions_gained_mean": float(dl.mean()), "cases_more": int((dl > 0.5).sum()), "cases_fewer": int((dl < -0.5).sum())}
        out[kind] = row
    out["sources"] = dict(collections.Counter(c["source"] for c in cases.values()))
    (DATA / "derived" / "auditoria_beta_recorte.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
