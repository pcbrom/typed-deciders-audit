#!/usr/bin/env python3
"""Report cards of the battery for the appendix (fonte/tables/battery_cards.tex), one block per domain with the same
layout: per decider, accuracy and ECE on the report items, violation rate and decisions per 1,000 of each rule at 0.90
and 0.95 against the plug-in population, the Beta promise at 0.95 calibrated on clean text and checked under typos and
the mixed profile, and the accuracy with an empty state. Bold, within each block: the highest accuracy, the lowest ECE,
and in each promise column the decider that keeps the promise (upper bound at most 0.05) with the most decisions.

Usage: tabela_boletins.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve()
D = HERE.parents[2] / "data" / "derived" / "bateria"
OUT = HERE.parents[3] / "fonte" / "tables" / "battery_cards.tex"
NAMES = {"massive_pt": "MASSIVE pt-PT (reference)", "news": "News Category", "finance": "Financial tweets", "liar2": "LIAR2", "fake_pt": "Fake.Br (pt)"}
DECIDERS = (("jev", "Jev 1.13"), ("released", "Laya"), ("ft0", "Laya-ft, seed 0"), ("ft1", "Laya-ft, seed 1"), ("ft2", "Laya-ft, seed 2"))
COLS = [("gate", f"all/{t}", rule) for t in ("0.9", "0.95") for rule in ("point", "beta", "ltt")] + \
       [("noise_promise", f"{c}/0.95", "beta") for c in ("typo_0.10", "misto_0.10")]


def cell(x):
    return f"{x['violation']:.2f}/{x['decisions']:.0f}"


def main():
    R = json.loads((D / "analise_bateria.json").read_text())
    lines = []
    for d, name in NAMES.items():
        r = R[d]
        top = str(r["cost"]["ft_size_largest"])
        acc = {"jev": r["accuracy"]["jev"]["acc"], "released": r["accuracy"]["released"]["acc"],
               **{f"ft{s}": r["labels"]["sizes"][top]["acc_per_seed"][s] for s in (0, 1, 2)}}
        ece = {"jev": r["ece"]["jev"], "released": r["ece"]["released"], **{f"ft{s}": r["ece"]["ft_per_seed"][s] for s in (0, 1, 2)}}
        emp = {m: r["empty"][m] for m, _ in DECIDERS}
        best = []
        for sec, key, rule in COLS:
            ok = {m: r[sec][m][key][rule] for m, _ in DECIDERS if r[sec][m][key][rule]["violation_upper95"] <= 0.05 and round(r[sec][m][key][rule]["decisions"]) >= 1}
            top_dec = max((round(v["decisions"]) for v in ok.values()), default=None)
            best.append({m for m, v in ok.items() if round(v["decisions"]) == top_dec})
        lines.append(f"\\multicolumn{{12}}{{@{{}}l}}{{\\textit{{{name}}}, {r['options']} options, fine-tunes on {int(top):,} labels}} \\\\")
        for m, label in DECIDERS:
            cells = [label, f"\\textbf{{{acc[m]:.3f}}}" if f"{acc[m]:.3f}" == f"{max(acc.values()):.3f}" else f"{acc[m]:.3f}",
                     f"\\textbf{{{ece[m]:.3f}}}" if f"{ece[m]:.3f}" == f"{min(ece.values()):.3f}" else f"{ece[m]:.3f}"]
            for (sec, key, rule), b in zip(COLS, best):
                s = cell(r[sec][m][key][rule])
                cells.append(f"\\textbf{{{s}}}" if m in b else s)
            cells.append(f"{emp[m]:.3f}")
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    OUT.write_text("\n".join(lines) + "\n")
    print(len(lines), "lines")


if __name__ == "__main__":
    main()
