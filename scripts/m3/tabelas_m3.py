#!/usr/bin/env python3
"""LaTeX rows of the M3 tables of the article: gates on ETS over all series (tab:m3-gates) and deciders on the
sample (tab:m3-deciders), from analise_m3_ets.json and analise_tipados_m3.json.

Usage: tabelas_m3.py OUT_DIR
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from common import DATA  # noqa: E402

D = DATA / "derived" / "m3"
GATES = [("static_point", "Point, calibrated once"), ("static_ltt", "Learn then Test, calibrated once"),
         ("sliding12_point", "Point, sliding"), ("sliding12_beta", "Beta rule, sliding"), ("sliding12_ltt", "Learn then Test, sliding"),
         ("recal", "Forgetting Beta per bin"), ("dynamic", "Dynamic per bin"), ("dynamic_lb", "Dynamic per bin, lower bound")]
DECIDERS = [("ETS", "ETS"), ("Jev", "Jev"), ("Laya released", "Laya as released"), ("Laya fine-tuned, mean of seeds", "Laya fine-tuned")]


def gate_rows():
    t = json.loads((D / "analise_m3_ets.json").read_text())["by_group"]["ALL"]["targets"]
    cols = [(tg, key) for tg in ("0.7", "0.8") for key in ("origins_below", "decisions_per_1000")]
    low = {tg: min(round(t[tg][g]["origins_below"], 2) for g, _ in GATES) for tg in ("0.7", "0.8")}
    rows = []
    for g, label in GATES:
        cells = []
        for tg, key in cols:
            v = t[tg][g][key]
            if key == "origins_below":
                s = f"{v:.2f}"
                cells.append(rf"\textbf{{{s}}}" if round(v, 2) == low[tg] else s)
            else:
                cells.append(f"{v:.0f}")
        rows.append(" & ".join([label] + cells) + r" \\")
    return rows


def ci(x):
    return f"$[{x[0]:.3f}, {x[1]:.3f}]$"


def decider_rows():
    d = json.loads((D / "analise_tipados_m3.json").read_text())["deciders"]
    best = max(d[k]["accuracy"] for k, _ in DECIDERS)
    rows = []
    for k, label in DECIDERS:
        r = d[k]
        acc = f"{r['accuracy']:.3f}"
        acc = rf"\textbf{{{acc}}}" if r["accuracy"] == best else acc
        diff = "--" if r["minus_ets"] is None else f"${r['minus_ets']['mean']:+.3f}$ {ci(r['minus_ets']['IC95'])}"
        rows.append(" & ".join([label, f"{acc} {ci(r['accuracy_IC95'])}", f"{r['macro_f1']:.3f}", f"{r['share_higher']:.2f}",
                                diff, f"{r['ceiling']['100']:.2f}"]) + r" \\")
    return rows


if __name__ == "__main__":
    out = Path(sys.argv[1])
    (out / "m3_gates.tex").write_text("\n".join(gate_rows() + [r"\bottomrule"]) + "\n")
    (out / "m3_deciders.tex").write_text("\n".join(decider_rows() + [r"\bottomrule"]) + "\n")
