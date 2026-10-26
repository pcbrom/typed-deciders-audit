#!/usr/bin/env python3
"""Summary table of the multi-domain battery (fonte/tables/battery.tex) from analise_bateria.json.
Per domain: accuracy of Jev, released Laya and the largest fine-tune (mean of seeds), labels from which the
fine-tune is non-inferior and superior to Jev, Jev's ECE, the cases (of 10: two targets, five deciders with the
three seeds) in which each rule keeps the promise, the median decisions per 1,000 of the fine-tunes at 0.95 for
the Beta rule and Learn then Test, and the seeds in which the Beta promise of the fine-tune survives typos.
Bold: the highest accuracy of each row and the rule that keeps the promise in most cases, where they differ.

Usage: tabela_bateria.py
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
D = HERE.parents[2] / "data" / "derived" / "bateria"
OUT = HERE.parents[3] / "fonte" / "tables" / "battery.tex"
NAMES = {"massive_pt": "MASSIVE pt (ref.)", "news": "News Category", "finance": "Financial tweets", "liar2": "LIAR2", "fake_pt": "Fake.Br (pt)"}
MODELS = ("jev", "released", "ft0", "ft1", "ft2")


def kept(r, rule):
    return sum(r["gate"][m][f"all/{t}"][rule]["violation_upper95"] <= 0.05 for m in MODELS for t in ("0.9", "0.95"))


def main():
    R = json.loads((D / "analise_bateria.json").read_text())
    lines = []
    for d, name in NAMES.items():
        r = R[d]
        top = str(r["cost"]["ft_size_largest"])
        accs = [r["accuracy"]["jev"]["acc"], r["accuracy"]["released"]["acc"], r["labels"]["sizes"][top]["acc_mean"]]
        cells = [name, str(r["options"])]
        cells += [f"\\textbf{{{a:.3f}}}" if a == max(accs) else f"{a:.3f}" for a in accs]
        ni, sup = r["labels"]["non_inferior_from"], r["labels"]["superior_from"]
        cells += [f"{ni:,}" if ni else "none", f"{sup:,}" if sup else "none", f"{r['ece']['jev']:.3f}"]
        k = {rule: kept(r, rule) for rule in ("point", "beta", "ltt")}
        cells += [f"\\textbf{{{v}}}" if v == max(k.values()) and len(set(k.values())) > 1 else str(v) for v in k.values()]
        dec = {rule: np.median([r["gate"][f"ft{s}"]["all/0.95"][rule]["decisions"] for s in (0, 1, 2)]) for rule in ("beta", "ltt")}
        cells += [f"\\textbf{{{v:.0f}}}" if v == max(dec.values()) and len({f"{x:.0f}" for x in dec.values()}) > 1 else f"{v:.0f}" for v in dec.values()]
        typo = sum(r["noise_promise"][f"ft{s}"]["typo_0.10/0.95"]["beta"]["violation_upper95"] <= 0.05 for s in (0, 1, 2))
        cells.append(f"{typo} of 3" if max(dec.values()) > 0 else "no decisions")
        lines.append(" & ".join(cells) + " \\\\")
        if d == "massive_pt":
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
