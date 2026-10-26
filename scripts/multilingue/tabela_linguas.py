#!/usr/bin/env python3
"""Per-locale reference table for the article (fonte/tables/languages.tex), from analise_multilingue.json:
accuracy on the report items, and violation rate of the Beta rule at 0.95 calibrated in en-US and in the
locale itself. Bold, within each row: the highest accuracy and the lowest violation rate of each group, unless all are equal.

Usage: tabela_linguas.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve()
D = HERE.parents[2] / "data" / "derived" / "multilingue"
OUT = HERE.parents[3] / "fonte" / "tables" / "languages.tex"
MODELS = ("jev", "laya_multilingual", "laya_ft")


def bold(values, fmt, best):
    b = best(values)
    if len({fmt(v) for v in values}) == 1:
        return [fmt(v) for v in values]
    return [f"\\textbf{{{fmt(v)}}}" if fmt(v) == fmt(b) else fmt(v) for v in values]


def main():
    d = json.loads((D / "analise_multilingue.json").read_text())
    acc, vio = (lambda v: f"{v:.3f}"), (lambda v: f"{v:.2f}")
    lines = []
    for r in d["locales"]:
        cells = [r["locale"], r["script"]]
        cells += bold([r[m]["accuracy"] for m in MODELS], acc, max)
        cells += bold([r[m]["promise_en"]["beta"]["violation"] for m in MODELS], vio, min)
        cells += bold([r[m]["promise_own"]["beta"]["violation"] for m in MODELS[1:]], vio, min)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    OUT.write_text("\n".join(lines) + "\n")
    print(len(lines) - 1, "rows")


if __name__ == "__main__":
    main()
