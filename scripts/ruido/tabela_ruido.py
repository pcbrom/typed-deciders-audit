#!/usr/bin/env python3
"""Table of robustness to natural noise for the article (fonte/tables/noise.tex), from analise_ruido.json.
Bold: the highest accuracy and the lowest violation rate of each column within a language.

Usage: tabela_ruido.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve()
D = HERE.parents[2] / "data" / "derived" / "ruido"
OUT = HERE.parents[3] / "fonte" / "tables" / "noise.tex"
NAMES = {"jev": "Jev 1.13", "laya_multilingual": "Laya multilingual", "laya_ft": "Laya multilingual, fine-tuned"}
COLS = ("limpo", "typo_0.10", "misto_0.10", "misto_0.20")


def main():
    d = json.loads((D / "analise_ruido.json").read_text())["robustez"]
    lines = []
    for loc in ("pt", "en"):
        rows = {m: d[f"{m}_{loc}"] for m in NAMES}
        best = {c: max(r[c]["acc"] for r in rows.values()) for c in COLS}
        vio = {m: (r["limpo"]["portao"]["beta_0.95"]["violacao"], r["misto_0.10"]["portao"]["beta_0.95"]["violacao"]) for m, r in rows.items()}
        bv = (min(v[0] for v in vio.values()), min(v[1] for v in vio.values()))
        for m, r in rows.items():
            cells = [NAMES[m], loc]
            for c in COLS:
                s = f"{r[c]['acc']:.3f}"
                cells.append(f"\\textbf{{{s}}}" if r[c]["acc"] == best[c] else s)
            for k in (0, 1):
                s = f"{vio[m][k]:.2f}"
                cells.append(f"\\textbf{{{s}}}" if vio[m][k] == bv[k] else s)
            lines.append(" & ".join(cells) + " \\\\")
        if loc == "pt":
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
