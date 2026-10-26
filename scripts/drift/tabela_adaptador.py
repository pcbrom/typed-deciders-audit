#!/usr/bin/env python3
"""LaTeX rows of the adapter table of the article (tab:adapter), from analysis_adaptador.json: mean cost
per item and share of windows below the promise per arm on S1, S2, D1 and D3, the lowest per column in bold.

Usage: tabela_adaptador.py ANALYSIS_JSON OUT_TEX
"""

import json
import sys
from pathlib import Path

ARMS = [("recal", "Recalibration only"), ("recal_lam08", r"Recalibration, $\lambda=0.80$"), ("ltt_window", "LTT per window"), ("author", "Author's loop"),
        ("periodic", "Periodic"), ("dyn_level", "Dynamic, level"), ("dyn_season", "Dynamic, seasonal"),
        ("dyn_season_exog", "Dynamic, seasonal and inputs"), ("dyn_season_exog_lb", "Dynamic, same, lower bound")]
COLS = [(s, key, fmt) for s in ("S1", "S2", "D1", "D3") for key, fmt in (("custo", "{:.3f}"), ("janelas_abaixo", "{:.2f}"))]


def main(src, out):
    med = json.loads(Path(src).read_text())["medias"]
    cells = {}
    for s, key, fmt in COLS:
        shown = {arm: fmt.format(med[f"{arm}|{s}"][key]) for arm, _ in ARMS if f"{arm}|{s}" in med}
        low = min(float(v) for v in shown.values())
        cells[(s, key)] = {a: rf"\textbf{{{v}}}" if float(v) == low else v for a, v in shown.items()}
    rows = [" & ".join([label] + [cells[(s, key)].get(arm, "--") for s, key, _ in COLS]) + r" \\" for arm, label in ARMS]
    Path(out).write_text("\n".join(rows + [r"\bottomrule"]) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
