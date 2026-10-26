#!/usr/bin/env python3
"""LaTeX rows of the drift tables of the article: the main run (tab:drift) and the efficiency at equal
promise (tab:drift-eff), from analysis_main.json and eficiencia.json.

Usage: tabelas_drift.py MAIN_JSON EFICIENCIA_JSON OUT_DIR
"""

import json
import sys
from pathlib import Path

ARMS = [("author", "Author's loop"), ("periodic", "Periodic ($K=3{,}000$)"), ("ddm", "DDM"), ("adwin", "ADWIN"),
        ("ph", "Page--Hinkley"), ("recal", "Recalibration only"), ("ltt_window", "LTT per window"), ("aci", "ACI"),
        ("frozen", "Frozen")]
NAME = dict(ARMS)
NAME["periodic"] = "Periodic"


def bold_min(values, fmt):
    """Format a column, in bold where the value equals the column minimum at the printed precision."""
    shown = [fmt.format(v) for v in values]
    low = min(float(s) for s in shown)
    return [rf"\textbf{{{s}}}" if float(s) == low else s for s in shown]


def main_rows(sec):
    cols = [("custo10", d, "{:.3f}", True) for d in ("D1", "D2", "D3", "control")]
    cols += [("janelas_abaixo", d, "{:.2f}", True) for d in ("D1", "D2", "D3")]
    cols += [("retreinos", d, "{:.1f}", False) for d in ("D1", "D3", "control")]
    table = []
    for key, d, fmt, rank in cols:
        vals = [sec[f"{a}|{d}"][key] for a, _ in ARMS]
        table.append(bold_min(vals, fmt) if rank else [fmt.format(v) for v in vals])
    return [" & ".join([label] + [c[i] for c in table]) + r" \\" for i, (_, label) in enumerate(ARMS)]


def verdict(v):
    """C clean gain, T trade-off, E equivalent, NI non-inferior, -- not shown."""
    res = v["recursos"]
    larger = any(res[k]["IC95"][0] > 0 for k in ("rotulos_por_1000", "retreinos", "gpu_s", "erros_por_1000"))
    if v["ganho_eficiencia"]:
        return "T" if larger else r"\textbf{C}"
    if v["equivalente"]:
        return "E"
    return "NI" if v["nao_inferior"] else "--"


def eff_rows(eff):
    out = []
    for me in ("author", "recal"):
        if out:
            out.append(r"\midrule")
        for other, _ in ARMS:
            if other == me:
                continue
            cells = [verdict(eff[f"{me}|{other}|{d}"]) for d in ("D1", "D2", "D3", "control")]
            out.append(" & ".join([NAME[me], NAME[other]] + cells) + r" \\")
    return out


if __name__ == "__main__":
    main_json, eff_json, out_dir = map(Path, sys.argv[1:4])
    sec = json.loads(main_json.read_text())["secundarias"]
    eff = json.loads(eff_json.read_text())["drift"]
    (out_dir / "drift_main.tex").write_text("\n".join(main_rows(sec) + [r"\bottomrule"]) + "\n")
    (out_dir / "drift_efficiency.tex").write_text("\n".join(eff_rows(eff) + [r"\bottomrule"]) + "\n")
