#!/usr/bin/env python3
"""Cost table of the article: accuracy, labels, training, inference and price per 1,000 decisions, from the
data files only (Jev tokens per call from the 1,200 G0 calls, Laya CPU time per item, label curve and the
G4 original-head training times). Bold marks the best value of each column where an order exists.

Usage: tabela_custo.py
"""

import json

import numpy as np

from common import DATA

PRICE = 0.042e-6          # US$ per input token, stated by the vendor in October 2026; output tokens free
OUT = DATA.parent / ".." / "fonte" / "tables" / "cost.tex"


def main():
    g0 = json.loads((DATA / "derived" / "explicacao" / "jev_g0.json").read_text())["rows"]
    tokens = float(np.mean([r["input_tokens"] for r in g0 if r.get("input_tokens")]))
    laya = json.loads((DATA / "raw" / "laya_pt.json").read_text())["checkpoints"]
    jev_acc = json.loads((DATA / "raw" / "label_curve_pt.json").read_text())["referencia_jev"]
    curve = json.loads((DATA / "raw" / "label_curve_pt.json").read_text())["pontos"]
    t4000 = np.mean([p["train_only_s"] for p in curve if p["n"] == 4000])
    # accuracy at 4,000 labels from the paired measurement used for non-inferiority (three seeds, drift recipe)
    acc4000 = float(np.mean(json.loads((DATA / "derived" / "rotulos_nao_inferioridade.json").read_text())["sizes"]["4000"]["acc_por_semente"]))
    g4 = [json.loads((DATA / "derived" / "explicacao" / f"g4_original_s{s}.json").read_text())["train_s"] for s in (0, 1, 2)]
    rows = [("Jev 1.13", jev_acc, "0", "none", "API, not measured", f"{1000 * tokens * PRICE:.3f}", "yes"),
            ("Laya multilingual", laya["multilingual"]["accuracy"], "0", "none", f"{laya['multilingual']['ms_por_item']:.1f} ms, CPU", "local", "no"),
            ("Laya multilingual, 4,000 labels", acc4000, "4,000", f"{t4000:.0f} s, one GPU", "as above", "local", "no"),
            ("Laya multilingual, fine-tuned", laya["ft-multilingual"]["accuracy"], "11,274",
             f"{min(g4):.0f} to {max(g4):.0f} s, one GPU", f"{laya['ft-multilingual']['ms_por_item']:.1f} ms, CPU", "local", "no")]
    best_acc = max(r[1] for r in rows)
    lines = []
    for name, acc, lab, train, inf, price, leaves in rows:
        a = f"{acc:.3f}"
        cells = [name, f"\\textbf{{{a}}}" if acc == best_acc else a, f"\\textbf{{{lab}}}" if lab == "0" else lab,
                 train, inf, price, f"\\textbf{{{leaves}}}" if leaves == "no" else leaves]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    OUT.resolve().write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"tokens per call {tokens:.1f}, price per decision {tokens * PRICE:.3e}")


if __name__ == "__main__":
    main()
