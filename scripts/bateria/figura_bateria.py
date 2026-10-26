#!/usr/bin/env python3
"""Label curves of the multi-domain battery (figures/fig_battery): per domain, accuracy of the fine-tuned Laya on the
report items against the labels used (mean and seeds), with Jev without labels and the released Laya as lines.
Green where the fine-tune is non-inferior to Jev, gold where it is superior, red where it is neither.

Run with the interpreter of arquivos/modelos_decisao/.venv.
Usage: figura_bateria.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from figures import COLOR, INK, MUTED, arrow, save  # noqa: E402
from common import DATA  # noqa: E402

ARROW = {"news": (0.25, 0.45), "finance": (0.06, 0.26), "liar2": (0.45, 0.70), "fake_pt": (0.08, 0.36)}
NAMES = {"news": "News Category (en, 5 options)", "finance": "Financial tweets (en, 3)", "liar2": "LIAR2 (en, 6)", "fake_pt": "Fake.Br (pt, 2)"}


def main():
    R = json.loads((DATA / "derived" / "bateria" / "analise_bateria.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    for ax, (d, title) in zip(axes.flat, NAMES.items()):
        r = R[d]
        sizes = sorted(int(s) for s in r["labels"]["sizes"])
        lab = r["labels"]["sizes"]
        edges = np.sqrt(np.array(sizes[:-1]) * np.array(sizes[1:]))
        lo_edges = np.r_[sizes[0] / 1.3, edges]
        hi_edges = np.r_[edges, sizes[-1] * 1.3]
        for s, a, b in zip(sizes, lo_edges, hi_edges):
            v = lab[str(s)]
            col = "#E0A100" if v["superior"] else "#5B8C5A" if v["non_inferior"] else "#B5452E"
            ax.axvspan(a, b, color=col, alpha=0.18 if col != "#E0A100" else 0.30, lw=0, zorder=0)
        for k, s in enumerate(sizes):
            ax.scatter([s] * len(lab[str(s)]["acc_per_seed"]), lab[str(s)]["acc_per_seed"], s=9, color="#8FB3DB", zorder=3)
        ax.plot(sizes, [lab[str(s)]["acc_mean"] for s in sizes], color=COLOR["ft-multilingual"], marker="o", ms=3.5, lw=1.4,
                label="Laya multilingual, fine-tuned")
        ax.axhline(r["accuracy"]["jev"]["acc"], color=COLOR["jev"], lw=1.3, label="Jev 1.13, no labels")
        ax.axhline(r["accuracy"]["released"]["acc"], color=MUTED, lw=1.0, ls=(0, (4, 3)), label="Laya multilingual, released")
        ax.axhline(r["chance"], color=MUTED, lw=0.7, ls=":", label="chance")
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s:,}" for s in sizes], fontsize=7)
        ax.minorticks_off()
        ax.set_xlim(lo_edges[0], hi_edges[-1])
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=9.5, loc="left")
        y0, y1 = ARROW[d]
        arrow(ax, (0.80, y0), (0.80, y1), "better")
    for ax in axes[1]:
        ax.set_xlabel("labelled items (log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy, report items")
    h, l = axes[0, 0].get_legend_handles_labels()
    from matplotlib.patches import Patch
    h += [Patch(color="#E0A100", alpha=0.45), Patch(color="#5B8C5A", alpha=0.3), Patch(color="#B5452E", alpha=0.3)]
    l += ["fine-tune superior to Jev", "non-inferior (2 points)", "neither"]
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.005))
    save(fig, "fig_battery", rect=(0, 0.07, 1, 1))


if __name__ == "__main__":
    main()
