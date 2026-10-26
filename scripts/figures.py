#!/usr/bin/env python3
"""Figures of the article (vector PDF), from data/derived and data/raw.

Each plot with a direction of improvement carries an arrow from worse to better.

Usage: figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib import font_manager

from common import DATA, FIGURES

for name in ("lmroman10-regular.otf", "lmroman10-bold.otf"):
    for base in (Path.home() / ".TinyTeX/texmf-dist/fonts/opentype/public/lm", Path("/usr/share/texmf/fonts/opentype/public/lm"),
                 Path("/usr/share/texlive/texmf-dist/fonts/opentype/public/lm")):
        if (base / name).exists():
            font_manager.fontManager.addfont(str(base / name))
            break
plt.rcParams.update({"font.family": ["Latin Modern Roman", "DejaVu Serif"], "mathtext.fontset": "cm", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#DDDDDD",
                     "grid.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42})
INK, MUTED = "#14213D", "#666666"
COLOR = {"jev": "#C2620F", "multilingual": "#8FB3DB", "root": "#5B8C5A", "typed-decisions": "#7A5195",
         "ft-multilingual": "#2F6FB0", "ft-typed-decisions": "#1B3F6B"}
LABEL = {"jev": "Jev 1.13", "multilingual": "Laya multilingual", "root": "Laya English", "typed-decisions": "Laya typed-decisions",
         "ft-multilingual": "Laya multilingual, fine-tuned", "ft-typed-decisions": "Laya typed-decisions, fine-tuned"}


def arrow(ax, start, end, text):
    ax.annotate("", xy=end, xytext=start, xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.7", color=INK, lw=1.4))
    ax.annotate(text, xy=end, xytext=(4, 0), textcoords="offset points", xycoords="axes fraction",
                fontsize=10, color=INK, va="center")


def unit_square(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")


def save(fig, name, rect=None):
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=200)
    plt.close(fig)


def reliability_and_roc(m):
    shown = {"pt": ["jev", "multilingual", "ft-multilingual"], "en": ["jev", "root", "typed-decisions"]}
    for kind in ("reliability", "roc"):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
        for ax, loc in zip(axes, ("pt", "en")):
            res = m["results"][loc]
            ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (3, 3)))
            for key in shown[loc]:
                r = res[key]
                if kind == "reliability":
                    b = np.array(r["curves"]["reliability"])
                    ece_txt = r["ece_after_temperature"] if r["ece_after_temperature"] is not None else r["ece_as_returned"]
                    ax.plot(b[:, 0], b[:, 1], marker="o", ms=3.5, lw=1.4, color=COLOR[key], label=f"{LABEL[key]} (ECE {ece_txt:.3f})")
                else:
                    x, y = r["curves"]["roc"]
                    ax.plot(x, y, lw=1.4, color=COLOR[key], label=f"{LABEL[key]} (AUC {r['auroc']:.3f})")
            unit_square(ax)
            ax.set_title("MASSIVE pt-PT" if loc == "pt" else "MASSIVE en-US", fontsize=11, loc="left")
            if kind == "reliability":
                ax.set_xlabel("stated confidence")
                ax.set_ylabel("observed accuracy")
                arrow(ax, (0.30, 0.62), (0.44, 0.48), "better")
                ax.legend(loc="upper left", fontsize=7.5)
            else:
                ax.set_xlabel("fraction of wrong answers accepted")
                ax.set_ylabel("fraction of correct answers accepted")
                arrow(ax, (0.55, 0.45), (0.30, 0.70), "better")
                ax.legend(loc="lower right", fontsize=7.5)
        save(fig, "fig_reliability" if kind == "reliability" else "fig_roc")


def deletion():
    configs = {"pt": ["ft-multilingual", "multilingual"], "en": ["root", "multilingual"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
    for ax, loc in zip(axes, ("pt", "en")):
        for key in configs[loc]:
            f = DATA / "derived" / f"deletion_{loc}_{key}.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text())["summary"]
            ax.plot(s["grid"], s["curve_most_important_first"], color=COLOR[key], lw=1.6,
                    label=f"{LABEL[key]}, most important first ({s['auc_most_important_first']:.2f})")
            ax.plot(s["grid"], s["curve_random"], color=COLOR[key], lw=1.0, ls=":",
                    label=f"{LABEL[key]}, random order ({s['auc_random']:.2f})")
        unit_square(ax)
        ax.set_title("MASSIVE pt-PT" if loc == "pt" else "MASSIVE en-US", fontsize=11, loc="left")
        ax.set_xlabel("fraction of words removed")
        ax.set_ylabel("probability of the original decision, relative")
        arrow(ax, (0.62, 0.62), (0.42, 0.42), "more faithful")
        ax.legend(loc="lower left", fontsize=6.5)
    save(fig, "fig_deletion")


def gate():
    """Violation against the plug-in population at 99%, with the one-sided 95% upper bound as error bar."""
    pop = json.loads((DATA / "derived" / "population_gate.json").read_text())["results"]
    rows = [("pt", "jev"), ("pt", "ft-multilingual"), ("pt", "ft-typed-decisions"),
            ("en", "jev"), ("en", "typed-decisions"), ("en", "root")]
    rules = (("point", "point rule", "#9A9A9A"), ("beta_ascending", "Beta rule, ascending scan", "#2F6FB0"),
             ("beta_fixed_sequence", "Beta rule, fixed sequence", "#8FB3DB"),
             ("ltt", "Learn then Test", "#5B8C5A"), ("sgr", "SGR", "#7A5195"))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(rows))
    w = 0.16
    for i, (key, name, col) in enumerate(rules):
        off = (i - 2) * w
        res = [pop[f"{loc}/{k}/ncal1000/0.99"][key]["plugin"] for loc, k in rows]
        vals = np.array([r["violation"] for r in res])
        up = np.array([r["violation_upper95"] for r in res]) - vals
        ax.bar(x + off, vals, width=w, color=col, label=name, yerr=[np.zeros_like(up), up],
               error_kw=dict(ecolor=INK, elinewidth=0.6, capsize=1.5))
    ax.axhspan(0, 0.05, color="#5B8C5A", alpha=0.12, lw=0, zorder=0)
    ax.axhline(0.05, color=COLOR["jev"], lw=1.0, ls=(0, (4, 3)))
    ax.text(len(rows) - 0.45, 0.068, "tolerance 0.05, promise kept below", fontsize=7, color=COLOR["jev"], ha="right")
    # coverage gains of the ascending Beta rule over Learn then Test with a 95% interval above zero, item-level uncertainty
    bi = [k for k, _, _ in rules].index("beta_ascending")
    for i, (loc, k) in enumerate(rows):
        r = pop[f"{loc}/{k}/ncal1000/0.99"]["beta_ascending"]
        d, sd = r["plugin"]["minus_ltt_decisions"], r["nested"]["minus_ltt_decisions"]["sd_total"]
        if d - 1.96 * sd > 0:
            ax.annotate(f"+{d:.0f} decisions\nper 1,000 vs LTT", xy=(i + (bi - 2) * w, r["plugin"]["violation_upper95"] + 0.01),
                        xytext=(0, 18), textcoords="offset points", ha="center", fontsize=6.5, color="#2F6FB0",
                        arrowprops=dict(arrowstyle="-", color="#2F6FB0", lw=0.6))
    ax.set_xticks(x)
    short = {"jev": "Jev 1.13", "ft-multilingual": "Laya multilingual\nfine-tuned", "ft-typed-decisions": "Laya typed-dec.\nfine-tuned",
             "typed-decisions": "Laya typed-dec.", "root": "Laya English"}
    ax.set_xticklabels([f"{short[k]}\n({l})" for l, k in rows], fontsize=7.5)
    ax.set_ylabel("violation rate, plug-in population")
    ax.set_xlim(-1.05, len(rows) - 0.5)
    ax.set_ylim(0, 0.8)
    ax.legend(loc="upper center", ncol=3, fontsize=7)
    arrow(ax, (0.03, 0.62), (0.03, 0.30), "better")
    save(fig, "fig_gate")


def accuracy_coverage(m):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True)
    for ax, key in zip(axes, ("jev", "ft-multilingual")):
        r = m["results"]["pt"][key]
        c = np.array(r["curves"]["accuracy_coverage"])
        # coverage bands of the Beta rule: central 90% of its splits that kept (green) or broke (red) the promise
        beta = np.array([p for p in r["gate"]["0.99"]["beta"]["splits_coverage_accuracy"] if p[1] is not None])
        if len(beta):
            for kept, col, lab in ((True, "#5B8C5A", "Beta rule keeps the promise"), (False, "#B5452E", "Beta rule breaks it")):
                sel = beta[(beta[:, 1] >= 0.99) == kept, 0]
                if len(sel) >= 10:
                    lo, hi = np.percentile(sel, [5, 95])
                    ax.axvspan(lo, hi, color=col, alpha=0.13, lw=0, zorder=0, label=f"{lab} (5th to 95th percentile of splits)")
        ax.plot(c[:, 0], c[:, 1], color=INK, lw=1.2, label="all thresholds, reported half")
        for rule, col, mk in (("point", "#9A9A9A", "o"), ("beta", "#2F6FB0", "^")):
            pts = [p for p in r["gate"]["0.99"][rule]["splits_coverage_accuracy"] if p[1] is not None]
            if pts:
                pts = np.array(pts)
                ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.35, color=col, marker=mk, edgecolor="none",
                           label=("point rule" if rule == "point" else "Beta rule") + ", 1,000 splits")
        ax.axhline(0.99, color=COLOR["jev"], lw=1.0, ls=(0, (4, 3)))
        ax.set_xlim(0.6, 1.0)
        ax.set_ylim(0.95, 1.0)
        ax.set_title(LABEL[key] + " (pt)", fontsize=11, loc="left")
        ax.set_xlabel("coverage (decided without a person)")
        arrow(ax, (0.08, 0.20), (0.30, 0.42), "better")
    axes[0].set_ylabel("accuracy among accepted")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, markerscale=4, bbox_to_anchor=(0.5, -0.01))
    save(fig, "fig_accuracy_coverage", rect=(0, 0.12, 1, 1))


def cost(m):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    res = m["results"]["pt"]
    for key, src, ls, lab in (("jev", "cost_10", "-", "Jev 1.13"), ("ft-multilingual", "cost_10", "-", "Laya multilingual, fine-tuned"),
                              ("multilingual", "cost_10", "--", "Laya multilingual, calibrated"),
                              ("multilingual", "cost_10_as_returned", ":", "Laya multilingual, as returned")):
        c = np.array(res[key]["curves"][src])
        ax.plot(c[:, 0], c[:, 1], color=COLOR[key] if src == "cost_10" else "#7F7F7F", lw=1.5, ls=ls, label=lab)
    ax.axvline(0.9, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.text(0.905, 1.75, r"$\tau^*=0.9$", fontsize=9, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)
    ax.set_xlabel(r"gate threshold $\tau$")
    ax.set_ylabel("cost per item (one review = 1)")
    ax.legend(loc="upper left", fontsize=7.5)
    arrow(ax, (0.30, 0.68), (0.30, 0.42), "better")
    save(fig, "fig_cost")


def breakeven():
    """Generic break-even of a local fine-tune against a priced API. The fine-tune pays off when it is
    non-inferior in accuracy (labels at least L_ni, measured) and cheaper, L c <= B with B = N p the API
    spend that N decisions would cost. The cost boundary L = B / c is a hyperbola."""
    g0 = json.loads((DATA / "derived" / "explicacao" / "jev_g0.json").read_text())["rows"]
    p = float(np.mean([r["input_tokens"] for r in g0 if r.get("input_tokens")])) * 0.042e-6
    ni = json.loads((DATA / "derived" / "rotulos_nao_inferioridade.json").read_text())
    l_ni, l_sup = ni["rotulos_nao_inferioridade"], ni["rotulos_superioridade"]
    c = np.linspace(0.003, 1.0, 500)
    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    budgets = (200, 1000, 3000, 10000)
    lines = ("#8CC08A", "#5B8C5A", "#3D6B3C", "#1F3F1E")
    for B in budgets:
        top = np.minimum(B / c, 1e9)
        ax.fill_between(c, l_ni, top, where=top > l_ni, color="#5B8C5A", alpha=0.16, lw=0, zorder=0)
    ax.axhspan(0, l_ni, color="#B5452E", alpha=0.14, lw=0, zorder=0)
    if l_sup:
        for B in budgets:
            top = np.minimum(B / c, 1e9)
            ax.fill_between(c, l_sup, top, where=top > l_sup, color="#E0A100", alpha=0.45, lw=0, zorder=0)
        ax.text(0.30, l_sup + 700, "gold: more accurate and cheaper, Laya beats Jev", fontsize=7.5, color="#7A5600", ha="left")
    ax.text(0.5, l_ni / 2, f"fewer than {l_ni:,} labels: Laya is not non-inferior to Jev,\nso no price of a label makes the fine-tune pay off",
            fontsize=7.5, color="#8A2E1C", ha="center", va="center", bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
    for B, col in zip(budgets, lines):
        y = B / c
        ax.plot(c, np.where(y >= l_ni, y, np.nan), color=col, lw=1.4, label=f"US${B:,} of API spend avoided ({B / p / 1e6:,.0f} million Jev decisions)")
        ax.plot(c, np.where(y <= l_ni, y, np.nan), color=col, lw=0.9, ls=(0, (2, 2)), alpha=0.6)
    ax.axhline(l_ni, color=INK, lw=1.2)
    ax.text(0.99, l_ni + 250, f"{l_ni:,} labels: non-inferior to Jev (margin 2 points)", fontsize=7, color=INK, ha="right")
    if l_sup and l_sup != l_ni:
        ax.axhline(l_sup, color=COLOR["jev"], lw=1.0, ls=(0, (4, 3)))
        ax.text(0.30, l_sup - 650, f"{l_sup:,} labels: more accurate than Jev", fontsize=7, color=COLOR["jev"], ha="left")
    ax.text(0.62, 7000, "green, between the solid line and a curve:\nnon-inferior and cheaper, the fine-tune pays off", fontsize=7.5, color="#2E5A2D", ha="center")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 13000)
    ax.set_xlabel("cost of one label (US$)")
    ax.set_ylabel("labels used in the fine-tune")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    # right axis: measured accuracy difference against Jev at the label counts of the grid, no interpolation
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    shown = [s for s in (1000, 2000, 3000, 4000, 6000, 8000, 11274) if str(s) in ni["sizes"] and s <= ax.get_ylim()[1]]
    ax2.set_yticks(shown)
    ax2.set_yticklabels([f"{100 * ni['sizes'][str(s)]['dif_media']:+.1f}".replace("+0.0", "0.0").replace("-", "−") for s in shown], fontsize=7.5, color=MUTED)
    ax2.tick_params(axis="y", colors=MUTED, length=3)
    ax2.set_ylabel("Laya minus Jev, accuracy (points, measured)", color=MUTED, fontsize=9)
    arrow(ax, (0.80, 0.63), (0.80, 0.76), "more accurate")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    save(fig, "fig_breakeven", rect=(0, 0.2, 1, 1))


def adversarial():
    """Greedy typo attack on Laya: decisions that survive each budget, and clean confidence by outcome."""
    from scipy.stats import beta as B
    d = json.loads((DATA / "derived" / "ruido" / "ataque.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
    styles = {"ft": (COLOR["ft-multilingual"], "Laya multilingual, fine-tuned"), "multilingual": (COLOR["multilingual"], "Laya multilingual")}
    for (m, loc) in (("ft", "pt"), ("ft", "en"), ("multilingual", "pt"), ("multilingual", "en")):
        rows = d[f"{m}_{loc}"]["itens"]
        n = len(rows)
        alive = [n - sum(1 for r in rows if r["steps"] is not None and r["steps"] <= b) for b in range(4)]
        s = np.array(alive) / n
        lo = [B.ppf(0.025, k, n - k + 1) if k else 0.0 for k in alive]
        hi = [B.ppf(0.975, k + 1, n - k) if k < n else 1.0 for k in alive]
        col, lab = styles[m]
        ls = "-" if loc == "pt" else "--"
        a1.fill_between(range(4), lo, hi, color=col, alpha=0.12, lw=0)
        a1.plot(range(4), s, color=col, ls=ls, marker="o", ms=3.5, lw=1.5, label=f"{lab} ({loc})")
    a1.set_xticks(range(4))
    a1.set_xlim(0, 3)
    a1.set_ylim(0, 1.02)
    a1.set_xlabel("typos applied by the attack")
    a1.set_ylabel("decisions not yet flipped")
    a1.legend(loc="lower left", fontsize=6.5)
    arrow(a1, (0.62, 0.38), (0.62, 0.66), "more robust")
    groups = ["1 typo", "2 typos", "3 typos", "resisted 3"]
    rng = np.random.default_rng(0)
    for loc, mk in (("pt", "o"), ("en", "^")):
        for r in d[f"ft_{loc}"]["itens"]:
            gi = 3 if r["steps"] is None else r["steps"] - 1
            x = float(np.log(np.clip(r["p0"], 1e-6, 1 - 1e-6) / np.clip(1 - r["p0"], 1e-6, 1)))
            a2.scatter(x, gi + rng.uniform(-0.18, 0.18) + (0.0 if loc == "pt" else 0.0), s=7, marker=mk, alpha=0.45,
                       color=COLOR["ft-multilingual"] if loc == "pt" else "#7A5195", edgecolor="none",
                       label=None)
    a2.scatter([], [], s=12, marker="o", color=COLOR["ft-multilingual"], label="pt")
    a2.scatter([], [], s=12, marker="^", color="#7A5195", label="en")
    a2.set_yticks(range(4))
    a2.set_yticklabels(groups)
    a2.set_ylim(3.8, -0.45)
    a2.set_xlabel("confidence of the clean decision (log-odds)")
    a2.set_title("fine-tuned Laya: flipped after", fontsize=9, loc="left")
    for xv, lab in ((np.log(99), "0.99"), (np.log(9999), "0.9999")):
        a2.axvline(xv, color=MUTED, lw=0.7, ls=(0, (3, 3)))
        a2.text(xv + 0.15, 3.66, lab, fontsize=7, color=MUTED, ha="left")
    a2.legend(loc="lower left", fontsize=7)
    save(fig, "fig_adversarial")


def languages():
    """The 0.95 promise calibrated in en-US and checked in every MASSIVE locale, and calibrated in each locale."""
    d = json.loads((DATA / "derived" / "multilingue" / "analise_multilingue.json").read_text())
    rows = [r for r in d["locales"] if r["locale"] != "en-US"]
    series = {"jev": (COLOR["jev"], "Jev 1.13", "o"), "laya_multilingual": (COLOR["multilingual"], "Laya multilingual", "s"),
              "laya_ft": (COLOR["ft-multilingual"], "Laya multilingual, fine-tuned (pt)", "^")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
    a1.axhspan(-0.02, 0.05, color="#2E8B57", alpha=0.13, lw=0)
    a1.axhspan(0.05, 1.04, color="#C0392B", alpha=0.06, lw=0)
    a1.text(-45, 0.075, "promise travels (violation at most 0.05)", fontsize=7, color="#1E6B40", va="bottom")
    for m, (col, lab, mk) in series.items():
        x = [100 * r[m]["minus_en"] for r in rows]
        y = [r[m]["promise_en"]["beta"]["violation"] for r in rows]
        a1.scatter(x, y, s=16, marker=mk, color=col, alpha=0.85, edgecolor="none", label=lab)
    a1.set_xscale("symlog", linthresh=2, linscale=1.2)
    a1.set_xlim(-50, 0.5)
    a1.set_xticks([-40, -20, -10, -5, -2, -1, 0])
    a1.set_xticklabels(["$-40$", "$-20$", "$-10$", "$-5$", "$-2$", "$-1$", "0"])
    a1.minorticks_off()
    a1.set_ylim(-0.02, 1.04)
    a1.set_xlabel("accuracy minus en-US (points, linear within 2)")
    a1.set_ylabel("violation rate, Beta rule calibrated in en-US")
    a1.legend(loc="lower left", bbox_to_anchor=(0.0, 0.14), fontsize=6.5)
    rules = ("point", "beta", "ltt")
    bars = (("jev", "promise_en", "Jev, calibrated in en-US"), ("laya_multilingual", "promise_own", "Laya, calibrated in the locale"),
            ("laya_ft", "promise_own", "Laya fine-tuned, calibrated in the locale"))
    w = 0.26
    for k, (m, key, lab) in enumerate(bars):
        held = [sum(r[m][key][ru]["violation_upper95"] <= 0.05 for r in rows) for ru in rules]
        dec = [np.median([r[m][key][ru]["decisions"] for r in rows]) for ru in rules]
        xs = np.arange(3) + (k - 1) * w
        a2.bar(xs, held, w, color=series[m][0], label=lab)
        for x, h, dv in zip(xs, held, dec):
            a2.text(x, h + 0.8, f"{dv:.0f}", fontsize=6.5, ha="center", color=MUTED)
    a2.set_xticks(range(3))
    a2.set_xticklabels(["point", "Beta, ascending", "LTT"])
    a2.set_ylim(0, len(rows))
    a2.set_ylabel(f"locales where the promise holds (of {len(rows)})")
    a2.legend(loc="upper left", fontsize=6.5)
    a2.set_title("numbers over bars: median decisions per 1,000", fontsize=7.5, loc="left", color=MUTED)
    arrow(a1, (0.86, 0.62), (0.86, 0.36), "better")
    arrow(a2, (0.665, 0.56), (0.665, 0.78), "better")
    save(fig, "fig_languages")


def label_curve():
    d = json.loads((DATA / "raw" / "label_curve_pt.json").read_text())
    pts = d["pontos"]
    ns = sorted({p["n"] for p in pts})
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.axhline(d["referencia_jev"], color=COLOR["jev"], lw=1.2, label="Jev 1.13, no labels")
    means = []
    for n in ns:
        acc = [p["acc"] for p in pts if p["n"] == n]
        ax.scatter([n] * len(acc), acc, s=12, color="#8FB3DB", zorder=3)
        means.append(np.mean(acc))
    ax.plot(ns, means, color=COLOR["ft-multilingual"], marker="o", ms=4, lw=1.4, label="Laya multilingual, fine-tuned")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n:,}" for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("labelled utterances (log scale)")
    ax.set_ylabel("accuracy, reported half")
    ax.legend(loc="lower right", fontsize=8)
    arrow(ax, (0.05, 0.45), (0.05, 0.85), "better")
    save(fig, "fig_label_curve")


def main():
    m = json.loads((DATA / "derived" / "metrics.json").read_text())
    reliability_and_roc(m)
    deletion()
    gate()
    accuracy_coverage(m)
    cost(m)
    label_curve()
    breakeven()
    adversarial()
    languages()
    print(sorted(p.name for p in FIGURES.glob("*.pdf")))


if __name__ == "__main__":
    main()
