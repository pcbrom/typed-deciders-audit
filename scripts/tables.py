#!/usr/bin/env python3
"""LaTeX bodies of every table of the article, generated from data/derived so that no number is
typed by hand. Each file in tables/ holds the rows after the header \\midrule up to and including
\\bottomrule (an \\input followed by \\bottomrule inside a tabular breaks the alignment), and the
article includes it with \\input. Every number is rounded half up on its stored decimal value.

Bold marks, per language or case, the best value under the criterion stated in each caption.

Usage: tables.py   (writes tables/*.tex)
"""

import json
import math
from decimal import ROUND_HALF_UP, Decimal


from common import DATA, ROOT

OUT = ROOT / "tables"
ORDER = {"pt": ["jev", "multilingual", "root", "typed-decisions", "ft-multilingual", "ft-typed-decisions"],
         "en": ["jev", "root", "typed-decisions", "multilingual"]}
NAME = {"jev": "Jev 1.13", "multilingual": "Laya multilingual", "root": "Laya English", "typed-decisions": "Laya typed-decisions",
        "ft-multilingual": "Laya multilingual, FT", "ft-typed-decisions": "Laya typed-decisions, FT"}
GATE_RULES = ["point", "beta_ascending", "beta_fixed_sequence", "ltt", "sgr"]
GATE_CASES = [("pt", "jev"), ("pt", "ft-multilingual"), ("pt", "ft-typed-decisions"), ("en", "jev"), ("en", "typed-decisions"), ("en", "root")]


def dec(x):
    """Decimal of the value as stored in the JSON, so that rounding acts on the stored digits."""
    return x if isinstance(x, Decimal) else Decimal(repr(float(x)))


def rd(x, nd, sign=False):
    """Round half up on the stored decimal value (one rule for every number of the article)."""
    q = dec(x).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return f"{q:+}" if sign and q != 0 else (f"+{q}" if sign else f"{q}")


def bold(s, on):
    return f"\\textbf{{{s}}}" if on else s


def pval(p):
    if p < 1e-30:
        return f"$<10^{{-{int(math.floor(-math.log10(p) / 10) * 10)}}}$"
    if p < 0.01:
        e = int(math.floor(math.log10(p)))
        return f"${rd(p / 10 ** e, 1)}\\times10^{{{e}}}$"
    return rd(p, 2)


def write(name, lines):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(lines + ["\\bottomrule"]) + "\n", encoding="utf-8")


def accuracy(m):
    lines = []
    for loc in ("pt", "en"):
        rs = m["results"][loc]
        best_acc = max(rs[k]["accuracy"] for k in ORDER[loc])
        best_f1 = max(rs[k]["macro_f1"] for k in ORDER[loc])
        for i, k in enumerate(ORDER[loc]):
            r = rs[k]
            lo, hi = r["accuracy_ci95"]
            cells = [f"\\multirow{{{len(ORDER[loc])}}}{{*}}{{{loc}}}" if i == 0 else "", NAME[k],
                     f"{bold(f'{rd(r['accuracy'], 3)}', r['accuracy'] == best_acc)} [{rd(lo, 3)}, {rd(hi, 3)}]",
                     bold(f"{rd(r['macro_f1'], 3)}", r["macro_f1"] == best_f1)]
            v = r["vs_jev"]
            if v:
                a, b = v["ci95_pp"]
                cells += [f"${rd(v['diff_pp'], 1, True)}$ [${rd(a, 1, True)}$, ${rd(b, 1, True)}$]", pval(v["p"])]
            else:
                cells += ["", ""]
            lines.append(" & ".join(cells) + " \\\\")
        if loc == "pt":
            lines.append("\\midrule")
    write("accuracy", lines)


def calibration(m):
    lines = []
    cols = [("ece_as_returned", min), ("ece_after_temperature", min), ("brier", min), ("nll", min), ("auroc", max)]
    for loc in ("pt", "en"):
        rs = m["results"][loc]
        best = {c: f([rs[k][c] for k in ORDER[loc] if rs[k][c] is not None]) for c, f in cols}
        med = min(rs[k]["ece_by_scenario"]["median"] for k in ORDER[loc])
        worst = min(rs[k]["ece_by_scenario"]["worst"]["ece"] for k in ORDER[loc])
        for i, k in enumerate(ORDER[loc]):
            r = rs[k]
            num = lambda c: "--" if r[c] is None else bold(f"{rd(r[c], 3)}", r[c] == best[c])
            lo, hi = r["auroc_ci95"]
            s = r["ece_by_scenario"]
            cells = [f"\\multirow{{{len(ORDER[loc])}}}{{*}}{{{loc}}}" if i == 0 else "", NAME[k].replace("typed-decisions, FT", "typed-dec., FT"),
                     "--" if r["temperature"] is None else f"{rd(r['temperature'], 2)}",
                     num("ece_as_returned"), num("ece_after_temperature"), num("brier"), num("nll"),
                     f"{num('auroc')} [{rd(lo, 3)}, {rd(hi, 3)}]",
                     f"{bold(f'{rd(s['median'], 3)}', s['median'] == med)} / {bold(f'{rd(s['worst']['ece'], 3)}', s['worst']['ece'] == worst)}"]
            lines.append(" & ".join(cells) + " \\\\")
        if loc == "pt":
            lines.append("\\midrule")
    write("calibration", lines)


def deletion():
    rows = [("pt", "ft-multilingual"), ("pt", "multilingual"), ("en", "multilingual"), ("en", "root")]
    s = {r: json.loads((DATA / "derived" / f"deletion_{r[0]}_{r[1]}.json").read_text())["summary"] for r in rows}
    gap = {k: dec(s[k]["auc_random"]) - dec(s[k]["auc_most_important_first"]) for k in rows}
    lines = []
    for loc in ("pt", "en"):
        best = max(gap[k] for k in rows if k[0] == loc)
        for k in rows:
            if k[0] != loc:
                continue
            x = s[k]
            lines.append(f"{loc} & {NAME[k[1]]} & {rd(x['auc_most_important_first'], 3)} & {rd(x['auc_random'], 3)} & "
                         f"{bold(rd(gap[k], 3), gap[k] == best)} & ${rd(x['spearman_p0_auc'], 2, True)}$ \\\\")
        if loc == "pt":
            lines.append("\\midrule")
    write("deletion", lines)


def gate_population(p):
    """Violation against the plug-in population, with the one-sided 95% upper bound, and decisions.
    Bold: per case, the most decisions among the rules whose upper bound is at most 0.05."""
    lines = []
    for t in ("0.99", "0.95"):
        for loc, key in GATE_CASES:
            row = p["results"][f"{loc}/{key}/ncal1000/{t}"]
            ok = {r: row[r]["plugin"]["violation_upper95"] <= 0.05 for r in GATE_RULES}
            top = max((row[r]["plugin"]["decisions"] for r in GATE_RULES if ok[r]), default=None)
            cells = [f"{rd(float(t) * 100, 0)}\\%", loc, NAME[key].replace("typed-decisions, FT", "typed-dec., FT")]
            for r in GATE_RULES:
                x = row[r]["plugin"]
                txt = f"{rd(x['violation'], 3)} ({rd(x['violation_upper95'], 3)}) / {rd(x['decisions'], 0)}"
                cells.append(bold(txt, ok[r] and top is not None and x["decisions"] == top))
            lines.append(" & ".join(cells) + " \\\\")
        if t == "0.99":
            lines.append("\\midrule")
    write("gate_population", lines)


def gate_versus_ltt(p):
    """Paired difference of the two Beta rules against LTT: decisions and violation, with the item SD
    from the nested bootstrap. Bold: differences whose plug-in value exceeds two item SDs."""
    lines = []
    for t in ("0.99", "0.95"):
        for loc, key in GATE_CASES:
            row = p["results"][f"{loc}/{key}/ncal1000/{t}"]
            cells = [f"{rd(float(t) * 100, 0)}\\%", loc, NAME[key].replace("typed-decisions, FT", "typed-dec., FT")]
            for r in ("beta_ascending", "beta_fixed_sequence"):
                pl, ne = row[r]["plugin"], row[r]["nested"]
                dd, sd = pl["minus_ltt_decisions"], ne["minus_ltt_decisions"]["sd_item"]
                dv, sv = pl["minus_ltt_violation"], ne["minus_ltt_violation"]["sd_item"]
                cells += [bold(f"${rd(dd, 1, True)}$ ({rd(sd, 1)})", abs(dd) > 2 * sd and abs(dd) > 0), f"${rd(dv, 3, True)}$ ({rd(sv, 3)})"]
            lines.append(" & ".join(cells) + " \\\\")
        if t == "0.99":
            lines.append("\\midrule")
    write("gate_versus_ltt", lines)


def gate_reporthalf(a):
    """Appendix: broken promises on the report half and decisions, every system, three targets."""
    labels = {"point": "point", "beta_ascending": "beta q=0.05 uniform min 20 ascending", "ltt": "ltt delta=0.05"}
    lines = []
    for loc in ("pt", "en"):
        for key in ORDER[loc]:
            cells = [loc, NAME[key].replace("typed-decisions, FT", "typed-dec., FT")]
            for t in ("0.95", "0.99", "0.995"):
                calib = "as_returned" if key == "jev" else "temperature"
                rows = {x["label"]: x for x in a[f"{loc}/{key}/{calib}/ncal1000/{t}"]}
                group = [rows[labels[r]] for r in ("point", "beta_ascending", "ltt")]
                # best: fewest broken promises at two decimals, then most decisions; abstaining breaks none
                def rank(x):
                    return (0.0, 0) if x["qualified_splits"] == 0 else (round(x["broken_per_split"], 2), -round(x["decisions_per_1000"]))
                best = min(rank(x) for x in group)
                for x in group:
                    s = "none" if x["qualified_splits"] == 0 else f"{rd(x['broken_per_split'], 2)} / {rd(x['decisions_per_1000'], 0)}"
                    cells.append(f"\\textbf{{{s}}}" if rank(x) == best else s)
            lines.append(" & ".join(cells) + " \\\\")
        if loc == "pt":
            lines.append("\\midrule")
    write("gate_reporthalf", lines)


def main():
    m = json.loads((DATA / "derived" / "metrics.json").read_text())
    accuracy(m)
    calibration(m)
    deletion()
    pf = DATA / "derived" / "population_gate.json"
    if pf.exists():
        p = json.loads(pf.read_text())
        gate_population(p)
        gate_versus_ltt(p)
    gate_reporthalf(json.loads((DATA / "derived" / "ablation_gate.json").read_text())["results"])
    print(sorted(f.name for f in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
