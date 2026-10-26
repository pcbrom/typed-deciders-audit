#!/usr/bin/env python3
"""Table of the audit of the Beta rule by kind of decider (fonte/tables/beta_audit.tex), from auditoria_beta_recorte.json.
Per kind: cases, cases kept by each rule, the Beta cases undetermined and broken, mean violation rate and mean decisions
per 1,000 of each rule. Bold: the rule that keeps the promise in most cases, the lowest mean violation and, among the
bounded rules, the most decisions.

Usage: tabela_auditoria_beta.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[1] / "data" / "derived" / "auditoria_beta_recorte.json"
OUT = HERE.parents[2] / "fonte" / "tables" / "beta_audit.tex"
KINDS = (("jev", "Jev 1.13"), ("laya_released", "Laya as released"), ("laya_finetuned", "Laya fine-tuned"))
RULES = ("point", "beta", "ltt")


def bold_if(s, cond):
    return f"\\textbf{{{s}}}" if cond else s


def main():
    d = json.loads(SRC.read_text())
    lines = []
    for k, name in KINDS:
        r = d[k]
        kept = {ru: r[ru]["kept"] for ru in RULES}
        viol = {ru: r[ru]["mean_violation"] for ru in RULES}
        dec = {ru: r[ru]["mean_decisions"] for ru in RULES}
        best_dec = max(dec[ru] for ru in ("beta", "ltt"))
        cells = [name, str(r["cases"])]
        cells += [bold_if(str(kept[ru]), kept[ru] == max(kept.values())) for ru in RULES]
        cells += [str(r["beta"]["undetermined"]), str(r["beta"]["broken"])]
        cells += [bold_if(f"{viol[ru]:.3f}", f"{viol[ru]:.3f}" == f"{min(viol.values()):.3f}") for ru in RULES]
        cells += [f"{dec['point']:.0f}"] + [bold_if(f"{dec[ru]:.0f}", dec[ru] == best_dec) for ru in ("beta", "ltt")]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
