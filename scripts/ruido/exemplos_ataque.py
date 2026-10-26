#!/usr/bin/env python3
"""Adversarial examples for the article (implementation_notes.md, adversarial examples): the median-
confidence item among those flipped with one typo (pt and en) and with two typos (pt), rerun with the
same seeds to record the edited text. Output: data/derived/ruido/exemplos_ataque.json and
fonte/tables/adversarial.tex.

Run with the interpreter of arquivos/modelos_decisao/.venv.
Usage: exemplos_ataque.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "drift"))
import ruido as R  # noqa: E402
from common import DATA, load_items  # noqa: E402
from laya_ruido import MODELS  # noqa: E402
from model import Learner  # noqa: E402

D = DATA / "derived" / "ruido"
OUT = HERE.parents[3] / "fonte" / "tables" / "adversarial.tex"


def rerun(L, it, item, loc, budget):
    """The greedy attack of ataque.py, returning the edited utterance and the probabilities."""
    words = it["utterance"].split()
    p0 = L.predict([{"id": f"x:{loc}:{item}:base", "text": it["utterance"], "options": it["options"]}])[0]
    j = int(np.argmax(p0))
    for step in range(1, budget + 1):
        cands = []
        for w, word in enumerate(words):
            if sum(ch.isalpha() for ch in word) < 2:
                continue
            for e, kind in enumerate(R.ERRORS):
                rng = np.random.default_rng(np.random.SeedSequence([20261023, 13, item, step, w, e]))
                new = R.keyboard_error(word, kind, rng)
                if new != word:
                    cands.append(words[:w] + [new] + words[w + 1:])
        P = L.predict([{"id": f"x:{loc}:{item}:{step}:{k}", "text": " ".join(c), "options": it["options"]} for k, c in enumerate(cands)])
        L._enc.clear()
        best = int(np.argmin(np.log(np.clip(P[:, j], 1e-12, 1))))
        words = cands[best]
        if int(np.argmax(P[best])) != j:
            k = int(np.argmax(P[best]))
            return {"clean": it["utterance"], "edited": " ".join(words), "steps": step, "before": it["options"][j], "p_before": float(p0[j]),
                    "after": it["options"][k], "p_after": float(P[best][k]), "gold": it["intent"]}
    return None


def median_item(rows, steps):
    sel = sorted([r for r in rows if r["steps"] == steps], key=lambda r: r["p0"])
    return sel[(len(sel) - 1) // 2] if sel else None


def tex(s):
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def main():
    att = json.loads((D / "ataque.json").read_text())
    L = Learner(MODELS["ft"])
    out = []
    for loc, steps in (("pt", 1), ("en", 1), ("pt", 2)):
        items, _ = load_items(loc)
        r = median_item(att[f"ft_{loc}"]["itens"], steps)
        ex = rerun(L, items[r["item"]], r["item"], loc, steps)
        out.append({"loc": loc, "item": r["item"], **(ex or {"rerun_differs": True})})
        print(loc, steps, out[-1], flush=True)
    (D / "exemplos_ataque.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    lines = []
    for e in out:
        if "edited" not in e:
            continue
        lines.append(f"{e['loc']} & {tex(e['clean'])} & {tex(e['edited'])} & \\texttt{{{tex(e['before'])}}} ({e['p_before']:.{5 if e['p_before'] > 0.9999 else 4}f}) & "
                     f"\\texttt{{{tex(e['after'])}}} ({e['p_after']:.2f}) \\\\")
    lines.append("\\bottomrule")
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
