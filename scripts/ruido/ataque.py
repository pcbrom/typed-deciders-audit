#!/usr/bin/env python3
"""Greedy adversarial typos against Laya (implementation_notes.md): at each step, four candidate typos
per word (one per keyboard error), the edit that most lowers the log-probability of the clean decision
is applied; stop when the decision changes or after 3 edits. Items: the 200 report items per language of
validations 2 and 4. Output: data/derived/ruido/ataque.json.

Run with the interpreter of arquivos/modelos_decisao/.venv.
Usage: ataque.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as B

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "drift"))
sys.path.insert(0, str(HERE.parents[1] / "explicacao"))
import ruido as R  # noqa: E402
from common import DATA, load_items  # noqa: E402
from jev_validations import selected  # noqa: E402
from laya_ruido import MODELS  # noqa: E402
from model import Learner  # noqa: E402

BUDGET = 3


def ci(k, n):
    return [round(float(B.ppf(0.025, k, n - k + 1)) if k else 0.0, 4), round(float(B.ppf(0.975, k + 1, n - k)) if k < n else 1.0, 4)]


def attack(L, it, item, loc):
    words = it["utterance"].split()
    base = {"id": f"a:{loc}:{item}:base", "text": it["utterance"], "options": it["options"]}
    p0 = L.predict([base])[0]
    j = int(np.argmax(p0))
    for step in range(1, BUDGET + 1):
        cands = []
        for w, word in enumerate(words):
            if sum(ch.isalpha() for ch in word) < 2:
                continue
            for e, kind in enumerate(R.ERRORS):
                rng = np.random.default_rng(np.random.SeedSequence([20261023, 13, item, step, w, e]))
                new = R.keyboard_error(word, kind, rng)
                if new != word:
                    cands.append(words[:w] + [new] + words[w + 1:])
        if not cands:
            return j, float(p0[j]), None
        batch = [{"id": f"a:{loc}:{item}:{step}:{k}", "text": " ".join(c), "options": it["options"]} for k, c in enumerate(cands)]
        P = L.predict(batch)
        L._enc.clear()
        best = int(np.argmin(np.log(np.clip(P[:, j], 1e-12, 1))))
        words = cands[best]
        if int(np.argmax(P[best])) != j:
            return j, float(p0[j]), step
    return j, float(p0[j]), None


def main():
    out = {}
    for name, ckpt in MODELS.items():
        L = Learner(ckpt)
        for loc in ("pt", "en"):
            items, gold = load_items(loc)
            rows = []
            for i in selected(loc, items):
                j, p, step = attack(L, items[i], int(i), loc)
                L._enc.clear()
                rows.append({"item": int(i), "j": j, "correct": bool(j == gold[i]), "p0": p, "steps": step})
            n = len(rows)
            res = {"n": n}
            for b in range(1, BUDGET + 1):
                k = sum(1 for r in rows if r["steps"] is not None and r["steps"] <= b)
                res[f"sucesso_ate_{b}"] = {"taxa": round(k / n, 4), "IC95": ci(k, n)}
            att = [r["p0"] for r in rows if r["steps"] is not None]
            res["p0_mediana_atacados"] = round(float(np.median(att)), 4) if att else None
            res["p0_mediana_resistentes"] = round(float(np.median([r["p0"] for r in rows if r["steps"] is None])), 4)
            res["p0_ge_0.99_atacados"] = round(float(np.mean([x >= 0.99 for x in att])), 4) if att else None
            out[f"{name}_{loc}"] = {**res, "itens": rows}
            print(name, loc, {k: v for k, v in res.items()}, flush=True)
    (DATA / "derived" / "ruido" / "ataque.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
