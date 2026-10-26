#!/usr/bin/env python3
"""Structural share of a decision (implementation_notes.md): the decision on the empty state against
the decision on the full utterance and against gold, with the chance level from the marginal option
positions, on the exact games of Laya and Jev.

Usage: parcela_estrutural.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_items  # noqa: E402

DER = Path(__file__).resolve().parents[2] / "data/derived/explicacao"


def summary(e_arg, f_arg, gold, pj0):
    e_arg, f_arg, gold = map(np.asarray, (e_arg, f_arg, gold))
    n = len(e_arg)
    fe, ff = np.bincount(e_arg, minlength=5) / n, np.bincount(f_arg, minlength=5) / n

    def prop(x):
        p = float(np.mean(x))
        h = 1.96 * np.sqrt(p * (1 - p) / n)
        return {"p": round(p, 4), "IC95": [round(p - h, 4), round(p + h, 4)]}
    return {"n": n, "igual_decisao": prop(e_arg == f_arg), "acaso": round(float(fe @ ff), 4),
            "acerto_vazio": prop(e_arg == gold), "acerto_cheio": prop(f_arg == gold), "p_decisao_no_vazio_mediana": round(float(np.median(pj0)), 4)}


def main():
    out = {}
    for case in ("pt_multilingual", "pt_multilingual_ft", "pt_typed", "en_multilingual", "en_typed"):
        z = np.load(DER / f"laya_games_{case}.npz")
        _, gold = load_items(str(z["locale"]) if "locale" in z else case[:2])
        e, f, g, p = [], [], [], []
        for k in range(len(z["d"])):
            P = z["probs"][z["offsets"][k]:z["offsets"][k + 1]]
            j = int(np.argmax(P[-1]))
            e.append(int(np.argmax(P[0]))); f.append(j); g.append(int(gold[int(z["item"][k])])); p.append(float(P[0][j]))
        out[case] = summary(e, f, g, p)
    rows = [json.loads(line) for line in open(DER / "jev_games.jsonl")]
    for loc in ("pt", "en"):
        _, gold = load_items(loc)
        for v in ("jev-1.13.0", "jev-1.14.0"):
            e, f, g, p = [], [], [], []
            for r in rows:
                if r["loc"] != loc:
                    continue
                pr = r["versions"][v]["probs"]
                E, F = np.array(pr["c0"]), np.array(pr[f"c{(1 << r["d"]) - 1}"])
                j = int(np.argmax(F))
                e.append(int(np.argmax(E))); f.append(j); g.append(int(gold[r["item"]])); p.append(float(E[j]))
            out[f"jev_{loc}_{v[4:8]}"] = summary(e, f, g, p)
    (DER / "parcela_estrutural.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v["igual_decisao"], "acaso", v["acaso"], "acerto_vazio", v["acerto_vazio"]["p"], "acerto_cheio", v["acerto_cheio"]["p"], "p_vazio", v["p_decisao_no_vazio_mediana"])


if __name__ == "__main__":
    main()
