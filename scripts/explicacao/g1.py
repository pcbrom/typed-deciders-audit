#!/usr/bin/env python3
"""Gate G1 of the explanation front (preregistration/explicacao_portoes.md): does order two suffice?

For each Laya case, the median over items of the least-squares order-two share R2<=2 of the exact
game y(S) = logit p_j(x_S), j decided on the full utterance. Pair classification: report items with
d >= 5 in the four G5 cases (decisive), chunked items only as sensitivity. R2: all items of the two
R2 cases. Thresholds: median >= 0.6 goes on, 0.4 to 0.6 goes on with a caveat, below 0.4 fails.

Usage: g1.py
"""

import json
from pathlib import Path

import numpy as np

import games as G

DER = Path(__file__).resolve().parents[2] / "data/derived/explicacao"
CASES = ["pt_multilingual", "en_multilingual", "pt_typed", "en_typed"]
R2_CASES = ["r2_pt", "r2_en"]


def verdict(m):
    return "segue" if m >= 0.6 else ("segue com ressalva" if m >= 0.4 else "reprova")


def shares(case, dmin):
    z = np.load(DER / f"laya_games_{case}.npz")
    out = []
    for k in range(len(z["d"])):
        d = int(z["d"][k])
        if d < dmin:
            continue
        probs = z["probs"][z["offsets"][k]:z["offsets"][k + 1]]
        j = int(np.argmax(probs[(1 << d) - 1]))
        v = G.value(probs, j)
        out.append((d, G.r2_order2_ls(v, d), G.r2_order2(v, G.mobius(v), d)))
    return out


def main():
    res = {"classificacao": {}, "r2": {}}
    for case in CASES:
        s = shares(case, 5)
        ls = np.array([x[1] for x in s]); tr = np.array([x[2] for x in s])
        ch = np.array([x[1] for x in s if x[0] == 8])
        med = float(np.nanmedian(ls))
        res["classificacao"][case] = {"n_itens": len(s), "mediana_mq": round(med, 4), "veredito": verdict(med),
                                      "mediana_truncada": round(float(np.nanmedian(tr)), 4),
                                      "sensibilidade_d8": {"n": int(len(ch)), "mediana_mq": round(float(np.nanmedian(ch)), 4) if len(ch) else None},
                                      "quartis_mq": [round(float(q), 4) for q in np.nanpercentile(ls, [25, 75])]}
    for case in R2_CASES:
        s = shares(case, 0)
        ls = np.array([x[1] for x in s])
        med = float(np.nanmedian(ls))
        res["r2"][case] = {"n_itens": len(s), "mediana_mq": round(med, 4), "veredito": verdict(med),
                           "mediana_truncada": round(float(np.nanmedian([x[2] for x in s])), 4)}
    (DER / "g1.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
