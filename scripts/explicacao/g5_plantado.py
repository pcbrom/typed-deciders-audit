#!/usr/bin/env python3
"""Seventh G5 fit: the planted model of seed 0 on the pt report items plus its planted test items. The
criterion holds if the planted intents X, X2 and X3 are among the top 10% of the intent intercepts
alpha_c (preregistration/explicacao_portoes.md, G5, and implementation_notes.md).

Run with the interpreter of arquivos/modelos_decisao/.venv (enumeration on the GPU), then the fit runs
with the system interpreter: g5_plantado.py --enumerate, then g5_plantado.py --fit.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from common import EXTERNAL  # noqa: E402
DER = HERE.parents[2] / "data/derived/explicacao"
WEIGHTS = (EXTERNAL / "laya-ajustado/plantado/plantado_s0.pt")
TARGETS = ("qa_definition", "play_game", "qa_maths")


def enumerate_report():
    import torch
    sys.path.insert(0, str(HERE.parents[1] / "drift"))
    from common import load_items
    from jev_enumerate import coalition_text, units
    from model import Learner
    items, gold = load_items("pt")
    L = Learner("multilingual")
    L.load_state(torch.load(WEIGHTS, map_location="cpu"))
    rows, offsets, ds, ids = [], [0], [], []
    for i in range(1000, 2000):
        us = units(items[i]["utterance"])
        batch = [{"id": f"pp:{i}:{m}", "text": coalition_text(us, m), "options": items[i]["options"], "gold": int(gold[i])}
                 for m in range(2 ** len(us))]
        rows.append(L.predict(batch, bs=128)); L._enc.clear()
        offsets.append(offsets[-1] + len(batch)); ds.append(len(us)); ids.append(i)
    np.savez_compressed(DER / "laya_games_plantado_s0_pt.npz", probs=np.concatenate(rows).astype(np.float32), offsets=np.array(offsets),
                        d=np.array(ds), item=np.array(ids), checkpoint=str(WEIGHTS), locale="pt", kind="main")


def fit_planted():
    import games as G
    from common import load_items
    from g5 import deletion_percentile, fit
    items, _ = load_items("pt")
    rows = []
    z = np.load(DER / "laya_games_plantado_s0_pt.npz")
    for k in range(len(z["d"])):
        d, item = int(z["d"][k]), int(z["item"][k])
        if d < 2:
            continue
        P = z["probs"][z["offsets"][k]:z["offsets"][k + 1]]
        j = int(np.argmax(P[(1 << d) - 1]))
        v = G.value(P, j)
        rows.append({"u": deletion_percentile(v, d, np.argsort(-G.shapley(G.mobius(v), d), kind="stable")), "d": d,
                     "logit_p0": float(v[(1 << d) - 1]), "intent": items[item]["options"][j], "src": "relato"})
    pz = np.load(DER / "plantio_s0.npz")
    meta = json.loads(str(pz["meta"]))
    for t in ("unico", "redundante", "sinergico"):
        for n, x in enumerate(meta[t]):
            d = int(pz[f"{t}_d"][n])
            P = pz[f"{t}_probs"][pz[f"{t}_offsets"][n]:pz[f"{t}_offsets"][n + 1]]
            j = int(np.argmax(P[(1 << d) - 1]))
            v = G.value(P, j)
            rows.append({"u": deletion_percentile(v, d, np.argsort(-G.shapley(G.mobius(v), d), kind="stable")), "d": d,
                         "logit_p0": float(v[(1 << d) - 1]), "intent": x["options"][j], "src": t})
    smp, rhat, intents = fit(rows)
    if rhat > 1.01:
        smp, rhat, intents = fit(rows, warmup=4000)
    alpha_c = np.asarray(smp["alpha"])[:, None] + np.asarray(smp["sigma_alpha"])[:, None] * np.asarray(smp["z_alpha"])
    means = alpha_c.mean(0)
    rank = {intents[i]: r + 1 for r, i in enumerate(np.argsort(-means))}
    top = math.ceil(0.1 * len(intents))
    out = {"n_itens": len(rows), "n_intencoes": len(intents), "top": top, "r_hat_max": round(rhat, 4),
           "postos_alvos": {t: rank.get(t) for t in TARGETS}, "criterio": all(rank.get(t, 10 ** 6) <= top for t in TARGETS),
           "P_b_pos": round(float((np.asarray(smp["b"]) > 0).mean()), 4)}
    (DER / "g5_plantado_s0.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--enumerate", action="store_true")
    ap.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    if a.enumerate:
        enumerate_report()
    if a.fit:
        fit_planted()
