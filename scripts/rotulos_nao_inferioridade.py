#!/usr/bin/env python3
"""Labels needed for the fine-tuned Laya to be non-inferior to Jev in accuracy (implementation_notes.md):
fine-tunes on subsets of the pt-PT training partition, paired accuracy difference against the frozen Jev
answers on the report items, item bootstrap, margin of 2 points (the G4 margin).

Run with the interpreter of arquivos/modelos_decisao/.venv.
Usage: rotulos_nao_inferioridade.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "drift"))
from common import DATA, load_items, load_systems  # noqa: E402
from data import make_item, massive  # noqa: E402
from model import Learner  # noqa: E402

SIZES = (1000, 1500, 2000, 2500, 3000, 3500, 4000, 6000, 8000, 11274)
SEEDS = (0, 1, 2)
MARGIN = 0.02


def pool():
    items, _ = load_items("pt")
    test = {it["utterance"] for it in items}
    rows = [r for r in massive("pt-PT") if r["partition"] == "train" and r["text"] not in test]
    intents = sorted({r["intent"] for r in rows})
    return rows, intents


def main():
    rows, intents = pool()
    items, gold = load_items("pt")
    rep = [{"id": f"pt:{i}", "text": it["utterance"], "options": it["options"], "gold": int(gold[i])} for i, it in enumerate(items)][1000:]
    g = gold[1000:]
    jev_ok = load_systems("pt")["jev"][1000:].argmax(1) == g
    out = {"n_pool": len(rows), "jev_acc": round(float(jev_ok.mean()), 4), "sizes": {}}
    for size in SIZES:
        oks, dropped = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(np.random.SeedSequence([20261023, size, seed]))
            sub = [rows[i] for i in rng.choice(len(rows), min(size, len(rows)), replace=False)]
            torch.manual_seed(seed)
            L = Learner("multilingual")
            train = []
            for r in sub:
                it = make_item(r, intents)
                try:
                    L.encode(it)
                    train.append(it)
                except ValueError:
                    pass
            dropped.append(len(sub) - len(train))
            L.finetune(train, epochs=3, batch=32, seed=seed)
            oks.append(L.predict(rep).argmax(1) == g)
            del L
            torch.cuda.empty_cache()
        d = (np.array(oks, float) - jev_ok[None, :]).mean(0)
        rng = np.random.default_rng(np.random.SeedSequence([20261023, 1]))
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out["sizes"][str(size)] = {"acc_por_semente": [round(float(o.mean()), 4) for o in oks], "dif_media": round(float(d.mean()), 4),
                                   "IC95": [round(float(lo), 4), round(float(hi), 4)], "nao_inferior": bool(lo > -MARGIN),
                                   "superior": bool(lo > 0), "descartados": dropped}
        print(size, out["sizes"][str(size)], flush=True)
    ok_sizes = [s for s in SIZES if out["sizes"][str(s)]["nao_inferior"]]
    out["rotulos_nao_inferioridade"] = next((s for s in SIZES if all(out["sizes"][str(t)]["nao_inferior"] for t in SIZES if t >= s)), None)
    out["rotulos_superioridade"] = next((s for s in SIZES if all(out["sizes"][str(t)]["superior"] for t in SIZES if t >= s)), None)
    (DATA / "derived" / "rotulos_nao_inferioridade.json").write_text(json.dumps(out, indent=1))
    print("nao inferior a partir de", out["rotulos_nao_inferioridade"], "| superior a partir de", out["rotulos_superioridade"])


if __name__ == "__main__":
    main()
