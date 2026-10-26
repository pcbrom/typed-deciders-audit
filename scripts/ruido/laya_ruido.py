#!/usr/bin/env python3
"""Laya under natural noise (implementation_notes.md): probabilities of the released multilingual
checkpoint and of the fine-tuned checkpoint of the article, on all items clean and on the report items
under every noise condition. Output: data/derived/ruido/laya_<model>_<loc>.npz.

Run with the interpreter of arquivos/modelos_decisao/.venv.
Usage: laya_ruido.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "drift"))
import ruido as R  # noqa: E402
from common import DATA, load_items, EXTERNAL  # noqa: E402
from model import Learner  # noqa: E402

MODELS = {"multilingual": "multilingual", "ft": str(EXTERNAL / "laya-ajustado/massive-pt-multilingual")}
OUT = DATA / "derived" / "ruido"


def items_for(loc, condition, idx):
    items, gold = load_items(loc)
    return [{"id": f"{loc}:{condition}:{i}", "text": R.apply(items[i]["utterance"], condition, i), "options": items[i]["options"],
             "gold": int(gold[i])} for i in idx]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, ckpt in MODELS.items():
        L = Learner(ckpt)
        for loc in ("pt", "en"):
            arrays = {"limpo_todos": L.predict(items_for(loc, "limpo", range(2000)))}
            for c in R.conditions(loc)[1:]:
                arrays[c] = L.predict(items_for(loc, c, range(1000, 2000)))
                L._enc.clear()
            np.savez_compressed(OUT / f"laya_{name}_{loc}.npz", **arrays)
            print(name, loc, "ok", flush=True)
        del L
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
