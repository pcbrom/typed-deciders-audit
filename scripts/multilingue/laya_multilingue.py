#!/usr/bin/env python3
"""Laya multilingual as released and fine-tuned in Portuguese on the parallel items of every MASSIVE 1.1
locale (implementation_notes.md, languages without fine-tuning). Output:
data/derived/multilingue/laya_<model>.npz with one probability array per locale.

Run with the interpreter of arquivos/modelos_decisao/.venv.
Usage: laya_multilingue.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "drift"))
sys.path.insert(0, str(HERE.parents[1] / "ruido"))
from common import DATA  # noqa: E402
from laya_ruido import MODELS  # noqa: E402
from model import Learner  # noqa: E402

ITEMS = DATA / "items" / "multilingue"


def main():
    out = DATA / "derived" / "multilingue"
    out.mkdir(parents=True, exist_ok=True)
    locales = sorted(p.stem.split("_")[1] for p in ITEMS.glob("massive_*_k5.json"))
    for name, ckpt in MODELS.items():
        L = Learner(ckpt)
        arrays = {}
        for loc in locales:
            items = json.loads((ITEMS / f"massive_{loc}_k5.json").read_text())["items"]
            batch = [{"id": it["id"], "text": it["utterance"], "options": it["options"]} for it in items]
            arrays[loc] = L.predict(batch)
            L._enc.clear()
        np.savez_compressed(out / f"laya_{name}.npz", **arrays)
        print(name, len(arrays), "locales", flush=True)
        del L
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
