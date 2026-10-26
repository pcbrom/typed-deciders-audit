#!/usr/bin/env python3
"""Training with noise as regularisation (implementation_notes.md): the G4 original head recipe, pool and
seeds, with half of the training items passed through an operator and level drawn uniformly; and the
G4 original heads of the same seeds as the reference. Both evaluated on every noise condition of the
report items, pt and en. Output: data/derived/ruido/treino_<head>_s<seed>.npz.

Run with the interpreter of arquivos/modelos_decisao/.venv.
Usage: treino_ruido.py --seed 0
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "drift"))
sys.path.insert(0, str(HERE.parents[1] / "explicacao"))
import aditiva as AD  # noqa: E402
import ruido as R  # noqa: E402
from common import DATA, load_items  # noqa: E402
from model import Learner  # noqa: E402

CHOICES = ["typo_0.05", "typo_0.10", "typo_0.20", "ordem_0.10", "caixa", "acento", "misto_0.05", "misto_0.10", "misto_0.20"]


def evaluate(L, tag):
    arrays = {}
    for loc in ("pt", "en"):
        items, gold = load_items(loc)
        for c in R.conditions(loc):
            its = [{"id": f"{tag}:{loc}:{c}:{i}", "text": R.apply(items[i]["utterance"], c, i), "options": items[i]["options"],
                    "gold": int(gold[i])} for i in range(1000, 2000)]
            arrays[f"{loc}|{c}"] = L.predict(its)
            L._enc.clear()
    return arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    out = DATA / "derived" / "ruido"
    ref = Learner("multilingual")
    ref.load_state(torch.load(AD.WEIGHTS / f"original_s{a.seed}.pt", map_location="cpu"))
    np.savez_compressed(out / f"treino_original_s{a.seed}.npz", **evaluate(ref, "orig"))
    del ref
    torch.cuda.empty_cache()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    L = Learner("multilingual")
    items = []
    for k, it in enumerate(AD.train_items()):
        rng = np.random.default_rng(np.random.SeedSequence([20261023, 14, a.seed, k]))
        if rng.random() < 0.5:
            it = {**it, "id": it["id"] + ":ruido", "text": R.apply(it["text"], CHOICES[int(rng.integers(len(CHOICES)))], k)}
        try:
            L.encode(it)
            items.append(it)
        except ValueError:
            pass
    secs = L.finetune(items, epochs=3, batch=32, seed=a.seed)
    print("seed", a.seed, "train_s", round(secs, 1), "items", len(items), flush=True)
    np.savez_compressed(out / f"treino_ruido_s{a.seed}.npz", **evaluate(L, "ruido"))


if __name__ == "__main__":
    main()
