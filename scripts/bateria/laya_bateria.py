#!/usr/bin/env python3
"""Laya on the multi-domain battery (implementation_notes.md, multi-domain battery): the released multilingual
checkpoint and fine-tunes on subsets of each domain's pool (sizes x seeds). Every model answers the 2,000 clean
items; the released checkpoint and the largest fine-tunes also answer the noisy report items and the empty state.
Output per domain: data/derived/bateria/laya_<domain>.npz (keys model|size|seed|condition) and
laya_<domain>.json (training seconds and dropped items). Skips domains already written.

Run with the interpreter of arquivos/modelos_decisao/.venv.
Usage: laya_bateria.py [--domains news finance liar2]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "drift"))
sys.path.insert(0, str(HERE.parents[1] / "ruido"))
import ruido as R  # noqa: E402
from common import DATA  # noqa: E402
from dominios import DOMAINS, load, pool_item  # noqa: E402
from model import Learner  # noqa: E402

OUT = DATA / "derived" / "bateria"
SIZES, SEEDS, CONDITIONS = (250, 500, 1000, 2000, 4000, 8000), (0, 1, 2), ("typo_0.10", "misto_0.10")


def variants(items):
    """Clean items, noisy report items and empty-state report items, with distinct ids for the encoder cache."""
    out = {"clean": items}
    for c in CONDITIONS:
        out[c] = [dict(items[k], id=f"{items[k]['id']}|{c}", text=R.apply(items[k]["text"], c, k)) for k in range(1000, 2000)]
    out["empty"] = [dict(items[k], id=f"{items[k]['id']}|empty", text="") for k in range(1000, 2000)]
    return out


def answer(L, sets, full):
    return {c: L.predict(v) for c, v in sets.items() if full or c == "clean"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for domain in args.domains:
        if (OUT / f"laya_{domain}.npz").exists():
            print(domain, "already done", flush=True)
            continue
        d = DOMAINS.index(domain)
        items, pool, labels = load(domain)
        sets = variants(items)
        arrays, meta = {}, {"sizes": {}, "n_pool": len(pool)}
        L = Learner("multilingual")
        for c, P in answer(L, sets, True).items():
            arrays[f"released|0|0|{c}"] = P
        del L
        sizes = [s for s in SIZES if s < len(pool)] + ([len(pool)] if len(pool) < SIZES[-1] else [])
        for size in sizes:
            for seed in SEEDS:
                rng = np.random.default_rng(np.random.SeedSequence([20261023, 31, d, size, seed]))
                sub = [pool[i] for i in rng.choice(len(pool), size, replace=False)]
                torch.manual_seed(seed)
                L = Learner("multilingual")
                train, dropped = [], 0
                for r in sub:
                    it = pool_item(domain, r, labels)
                    try:
                        L.encode(it)
                        train.append(it)
                    except ValueError:
                        dropped += 1
                secs = L.finetune(train, epochs=3, batch=32, seed=seed)
                for c, P in answer(L, sets, size == sizes[-1]).items():
                    arrays[f"ft|{size}|{seed}|{c}"] = P
                freq = {}
                for it in train:
                    freq[it["options"][it["gold"]]] = freq.get(it["options"][it["gold"]], 0) + 1
                meta["sizes"].setdefault(str(size), []).append({"seed": seed, "seconds": round(secs, 1), "dropped": dropped, "label_counts": freq})
                print(domain, size, seed, round(secs, 1), "s", "acc", round(float((arrays[f"ft|{size}|{seed}|clean"][1000:].argmax(1) == np.array([it["gold"] for it in items[1000:]])).mean()), 3), flush=True)
                del L
                torch.cuda.empty_cache()
        np.savez_compressed(OUT / f"laya_{domain}.npz", **arrays)
        (OUT / f"laya_{domain}.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
