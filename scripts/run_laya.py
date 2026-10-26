#!/usr/bin/env python3
"""Run Laya checkpoints on the MASSIVE items and store the probabilities per item.

Checkpoints of the pinned weights revision: root (English), multilingual, typed-decisions. The
fine-tuned checkpoints are produced with the recipe of github.com/pcbrom/laya-jev
(scripts/laya_ajuste.py, MASSIVE pt-PT training split with the 240 test duplicates removed) and
passed here by local path. Inference runs on CPU with bf16 autocast, in batches of 8 sorted by
length, the configuration under which data/raw/laya_<locale>.json was produced.

Usage: run_laya.py --locale pt --checkpoints multilingual,root,typed-decisions \
                   [--finetuned ft-multilingual=PATH,ft-typed-decisions=PATH] [--weights DIR]
Env:   LAYA_THREADS (default 6). Weights are downloaded from the Hugging Face Hub at the pinned
       revision unless --weights points to a local copy of convaiinnovations/laya.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import laya
from laya.common import collate_items

from common import DATA, load_items, question

REPO_ID = "convaiinnovations/laya"
REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
LAYA_COMMIT = "4066d5d5fbf08b66c6757ddeedbd797bd7655bc0"
SUBFOLDER = {"root": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}


def load_agent(name, weights, finetuned):
    if name in finetuned:
        agent = laya.load(finetuned[name], device="cpu")
    elif weights:
        sub = SUBFOLDER[name]
        agent = laya.load(os.path.join(weights, sub) if sub else weights, device="cpu")
    else:
        kw = {"subfolder": SUBFOLDER[name]} if SUBFOLDER[name] else {}
        agent = laya.load(REPO_ID, revision=REVISION, device="cpu", **kw)
    agent.amp_enabled, agent.dtype = True, torch.bfloat16
    agent.model.eval()
    return agent


@torch.no_grad()
def probabilities(agent, items, batch=8):
    enc, internals = [], []
    for it in items:
        internal = {"intent": agent._to_internal(question(it["options"]))}
        enc.append(agent._encode_state({"utterance": it["utterance"]}, ["intent"], internal))
        internals.append(internal)
    order = sorted(range(len(items)), key=lambda i: len(enc[i][0]["ids"]))
    out = [None] * len(items)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        b = collate_items([enc[i] for i in idx], agent.tok.pad_token_id)
        logits, act = agent._forward(b)
        for row, i in enumerate(idx):
            a = agent._decode_answers(logits, act, enc[i], ["intent"], internals[i], row)["intent"]
            out[i] = [round(float(a["probabilities"][o]), 5) for o in items[i]["options"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", required=True, choices=("pt", "en"))
    ap.add_argument("--checkpoints", default="multilingual,root,typed-decisions")
    ap.add_argument("--finetuned", default="", help="name=path pairs, comma separated")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("LAYA_THREADS", 6)))
    items, gold = load_items(args.locale)
    finetuned = dict(kv.split("=", 1) for kv in args.finetuned.split(",") if kv)
    names = [n for n in args.checkpoints.split(",") if n] + list(finetuned)
    out = {"note": "Laya probabilities per item and checkpoint, CPU, bf16 autocast, batches of 8 sorted by length; "
                   "same order as data/items", "laya_code_commit": LAYA_COMMIT, "weights_revision": REVISION,
           "checkpoints": {}, "probs": {}}
    for name in names:
        agent = load_agent(name, args.weights, finetuned)
        probabilities(agent, items[:16])    # warm-up
        t0 = time.perf_counter()
        p = probabilities(agent, items)
        dt = time.perf_counter() - t0
        acc = float(np.mean(np.argmax(p, axis=1) == gold))
        out["checkpoints"][name] = {"ms_por_item": round(1000 * dt / len(items), 1), "accuracy": round(acc, 4)}
        out["probs"][name] = p
        print(f"{name}: accuracy {acc:.4f}, {1000 * dt / len(items):.1f} ms per item")
    with open(DATA / "raw" / f"laya_{args.locale}.json", "w") as f:
        json.dump(out, f, indent=0)


if __name__ == "__main__":
    main()
