#!/usr/bin/env python3
"""Deletion curves by occlusion for a Laya checkpoint on the reported half (items 1000 to 1999).

For each utterance the decided option is the one with the highest calibrated probability p0.
Word importance is the drop in that option's probability when the word alone is removed. The
deletion curve removes words cumulatively, most important first, and records
p(decided option) / p0, clipped at 1, against the fraction of words removed (step interpolation
on an 11-point grid). The reference removes words in random order (3 seeds, fixed generator).
The temperature is fitted on the calibration half from data/raw/laya_<locale>.json.

Jev is not measured: the published answers are frozen, and occlusion needs new calls.

Usage: deletion.py --locale en --checkpoint root [--finetuned-path DIR] [--weights DIR]
Env:   LAYA_THREADS (default 12)
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from common import CAL, DATA, load_items, load_systems, log_probs, fit_temperature
from run_laya import load_agent, question
from laya.common import collate_items

GRID = np.linspace(0, 1, 11)


@torch.no_grad()
def probs_batch(agent, pairs, batch=32):
    enc, internals = [], []
    for utt, opts in pairs:
        internal = {"intent": agent._to_internal(question(opts))}
        enc.append(agent._encode_state({"utterance": utt}, ["intent"], internal))
        internals.append(internal)
    order = sorted(range(len(pairs)), key=lambda i: len(enc[i][0]["ids"]))
    out = [None] * len(pairs)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        b = collate_items([enc[i] for i in idx], agent.tok.pad_token_id)
        logits, act = agent._forward(b)
        for row, i in enumerate(idx):
            a = agent._decode_answers(logits, act, enc[i], ["intent"], internals[i], row)["intent"]
            out[i] = np.array([a["probabilities"][o] for o in pairs[i][1]], dtype=float)
    return out


def calibrate(p, t):
    z = np.log(np.clip(p, 1e-6, 1.0)) / t
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def on_grid(values):
    n = len(values) - 1
    return np.array([values[int(np.floor(x * n + 1e-9))] for x in GRID])


def area(y):
    return float(np.sum(np.diff(GRID) * (y[1:] + y[:-1]) / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", required=True, choices=("pt", "en"))
    ap.add_argument("--checkpoint", required=True,
                    choices=("root", "multilingual", "typed-decisions", "ft-multilingual", "ft-typed-decisions"))
    ap.add_argument("--finetuned-path", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("LAYA_THREADS", 12)))

    items, gold = load_items(args.locale)
    p_items = load_systems(args.locale)[args.checkpoint]
    t = fit_temperature(log_probs(p_items[CAL]), gold[CAL])
    finetuned = {args.checkpoint: args.finetuned_path} if args.checkpoint.startswith("ft-") else {}
    agent = load_agent(args.checkpoint, args.weights, finetuned)
    rng = np.random.default_rng(20261020)

    plan, meta = [], []
    for i in range(1000, 1000 + args.n):
        words, opts = items[i]["utterance"].split(), items[i]["options"]
        n = len(words)
        perms = [rng.permutation(n).tolist() for _ in range(3)]
        meta.append({"i": i, "words": words, "opts": opts, "start": len(plan), "n": n})
        plan += [(items[i]["utterance"], opts)] + [(" ".join(words[:k] + words[k + 1:]), opts) for k in range(n)]
        for perm in perms:
            for j in range(1, n + 1):
                drop = set(perm[:j])
                plan.append((" ".join(w for k, w in enumerate(words) if k not in drop), opts))
    t0 = time.perf_counter()
    Q = [calibrate(p, t) for p in probs_batch(agent, plan)]
    second = []
    for m in meta:
        s, n = m["start"], m["n"]
        dec = int(np.argmax(Q[s]))
        imp = np.array([Q[s][dec] - Q[s + 1 + k][dec] for k in range(n)])
        order = np.argsort(-imp, kind="stable").tolist()
        m.update({"dec": dec, "p0": float(Q[s][dec]), "imp": imp, "mstart": len(second)})
        for j in range(1, n + 1):
            drop = set(order[:j])
            second.append((" ".join(w for k, w in enumerate(m["words"]) if k not in drop), m["opts"]))
    R = [calibrate(p, t) for p in probs_batch(agent, second)]
    seconds = time.perf_counter() - t0

    rows, morf_all, rand_all = [], [], []
    for m in meta:
        n, s, dec, p0 = m["n"], m["start"], m["dec"], m["p0"]
        morf = on_grid([1.0] + [min(1.0, R[m["mstart"] + j][dec] / p0) for j in range(n)])
        base = s + 1 + n
        rand = np.mean([on_grid([1.0] + [min(1.0, Q[base + k * n + j][dec] / p0) for j in range(n)])
                        for k in range(3)], axis=0)
        morf_all.append(morf)
        rand_all.append(rand)
        rows.append({"index": m["i"], "words": m["words"], "gold": items[m["i"]]["intent"], "decision": m["opts"][dec],
                     "correct": bool(dec == gold[m["i"]]), "p0": round(p0, 6),
                     "importance": [round(float(x), 5) for x in m["imp"]],
                     "auc_most_important_first": round(area(morf), 4), "auc_random": round(area(rand), 4)})
    morf_mean, rand_mean = np.mean(morf_all, axis=0), np.mean(rand_all, axis=0)
    p0s = np.array([r["p0"] for r in rows])
    aucs = np.array([r["auc_most_important_first"] for r in rows])
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable")
    summary = {"locale": args.locale, "checkpoint": args.checkpoint, "temperature": round(t, 4), "n": len(rows),
               "mean_words": round(float(np.mean([len(r["words"]) for r in rows])), 2),
               "forward_items": len(plan) + len(second), "seconds": round(seconds, 1), "grid": GRID.round(2).tolist(),
               "curve_most_important_first": morf_mean.round(4).tolist(), "curve_random": rand_mean.round(4).tolist(),
               "auc_most_important_first": round(area(morf_mean), 4), "auc_random": round(area(rand_mean), 4),
               "spearman_p0_auc": round(float(np.corrcoef(rank(p0s), rank(aucs))[0, 1]), 4)}
    with open(DATA / "derived" / f"deletion_{args.locale}_{args.checkpoint}.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "items": rows}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
