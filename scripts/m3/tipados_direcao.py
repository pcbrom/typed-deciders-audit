#!/usr/bin/env python3
"""Typed deciders on the direction of the next value of M3 monthly (implementation_notes.md, gate on time series).

Sample: 150 series, 25 per category, drawn with SeedSequence([20261023, 40]), at the last 12 origins. The state holds
the last 24 values before the origin, the question asks whether the next value is higher than the last one.

  jev    Jev 1.13, appended to data/derived/m3/jev.jsonl with model reported and UTC time, stops before US$0.05
  laya   Laya multilingual as released and fine-tuned (seeds 0 to 2) in sliding blocks, on 8,000 decisions of the other
         series at the 24 origins before each block of 6 test origins: data/derived/m3/laya.npz (run with the interpreter of arquivos/modelos_decisao/.venv)

Usage: tipados_direcao.py jev|laya
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "drift"))
sys.path.insert(0, str(HERE.parents[1] / "explicacao"))
from common import DATA  # noqa: E402
from ets_direcao import ORIGINS, load  # noqa: E402

OUT = DATA / "derived" / "m3"
OPTIONS = ["higher", "not higher"]
INSTRUCTION = "Will the next value of the monthly series in `history` be higher than its last value?"
LAST, HIST, CAP, PRICE, VERSION = 12, 24, 0.05, 0.042e-6, "jev-1.13.0"
BLOCK, TRAIN_ORIGINS = 6, 24                   # sliding blocks: train on the 24 origins before each block of 6


def fmt(v):
    return f"{v:.6g}"


def item(sn, y, t, tag):
    """Decision at time t (index of y_t), state from y_{t-24}..y_{t-1}."""
    return {"id": f"{sn}:{t}:{tag}", "text": "", "options": OPTIONS, "gold": 0 if y[t] > y[t - 1] else 1,
            "instruction": INSTRUCTION, "state": {"history": ", ".join(fmt(v) for v in y[t - HIST:t])}}


def sample():
    series = load()
    types = sorted({tp for _, tp, _ in series})
    rng = np.random.default_rng(np.random.SeedSequence([20261023, 40]))
    pick = []
    for tp in types:
        idx = [i for i, (_, t, _) in enumerate(series) if t == tp]
        pick += sorted(rng.choice(idx, 25, replace=False).tolist())
    return series, pick


def eval_items(series, pick):
    out = []
    for i in pick:
        sn, _, y = series[i]
        L = len(y)
        for k, t in enumerate(range(L - ORIGINS, L)):
            if k >= ORIGINS - LAST:
                out.append((i, k, item(sn, y, t, "eval")))
    return out


def run_jev():
    sys.path.insert(0, str(HERE.parents[1] / "explicacao"))
    from jev_packing_test import post
    key = os.environ["TYPESAFE_API_KEY"]
    series, pick = sample()
    jobs = eval_items(series, pick)
    path = OUT / "jev.jsonl"
    done = {(json.loads(l)["series"], json.loads(l)["origin"]) for l in open(path)} if path.exists() else set()
    todo = [j for j in jobs if (j[0], j[1]) not in done]
    q = {"type": "choice", "instructions": INSTRUCTION, "criteria": {o: o for o in OPTIONS}}
    tokens, per_call = 0, None
    with open(path, "a") as f, ThreadPoolExecutor(max_workers=6) as ex:
        for s in range(0, len(todo), 120):
            batch = todo[s:s + 120]
            if per_call and (tokens + per_call * len(batch)) * PRICE > CAP:
                print("stopping before the cap | US$", round(tokens * PRICE, 4), flush=True)
                return
            for (i, k, it), (r, err) in zip(batch, ex.map(lambda j: post(key, {"model": VERSION, "state": j[2]["state"], "questions": {"label": q}}), batch)):
                if err:
                    print("stopping on error", err, flush=True)
                    return
                row = {"series": i, "origin": k, "model_reported": r.get("model"), "answered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "input_tokens": r["usage"]["input_tokens"], "probs": [r["answers"]["label"]["probabilities"][o] for o in OPTIONS], "gold": it["gold"]}
                f.write(json.dumps(row) + "\n")
                tokens += row["input_tokens"]
            f.flush()
            per_call = tokens / (s + len(batch))
            print(s + len(batch), "of", len(todo), "| US$", round(tokens * PRICE, 4), flush=True)


def run_laya():
    import torch
    from model import Learner
    series, pick = sample()
    jobs = eval_items(series, pick)
    ev = [it for _, _, it in jobs]
    chosen = set(pick)
    origin = np.array([k for _, k, _ in jobs])
    arrays = {"series": np.array([i for i, _, _ in jobs]), "origin": origin, "gold": np.array([it["gold"] for it in ev])}
    L = Learner("multilingual")
    arrays["released"] = L.predict(ev)
    del L
    for seed in (0, 1, 2):
        P = np.zeros((len(ev), len(OPTIONS)), dtype=np.float32)
        for b0 in range(ORIGINS - LAST, ORIGINS, BLOCK):       # sliding blocks of test origins
            pool = []
            for i, (sn, _, y) in enumerate(series):
                if i in chosen:
                    continue
                L_ = len(y)
                for k in range(b0 - TRAIN_ORIGINS, b0):
                    pool.append(item(sn, y, L_ - ORIGINS + k, f"train{b0}"))
            rng = np.random.default_rng(np.random.SeedSequence([20261023, 41, seed, b0]))
            sub = [pool[j] for j in rng.choice(len(pool), 8000, replace=False)]
            torch.manual_seed(seed)
            L = Learner("multilingual")
            secs = L.finetune(sub, epochs=3, batch=32, seed=seed)
            mask = (origin >= b0) & (origin < b0 + BLOCK)
            P[mask] = L.predict([ev[j] for j in np.nonzero(mask)[0]])
            print("seed", seed, "block", b0, round(secs, 1), "s acc", round(float((P[mask].argmax(1) == arrays["gold"][mask]).mean()), 3), flush=True)
            del L
            torch.cuda.empty_cache()
        arrays[f"ft{seed}"] = P
    print("released acc", round(float((arrays["released"].argmax(1) == arrays["gold"]).mean()), 3))
    np.savez_compressed(OUT / "laya.npz", **arrays)


if __name__ == "__main__":
    {"jev": run_jev, "laya": run_laya}[sys.argv[1]]()
