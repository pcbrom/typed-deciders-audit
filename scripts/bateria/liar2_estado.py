#!/usr/bin/env python3
"""LIAR2 with a structured state (implementation_notes.md, when no promise can be made, and the half-split note that
supersedes its fine-tune): statement, speaker and the speaker's record of ratings in a fixed half of the training
split; fine-tunes draw their rows from the other half.

  jev    Jev 1.13 on the 2,000 battery items, appended to data/derived/bateria/jev_liar2_estado_metade.jsonl (resumes)
  laya   Laya released and fine-tuned on 8,000 rows of the other half, seeds 0 to 2:
         data/derived/bateria/laya_liar2_estado_metade.npz (run with the interpreter of arquivos/modelos_decisao/.venv)

Usage: liar2_estado.py jev|laya
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict
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
from dominios import FIXED, RAW, load  # noqa: E402

OUT = DATA / "derived" / "bateria"
OPTIONS = FIXED["liar2"][0]
INSTRUCTION = "How true is the claim in `statement`, as rated by a fact-checker? `speaker_record` counts earlier ratings of statements by `speaker`."
VERSION, PRICE = "jev-1.13.0", 0.042e-6


def record(counts):
    return ", ".join(f"{o}: {counts.get(o, 0)}" for o in OPTIONS)


def build():
    """Battery items and pool rows with the structured state."""
    train = list(csv.DictReader(open(RAW / "liar2" / "train.csv", encoding="utf-8")))
    test = {int(r["id"]): r for r in csv.DictReader(open(RAW / "liar2" / "test.csv", encoding="utf-8"))}
    perm = np.random.default_rng(np.random.SeedSequence([20261023, 37])).permutation(len(train))
    record_half = {int(train[i]["id"]) for i in perm[:len(train) // 2]}
    by_speaker = defaultdict(Counter)
    for r in train:
        if int(r["id"]) in record_half:
            by_speaker[r["speaker"]][OPTIONS[int(r["label"])]] += 1
    items, pool, _ = load("liar2")
    out = []
    for it in items:
        r = test[it["source_id"]]
        out.append(dict(it, id=f"{it['id']}|state", instruction=INSTRUCTION,
                        state={"statement": it["text"], "speaker": r["speaker"], "speaker_record": record(by_speaker[r["speaker"]])}))
    train_by_id = {int(r["id"]): r for r in train}
    rows = []
    for p in pool:
        if p["id"] in record_half:
            continue
        r = train_by_id[p["id"]]
        rows.append({"id": f"{p['id']}:{p['label']}|half", "text": p["text"], "options": list(OPTIONS), "gold": OPTIONS.index(p["label"]),
                     "instruction": INSTRUCTION, "state": {"statement": p["text"], "speaker": r["speaker"], "speaker_record": record(by_speaker[r["speaker"]])}})
    return out, rows


def run_jev():
    from jev_packing_test import post
    key = os.environ["TYPESAFE_API_KEY"]
    items, _ = build()
    path = OUT / "jev_liar2_estado_metade.jsonl"
    done = {json.loads(l)["item"] for l in open(path)} if path.exists() else set()
    todo = [(k, it) for k, it in enumerate(items) if k not in done]
    q = {"type": "choice", "instructions": INSTRUCTION, "criteria": {o: o for o in OPTIONS}}
    tokens = 0
    with open(path, "a") as f, ThreadPoolExecutor(max_workers=6) as ex:
        for s in range(0, len(todo), 300):
            batch = todo[s:s + 300]
            for (k, it), (r, err) in zip(batch, ex.map(lambda x: post(key, {"model": VERSION, "state": x[1]["state"], "questions": {"label": q}}), batch)):
                if err:
                    print("stopping on error", err, "at", k, flush=True)
                    return
                row = {"item": k, "model_reported": r.get("model"), "answered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "input_tokens": r["usage"]["input_tokens"], "probs": [r["answers"]["label"]["probabilities"][o] for o in OPTIONS]}
                f.write(json.dumps(row) + "\n")
                tokens += row["input_tokens"]
            f.flush()
            print(s + len(batch), "of", len(todo), "| US$", round(tokens * PRICE, 4), flush=True)


def run_laya():
    import torch
    from model import Learner
    items, rows = build()
    arrays = {}
    L = Learner("multilingual")
    arrays["released"] = L.predict(items)
    del L
    for seed in (0, 1, 2):
        rng = np.random.default_rng(np.random.SeedSequence([20261023, 34, seed]))
        sub = [rows[i] for i in rng.choice(len(rows), 8000, replace=False)]
        torch.manual_seed(seed)
        L = Learner("multilingual")
        train = []
        for it in sub:
            try:
                L.encode(it)
                train.append(it)
            except ValueError:
                pass
        secs = L.finetune(train, epochs=3, batch=32, seed=seed)
        arrays[f"ft{seed}"] = L.predict(items)
        gold = np.array([it["gold"] for it in items[1000:]])
        print("seed", seed, "trained on", len(train), round(secs, 1), "s, acc", round(float((arrays[f"ft{seed}"][1000:].argmax(1) == gold).mean()), 3), flush=True)
        del L
        torch.cuda.empty_cache()
    np.savez_compressed(OUT / "laya_liar2_estado_metade.npz", **arrays)


if __name__ == "__main__":
    {"jev": run_jev, "laya": run_laya}[sys.argv[1]]()
