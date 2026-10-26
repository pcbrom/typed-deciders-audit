#!/usr/bin/env python3
"""Exact coalition game of Jev on the report items (preregistration/explicacao_portoes.md and the
declared deviation of 2026-10-21 in preregistration/deviations.md).

Units are the words of an utterance with at most 8 words, otherwise 8 contiguous spans of nearly
equal size. For every item, all 2^d coalitions are asked in one request, each coalition as its own
choice question with the kept words inside the instructions and a fixed neutral state.

Order and stopping rule: the report items of each language are shuffled once with
SeedSequence([20261021, 50]); languages are interleaved item by item, and both pinned versions are
asked for an item before the next item starts. 429 and 529 are retried with exponential backoff;
the first other error stops the run and is recorded. Results are appended item by item to
data/derived/explicacao/jev_games.jsonl, so a run resumes where it stopped, and the measured cost
and the projection of the remainder are written to jev_games_cost.json.

The key is read from TYPESAFE_API_KEY and never written.
Usage: TYPESAFE_API_KEY=... jev_enumerate.py [--workers 3] [--max-items N]
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, load_items  # noqa: E402
from jev_packing_test import NEUTRAL, criteria, post  # noqa: E402

VERSIONS = ("jev-1.13.0", "jev-1.14.0")
PRICE = 0.042 / 1e6
OUT = DATA / "derived" / "explicacao" / "jev_games.jsonl"
COST = DATA / "derived" / "explicacao" / "jev_games_cost.json"


def units(text, max_units=8):
    words = text.split()
    if len(words) <= max_units:
        return [[w] for w in words]
    edges = np.linspace(0, len(words), max_units + 1).round().astype(int)
    return [words[a:b] for a, b in zip(edges[:-1], edges[1:])]


def coalition_text(us, mask):
    return " ".join(w for i, u in enumerate(us) if mask >> i & 1 for w in u)


def body(item, version):
    us = units(item["utterance"])
    qs = {f"c{m}": {"type": "choice", "instructions": f"What is the user asking for in this utterance: \"{coalition_text(us, m)}\"?",
                    "criteria": criteria(item["options"])} for m in range(2 ** len(us))}
    return {"model": version, "state": NEUTRAL, "questions": qs}, len(us)


def order():
    rng = np.random.default_rng(np.random.SeedSequence([20261021, 50]))
    per = {loc: list(rng.permutation(np.arange(1000, 2000))) for loc in ("pt", "en")}
    return [(loc, int(per[loc][k])) for k in range(1000) for loc in ("pt", "en")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-items", type=int, default=0)
    a = ap.parse_args()
    key = os.environ["TYPESAFE_API_KEY"]
    items = {loc: load_items(loc)[0] for loc in ("pt", "en")}
    done = set()
    if OUT.exists():
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["loc"], r["item"]))
    todo = [x for x in order() if x not in done]
    if a.max_items:
        todo = todo[:a.max_items]
    stop, lock = threading.Event(), threading.Lock()
    stats = {"stopped_by": None}

    def work(job):
        if stop.is_set():
            return
        loc, i = job
        it = items[loc][i]
        rec = {"loc": loc, "item": i, "options": it["options"], "versions": {}}
        for v in VERSIONS:
            b, d = body(it, v)
            r, err = post(key, b)
            if err:
                stats["stopped_by"] = f"{err} at {loc}/{i}/{v}"
                stop.set()
                return
            rec["d"] = d
            rec["versions"][v] = {"model_reported": r.get("model"), "input_tokens": r["usage"]["input_tokens"],
                                  "probs": {q: [ans["probabilities"][o] for o in it["options"]] for q, ans in r["answers"].items()}}
        with lock:
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    rows = [json.loads(line) for line in open(OUT)] if OUT.exists() else []
    tok = sum(v["input_tokens"] for r in rows for v in r["versions"].values())
    coal = sum(2 ** r["d"] for r in rows)
    remaining = [x for x in order() if x not in {(r["loc"], r["item"]) for r in rows}]
    rem_coal = sum(2 ** len(units(items[loc][i]["utterance"])) for loc, i in remaining)
    per_coal = tok / (coal * len(VERSIONS)) if coal else None
    cost = {"items_complete": len(rows), "items_remaining": len(remaining), "tokens": tok, "spent_usd": round(tok * PRICE, 4),
            "tokens_per_coalition_measured": round(per_coal, 1) if per_coal else None,
            "projected_remaining_usd": round(rem_coal * len(VERSIONS) * per_coal * PRICE, 4) if per_coal else None,
            "stopped_by": stats["stopped_by"], "seconds_this_run": round(time.time() - t0, 1)}
    json.dump(cost, open(COST, "w"), indent=1)
    print(json.dumps(cost, indent=1))


if __name__ == "__main__":
    main()
