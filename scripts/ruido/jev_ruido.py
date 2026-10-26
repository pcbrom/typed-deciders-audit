#!/usr/bin/env python3
"""Jev 1.13 under natural noise (implementation_notes.md): fresh answers on the report items, clean and
under every noise condition, in the preregistered G0 format (utterance in the state). Appends to
data/derived/ruido/jev.jsonl and resumes. Stops when the projected spend would exceed US$0.50.
The key is read from TYPESAFE_API_KEY and never written.

Usage: TYPESAFE_API_KEY=... jev_ruido.py
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "explicacao"))
import ruido as R  # noqa: E402
from common import DATA, load_items, question  # noqa: E402
from jev_packing_test import post  # noqa: E402

OUT = DATA / "derived" / "ruido" / "jev.jsonl"
PRICE, CAP, VERSION = 0.042e-6, 0.50, "jev-1.13.0"


def main():
    key = os.environ["TYPESAFE_API_KEY"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done, tokens = set(), 0
    if OUT.exists():
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["loc"], r["condition"], r["item"]))
            tokens += r.get("input_tokens", 0)
    jobs = []
    for loc in ("pt", "en"):
        items, _ = load_items(loc)
        for c in R.conditions(loc):
            for i in range(1000, 2000):
                if (loc, c, i) not in done:
                    jobs.append({"loc": loc, "condition": c, "item": i, "options": items[i]["options"],
                                 "body": {"model": VERSION, "state": {"utterance": R.apply(items[i]["utterance"], c, i)},
                                          "questions": {"intent": question(items[i]["options"])}}})
    print("pending", len(jobs), "spent so far US$", round(tokens * PRICE, 4), flush=True)
    with open(OUT, "a") as f, ThreadPoolExecutor(max_workers=6) as ex:
        for start in range(0, len(jobs), 300):
            batch = jobs[start:start + 300]
            avg = tokens / max(1, len(done)) if done else 400
            if (tokens + avg * len(batch)) * PRICE > CAP:
                print("stopping: projected spend above the cap", flush=True)
                break
            for j, (r, err) in zip(batch, ex.map(lambda j: post(key, j["body"]), batch)):
                if err:
                    print("error", err, "on", j["loc"], j["condition"], j["item"], flush=True)
                    return
                row = {k: j[k] for k in ("loc", "condition", "item")}
                row["input_tokens"] = r["usage"]["input_tokens"]
                row["probs"] = [r["answers"]["intent"]["probabilities"][o] for o in j["options"]]
                f.write(json.dumps(row) + "\n")
                tokens += row["input_tokens"]
                done.add((j["loc"], j["condition"], j["item"]))
            f.flush()
            print(len(done), "answers, US$", round(tokens * PRICE, 4), flush=True)


if __name__ == "__main__":
    main()
