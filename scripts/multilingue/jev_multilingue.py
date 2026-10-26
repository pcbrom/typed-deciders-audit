#!/usr/bin/env python3
"""Jev 1.13 on the report items of every MASSIVE 1.1 locale other than pt-PT and en-US
(implementation_notes.md, languages without fine-tuning), G0 format, six requests in parallel. Appends to
data/derived/multilingue/jev.jsonl and resumes. Any non-retryable error, a refusal for lack of credit
included, stops the run and is printed with its code. The key is read from TYPESAFE_API_KEY.

Usage: TYPESAFE_API_KEY=... jev_multilingue.py
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "explicacao"))
from common import DATA, question  # noqa: E402
from jev_packing_test import post  # noqa: E402

ITEMS = DATA / "items" / "multilingue"
OUT = DATA / "derived" / "multilingue" / "jev.jsonl"
PRICE, VERSION = 0.042e-6, "jev-1.13.0"


def main():
    key = os.environ["TYPESAFE_API_KEY"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done, tokens = set(), 0
    if OUT.exists():
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["locale"], r["item"]))
            tokens += r["input_tokens"]
    locales = sorted(p.stem.split("_")[1] for p in ITEMS.glob("massive_*_k5.json") if p.stem.split("_")[1] not in ("pt-PT", "en-US"))
    jobs = []
    for loc in locales:
        items = json.loads((ITEMS / f"massive_{loc}_k5.json").read_text())["items"]
        for i in range(1000, 2000):
            if (loc, i) not in done:
                jobs.append({"locale": loc, "item": i, "options": items[i]["options"],
                             "body": {"model": VERSION, "state": {"utterance": items[i]["utterance"]},
                                      "questions": {"intent": question(items[i]["options"])}}})
    print("pending", len(jobs), "spent so far US$", round(tokens * PRICE, 4), flush=True)
    with open(OUT, "a") as f, ThreadPoolExecutor(max_workers=6) as ex:
        for start in range(0, len(jobs), 600):
            batch = jobs[start:start + 600]
            for j, (r, err) in zip(batch, ex.map(lambda j: post(key, j["body"]), batch)):
                if err:
                    print("stopping on error", err, "at", j["locale"], j["item"], "| answers", len(done), "| US$", round(tokens * PRICE, 4), flush=True)
                    return
                row = {"locale": j["locale"], "item": j["item"], "input_tokens": r["usage"]["input_tokens"],
                       "probs": [r["answers"]["intent"]["probabilities"][o] for o in j["options"]]}
                f.write(json.dumps(row) + "\n")
                tokens += row["input_tokens"]
                done.add((j["locale"], j["item"]))
            f.flush()
            print(len(done), "answers, US$", round(tokens * PRICE, 4), flush=True)
    print("done", len(done), "answers, US$", round(tokens * PRICE, 4), flush=True)


if __name__ == "__main__":
    main()
