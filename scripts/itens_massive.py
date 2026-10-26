#!/usr/bin/env python3
"""Builds the static MASSIVE items of the article (data/items/massive_<pt|en>_k5.json) from the MASSIVE 1.1 release.

The rule, with Python's `random` module and seed 13: keep the test partition in file order, shuffle it, take the first
2,000 rows, and for each row in that order draw four distractors from the other intents of the test partition (sorted
intent list) and shuffle them with the gold intent, the same generator continuing from the shuffle. MASSIVE lists the
same identifiers in the same order in every locale, so the procedure gives parallel items in pt-PT and en-US. It is the
procedure that produced the items of the companion lesson repository (github.com/pcbrom/laya-jev, data/massive_pt/test.jsonl).

Usage: itens_massive.py --massive DIR [--locale pt-PT] [--check]
       DIR holds pt-PT.jsonl and en-US.jsonl of MASSIVE 1.1 (CC BY 4.0). --check compares with the frozen item files.
"""

import argparse
import json
import random
import sys
from pathlib import Path

SEED, N, K = 13, 2000, 5
FROZEN = Path(__file__).resolve().parents[1] / "data" / "items"
SHORT = {"pt-PT": "pt", "en-US": "en"}


def build(path):
    rows = [r for r in (json.loads(line) for line in open(path, encoding="utf-8")) if r["partition"] == "test"]
    intents = sorted({r["intent"] for r in rows})
    rng = random.Random(SEED)
    rng.shuffle(rows)
    items = []
    for r in rows[:N]:
        options = rng.sample([x for x in intents if x != r["intent"]], K - 1) + [r["intent"]]
        rng.shuffle(options)
        items.append({"massive_id": r["id"], "utterance": r["utt"], "options": options, "intent": r["intent"]})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--massive", required=True)
    ap.add_argument("--locale", default="pt-PT", choices=sorted(SHORT))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    items = build(Path(a.massive) / f"{a.locale}.jsonl")
    if a.check:
        frozen = json.loads((FROZEN / f"massive_{SHORT[a.locale]}_k5.json").read_text(encoding="utf-8"))["items"]
        same = sum((g["utterance"], g["intent"], g["options"]) == (f["utterance"], f["intent"], f["options"]) for g, f in zip(items, frozen))
        print(f"{a.locale}: {same} of {len(frozen)} items identical to the frozen file")
        sys.exit(0 if same == len(frozen) else 1)
    json.dump({"source": f"MASSIVE 1.1 ({a.locale}), test split, {K} options per question, seed {SEED}", "n": len(items), "items": items},
              sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
