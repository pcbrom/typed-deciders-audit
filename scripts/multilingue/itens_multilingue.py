#!/usr/bin/env python3
"""Parallel items in every MASSIVE 1.1 locale (implementation_notes.md, languages without fine-tuning):
the 2,000 items of the article mapped to MASSIVE identifiers by their pt-PT and en-US utterances and
intents, then the utterance of each locale with the same identifier, keeping options and gold.

Usage: itens_multilingue.py
"""

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from common import DATA, EXTERNAL  # noqa: E402

SRC = (EXTERNAL / "dados/massive_1.1_completo/1.1/data")
OUT = DATA / "items" / "multilingue"


def test_rows(loc):
    return [json.loads(line) for line in open(SRC / f"{loc}.jsonl", encoding="utf-8") if json.loads(line)["partition"] == "test"]


def main():
    by_text = {}
    for loc in ("pt-PT", "en-US"):
        m = collections.defaultdict(set)
        for r in test_rows(loc):
            m[(r["utt"], r["intent"])].add(r["id"])
        by_text[loc] = m
    P = json.loads((DATA / "items" / "massive_pt_k5.json").read_text())["items"]
    E = json.loads((DATA / "items" / "massive_en_k5.json").read_text())["items"]
    ids = []
    for a, b in zip(P, E):
        inter = by_text["pt-PT"][(a["utterance"], a["intent"])] & by_text["en-US"][(b["utterance"], b["intent"])]
        assert len(inter) == 1, a["id"]
        ids.append(inter.pop())
    OUT.mkdir(parents=True, exist_ok=True)
    locales = sorted(p.stem for p in SRC.glob("*.jsonl"))
    for loc in locales:
        rows = {r["id"]: r for r in test_rows(loc)}
        items = []
        for k, (mid, a) in enumerate(zip(ids, P)):
            r = rows[mid]
            assert r["intent"] == a["intent"], (loc, mid)
            items.append({"id": f"{loc}-test-{k}", "massive_id": mid, "utterance": r["utt"], "options": a["options"], "intent": a["intent"]})
        (OUT / f"massive_{loc}_k5.json").write_text(json.dumps({"source": f"MASSIVE 1.1 ({loc}), test split, items parallel to the article", "n": len(items),
                                                                 "items": items}, ensure_ascii=False))
    print(len(locales), "locales")


if __name__ == "__main__":
    main()
