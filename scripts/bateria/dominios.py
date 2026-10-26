"""Domains of the multi-domain battery (implementation_notes.md, multi-domain battery): 2,000 items per domain
(0 to 999 calibration half, 1000 to 1999 report half) and the training pool, both deterministic.

Item: {"id", "text", "options", "gold", "instruction", "source_id"}; the same layout serves Jev and the Learner.
Usage as a script writes data/items/bateria/<domain>.json and prints counts.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "drift"))
from common import DATA, INSTRUCTION, load_items, EXTERNAL  # noqa: E402
from data import NEWS, NEWS_INSTRUCTION, make_item, massive  # noqa: E402

RAW = (EXTERNAL / "dados/bateria")
DOMAINS = ("news", "finance", "liar2", "fake_pt", "massive_pt")
N_ITEMS, WORDS = 2000, 150
FIXED = {"finance": (["bearish", "bullish", "neutral"], "What is the market sentiment of the financial tweet in `utterance`?"),
         "liar2": (["pants on fire", "false", "barely true", "half true", "mostly true", "true"],
                   "How true is the claim in `utterance`, as rated by a fact-checker?"),
         "fake_pt": (["fake news", "real news"], "Is the news in `utterance` fake or real?")}
LOCALE = {"news": "en", "finance": "en", "liar2": "en", "fake_pt": "pt", "massive_pt": "pt"}


def _rows(domain):
    """(evaluation rows, training rows), each {"id", "text", "label"} with label a name."""
    if domain == "news":
        rows = [json.loads(line) for line in open(NEWS, encoding="utf-8")]
        rows = [{"id": 10_000_000 + k, "text": r["headline"], "label": r["category"]} for k, r in enumerate(rows) if r["headline"].strip()]
        return rows, rows
    opts = FIXED[domain][0]
    if domain == "finance":
        read = lambda f: [{"id": k, "text": r["text"], "label": opts[int(r["label"])]} for k, r in enumerate(csv.DictReader(open(RAW / "tfn" / f, encoding="utf-8")))]
        return read("sent_valid.csv"), read("sent_train.csv")
    if domain == "liar2":
        read = lambda f: [{"id": int(r["id"]), "text": r["statement"], "label": opts[int(r["label"])]} for r in csv.DictReader(open(RAW / "liar2" / f, encoding="utf-8"))]
        return read("test.csv"), read("train.csv")
    base = next((RAW / "fakebr" / "src").glob("*/size_normalized_texts"))
    rows = []
    for cls, label in (("fake", opts[0]), ("true", opts[1])):
        for p in sorted(base.glob(f"{cls}/*.txt"), key=lambda p: int(p.stem)):
            words = p.read_text(encoding="utf-8").split()
            rows.append({"id": f"{cls}/{p.stem}", "pair": int(p.stem), "text": " ".join(words[:WORDS]), "label": label})
    return rows, rows


def _item(domain, row, labels):
    if domain == "massive_pt":
        it = make_item({"id": row["id"], "text": row["text"], "intent": row["label"], "scenario": row["scenario"]}, labels)
        it["instruction"] = INSTRUCTION
    elif domain == "news":
        it = make_item({"id": row["id"], "text": row["text"], "intent": row["label"], "scenario": "news"}, labels)
        it["instruction"] = NEWS_INSTRUCTION
    else:
        opts, instruction = FIXED[domain]
        it = {"id": f"{row['id']}:{row['label']}", "text": row["text"], "options": list(opts), "gold": opts.index(row["label"]), "instruction": instruction}
    it["source_id"] = row["id"]
    return it


def load(domain):
    """(items, pool, labels): the 2,000 evaluation items and the training pool as items."""
    d = DOMAINS.index(domain)
    if domain == "massive_pt":
        raw, gold = load_items("pt")
        items = [{"id": f"pt:{k}", "text": it["utterance"], "options": it["options"], "gold": int(gold[k]), "instruction": INSTRUCTION,
                  "source_id": it["id"]} for k, it in enumerate(raw)]
        test = {it["utterance"] for it in raw}
        pool = [{"id": int(r["id"]), "text": r["text"], "label": r["intent"], "scenario": r["scenario"]}
                for r in massive("pt-PT") if r["partition"] == "train" and r["text"] not in test]
        return items, pool, sorted({r["label"] for r in pool})
    ev, tr = _rows(domain)
    labels = sorted({r["label"] for r in ev + tr})
    rng = np.random.default_rng(np.random.SeedSequence([20261023, 30, d]))
    if domain == "fake_pt":
        pairs = sorted({r["pair"] for r in ev})
        chosen_pairs = set(rng.choice(pairs, N_ITEMS // 2, replace=False).tolist())
        idx = [i for i, r in enumerate(ev) if r["pair"] in chosen_pairs]
        pick = np.array(idx)[rng.permutation(len(idx))]
        items = [_item(domain, ev[i], labels) for i in pick]
        return items, [r for r in tr if r["pair"] not in chosen_pairs], labels
    pick = rng.choice(len(ev), N_ITEMS, replace=False)
    items = [_item(domain, ev[i], labels) for i in pick]
    texts, chosen = {it["text"] for it in items}, {ev[i]["id"] for i in pick}
    pool = [r for r in tr if r["text"] not in texts and not (tr is ev and r["id"] in chosen)]
    return items, pool, labels


def pool_item(domain, row, labels):
    return _item(domain, row, labels)


def main():
    out = DATA / "items" / "bateria"
    out.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        items, pool, labels = load(domain)
        gold = np.array([it["gold"] for it in items])
        (out / f"{domain}.json").write_text(json.dumps({"domain": domain, "locale": LOCALE[domain], "n": len(items), "n_pool": len(pool),
                                                        "labels": labels, "items": items}, ensure_ascii=False))
        print(domain, len(items), "pool", len(pool), "options", len(items[0]["options"]), "gold counts", np.bincount(gold).tolist())


if __name__ == "__main__":
    main()
