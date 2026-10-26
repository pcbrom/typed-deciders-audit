#!/usr/bin/env python3
"""Validations 2 and 4 of the explanation front on Jev (preregistration/implementation_notes.md).

For 200 report items per language: the exact game under the replacement operator (removed units
replaced by training-unigram words without high PMI with the options), and the exact game under two
alternative option configurations and a placebo with the original options in permuted order.
One packed request per item, configuration and version, with the stopping rule of jev_enumerate.

The key is read from TYPESAFE_API_KEY and never written.
Usage: TYPESAFE_API_KEY=... jev_validations.py [--workers 3]
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, load_items, EXTERNAL  # noqa: E402
from jev_enumerate import PRICE, VERSIONS, units  # noqa: E402
from jev_packing_test import NEUTRAL, criteria, post  # noqa: E402

MASSIVE = (EXTERNAL / "dados/1.1/data")
LOCALE = {"pt": "pt-PT", "en": "en-US"}
GAMES = DATA / "derived" / "explicacao" / "jev_games.jsonl"
OUT = DATA / "derived" / "explicacao" / "jev_validations.jsonl"
COST = DATA / "derived" / "explicacao" / "jev_validations_cost.json"


def selected(loc, items):
    rng = np.random.default_rng(np.random.SeedSequence([20261021, 52]))
    picks = {}
    for lo in ("pt", "en"):
        its = load_items(lo)[0] if lo != loc else items
        pool = [i for i in range(1000, 2000) if len(units(its[i]["utterance"])) >= 2]
        picks[lo] = sorted(int(i) for i in rng.choice(pool, 200, replace=False))
    return picks[loc]


class Vocabulary:
    """Training unigram of a locale and PMI of each word with each intent (add-one smoothing)."""

    def __init__(self, loc):
        rows = [json.loads(line) for line in open(MASSIVE / f"{LOCALE[loc]}.jsonl", encoding="utf-8")]
        rows = [r for r in rows if r["partition"] == "train"]
        self.word, self.joint, self.intent = Counter(), defaultdict(Counter), Counter()
        for r in rows:
            ws = set(w.lower() for w in r["utt"].split())
            self.intent[r["intent"]] += 1
            for w in ws:
                self.word[w] += 1
                self.joint[r["intent"]][w] += 1
        self.n = len(rows)
        self.words = sorted(self.word)
        self.p = np.array([self.word[w] for w in self.words], float)
        self.p /= self.p.sum()

    def pmi(self, w, intent):
        pw = (self.word[w] + 1) / (self.n + 2)
        pwi = (self.joint[intent][w] + 1) / (self.intent[intent] + 2)
        return math.log(pwi / pw)

    def replacement(self, options, item, unit, pos):
        rng = np.random.default_rng(np.random.SeedSequence([20261021, 53, item, unit, pos]))
        while True:
            w = self.words[int(rng.choice(len(self.words), p=self.p))]
            if all(self.pmi(w, o) <= 1.0 for o in options):
                return w


def packed(texts, options, version):
    return {"model": version, "state": NEUTRAL,
            "questions": {f"c{m}": {"type": "choice", "instructions": f"What is the user asking for in this utterance: \"{t}\"?",
                                    "criteria": criteria(options)} for m, t in enumerate(texts)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    key = os.environ["TYPESAFE_API_KEY"]
    games = {}
    for line in open(GAMES):
        r = json.loads(line)
        games[(r["loc"], r["item"])] = r
    jobs = []
    for loc in ("pt", "en"):
        items, _ = load_items(loc)
        intents = sorted({json.loads(line)["intent"] for line in open(MASSIVE / f"{LOCALE[loc]}.jsonl", encoding="utf-8")})
        vocab = Vocabulary(loc)
        for i in selected(loc, items):
            it, us = items[i], units(items[i]["utterance"])
            d = len(us)
            repl = [[vocab.replacement(it["options"], i, u, k) for k in range(len(us[u]))] for u in range(d)]
            op_texts = [" ".join(w for u in range(d) for w in (us[u] if m >> u & 1 else repl[u])) for m in range(2 ** d)]
            plain = [" ".join(w for u in range(d) if m >> u & 1 for w in us[u]) for m in range(2 ** d)]
            for v in VERSIONS:
                full = np.array(games[(loc, i)]["versions"][v]["probs"][f"c{2 ** d - 1}"])
                j, k = (int(x) for x in np.argsort(-full, kind="stable")[:2])
                keep = {j, k}
                jobs.append((loc, i, v, "operator", it["options"], op_texts, d, j, k))
                for c in (1, 2):
                    rng = np.random.default_rng(np.random.SeedSequence([20261021, 54, i, c]))
                    pool = [x for x in intents if x not in it["options"]]
                    new = list(rng.choice(pool, 3, replace=False))
                    opts = [it["options"][p] if p in keep else str(new.pop()) for p in range(5)]
                    jobs.append((loc, i, v, f"alt{c}", opts, plain, d, j, k))
                perm = np.random.default_rng(np.random.SeedSequence([20261021, 55, i])).permutation(5)
                jobs.append((loc, i, v, "placebo", [it["options"][p] for p in perm], plain, d, j, k))
    done = set()
    if OUT.exists():
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["loc"], r["item"], r["version"], r["kind"]))
    todo = [jb for jb in jobs if jb[:4] not in done]
    stop, lock, stats = threading.Event(), threading.Lock(), {"stopped_by": None}

    def work(jb):
        if stop.is_set():
            return
        loc, i, v, kind, opts, texts, d, j, k = jb
        r, err = post(key, packed(texts, opts, v))
        if err:
            stats["stopped_by"] = f"{err} at {loc}/{i}/{v}/{kind}"
            stop.set()
            return
        rec = {"loc": loc, "item": i, "version": v, "kind": kind, "options": opts, "d": d, "j": j, "k_star": k,
               "model_reported": r.get("model"), "input_tokens": r["usage"]["input_tokens"],
               "probs": {q: [ans["probabilities"][o] for o in opts] for q, ans in r["answers"].items()}}
        if kind == "operator":
            rec["texts"] = texts
        with lock:
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    rows = [json.loads(line) for line in open(OUT)] if OUT.exists() else []
    tok = sum(r["input_tokens"] for r in rows)
    cost = {"requests_done": len(rows), "requests_total": len(jobs), "tokens": tok, "spent_usd": round(tok * PRICE, 4),
            "stopped_by": stats["stopped_by"], "seconds_this_run": round(time.time() - t0, 1)}
    json.dump(cost, open(COST, "w"), indent=1)
    print(json.dumps(cost, indent=1))


if __name__ == "__main__":
    main()
