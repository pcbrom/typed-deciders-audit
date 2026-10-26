#!/usr/bin/env python3
"""Micro-test before the explanation front: can the coalitions of one item be packed as separate
questions of a single Jev request without changing the answers, and what does each format cost?

Formats, for 20 report items (10 pt, 10 en) and 4 coalitions each (the full utterance, the
utterance without its first word, without its second word, and its first three words):
  A   one request per coalition, text in the state (as in the frozen collection)
  A1  one request per coalition, text inside the question, a fixed neutral state
  B   one request per item, every coalition as its own question with the text inside it
A1 against B isolates packing (the docs give each question the context of the state plus that
question only); A against A1 shows whether moving the text from the state to the question changes
the answer. Both Jev versions are queried by their pinned ids. Billable tokens come from
usage.input_tokens.

The key is read from TYPESAFE_API_KEY and never written.
Usage: TYPESAFE_API_KEY=... jev_packing_test.py [--out ../../data/derived/explicacao/jev_packing_test.json]
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, load_items  # noqa: E402

URL = "https://api.typesafe.ai/v1/systemone"
VERSIONS = ("jev-1.13.0", "jev-1.14.0")
NEUTRAL = {"task": "intent of a short user utterance"}


def post(key, body, retries=6):
    data = json.dumps(body).encode()
    for attempt in range(retries):
        req = urllib.request.Request(URL, data=data, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read()), None
        except urllib.error.HTTPError as e:
            if e.code in (429, 529) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, f"HTTP {e.code}"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, type(e).__name__
    return None, "retries exhausted"


def criteria(options):
    return {o: o.replace("_", " ") for o in options}


def coalitions(words):
    full = list(range(len(words)))
    return {"N": full, "N-1": full[1:], "N-2": [i for i in full if i != 1], "first3": full[:3]}


def requests(item, version):
    words = item["utterance"].split()
    out = []
    for name, idx in coalitions(words).items():
        text = " ".join(words[i] for i in idx)
        out.append(("A", name, {"model": version, "state": {"utterance": text},
                                "questions": {"intent": {"type": "choice", "instructions": "What is the user asking for in `utterance`?",
                                                         "criteria": criteria(item["options"])}}}))
        out.append(("A1", name, {"model": version, "state": NEUTRAL,
                                 "questions": {"intent": {"type": "choice", "instructions": f"What is the user asking for in this utterance: \"{text}\"?",
                                                          "criteria": criteria(item["options"])}}}))
    packed = {name: {"type": "choice", "instructions": f"What is the user asking for in this utterance: \"{' '.join(words[i] for i in idx)}\"?",
                     "criteria": criteria(item["options"])} for name, idx in coalitions(words).items()}
    out.append(("B", None, {"model": version, "state": NEUTRAL, "questions": packed}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "derived" / "explicacao" / "jev_packing_test.json"))
    a = ap.parse_args()
    key = os.environ["TYPESAFE_API_KEY"]
    rng = np.random.default_rng(np.random.SeedSequence([20261021, 60]))
    jobs = []
    for loc in ("pt", "en"):
        items, _ = load_items(loc)
        pool = [i for i in range(1000, 2000) if len(items[i]["utterance"].split()) >= 4]
        for i in sorted(rng.choice(pool, 10, replace=False)):
            for v in VERSIONS:
                for fmt, name, body in requests(items[i], v):
                    jobs.append({"loc": loc, "item": int(i), "version": v, "format": fmt, "coalition": name,
                                 "options": items[i]["options"], "body": body})
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(lambda j: post(key, j["body"]), jobs))
    rows = []
    for j, (r, err) in zip(jobs, res):
        row = {k: j[k] for k in ("loc", "item", "version", "format", "coalition", "options")}
        row["error"] = err
        if r:
            row["model_reported"] = r.get("model")
            row["input_tokens"] = r["usage"]["input_tokens"]
            row["probs"] = {q: [ans["probabilities"][o] for o in j["options"]] for q, ans in r["answers"].items()}
        rows.append(row)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"measured": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0, 1), "rows": rows}, open(a.out, "w"), indent=0)
    print(f"{sum(1 for r in rows if not r['error'])}/{len(rows)} ok in {time.time() - t0:.1f} s; errors: "
          f"{sorted({r['error'] for r in rows if r['error']})}")


if __name__ == "__main__":
    main()
