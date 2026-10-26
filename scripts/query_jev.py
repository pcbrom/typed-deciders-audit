#!/usr/bin/env python3
"""Query Jev (TypeSafe System One API) on the MASSIVE items and store the answers as returned.

The API key is read from the TYPESAFE_API_KEY environment variable and is never written anywhere.
The published answers in data/raw/jev_<locale>.json were collected on 2026-09-26 with the alias
jev-latest, which the API reported as jev-1.13.0; pass --model jev-1.13.0 to pin that version.
Requests use a small thread pool with exponential backoff on 429/529 and on network errors.

Usage: TYPESAFE_API_KEY=... query_jev.py --locale pt [--model jev-1.13.0] [--workers 8] [--out PATH]
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from common import DATA, load_items, question

URL = "https://api.typesafe.ai/v1/systemone"


def ask(key, item, model, timeout=30, retries=5):
    body = {"model": model, "state": {"utterance": item["utterance"]},
            "questions": {"intent": question(item["options"])}}
    data = json.dumps(body).encode()
    for attempt in range(retries):
        req = urllib.request.Request(URL, data=data, headers={"Authorization": "Bearer " + key,
                                                             "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read()), None
        except urllib.error.HTTPError as e:
            if e.code in (429, 529) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, f"HTTP {e.code}"
        except Exception as e:  # network error: retry, then report
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, type(e).__name__
    return None, "retries exhausted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", required=True, choices=("pt", "en"))
    ap.add_argument("--model", default="jev-1.13.0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    key = os.environ["TYPESAFE_API_KEY"]
    items, _ = load_items(args.locale)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda it: ask(key, it, args.model), items))
    errors = {i: e for i, (_, e) in enumerate(results) if e}
    probs = [[r["answers"]["intent"]["probabilities"][o] for o in it["options"]] if r else None
             for it, (r, _) in zip(items, results)]
    reported = next((r.get("model") for r, _ in results if r), None)
    out = {"model": reported, "measured": time.strftime("%Y-%m-%d"), "endpoint": "POST " + URL,
           "note": "probabilities exactly as returned by the API; same order as data/items",
           "config": {"locale": args.locale, "n_sent": len(items), "n_ok": len(items) - len(errors),
                      "model_requested": args.model, "model_reported": reported},
           "errors": errors, "probs": probs}
    dest = args.out or DATA / "raw" / f"jev_{args.locale}.json"
    with open(dest, "w") as f:
        json.dump(out, f, indent=0)
    print(f"{len(items) - len(errors)}/{len(items)} answered, model {reported}, written to {dest}")


if __name__ == "__main__":
    main()
