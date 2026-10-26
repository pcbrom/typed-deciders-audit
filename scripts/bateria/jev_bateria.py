#!/usr/bin/env python3
"""Jev 1.13 on the multi-domain battery (implementation_notes.md, multi-domain battery): every item clean,
the report items under typo_0.10 and misto_0.10, and the empty state. One record per answer with model
reported, UTC timestamp, input tokens and probabilities, appended to data/derived/bateria/jev.jsonl, with
resume. Stops before the projected spend passes the cap and on any non-retryable error, printing why.

Usage: TYPESAFE_API_KEY=... jev_bateria.py [--domains news finance liar2] [--cap 0.60]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "ruido"))
sys.path.insert(0, str(HERE.parents[1] / "explicacao"))
import ruido as R  # noqa: E402
from common import DATA  # noqa: E402
from dominios import DOMAINS  # noqa: E402
from jev_packing_test import post  # noqa: E402

OUT = DATA / "derived" / "bateria" / "jev.jsonl"
PRICE, VERSION = 0.042e-6, "jev-1.13.0"
CONDITIONS = ("typo_0.10", "misto_0.10")


def question(it):
    return {"type": "choice", "instructions": it["instruction"], "criteria": {o: o.replace("_", " ") for o in it["options"]}}


def jobs_for(domain):
    items = json.loads((DATA / "items" / "bateria" / f"{domain}.json").read_text())["items"]
    out = [(domain, k, "clean", it, it["text"]) for k, it in enumerate(items)]
    out += [(domain, k, c, items[k], R.apply(items[k]["text"], c, k)) for c in CONDITIONS for k in range(1000, 2000)]
    fixed = domain != "news"
    out += [(domain, k, "empty", items[k], "") for k in ([1000] if fixed else range(1000, 2000))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS))
    ap.add_argument("--cap", type=float, default=0.60)
    args = ap.parse_args()
    key = os.environ["TYPESAFE_API_KEY"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done, tokens = set(), 0
    if OUT.exists():
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["domain"], r["item"], r["condition"]))
            tokens += r["input_tokens"]
    jobs = [j for d in args.domains for j in jobs_for(d) if (j[0], j[1], j[2]) not in done]
    print("pending", len(jobs), "spent so far US$", round(tokens * PRICE, 4), flush=True)
    per_call = None
    with open(OUT, "a") as f, ThreadPoolExecutor(max_workers=6) as ex:
        for start in range(0, len(jobs), 300):
            batch = jobs[start:start + 300]
            if per_call and (tokens + per_call * len(batch)) * PRICE > args.cap:
                print("stopping before the cap at", batch[0][:3], "| US$", round(tokens * PRICE, 4), flush=True)
                return
            bodies = [{"model": VERSION, "state": {"utterance": text}, "questions": {"label": question(it)}} for _, _, _, it, text in batch]
            for (domain, k, cond, it, _), (r, err) in zip(batch, ex.map(lambda b: post(key, b), bodies)):
                if err:
                    print("stopping on error", err, "at", domain, k, cond, "| US$", round(tokens * PRICE, 4), flush=True)
                    return
                row = {"domain": domain, "item": k, "condition": cond, "model_reported": r.get("model"),
                       "answered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "input_tokens": r["usage"]["input_tokens"],
                       "probs": [r["answers"]["label"]["probabilities"][o] for o in it["options"]]}
                f.write(json.dumps(row) + "\n")
                tokens += row["input_tokens"]
            f.flush()
            per_call = tokens / max(1, len(done) + start + len(batch))
            print(batch[-1][0], start + len(batch), "of", len(jobs), "| US$", round(tokens * PRICE, 4), flush=True)
    print("done | US$", round(tokens * PRICE, 4), flush=True)


if __name__ == "__main__":
    main()
