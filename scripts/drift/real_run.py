#!/usr/bin/env python3
"""Real streams of the drift experiment (News Category, implementation_notes.md, real streams).

Arms: the frozen model and every tuned arm with its selected configuration (selection.json).
Streams: ten transitions R2012 to R2021, five orders each (replicates 0 to 4), abrupt change, role 0.
Results are appended to data/derived/drift/real.jsonl, so an interrupted run resumes. Between runs the
script waits while /tmp/gpu_pause_orientacoes exists.

Usage: real_run.py [--replicates 5]
"""

import argparse
import json
import os
import time
from pathlib import Path

import data as D
import run as R

DER = Path(__file__).resolve().parents[2] / "data/derived/drift"
PAUSE = Path("/tmp/gpu_pause_orientacoes")


def plan(replicates):
    sel = json.loads((DER / "selection.json").read_text())["selected"]
    arms = [("frozen", {})] + [(a, sel[a]["cfg"]) for a in sorted(sel)]
    for rep in range(replicates):
        for drift in D.REAL:
            for arm, cfg in arms:
                yield arm, cfg, drift, "abrupt", rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=5)
    a = ap.parse_args()
    out = DER / "real.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            done.add((r["arm"], r["drift"], r["pattern"], r["replicate"]))
    for arm, cfg, drift, pattern, rep in plan(a.replicates):
        if (arm, drift, pattern, rep) in done:
            continue
        while PAUSE.exists():
            time.sleep(20)
        res = R.run(arm, cfg, drift, pattern, rep, role=0)
        with open(out, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps({k: res[k] for k in ("arm", "drift", "pattern", "replicate", "cost_per_item", "retrains",
                                              "promotions", "wall_s")}), flush=True)
    os.sync()


if __name__ == "__main__":
    main()
