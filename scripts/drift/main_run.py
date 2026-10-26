#!/usr/bin/env python3
"""Main run of the drift experiment (preregistration/drift.md, sections 5 to 9), on evaluation seeds.

Arms: the frozen model and every tuned arm with its selected configuration (selection.json).
Streams: R = 30 replicates of each synthetic drift (D1, D2, D3) in both patterns, and 30 replicates
of the control stream with no drift (its pattern label is "abrupt" and has no effect), role 0.
Order: replicate by replicate, so that an interruption leaves a sample balanced across drifts,
patterns and arms. Results are appended to data/derived/drift/main.jsonl, so an interrupted run
resumes. Between runs the script waits while /tmp/gpu_pause_orientacoes exists.

Usage: main_run.py [--replicates 30]
"""

import argparse
import json
import os
import time
from pathlib import Path

import run as R

DER = Path(__file__).resolve().parents[2] / "data/derived/drift"
PAUSE = Path("/tmp/gpu_pause_orientacoes")


def plan(replicates):
    sel = json.loads((DER / "selection.json").read_text())["selected"]
    arms = [("frozen", {})] + [(a, sel[a]["cfg"]) for a in sorted(sel)]
    for rep in range(replicates):
        for drift in ("D1", "D2", "D3", "control"):
            for pattern in (("abrupt",) if drift == "control" else ("abrupt", "gradual")):
                for arm, cfg in arms:
                    yield arm, cfg, drift, pattern, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=30)
    a = ap.parse_args()
    out = DER / "main.jsonl"
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
