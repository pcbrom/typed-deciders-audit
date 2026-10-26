#!/usr/bin/env python3
"""Tuning of the drift arms (preregistration/implementation_notes.md, section Tuning).

Twenty configurations per arm, each run on 12 tuning streams (role 1, 4,000 items, change in
[1,500, 2,500]): three synthetic drifts, two patterns, two replicates. Results are appended to
data/derived/drift/tuning.jsonl, so an interrupted run resumes. Between runs the script waits while
/tmp/gpu_pause_orientacoes exists, to yield the GPU to the other session on this machine.

Usage: tune.py [--arms author,periodic,...]
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

import run as R

PAUSE = Path("/tmp/gpu_pause_orientacoes")
OUT = Path(__file__).resolve().parents[2] / "data/derived/drift/tuning.jsonl"
PROMO_DET = ("always", "mcnemar", "seldonian")
GRIDS = {
    "author": [{"lam": l, "q": q, "promotion": "beta"} for l in (0.95, 0.97, 0.98, 0.99, 0.995) for q in (0.01, 0.05, 0.10, 0.20)],
    "periodic": [{"every": k, "promotion": pr} for k in (500, 1000, 2000, 3000, 4000) for pr in ("always", "mcnemar", "seldonian", "beta")],
    "ddm": [{"params": {"drift_threshold": t}, "promotion": pr, "all_labels": al}
            for t in (3.0, 2.5, 3.5, 4.0, 4.5) for pr in PROMO_DET for al in (False, True)],
    "adwin": [{"params": {"delta": d}, "promotion": pr, "all_labels": al}
              for d in (0.002, 0.0005, 0.001, 0.005, 0.01) for pr in PROMO_DET for al in (False, True)],
    "ph": [{"params": {"threshold": t}, "promotion": pr, "all_labels": al}
           for t in (50.0, 5.0, 10.0, 20.0, 100.0) for pr in PROMO_DET for al in (False, True)],
    "ltt_window": [{"q": q} for q in (0.01, 0.025, 0.05, 0.10, 0.20)],
    "aci": [{"gamma": g} for g in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)],
    "recal": [{"lam": l, "bins": b} for l in (0.9, 0.95, 0.98, 0.99, 1.0) for b in (5, 10, 15, 20)],
}
STREAMS = [(d, p, r) for d in ("D1", "D2", "D3") for p in ("abrupt", "gradual") for r in (0, 1)]


def configs(arm):
    grid = GRIDS[arm]
    if len(grid) <= 20:
        return grid
    rng = np.random.default_rng(np.random.SeedSequence([20261022, 77, list(GRIDS).index(arm)]))
    return [grid[i] for i in sorted(rng.choice(len(grid), 20, replace=False))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(GRIDS))
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["arm"], json.dumps(r["cfg"], sort_keys=True), r["drift"], r["pattern"], r["replicate"]))
    for arm in a.arms.split(","):
        for cfg in configs(arm):
            for d, p, rep in STREAMS:
                key = (arm, json.dumps(cfg, sort_keys=True), d, p, rep)
                if key in done:
                    continue
                while PAUSE.exists():
                    time.sleep(20)
                res = R.run(arm, cfg, d, p, rep, role=1, length=4000)
                res.pop("windows")
                with open(OUT, "a") as f:
                    f.write(json.dumps(res) + "\n")
            print(json.dumps({"arm": arm, "cfg": cfg, "done": time.strftime("%H:%M")}), flush=True)


if __name__ == "__main__":
    main()
