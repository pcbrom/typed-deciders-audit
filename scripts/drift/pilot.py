#!/usr/bin/env python3
"""Pilot of the drift experiment (preregistration/drift.md, section 8): 10 streams per synthetic
drift (5 abrupt, 5 gradual) on seeds disjoint from evaluation and tuning (role 3), for the author's
arm and three default baselines. It estimates the standard deviation of the paired cost
difference, the time per run and the drop of the frozen model; it does not change R.

Between runs the pilot waits while /tmp/gpu_pause_orientacoes exists, so the GPU can be yielded to
the other session on this machine without killing the process.

Usage: pilot.py [--out ../../data/derived/drift/pilot.jsonl]
"""

import argparse
import json
import os
import time
from pathlib import Path

import run as R

PAUSE = Path("/tmp/gpu_pause_orientacoes")
ARMS = [("author", {"lam": 0.99}), ("frozen", {}), ("periodic", {"every": 2000}), ("ddm", {})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "data/derived/drift/pilot.jsonl"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            done.add((r["arm"], r["drift"], r["pattern"], r["replicate"]))
    for drift in ("D1", "D2", "D3"):
        for rep in range(10):
            pattern = "abrupt" if rep < 5 else "gradual"
            for arm, cfg in ARMS:
                if (arm, drift, pattern, rep) in done:
                    continue
                while PAUSE.exists():
                    time.sleep(20)
                res = R.run(arm, cfg, drift, pattern, rep, role=3)
                with open(out, "a") as f:
                    f.write(json.dumps(res) + "\n")
                print(json.dumps({k: res[k] for k in ("arm", "drift", "pattern", "replicate", "cost_per_item", "retrains",
                                                      "promotions", "wall_s")}), flush=True)
    os.sync()


if __name__ == "__main__":
    main()
