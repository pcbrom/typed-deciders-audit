#!/usr/bin/env python3
"""Recalibration arm with lambda = 0.80, the level discount selected for the dynamic adapter, on the streams of the
adapter analysis (implementation_notes.md, recalibration with lambda 0.80): S1, S2 and the control, abrupt; D1 to
D3, abrupt and gradual; 30 replicates. Attribution check for the gain of dyn_level over the tuned recalibration.

Appends to data/derived/drift/recal_lambda08.jsonl, resumes, and waits while /tmp/gpu_pause_orientacoes exists.

Usage: recal_lambda_run.py
"""

import json
import time
from pathlib import Path

import run as R

DER = Path(__file__).resolve().parents[2] / "data/derived/drift"
PAUSE = Path("/tmp/gpu_pause_orientacoes")
CFG = {"bins": 20, "lam": 0.8}
LABEL = "recal_lam08"


def plan():
    for d in ("D1", "D3", "S1", "S2", "D2", "control"):
        for rep in range(30):
            for p in (("abrupt", "gradual") if d in ("D1", "D2", "D3") else ("abrupt",)):
                yield d, p, rep


def main():
    out = DER / "recal_lambda08.jsonl"
    done = {(r["drift"], r["pattern"], r["replicate"]) for r in map(json.loads, open(out))} if out.exists() else set()
    for d, p, rep in plan():
        if (d, p, rep) in done:
            continue
        while PAUSE.exists():
            time.sleep(20)
        res = R.run("recal", CFG, d, p, rep, role=0)
        with open(out, "a") as f:
            f.write(json.dumps({**res, "label": LABEL}) + "\n")
        print(json.dumps({"drift": d, "pattern": p, "replicate": rep, "cost": round(res["cost_per_item"], 4), "wall_s": res["wall_s"]}), flush=True)


if __name__ == "__main__":
    main()
