#!/usr/bin/env python3
"""Attribution ablation inside the author's arm (preregistration/drift.md, section 4): each variant swaps
one component of the selected configuration, on the same evaluation streams as the main run (30
replicates of D1, D2 and D3 in both patterns). Appends to data/derived/drift/ablation.jsonl, resumes,
and waits while /tmp/gpu_pause_orientacoes exists.

Usage: ablation_run.py [--replicates 30]
"""

import argparse
import json
import os
import time
from pathlib import Path

import run as R

DER = Path(__file__).resolve().parents[2] / "data/derived/drift"
PAUSE = Path("/tmp/gpu_pause_orientacoes")
VARIANTS = {"contagem_por_janela": {"counts": "window"}, "clopper_pearson": {"trigger": "cp"},
            "promocao_mcnemar": {"promotion": "mcnemar"}, "promocao_sempre": {"promotion": "always"},
            "requalificacao_ltt": {"requal": "ltt"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=30)
    a = ap.parse_args()
    base = json.loads((DER / "selection.json").read_text())["selected"]["author"]["cfg"]
    out = DER / "ablation.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            done.add((r["variant"], r["drift"], r["pattern"], r["replicate"]))
    for rep in range(a.replicates):
        for drift in ("D1", "D2", "D3"):
            for pattern in ("abrupt", "gradual"):
                for name, swap in VARIANTS.items():
                    if (name, drift, pattern, rep) in done:
                        continue
                    while PAUSE.exists():
                        time.sleep(20)
                    res = R.run("author", {**base, **swap}, drift, pattern, rep, role=0)
                    res["variant"] = name
                    with open(out, "a") as f:
                        f.write(json.dumps(res) + "\n")
                    print(json.dumps({k: res[k] for k in ("variant", "drift", "pattern", "replicate", "cost_per_item", "retrains")}), flush=True)
    os.sync()


if __name__ == "__main__":
    main()
