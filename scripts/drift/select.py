#!/usr/bin/env python3
"""Selection of the tuned configurations (preregistration/implementation_notes.md, section Tuning).

For each arm, the configuration with the lowest mean primary cost over its 12 tuning streams is
selected. The best state-of-the-art comparator is the selected configuration with the lowest mean
primary cost among the baseline arms (every arm except the author's and the frozen model). Only
configurations with all 12 tuning streams count. Writes data/derived/drift/selection.json.

Usage: select.py
"""

import collections
import json
import statistics
from pathlib import Path

DER = Path(__file__).resolve().parents[2] / "data/derived/drift"


def main():
    rows = [json.loads(l) for l in open(DER / "tuning.jsonl")]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["arm"], json.dumps(r["cfg"], sort_keys=True))].append(r["cost_per_item"])
    arms = collections.defaultdict(list)
    for (arm, cfg), costs in by.items():
        if len(costs) == 12:
            arms[arm].append({"cfg": json.loads(cfg), "mean_cost": statistics.mean(costs), "n_streams": len(costs)})
    selected = {arm: min(cs, key=lambda c: c["mean_cost"]) for arm, cs in arms.items()}
    baselines = {a: s for a, s in selected.items() if a != "author"}
    best = min(baselines, key=lambda a: baselines[a]["mean_cost"])
    out = {"selected": selected, "best_comparator": best, "n_configurations": {a: len(cs) for a, cs in arms.items()},
           "frozen": {"cfg": {}, "note": "no tuning, by design"}}
    (DER / "selection.json").write_text(json.dumps(out, indent=1))
    for a, s in sorted(selected.items(), key=lambda kv: kv[1]["mean_cost"]):
        print(f"{a:12s} {s['mean_cost']:.4f} {json.dumps(s['cfg'])}")
    print("best comparator:", best)


if __name__ == "__main__":
    main()
