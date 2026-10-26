#!/usr/bin/env python3
"""Runs of the dynamic-model adapter (implementation_notes.md, drift with a pattern).

  tune   dyn_level with delta in {0.80, 0.90, 0.95} on the tuning streams of the main experiment (role 1, D1 to D3 in
         both patterns, replicates 0 and 1, 4,000 items); selection by mean cost into adaptador_selection.json
  main   S1 and S2, 30 replicates: frozen and the tuned arms of the main run, and the four dynamic arms; D1 to D3 in
         both patterns and the control stream, 30 replicates: dyn_level and dyn_season_exog

Appends to data/derived/drift/adaptador_tuning.jsonl and adaptador_main.jsonl, resumes, and waits while
/tmp/gpu_pause_orientacoes exists.

Usage: adaptador_run.py tune|main
"""

import json
import statistics
import sys
import time
from pathlib import Path

import run as R

DER = Path(__file__).resolve().parents[2] / "data/derived/drift"
PAUSE = Path("/tmp/gpu_pause_orientacoes")
DELTAS = (0.80, 0.90, 0.95)
PERIOD_WINDOWS = 4


def dynamic_arms(delta):
    base = {"bins": 20, "delta": delta}
    return {"dyn_level": base, "dyn_season": {**base, "period": PERIOD_WINDOWS},
            "dyn_season_exog": {**base, "period": PERIOD_WINDOWS, "exog": True},
            "dyn_season_exog_lb": {**base, "period": PERIOD_WINDOWS, "exog": True, "criterion": "lower"}}


def append(path, res, extra):
    with open(path, "a") as f:
        f.write(json.dumps({**res, **extra}) + "\n")


def done_keys(path, fields):
    if not path.exists():
        return set()
    return {tuple(json.loads(l)[k] for k in fields) for l in open(path)}


def tune():
    out = DER / "adaptador_tuning.jsonl"
    done = done_keys(out, ("delta", "drift", "pattern", "replicate"))
    for delta in DELTAS:
        for d in ("D1", "D2", "D3"):
            for p in ("abrupt", "gradual"):
                for rep in (0, 1):
                    if (delta, d, p, rep) in done:
                        continue
                    while PAUSE.exists():
                        time.sleep(20)
                    res = R.run("dynamic", {"bins": 20, "delta": delta}, d, p, rep, role=1, length=4000)
                    append(out, {k: v for k, v in res.items() if k != "windows"}, {"delta": delta})
    rows = [json.loads(l) for l in open(out)]
    cost = {dl: statistics.mean(r["cost_per_item"] for r in rows if r["delta"] == dl) for dl in DELTAS}
    best = min(cost, key=cost.get)
    (DER / "adaptador_selection.json").write_text(json.dumps({"mean_cost": cost, "delta": best}, indent=1))
    print("tuning", cost, "selected", best, flush=True)


def plan(delta):
    sel = json.loads((DER / "selection.json").read_text())["selected"]
    dyn = dynamic_arms(delta)
    base = [("frozen", "frozen", {})] + [(a, a, sel[a]["cfg"]) for a in ("recal", "author", "periodic", "ltt_window", "aci")]
    base += [(name, "dynamic", cfg) for name, cfg in dyn.items()]
    for rep in range(30):
        for d in ("S1", "S2"):
            for name, arm, cfg in base:
                yield name, arm, cfg, d, "abrupt", rep
        for d in ("D1", "D2", "D3", "control"):
            for p in (("abrupt",) if d == "control" else ("abrupt", "gradual")):
                for name in ("dyn_level", "dyn_season_exog"):
                    yield name, "dynamic", dyn[name], d, p, rep


def main_runs():
    delta = json.loads((DER / "adaptador_selection.json").read_text())["delta"]
    out = DER / "adaptador_main.jsonl"
    done = done_keys(out, ("label", "drift", "pattern", "replicate"))
    for name, arm, cfg, d, p, rep in plan(delta):
        if (name, d, p, rep) in done:
            continue
        while PAUSE.exists():
            time.sleep(20)
        res = R.run(arm, cfg, d, p, rep, role=0)
        append(out, res, {"label": name})
        print(json.dumps({"label": name, "drift": d, "pattern": p, "replicate": rep, "cost": round(res["cost_per_item"], 4),
                          "wall_s": res["wall_s"]}), flush=True)


if __name__ == "__main__":
    {"tune": tune, "main": main_runs}[sys.argv[1]]()
