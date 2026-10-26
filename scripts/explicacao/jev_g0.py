#!/usr/bin/env python3
"""Gate G0 of the explanation front (preregistration/explicacao_portoes.md): stability of Jev.

100 report items per language (drawn with SeedSequence([20261021, 51])), 3 repetitions each, in
each pinned version, in the preregistered format (utterance in the state). For jev-1.13.0 the
answers are compared with the frozen collection; for jev-1.14.0, which has none, the repetitions
are compared with each other. Billable tokens are recorded from usage.input_tokens.

The key is read from TYPESAFE_API_KEY and never written.
Usage: TYPESAFE_API_KEY=... jev_g0.py
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, load_items, question  # noqa: E402
from jev_packing_test import post  # noqa: E402

VERSIONS, REPS, N = ("jev-1.13.0", "jev-1.14.0"), 3, 100
OUT = DATA / "derived" / "explicacao" / "jev_g0.json"


def main():
    key = os.environ["TYPESAFE_API_KEY"]
    rng = np.random.default_rng(np.random.SeedSequence([20261021, 51]))
    jobs = []
    for loc in ("pt", "en"):
        items, _ = load_items(loc)
        for i in sorted(rng.choice(np.arange(1000, 2000), N, replace=False)):
            for v in VERSIONS:
                for rep in range(REPS):
                    jobs.append({"loc": loc, "item": int(i), "version": v, "rep": rep, "options": items[i]["options"],
                                 "body": {"model": v, "state": {"utterance": items[i]["utterance"]},
                                          "questions": {"intent": question(items[i]["options"])}}})
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(lambda j: post(key, j["body"]), jobs))
    rows = []
    for j, (r, err) in zip(jobs, res):
        row = {k: j[k] for k in ("loc", "item", "version", "rep")}
        row["error"] = err
        if r:
            row["model_reported"] = r.get("model")
            row["input_tokens"] = r["usage"]["input_tokens"]
            row["probs"] = [r["answers"]["intent"]["probabilities"][o] for o in j["options"]]
        rows.append(row)
    frozen = {loc: json.loads((DATA / "raw" / f"jev_{loc}.json").read_text())["probs"] for loc in ("pt", "en")}
    out = {"measured": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0, 1), "rows": rows, "gate": {}}
    for v in VERSIONS:
        R = [r for r in rows if r["version"] == v and not r["error"]]
        if v == "jev-1.13.0":
            pairs = [(np.array(r["probs"]), np.array(frozen[r["loc"]][r["item"]])) for r in R]
        else:
            byitem = {}
            for r in R:
                byitem.setdefault((r["loc"], r["item"]), []).append(np.array(r["probs"]))
            pairs = [(a, b) for ps in byitem.values() for x, a in enumerate(ps) for b in ps[x + 1:]]
        arg = float(np.mean([a.argmax() == b.argmax() for a, b in pairs]))
        close = float(np.mean(np.concatenate([np.abs(a - b) <= 0.01 for a, b in pairs])))
        out["gate"][v] = {"pairs": len(pairs), "argmax_agreement": round(arg, 4), "probs_within_0.01": round(close, 4),
                          "passes": arg >= 0.98 and close >= 0.95,
                          "tokens": int(sum(r["input_tokens"] for r in R)), "errors": sum(1 for r in rows if r["version"] == v and r["error"])}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=0)
    print(json.dumps(out["gate"], indent=1), out["seconds"], "s")


if __name__ == "__main__":
    main()
