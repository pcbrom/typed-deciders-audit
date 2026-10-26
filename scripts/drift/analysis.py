#!/usr/bin/env python3
"""Analysis of the drift experiment (preregistration/drift.md, sections 6 and 9, and
implementation_notes.md, analysis of the main run).

Usage: analysis.py [--file main.jsonl] [--comparator periodic]
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

DER = Path(__file__).resolve().parents[2] / "data/derived/drift"


def paired(a, b):
    d = np.array(a) - np.array(b)
    n = len(d)
    if n < 2:
        return {"n": n}
    m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
    t = stats.t.ppf(0.975, n - 1)
    rng = np.random.default_rng(np.random.SeedSequence([20261022, 90]))
    boot = [d[rng.integers(0, n, n)].mean() for _ in range(10000)]
    return {"n": n, "media": round(m, 5), "IC95_t": [round(m - t * se, 5), round(m + t * se, 5)],
            "IC95_boot": [round(float(x), 5) for x in np.percentile(boot, [2.5, 97.5])], "dp": round(float(d.std(ddof=1)), 5)}


def cost100(r):
    return (100 * r["wrong_accepted"] + r["escalated"] + r["audited"]) / r["length"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="main.jsonl")
    ap.add_argument("--comparator", default=json.loads((DER / "selection.json").read_text())["best_comparator"])
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(DER / a.file)]
    by = {(r["arm"], r["drift"], r["pattern"], r["replicate"]): r for r in rows}
    arms = sorted({r["arm"] for r in rows})
    out = {"arquivo": a.file, "comparador": a.comparator, "por_drift": {}, "secundarias": {}}
    favour = 0
    for drift in ("D1", "D2", "D3"):
        keys = sorted({(p, rep) for (arm, d, p, rep) in by if d == drift})
        pairs = [(by[("author", drift, p, rep)], by[(a.comparator, drift, p, rep)]) for p, rep in keys
                 if ("author", drift, p, rep) in by and (a.comparator, drift, p, rep) in by]
        prim = paired([x["cost_per_item"] for x, _ in pairs], [y["cost_per_item"] for _, y in pairs])
        ni = paired([x["post_windows_below"] for x, _ in pairs], [y["post_windows_below"] for _, y in pairs])
        ok = bool(prim.get("n", 0) >= 2 and prim["IC95_t"][1] < 0)
        favour += ok
        out["por_drift"][drift] = {"custo_primario": prim, "autor_menor_com_IC": ok, "nao_inferioridade": ni,
                                   "nao_inferior": bool(ni.get("n", 0) >= 2 and ni["IC95_t"][1] <= 0.02)}
    ctrl = [(by[("author", "control", "abrupt", rep)], by[(a.comparator, "control", "abrupt", rep)])
            for rep in range(100) if ("author", "control", "abrupt", rep) in by and (a.comparator, "control", "abrupt", rep) in by]
    fr = paired([x["retrains"] for x, _ in ctrl], [y["retrains"] for _, y in ctrl])
    out["retreinos_falsos"] = {**fr, "nao_mais_que_comparador": bool(fr.get("n", 0) >= 2 and fr["IC95_t"][0] <= 0)}
    ni_all = all(v["nao_inferior"] for v in out["por_drift"].values())
    out["regra_do_titulo"] = {"drifts_a_favor": favour, "mantem_bayesian": bool(favour >= 2 and ni_all and out["retreinos_falsos"]["nao_mais_que_comparador"])}
    for arm in arms:
        for drift in ("D1", "D2", "D3", "control"):
            rs = [r for r in rows if r["arm"] == arm and r["drift"] == drift]
            if not rs:
                continue
            delays = [r["first_retrain_after_change"] for r in rs if r.get("first_retrain_after_change") is not None]
            out["secundarias"][f"{arm}|{drift}"] = {
                "n": len(rs), "custo10": round(statistics.mean(r["cost_per_item"] for r in rs), 4),
                "custo100": round(statistics.mean(cost100(r) for r in rs), 4),
                "janelas_abaixo": round(statistics.mean(b), 4) if (b := [r["post_windows_below"] for r in rs if r["post_windows_below"] is not None]) else None,
                "retreinos": round(statistics.mean(r["retrains"] for r in rs), 3), "promocoes": round(statistics.mean(r["promotions"] for r in rs), 3),
                "rotulos_por_1000": round(1000 * statistics.mean((r["escalated"] + r["audited"]) / r["length"] for r in rs), 1),
                "gpu_s": round(statistics.mean(r.get("gpu_s", 0.0) for r in rs), 2),
                "atraso_mediano": statistics.median(delays) if delays else None, "frac_com_retreino_pos_mudanca": round(len(delays) / len(rs), 3)}
    name = "analysis_" + Path(a.file).stem + ".json"
    (DER / name).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "secundarias"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
