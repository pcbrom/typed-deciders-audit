#!/usr/bin/env python3
"""Analysis of the attribution ablation and of the real streams of the drift experiment
(preregistration/implementation_notes.md, analysis of the ablation and the real streams).

ablation: each variant against the author's arm of the main run, paired by drift, pattern and replicate.
real: manipulation check from the cached initial models, then author minus periodic per transition,
summarised over the transitions that pass with transitions as clusters.

Usage: analysis_extra.py ablation | real [--dir data/derived/drift]
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from scipy import stats

from analysis import paired

DER = Path(__file__).resolve().parents[2] / "data/derived/drift"


def load(path):
    return [json.loads(line) for line in open(path)]


def ablation(der, name="ablation"):
    main = {(r["drift"], r["pattern"], r["replicate"]): r for r in load(der / "main.jsonl") if r["arm"] == "author"}
    rows = load(der / f"{name}.jsonl")
    out = {}
    for v in sorted({r["variant"] for r in rows}):
        out[v] = {}
        for drift in ("D1", "D2", "D3"):
            pairs = [(r, main[k]) for r in rows if r["variant"] == v and r["drift"] == drift
                     and (k := (r["drift"], r["pattern"], r["replicate"])) in main]
            cost = paired([x["cost_per_item"] for x, _ in pairs], [y["cost_per_item"] for _, y in pairs])
            out[v][drift] = {"custo": cost,
                             "janelas_abaixo": paired([x["post_windows_below"] for x, _ in pairs], [y["post_windows_below"] for _, y in pairs]),
                             "retreinos": paired([x["retrains"] for x, _ in pairs], [y["retrains"] for _, y in pairs]),
                             "rotulos_por_1000": paired([1000 * (x["escalated"] + x["audited"]) / x["length"] for x, _ in pairs],
                                                        [1000 * (y["escalated"] + y["audited"]) / y["length"] for _, y in pairs]),
                             "erros_por_1000": paired([1000 * x["wrong_accepted"] / x["length"] for x, _ in pairs],
                                                      [1000 * y["wrong_accepted"] / y["length"] for _, y in pairs]),
                             "componente_creditado": bool(cost.get("n", 0) >= 2 and cost["IC95_t"][0] > 0)}
            w = out[v][drift]["janelas_abaixo"]
            if w.get("n", 0) >= 2:
                out[v][drift]["janelas_nao_inferior"] = bool(w["IC95_t"][1] <= 0.02)
                out[v][drift]["janelas_superior"] = bool(w["IC95_t"][1] < 0)
    (der / f"analysis_{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({v: {d: {"media": x["custo"].get("media"), "IC95_t": x["custo"].get("IC95_t"), "creditado": x["componente_creditado"]}
                          for d, x in dd.items()} for v, dd in out.items()}, indent=1, ensure_ascii=False))


def manipulation(drifts):
    import data as D
    from model import initial_learner
    out = {}
    for drift in drifts:
        st = D.stream(drift, "abrupt", 0, 0)
        L = initial_learner(drift, D.initial_training(drift))
        acc = [float((L.predict(t).argmax(1) == np.array([it["gold"] for it in t])).mean()) for t in (st.test_pre, st.test_post)]
        out[drift] = {"acc_pre": round(acc[0], 4), "acc_post": round(acc[1], 4), "queda_pp": round(100 * (acc[0] - acc[1]), 2),
                      "passa": bool(acc[0] - acc[1] >= 0.02)}
    return out


def real(der, comparator):
    rows = load(der / "real.jsonl")
    by = {(r["arm"], r["drift"], r["replicate"]): r for r in rows}
    drifts = sorted({r["drift"] for r in rows})
    check = manipulation(drifts)
    per = {}
    for drift in drifts:
        reps = sorted({rep for (a, d, rep) in by if d == drift and a == "author" and (comparator, d, rep) in by})
        dc = [by[("author", drift, r)]["cost_per_item"] - by[(comparator, drift, r)]["cost_per_item"] for r in reps]
        dw = [by[("author", drift, r)]["post_windows_below"] - by[(comparator, drift, r)]["post_windows_below"] for r in reps]
        arms = sorted({a for (a, d, _) in by if d == drift})
        per[drift] = {**check[drift], "n_ordens": len(reps),
                      "dif_custo": round(float(np.mean(dc)), 5) if dc else None,
                      "dif_janelas_abaixo": round(float(np.mean(dw)), 5) if dw else None,
                      "custo_por_braco": {a: round(statistics.mean(by[(a, drift, r)]["cost_per_item"] for r in range(10) if (a, drift, r) in by), 4)
                                          for a in arms}}
    ok = [d for d in drifts if per[d]["passa"] and per[d]["dif_custo"] is not None]

    def cluster(key):
        x = np.array([per[d][key] for d in ok])
        if len(x) < 2:
            return {"n_transicoes": len(x)}
        m, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
        t = stats.t.ppf(0.975, len(x) - 1)
        return {"n_transicoes": len(x), "media": round(m, 5), "IC95_t": [round(m - t * se, 5), round(m + t * se, 5)]}

    out = {"comparador": comparator, "por_transicao": per, "transicoes_validas": ok,
           "resumo": {"custo": cluster("dif_custo"), "janelas_abaixo": cluster("dif_janelas_abaixo")}}
    (der / "analysis_real.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({"transicoes_validas": ok, "resumo": out["resumo"],
                      "checagem": {d: (per[d]["queda_pp"], per[d]["passa"]) for d in drifts}}, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("ablation", "ablation2", "real"))
    ap.add_argument("--dir", default=str(DER))
    ap.add_argument("--comparator", default=None)
    a = ap.parse_args()
    der = Path(a.dir)
    comp = a.comparator or json.loads((DER / "selection.json").read_text())["best_comparator"]
    if a.what == "real":
        real(der, comp)
    else:
        ablation(der, a.what)


if __name__ == "__main__":
    main()
