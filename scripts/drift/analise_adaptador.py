#!/usr/bin/env python3
"""Analysis of the dynamic-model adapter (implementation_notes.md, drift with a pattern; exploratory).

Primary comparisons, paired by stream (drift, pattern, replicate): dyn_season and dyn_season_exog against
the recalibration arm on S1 and S2, and dyn_level against the recalibration arm on every stream. The
recalibration arm of the D streams and the control comes from the main run, on the same streams. Each
comparison follows the efficiency protocol of the drift front: outcome the share of windows below the
promise (all windows, since S1 and S2 have no change point), margin 0.02, resources labels per 1,000
items, wrong actions per 1,000 and cost per item.

Usage: analise_adaptador.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

DER = Path(__file__).resolve().parents[2] / "data/derived/drift"
ALPHA, MARGIN = 0.95, 0.02
STREAMS = ("S1", "S2", "D1", "D2", "D3", "control")
LABELS = ("frozen", "recal", "recal_lam08", "author", "periodic", "ltt_window", "aci", "dyn_level", "dyn_season", "dyn_season_exog", "dyn_season_exog_lb")
PRIMARY = [(arm, s) for arm in ("dyn_season", "dyn_season_exog") for s in ("S1", "S2")] + [("dyn_level", s) for s in STREAMS]
SECONDARY = [("dyn_season_exog_lb", s) for s in ("S1", "S2")] + [("dyn_season_exog", s) for s in ("D1", "D2", "D3", "control")]
ATTRIBUTION = [("dyn_level", "recal_lam08", s) for s in STREAMS] + [("recal_lam08", "recal", s) for s in STREAMS]


def ci(d, level):
    d = np.asarray(d, float)
    m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    if se == 0:
        return m, m, m
    t = stats.t.ppf(0.5 + level / 2, len(d) - 1)
    return m, float(m - t * se), float(m + t * se)


def below(r):
    """Share of windows below the promise: post-change windows on D1 to D3, all windows otherwise."""
    if r["drift"] in ("D1", "D2", "D3") and r["post_windows_below"] is not None:
        return r["post_windows_below"]
    th = [w["theta"] for w in r["windows"] if w["theta"] is not None]
    return float(np.mean([x < ALPHA for x in th])) if th else 0.0


def resources(r):
    return {"rotulos_por_1000": 1000 * (r["escalated"] + r["audited"]) / r["length"],
            "erros_por_1000": 1000 * r["wrong_accepted"] / r["length"], "custo": r["cost_per_item"]}


def load():
    by = {}
    for r in map(json.loads, open(DER / "adaptador_main.jsonl")):
        by[(r["label"], r["drift"], r["pattern"], r["replicate"])] = r
    lam = DER / "recal_lambda08.jsonl"
    if lam.exists():
        for r in map(json.loads, open(lam)):
            by[(r["label"], r["drift"], r["pattern"], r["replicate"])] = r
    for r in map(json.loads, open(DER / "main.jsonl")):
        key = (r["arm"], r["drift"], r["pattern"], r["replicate"])
        if r["drift"] in ("D1", "D2", "D3", "control") and key not in by:
            by[key] = r
    return by


def compare(by, arm, stream, against="recal"):
    keys = sorted(k[2:] for k in by if k[0] == arm and k[1] == stream and (against, stream) + k[2:] in by)
    a, b = [by[(arm, stream) + k] for k in keys], [by[(against, stream) + k] for k in keys]
    m, lo95, hi95 = ci([below(x) - below(y) for x, y in zip(a, b)], 0.95)
    _, lo90, hi90 = ci([below(x) - below(y) for x, y in zip(a, b)], 0.90)
    res = {}
    for key in ("rotulos_por_1000", "erros_por_1000", "custo"):
        mm, lo, hi = ci([resources(x)[key] - resources(y)[key] for x, y in zip(a, b)], 0.95)
        res[key] = {"media": round(mm, 4), "IC95": [round(lo, 4), round(hi, 4)], "menor": bool(hi < 0), "maior": bool(lo > 0)}
    ni = bool(hi95 <= MARGIN)
    fewer = [k for k in ("rotulos_por_1000", "erros_por_1000") if res[k]["menor"]]
    larger = [k for k in ("rotulos_por_1000", "erros_por_1000") if res[k]["maior"]]
    if ni and fewer:
        verdict = "troca" if larger else "ganho limpo"
    elif ni and bool(lo90 >= -MARGIN and hi90 <= MARGIN):
        verdict = "equivalente"
    elif ni:
        verdict = "nao inferior"
    else:
        verdict = "nao inferioridade nao mostrada"
    return {"n": len(keys), "janelas_abaixo": {"media": round(m, 4), "IC95": [round(lo95, 4), round(hi95, 4)], "IC90": [round(lo90, 4), round(hi90, 4)]},
            "nao_inferior": ni, "recursos": res, "leitura": verdict}


def summary(by):
    out = {}
    for s in STREAMS:
        for arm in LABELS:
            rs = [r for k, r in by.items() if k[0] == arm and k[1] == s]
            if rs:
                out[f"{arm}|{s}"] = {"n": len(rs), "custo": round(float(np.mean([r["cost_per_item"] for r in rs])), 4),
                                     "janelas_abaixo": round(float(np.mean([below(r) for r in rs])), 4),
                                     "rotulos_por_1000": round(float(np.mean([resources(r)["rotulos_por_1000"] for r in rs])), 1),
                                     "erros_por_1000": round(float(np.mean([resources(r)["erros_por_1000"] for r in rs])), 2),
                                     "retreinos": round(float(np.mean([r["retrains"] for r in rs])), 2)}
    return out


def main():
    by = load()
    res = {"rotulo": "exploratorio, sem correcao de multiplicidade", "margem": MARGIN,
           "primarias": {f"{a}|{s}": compare(by, a, s) for a, s in PRIMARY},
           "secundarias": {f"{a}|{s}": compare(by, a, s) for a, s in SECONDARY},
           "atribuicao": {f"{a}|{b}|{s}": compare(by, a, s, against=b) for a, b, s in ATTRIBUTION if any(k[0] == a for k in by) and any(k[0] == b for k in by)},
           "medias": summary(by)}
    (DER / "analysis_adaptador.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for part in ("primarias", "secundarias", "atribuicao"):
        for k, v in res[part].items():
            print(part[:4], k, v["leitura"], "janelas", v["janelas_abaixo"]["media"], v["janelas_abaixo"]["IC95"],
                  "custo", v["recursos"]["custo"]["media"], v["recursos"]["custo"]["IC95"],
                  "rotulos", v["recursos"]["rotulos_por_1000"]["media"], "erros", v["recursos"]["erros_por_1000"]["media"])


if __name__ == "__main__":
    main()
