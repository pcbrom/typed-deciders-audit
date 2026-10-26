#!/usr/bin/env python3
"""Exploratory efficiency analysis: a method gains when its outcome is non-inferior within a margin
fixed in advance and it uses fewer resources (implementation_notes.md, exploratory efficiency analysis).

Drift: the author's loop and recalibration only against every other arm, per drift type.
Static gate: the author's Beta rule against Learn then Test on the plug-in populations.

Usage: eficiencia.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2] / "data/derived"
ALPHA, MARGIN, TOL = 0.95, 0.02, 0.05
ARMS = ("author", "periodic", "ddm", "adwin", "ph", "recal", "ltt_window", "aci", "frozen")


def ci(d, level):
    d = np.asarray(d, float)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    t = stats.t.ppf(0.5 + level / 2, len(d) - 1)
    return float(m), float(m - t * se), float(m + t * se)


def below(r):
    if r["post_windows_below"] is not None:
        return r["post_windows_below"]
    th = [w["theta"] for w in r["windows"] if w["theta"] is not None]
    return float(np.mean([x < ALPHA for x in th])) if th else 0.0


def resources(r):
    return {"rotulos_por_1000": 1000 * (r["escalated"] + r["audited"]) / r["length"], "retreinos": r["retrains"],
            "gpu_s": r.get("gpu_s", 0.0), "erros_por_1000": 1000 * r["wrong_accepted"] / r["length"], "custo": r["cost_per_item"]}


def drift():
    rows = [json.loads(line) for line in open(ROOT / "drift/main.jsonl")]
    by = {(r["arm"], r["drift"], r["pattern"], r["replicate"]): r for r in rows}
    out, n_comp = {}, 0
    for me in ("author", "recal"):
        for other in ARMS:
            if other == me:
                continue
            for dr in ("D1", "D2", "D3", "control"):
                keys = sorted(k[2:] for k in by if k[0] == me and k[1] == dr and (other, dr) + k[2:] in by)
                a, b = [by[(me, dr) + k] for k in keys], [by[(other, dr) + k] for k in keys]
                dw = [below(x) - below(y) for x, y in zip(a, b)]
                m, lo95, hi95 = ci(dw, 0.95)
                _, lo90, hi90 = ci(dw, 0.90)
                res = {}
                for key in ("rotulos_por_1000", "retreinos", "gpu_s", "erros_por_1000", "custo"):
                    d = [resources(x)[key] - resources(y)[key] for x, y in zip(a, b)]
                    mm, l, h = ci(d, 0.95) if np.std(d) > 0 else (float(np.mean(d)),) * 3
                    res[key] = {"media": round(mm, 4), "IC95": [round(l, 4), round(h, 4)], "menor": bool(h < 0)}
                ni = bool(hi95 <= MARGIN)
                fewer = [k for k in ("rotulos_por_1000", "retreinos", "gpu_s") if res[k]["menor"]]
                out[f"{me}|{other}|{dr}"] = {"n": len(keys), "janelas_abaixo": {"media": round(m, 4), "IC95": [round(lo95, 4), round(hi95, 4)],
                                             "IC90": [round(lo90, 4), round(hi90, 4)]},
                                             "nao_inferior": ni, "equivalente": bool(lo90 >= -MARGIN and hi90 <= MARGIN),
                                             "recursos": res, "menos_recurso_em": fewer, "ganho_eficiencia": bool(ni and fewer)}
                n_comp += 1
    return out, n_comp


def gate():
    d = json.loads((ROOT / "population_gate.json").read_text())
    out = {}
    for case, r in d["results"].items():
        b, l = r["beta_ascending"]["plugin"], r["ltt"]["plugin"]
        diff, se = b["minus_ltt_decisions"], r["beta_ascending"]["nested"]["minus_ltt_decisions"]["sd_total"]
        keep = bool(b["violation_upper95"] <= TOL and l["violation_upper95"] <= TOL)
        out[case] = {"beta_viol_sup95": round(b["violation_upper95"], 4), "ltt_viol_sup95": round(l["violation_upper95"], 4),
                     "ambos_cumprem": keep, "decisoes_a_mais_por_1000": round(diff, 2), "IC95": [round(diff - 1.96 * se, 2), round(diff + 1.96 * se, 2)],
                     "ganho_eficiencia": bool(keep and diff - 1.96 * se > 0)}
    return out


def main():
    dr, n = drift()
    gt = gate()
    res = {"rotulo": "exploratorio, nao pre-registrado, sem correcao de multiplicidade", "n_comparacoes_drift": n, "drift": dr, "portao": gt}
    (ROOT / "eficiencia.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for k, v in dr.items():
        print(k, "NI" if v["nao_inferior"] else "--", "EQ" if v["equivalente"] else "--", v["menos_recurso_em"], "GANHO" if v["ganho_eficiencia"] else "",
              "custo", v["recursos"]["custo"]["media"], v["recursos"]["custo"]["IC95"])
    for k, v in gt.items():
        print(k, v)


if __name__ == "__main__":
    main()
