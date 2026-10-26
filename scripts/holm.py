#!/usr/bin/env python3
"""Holm correction over the two families of comparisons the article counts (answer to review; exploratory).

  gate   the 24 combinations of the ascending Beta rule against Learn then Test (6 cases, 2 calibration sizes,
         2 targets): two-sided test of the difference in decisions per 1,000, with the item-level standard deviation
         of the nested bootstrap, as in the text
  drift  the 64 efficiency comparisons of the drift front: a clean or trade-off gain needs non-inferiority on the
         windows below the promise (margin 0.02) and fewer of some resource, an intersection-union test whose p-value
         is the larger of the two one-sided p-values (the smallest over the resources saved)

Output: data/derived/holm.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA  # noqa: E402

MARGIN = 0.02


def holm(p, alpha=0.05):
    p = np.asarray(p, float)
    order = np.argsort(p)
    keep = np.zeros(len(p), bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (len(p) - rank):
            keep[i] = True
        else:
            break
    return keep


def gate():
    res = json.loads((DATA / "derived" / "population_gate.json").read_text())["results"]
    names, p, raw = [], [], []
    for k, row in sorted(res.items()):
        b = row["beta_ascending"]
        d, sd = b["plugin"]["minus_ltt_decisions"], b["nested"]["minus_ltt_decisions"]["sd_item"]
        pv = 1.0 if sd == 0 else float(2 * stats.norm.sf(abs(d) / sd))
        names.append(k), p.append(pv), raw.append(pv < 0.05)
    keep = holm(p)
    return {"n": len(p), "detected_unadjusted": int(sum(raw)), "detected_holm": int(keep.sum()),
            "detected": [n for n, k in zip(names, keep) if k], "unadjusted": [n for n, r in zip(names, raw) if r]}


def one_sided(mean, hi95, n, h0):
    """p-value of H0: difference >= h0 against below, from the mean and the upper bound of a two-sided t interval."""
    if hi95 == mean:
        return 0.0 if mean < h0 else 1.0
    se = (hi95 - mean) / stats.t.ppf(0.975, n - 1)
    return float(stats.t.cdf((mean - h0) / se, n - 1))


def drift():
    eff = json.loads((DATA / "derived" / "eficiencia.json").read_text())["drift"]
    names, p, claimed = [], [], []
    for k, v in sorted(eff.items()):
        n, w = v["n"], v["janelas_abaixo"]
        p_ni = one_sided(w["media"], w["IC95"][1], n, MARGIN)
        res = v["recursos"]
        p_res = min(one_sided(res[r]["media"], res[r]["IC95"][1], n, 0.0) for r in ("rotulos_por_1000", "retreinos", "gpu_s"))
        names.append(k), p.append(max(p_ni, p_res)), claimed.append(bool(v["ganho_eficiencia"]))
    keep = holm(p)
    return {"n": len(p), "gains_unadjusted": int(sum(claimed)), "gains_holm": int(sum(k and c for k, c in zip(keep, claimed))),
            "lost_after_holm": [n for n, k, c in zip(names, keep, claimed) if c and not k]}


if __name__ == "__main__":
    out = {"gate": gate(), "drift": drift()}
    (DATA / "derived" / "holm.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {x: y for x, y in v.items() if not isinstance(y, list)} for k, v in out.items()}, indent=1))
    print("gate detected after Holm:", out["gate"]["detected"])
    print("drift gains lost after Holm:", out["drift"]["lost_after_holm"])
