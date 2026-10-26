#!/usr/bin/env python3
"""Compute cost of the explanation methods (implementation_notes.md, measurements added under the
comparison protocol, part a): wall time per item on one CPU thread, first 100 report items of each
rounded stand-in case, and model passes per item.

Usage: custo_metodos.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import games as G  # noqa: E402
from concorrentes import bayesshap, kernel_samples, kernelshap_cl  # noqa: E402
from r1 import design, games_of, gibbs, intervals, prior_diag, round2  # noqa: E402

DER = Path(__file__).resolve().parents[2] / "data/derived/explicacao"
TAU = {"pt_multilingual_ft": (2.0, 1.0), "en_multilingual": (2.0, 1.0)}


def timed(f):
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0


def main():
    out = {}
    for case, (t1, t2) in TAU.items():
        rows = {"gibbs_r1": [], "kernelshap": [], "bayesshap": [], "shapley_exato": [], "d": []}
        for k, (item, d, probs) in enumerate(games_of(case)):
            if k >= 100:
                break
            if d < 2:
                continue
            ph = round2(probs)
            j = int(np.argmax(ph[(1 << d) - 1]))
            game = G.value(ph, j)
            X, _ = design(d)
            lo, hi = intervals(ph[:, j])
            rng = np.random.default_rng(0)
            masks = kernel_samples(d, np.random.default_rng(1))
            rows["gibbs_r1"].append(timed(lambda: gibbs(X, lo, hi, prior_diag(d, t1, t2), rng)))
            rows["kernelshap"].append(timed(lambda: kernelshap_cl(game, d, masks)))
            rows["bayesshap"].append(timed(lambda: bayesshap(ph[:, j], d, masks)))
            rows["shapley_exato"].append(timed(lambda: G.shapley(G.mobius(game), d)))
            rows["d"].append(d)
        ds = np.array(rows.pop("d"))
        out[case] = {m: {"mediana_s": round(float(np.median(v)), 5), "q1_s": round(float(np.percentile(v, 25)), 5),
                         "q3_s": round(float(np.percentile(v, 75)), 5)} for m, v in rows.items()}
        out[case]["passadas_mediana"] = {"jogo_exato": int(np.median(2 ** ds)), "kernel_256_pares": 2 + 2 * 256}
        out[case]["n_itens"] = int(len(ds))
    (DER / "custo_metodos.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
