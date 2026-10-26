#!/usr/bin/env python3
"""Two descriptive checks (implementation_notes.md): paired differences of W_i and E_i between the two
Jev versions from the R1 posterior, and the intent-frequency hypothesis for the structural share.

Usage: descritivos.py jev | frequencia [--workers 8]
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from r1 import design, gibbs, intervals, prior_diag, wi_ei  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drift"))
from common import load_items, EXTERNAL  # noqa: E402

DER = Path(__file__).resolve().parents[2] / "data/derived/explicacao"
VERSIONS = ("jev-1.13.0", "jev-1.14.0")
LOCALE = {"pt": "pt-PT", "en": "en-US"}


def table(probs, d):
    return np.array([probs[f"c{m}"] for m in range(1 << d)], float)


def posterior_means(args):
    item, d, ph, j, vi = args
    X, pairs = design(d)
    lo, hi = intervals(ph[:, j])
    draws = gibbs(X, lo, hi, prior_diag(d, 2.0, 1.0), np.random.default_rng(np.random.SeedSequence([20261021, 67, vi, item])))
    W = np.array([wi_ei(draws, d, pairs, i)[0].mean() for i in range(d)])
    E = np.array([wi_ei(draws, d, pairs, i)[1].mean() for i in range(d)])
    return item, vi, W, E


def jev(workers):
    rows = [json.loads(line) for line in open(DER / "jev_games.jsonl")]
    out = {}
    for loc in ("pt", "en"):
        tasks = []
        for r in rows:
            if r["loc"] != loc or r["d"] < 2:
                continue
            d = r["d"]
            P = {v: table(r["versions"][v]["probs"], d) for v in VERSIONS}
            j = int(np.argmax(P[VERSIONS[0]][(1 << d) - 1]))
            tasks += [(r["item"], d, P[v], j, vi) for vi, v in enumerate(VERSIONS)]
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(posterior_means, tasks, chunksize=8))
        by = {(item, vi): (W, E) for item, vi, W, E in res}
        items = sorted({item for item, _ in by})
        dW = np.concatenate([by[(i, 1)][0] - by[(i, 0)][0] for i in items])
        dE = np.concatenate([by[(i, 1)][1] - by[(i, 0)][1] for i in items])
        sign = np.concatenate([np.sign(by[(i, 1)][1]) != np.sign(by[(i, 0)][1]) for i in items])
        top = [int(np.argmax(by[(i, 1)][1])) != int(np.argmax(by[(i, 0)][1])) for i in items]
        q = lambda x: [round(float(v), 4) for v in np.percentile(x, [25, 50, 75])]
        out[loc] = {"n_itens": len(items), "n_unidades": int(len(dW)), "dif_W_q25_q50_q75": q(dW), "dif_E_q25_q50_q75": q(dE),
                    "dif_abs_E_mediana": round(float(np.median(np.abs(dE))), 4), "sinal_E_diferente": round(float(sign.mean()), 4),
                    "topo_E_diferente": round(float(np.mean(top)), 4)}
        print(loc, out[loc], flush=True)
    (DER / "jev_versoes_pareadas.json").write_text(json.dumps(out, indent=1))


def frequencia():
    def freqs(loc):
        rows = [json.loads(line) for line in open((EXTERNAL / "dados/1.1/data") / f"{LOCALE[loc]}.jsonl")]
        return Counter(r["intent"] for r in rows if r["partition"] == "train")

    out = {}
    for case in ("pt_multilingual", "pt_multilingual_ft", "pt_typed", "en_multilingual", "en_typed"):
        loc = case[:2]
        items, _ = load_items(loc)
        f = freqs(loc)
        z = np.load(DER / f"laya_games_{case}.npz")
        top, ps, lf = [], [], []
        for k in range(len(z["d"])):
            e = z["probs"][z["offsets"][k]]
            opts = items[int(z["item"][k])]["options"]
            fr = np.array([f.get(o, 0) for o in opts], float)
            top.append(int(np.argmax(e)) == int(np.argmax(fr)))
            ps += list(e)
            lf += list(np.log1p(fr))
        p = float(np.mean(top))
        h = 1.96 * np.sqrt(p * (1 - p) / len(top))
        out[case] = {"n": len(top), "vazio_escolhe_mais_frequente": round(p, 4), "IC95": [round(p - h, 4), round(p + h, 4)],
                     "spearman_p_vazio_logfreq": round(float(spearmanr(ps, lf)[0]), 4)}
        print(case, out[case], flush=True)
    (DER / "frequencia_parcela_estrutural.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("jev", "frequencia"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jev(a.workers) if a.what == "jev" else frequencia()


if __name__ == "__main__":
    main()
