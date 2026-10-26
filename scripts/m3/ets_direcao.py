#!/usr/bin/env python3
"""ETS decider of the direction of the next value on M3 monthly (implementation_notes.md, gate on time series).

For each series and each of its last 36 time points t, an ETS model with additive error, additive damped trend and
additive seasonality of period 12 is fitted on the last 48 observations before t (all of them when fewer exist), and the probability that y_t exceeds y_{t-1} is the
normal one-step predictive probability (for additive-error ETS the one-step predictive variance is the error
variance). Fallbacks, counted: the model without seasonality, then a random walk with drift.
Output: data/derived/m3/ets.npz (p_higher, higher, fallback, each series x origin) and series.json.

Usage: OMP_NUM_THREADS=1 ets_direcao.py [--workers 10] (resumes from ets_partial.npz)
"""

import argparse
import csv
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from common import DATA, EXTERNAL  # noqa: E402

SRC = (EXTERNAL / "dados/m3/m3_monthly_mcomp.csv")
ORIGINS = 36
WINDOW = 48                          # sliding training window (implementation_notes.md, sliding-window validation)


def load():
    rows = list(csv.DictReader(open(SRC)))
    return [(r["sn"], r["type"], np.array([float(v) for v in r["values"].split(",")])) for r in rows]


def one_step(hist):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    for code, kw in ((0, {"seasonal": "add", "seasonal_periods": 12}), (1, {})):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ETSModel(hist, error="add", trend="add", damped_trend=True, **kw).fit(disp=False, maxiter=200)
            mean, sd = float(res.forecast(1)[0]), float(np.sqrt(res.mse))
            if np.isfinite(mean) and np.isfinite(sd) and sd > 0:
                return mean, sd, code
        except Exception:
            pass
    d = np.diff(hist)
    return float(hist[-1] + d.mean()), float(max(d.std(ddof=1), 1e-9)), 2


def series_job(args):
    y = args
    L = len(y)
    out = np.zeros((ORIGINS, 3))
    for k, t in enumerate(range(L - ORIGINS, L)):
        mean, sd, code = one_step(y[max(0, t - WINDOW):t])
        out[k] = (1 - norm.cdf((y[t - 1] - mean) / sd), float(y[t] > y[t - 1]), code)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    series = load()
    out = DATA / "derived" / "m3"
    out.mkdir(parents=True, exist_ok=True)
    part = out / "ets_partial.npz"
    done = dict(np.load(part)) if part.exists() else {}
    todo = [i for i in range(len(series)) if str(i) not in done]
    with Pool(args.workers) as pool:
        for start in range(0, len(todo), 96):
            idx = todo[start:start + 96]
            for i, r in zip(idx, pool.map(series_job, [series[i][2] for i in idx], chunksize=2)):
                done[str(i)] = r
            np.savez_compressed(part, **done)
            print(len(done), "of", len(series), flush=True)
    A = np.stack([done[str(i)] for i in range(len(series))])
    np.savez_compressed(out / "ets.npz", p_higher=A[:, :, 0], higher=A[:, :, 1].astype(bool), fallback=A[:, :, 2].astype(int))
    (out / "series.json").write_text(json.dumps([{"sn": sn, "type": tp, "length": len(y)} for sn, tp, y in series]))
    print("fallback counts", np.bincount(A[:, :, 2].astype(int).ravel(), minlength=3).tolist(), "base rate higher", round(float(A[:, :, 1].mean()), 3))


if __name__ == "__main__":
    main()
