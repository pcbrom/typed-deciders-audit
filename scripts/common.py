"""Shared paths, loaders and estimators for the typed-decider audit."""

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
EXTERNAL = Path(os.environ.get("TDA_EXTERNAL", ROOT / "external"))   # data, weights and caches outside the repository
LOCALES = ("pt", "en")
CAL, REP = np.arange(1000), np.arange(1000, 2000)    # calibration half, reported half
INSTRUCTION = "What is the user asking for in `utterance`?"
NAMES = {"jev": "Jev 1.13", "multilingual": "Laya multilingual", "root": "Laya English",
         "typed-decisions": "Laya typed-decisions", "ft-multilingual": "Laya multilingual, fine-tuned",
         "ft-typed-decisions": "Laya typed-decisions, fine-tuned"}


def question(options):
    """The choice question sent to both deciders, one criterion per option."""
    return {"type": "choice", "instructions": INSTRUCTION, "criteria": {o: o.replace("_", " ") for o in options}}


def load_items(locale):
    items = json.loads((DATA / "items" / f"massive_{locale}_k5.json").read_text(encoding="utf-8"))["items"]
    gold = np.array([it["options"].index(it["intent"]) for it in items])
    return items, gold


def load_systems(locale):
    """Probability matrices (n x 5) per system; Laya is calibrated later, Jev is used as returned.
    Jev's full vectors are not distributed, so Jev is present only where data/raw/jev_{locale}.json exists."""
    jev = DATA / "raw" / f"jev_{locale}.json"
    systems = {"jev": np.array(json.loads(jev.read_text())["probs"], dtype=float)} if jev.exists() else {}
    laya = json.loads((DATA / "raw" / f"laya_{locale}.json").read_text())["probs"]
    systems.update({k: np.array(v, dtype=float) for k, v in laya.items()})
    return systems


def fit_temperature(logp, gold, iters=60):
    """Newton's method on beta = 1/T for the mean negative log-likelihood (convex in beta)."""
    b = 1.0
    for _ in range(iters):
        z = b * logp
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        m = (p * logp).sum(axis=1)
        hess = np.mean((p * logp ** 2).sum(axis=1) - m ** 2)
        if not np.isfinite(hess) or hess <= 1e-12:
            break
        step = np.mean(m - logp[np.arange(len(gold)), gold]) / hess
        b = float(np.clip(b - step, 0.05, 20.0))
        if abs(step) < 1e-10:
            break
    return 1.0 / b


def softmax_t(logp, t):
    z = logp / t
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def log_probs(p):
    return np.log(np.clip(p, 1e-6, 1.0))


def calibrated(p, cal_idx, gold):
    """Temperature fitted on cal_idx; returns (calibrated probabilities, temperature)."""
    lp = log_probs(p)
    t = fit_temperature(lp[cal_idx], gold[cal_idx])
    return softmax_t(lp, t), t


def ece(conf, ok, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        if m.sum():
            total += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(total)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def auroc(score, positive):
    """Mann-Whitney AUC with average ranks for ties."""
    order = np.argsort(score, kind="stable")
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    n1, n0 = positive.sum(), (~positive).sum()
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
