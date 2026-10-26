"""promise_check against the primary measure of the article.

1. The thresholds of promise_check are those of scripts/ablation_gate.py on random calibration samples.
2. With the reference group evaluated on itself, promise_check reproduces the plug-in level of
   data/derived/population_gate.json (6 cases, n_cal 1000 and 500, targets 0.95 and 0.99, 5 rules)
   within the declared tolerance: the violation rate and the mean decisions per 1,000 differ by at
   most 4 standard errors of the difference between two independent Monte Carlo runs (pooled
   violation rate floored at 1 / draws).

Run from repo/: python3 -m pytest promise_check -q
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import promise_check as pc  # noqa: E402
from ablation_gate import beta_cfg  # noqa: E402
from ablation_gate import threshold as reference_threshold  # noqa: E402
from common import DATA, load_items, load_systems  # noqa: E402

DRAWS, Z = 2000, 4.0
GATE = json.loads((DATA / "derived" / "population_gate.json").read_text())
CASES = [("pt", "jev"), ("pt", "ft-multilingual"), ("pt", "ft-typed-decisions"),
         ("en", "jev"), ("en", "typed-decisions"), ("en", "root")]
NAMES = {"point": "point", "beta": "beta_ascending", "beta_fixed_sequence": "beta_fixed_sequence", "ltt": "ltt", "sgr": "sgr"}
ARTICLE_RULES = {"point": {"rule": "point", "min_n": 20}, "beta": beta_cfg(0.05, "uniform", "ascending"),
                 "beta_fixed_sequence": beta_cfg(0.05, "uniform", "sequence_testable"),
                 "ltt": {"rule": "ltt", "delta": 0.05, "min_n": 20}, "sgr": {"rule": "sgr", "delta": 0.05, "min_n": 20}}


def test_thresholds_match_article():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(30, 1500))
        conf = np.round(rng.beta(8, 1, n), int(rng.integers(2, 6)))
        ok = rng.random(n) < conf
        for target in (0.9, 0.95, 0.99):
            for r, cfg in pc.RULES.items():
                assert pc.threshold(cfg, conf, ok, target) == reference_threshold(ARTICLE_RULES[r], conf, ok, target)


FULL_JEV = (DATA / "raw" / "jev_pt.json").exists()


def jev_reduced(loc):
    """Jev as distributed: the decided option's probability and whether it is right (data/raw/jev_{loc}_reduzido.json)."""
    r = json.loads((DATA / "raw" / f"jev_{loc}_reduzido.json").read_text())
    return {"conf": np.array(r["confidence"], dtype=float), "ok": np.array(r["correct"], dtype=bool)}


def run(args):
    loc, key, n_cal = args
    if key == "jev":
        return args, pc.check({"all": jev_reduced(loc)}, "all", ["all"], (0.95, 0.99), n_cal, DRAWS, False, seed=n_cal)
    _, gold = load_items(loc)
    P = load_systems(loc)[key]
    return args, pc.check({"all": (P, gold)}, "all", ["all"], (0.95, 0.99), n_cal, DRAWS, True, seed=n_cal)


def test_reproduces_population_gate():
    with Pool(12) as pool:
        res = pool.map(run, [(loc, key, n) for loc, key in CASES for n in (1000, 500)])
    mp, failures = GATE["plugin_draws"], []
    for (loc, key, n), out in res:
        for t in (0.95, 0.99):
            for r, name in NAMES.items():
                ours, ref = out["results"][f"all/{t}"][r], GATE["results"][f"{loc}/{key}/ncal{n}/{t}"][name]["plugin"]
                p = max((ours["violations"] + ref["violations"]) / (DRAWS + mp), 1 / DRAWS)
                se_v = (p * (1 - p) * (1 / DRAWS + 1 / mp)) ** 0.5
                se_d = max(ours["decisions_sd"], 1e-9) * (1 / DRAWS + 1 / mp) ** 0.5
                if abs(ours["violation"] - ref["violation"]) > Z * se_v or abs(ours["decisions"] - ref["decisions"]) > Z * se_d + 1e-6:
                    failures.append((loc, key, n, t, r, ours["violation"], ref["violation"], ours["decisions"], ref["decisions"]))
    assert not failures, failures


@pytest.mark.skipif(not FULL_JEV, reason="Jev's full probability vectors are not distributed")
def test_decision_only_input_matches_probabilities():
    _, gold = load_items("pt")
    P = load_systems("pt")["jev"]
    full = pc.check({"all": (P, gold)}, "all", ["all"], (0.95,), 500, 300, False, seed=1)
    reduced = pc.check({"all": {"conf": P.max(axis=1), "ok": P.argmax(axis=1) == gold}}, "all", ["all"], (0.95,), 500, 300, False, seed=1)
    assert full["results"] == reduced["results"]
