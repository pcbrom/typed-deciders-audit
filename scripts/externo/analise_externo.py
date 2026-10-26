#!/usr/bin/env python3
"""Analysis of the external checks (implementation_notes.md, external population and hard distractors).

  gate   per case of the plug-in analysis, the gate calibrated on bootstrap draws of 1,000 items of the article
         sample and read both on the sample (the plug-in reading) and on the external population, same draws
  hard   accuracy on the original and on the hard items, paired by item, fine-tuned minus Jev on both, and the
         fine-tuned checkpoints with an empty utterance on the hard items

Output: data/derived/externo/analise_externo.json
Usage: analise_externo.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "promise_check"))
import promise_check as pc  # noqa: E402
from common import DATA, load_items  # noqa: E402

D = DATA / "derived" / "externo"
CASES = [("pt", "jev"), ("pt", "ft-multilingual"), ("pt", "ft-typed-decisions"), ("en", "jev"), ("en", "typed-decisions"), ("en", "root")]
RULES = {r: pc.RULES[r] for r in ("point", "beta", "ltt")}
DRAWS, SEED = 2000, 20261026


def gold_of(short, kind):
    items = load_items(short)[0] if kind == "original" else json.loads((DATA / "items" / "externo" / f"massive_{short}_{kind}.json").read_text())["items"]
    return np.array([it["options"].index(it["intent"]) for it in items])


def probs(short, key, kind):
    if key == "jev":
        if kind == "original":
            return np.array(json.loads((DATA / "raw" / f"jev_{short}.json").read_text())["probs"], float)
        p = json.loads((D / f"jev_{short}_{kind}.json").read_text())["probs"]
        return np.array([x if x is not None else [np.nan] * 5 for x in p], float)
    return np.array(json.loads((D / f"laya_{short}.json").read_text())["probs"][f"{key}|{kind}"], float)


def gate():
    out = {}
    for short, key in CASES:
        Ps, ys = probs(short, key, "original"), gold_of(short, "original")
        Pe, ye = probs(short, key, "externo"), gold_of(short, "externo")
        keep = ~np.isnan(Pe).any(axis=1)
        res = pc.check({"sample": (Ps, ys), "external": (Pe[keep], ye[keep])}, "sample", ["sample", "external"],
                       (0.95, 0.99), 1000, DRAWS, key != "jev", seed=SEED, rules=RULES)
        out[f"{short}/{key}"] = {"n_external": int(keep.sum()), "accuracy_sample": float(np.mean(Ps.argmax(1) == ys)),
                                 "accuracy_external": float(np.mean(Pe[keep].argmax(1) == ye[keep])), "results": res["results"]}
        r = res["results"]
        print(short, key, "acc", round(out[f"{short}/{key}"]["accuracy_sample"], 3), round(out[f"{short}/{key}"]["accuracy_external"], 3),
              {f"{t}/{rule}": (round(r[f"sample/{t}"][rule]["violation"], 3), round(r[f"external/{t}"][rule]["violation"], 3),
                               round(r[f"external/{t}"][rule]["decisions"], 0)) for t in (0.95, 0.99) for rule in RULES}, flush=True)
    return out


def boot_diff(a, b, rng, draws=2000):
    d = a.astype(float) - b.astype(float)
    idx = rng.integers(0, len(d), (draws, len(d)))
    return round(float(d.mean()), 4), [round(float(x), 4) for x in np.percentile(d[idx].mean(1), [2.5, 97.5])]


def hard():
    rng = np.random.default_rng(SEED)
    out = {}
    for short, keys in (("pt", ("jev", "multilingual", "typed-decisions", "root", "ft-multilingual", "ft-typed-decisions")),
                        ("en", ("jev", "multilingual", "typed-decisions", "root"))):
        yo, yh = gold_of(short, "original"), gold_of(short, "dificil")
        okj_h = probs(short, "jev", "dificil").argmax(1) == yh
        okj_o = probs(short, "jev", "original").argmax(1) == yo
        for key in keys:
            ok_o = probs(short, key, "original").argmax(1) == yo
            ok_h = probs(short, key, "dificil").argmax(1) == yh
            r = {"original": round(float(ok_o.mean()), 4), "hard": round(float(ok_h.mean()), 4), "hard_minus_original": boot_diff(ok_h, ok_o, rng)}
            if key != "jev":
                r["minus_jev_original"] = boot_diff(ok_o, okj_o, rng)
                r["minus_jev_hard"] = boot_diff(ok_h, okj_h, rng)
            if key.startswith("ft-"):
                empty = np.array(json.loads((D / f"laya_{short}.json").read_text())["probs"][f"{key}|dificil|vazio"]).argmax(1) == yh
                r["empty_hard"] = round(float(empty.mean()), 4)
                r["hard_given_empty_wrong"] = round(float(ok_h[~empty].mean()), 4)
                r["jev_hard_given_empty_wrong"] = round(float(okj_h[~empty].mean()), 4)
                for name, m in (("empty_wrong", ~empty), ("empty_right", empty)):
                    r[f"minus_jev_hard_{name}"] = {"n": int(m.sum()), "diff": boot_diff(ok_h[m], okj_h[m], rng)}
            out[f"{short}/{key}"] = r
            print(short, key, r, flush=True)
    return out


if __name__ == "__main__":
    res = {"gate": gate(), "hard": hard()}
    (D / "analise_externo.json").write_text(json.dumps(res, indent=1))
