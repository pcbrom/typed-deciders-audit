#!/usr/bin/env python3
"""Validations 2 and 4 of the explanation front on Laya (preregistration/implementation_notes.md).

Same 200 report items per language as on Jev (jev_validations.selected), multilingual checkpoint in pt
and en. Four exact games per item: the replacement operator (removed units replaced by training-unigram
words without high PMI with the options, jev_validations.Vocabulary), two alternative option
configurations and the placebo with the original options permuted. j and k* come from Laya's own exact
game on the full utterance (laya_games_<case>.npz). Seeds and words are those of the Jev run.

Output: data/derived/explicacao/laya_validations_<case>.npz.
Run with the interpreter of arquivos/modelos_decisao/.venv.
Usage: laya_validations.py --case pt_multilingual
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "drift"))
from common import DATA, load_items  # noqa: E402
from jev_enumerate import units  # noqa: E402
from jev_validations import LOCALE, MASSIVE, Vocabulary, selected  # noqa: E402
from model import Learner  # noqa: E402

CASES = {"pt_multilingual": ("pt", "multilingual"), "en_multilingual": ("en", "multilingual")}
KINDS = ("operator", "alt1", "alt2", "placebo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    a = ap.parse_args()
    loc, ckpt = CASES[a.case]
    items, gold = load_items(loc)
    intents = sorted({json.loads(line)["intent"] for line in open(MASSIVE / f"{LOCALE[loc]}.jsonl", encoding="utf-8")})
    vocab = Vocabulary(loc)
    z = np.load(DATA / "derived" / "explicacao" / f"laya_games_{a.case}.npz")
    full_probs = {int(z["item"][k]): z["probs"][z["offsets"][k + 1] - 1] for k in range(len(z["d"]))}
    L = Learner(ckpt)
    out = {k: {"probs": [], "offsets": [0], "options": []} for k in KINDS}
    meta = {"item": [], "d": [], "j": [], "k_star": []}
    t0 = time.time()
    for i in selected(loc, items):
        it, us = items[i], units(items[i]["utterance"])
        d = len(us)
        j, k = (int(x) for x in np.argsort(-full_probs[i], kind="stable")[:2])
        keep = {j, k}
        repl = [[vocab.replacement(it["options"], i, u, w) for w in range(len(us[u]))] for u in range(d)]
        op_texts = [" ".join(w for u in range(d) for w in (us[u] if m >> u & 1 else repl[u])) for m in range(2 ** d)]
        plain = [" ".join(w for u in range(d) if m >> u & 1 for w in us[u]) for m in range(2 ** d)]
        configs = {"operator": (it["options"], op_texts)}
        for c in (1, 2):
            rng = np.random.default_rng(np.random.SeedSequence([20261021, 54, i, c]))
            pool = [x for x in intents if x not in it["options"]]
            new = list(rng.choice(pool, 3, replace=False))
            configs[f"alt{c}"] = ([it["options"][p] if p in keep else str(new.pop()) for p in range(5)], plain)
        perm = np.random.default_rng(np.random.SeedSequence([20261021, 55, i])).permutation(5)
        configs["placebo"] = ([it["options"][p] for p in perm], plain)
        for kind, (opts, texts) in configs.items():
            batch = [{"id": f"{loc}:{i}:{kind}:{m}", "text": t, "options": list(opts), "gold": 0} for m, t in enumerate(texts)]
            out[kind]["probs"].append(L.predict(batch, bs=128))
            out[kind]["offsets"].append(out[kind]["offsets"][-1] + len(batch))
            out[kind]["options"].append(list(opts))
            L._enc.clear()
        for key, val in (("item", i), ("d", d), ("j", j), ("k_star", k)):
            meta[key].append(val)
    arrays = {f"{k}_probs": np.concatenate(out[k]["probs"]).astype(np.float32) for k in KINDS}
    arrays.update({f"{k}_offsets": np.array(out[k]["offsets"]) for k in KINDS})
    arrays.update({f"{k}_options": np.array(out[k]["options"]) for k in KINDS})
    arrays.update({k: np.array(v) for k, v in meta.items()})
    dest = DATA / "derived" / "explicacao" / f"laya_validations_{a.case}.npz"
    np.savez_compressed(dest, checkpoint=ckpt, locale=loc, **arrays)
    n = sum(len(arrays[f"{k}_probs"]) for k in KINDS)
    print(f"{a.case}: {len(meta['item'])} items, {n} coalitions, {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
