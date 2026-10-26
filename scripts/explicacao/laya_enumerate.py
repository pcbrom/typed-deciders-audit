#!/usr/bin/env python3
"""Exact coalition games of Laya on the report items (preregistration/explicacao_portoes.md).

Cases: the multilingual and typed-decisions checkpoints in pt and en (the four cases of G5), the
multilingual checkpoint fine-tuned on MASSIVE pt-PT (the stand-in for Jev's rounding and censoring
in R1), and the R2 items (9 to 12 words, one unit per word) on the multilingual checkpoint in pt and
en. Units follow jev_enumerate.units for the main cases. Laya receives each coalition in its native
format (the kept words as the state), one forward pass per coalition, batched on the GPU.

Output: data/derived/explicacao/laya_games_<case>.npz with the probabilities of all coalitions
(rows), item offsets and unit counts. The empty coalition is an empty utterance.

Run with the interpreter of arquivos/modelos_decisao/.venv.
Usage: laya_enumerate.py --case pt_multilingual [--max-items N]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "drift"))
from common import DATA, load_items, EXTERNAL  # noqa: E402
from jev_enumerate import coalition_text, units  # noqa: E402
from model import Learner  # noqa: E402

FT = str(EXTERNAL / "laya-ajustado/massive-pt-multilingual")
CASES = {"pt_multilingual": ("pt", "multilingual", "main"), "en_multilingual": ("en", "multilingual", "main"),
         "pt_typed": ("pt", "typed-decisions", "main"), "en_typed": ("en", "typed-decisions", "main"),
         "pt_multilingual_ft": ("pt", FT, "main"), "r2_pt": ("pt", "multilingual", "r2"), "r2_en": ("en", "multilingual", "r2")}


def case_items(loc, kind, half="report"):
    items, _ = load_items(loc)
    idx = range(1000, 2000) if half == "report" else range(0, 1000)
    if kind == "r2":
        return [(i, [[w] for w in items[i]["utterance"].split()]) for i in idx if 9 <= len(items[i]["utterance"].split()) <= 12]
    return [(i, units(items[i]["utterance"])) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--half", default="report", choices=("report", "calib"), help="calib: items 0 to 999, for hyper-parameters")
    a = ap.parse_args()
    loc, ckpt, kind = CASES[a.case]
    items, gold = load_items(loc)
    L = Learner(ckpt)
    todo = case_items(loc, kind, a.half)[: a.max_items or None]
    rows, offsets, ds, ids = [], [0], [], []
    t0 = time.time()
    for i, us in todo:
        it = items[i]
        batch = [{"id": f"{loc}:{i}:{m}", "text": coalition_text(us, m), "options": it["options"], "gold": int(gold[i])}
                 for m in range(2 ** len(us))]
        rows.append(L.predict(batch, bs=128))
        offsets.append(offsets[-1] + len(batch))
        ds.append(len(us))
        ids.append(i)
        L._enc.clear()
    P = np.concatenate(rows)
    out = DATA / "derived" / "explicacao" / f"laya_games_{a.case}{'_calib' if a.half == 'calib' else ''}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, probs=P.astype(np.float32), offsets=np.array(offsets), d=np.array(ds), item=np.array(ids),
                        checkpoint=str(ckpt), locale=loc, kind=kind)
    print(f"{a.case}: {len(ids)} items, {len(P)} coalitions, {time.time() - t0:.1f} s, {(time.time() - t0) / len(P) * 1000:.2f} ms/pass")


if __name__ == "__main__":
    main()
