#!/usr/bin/env python3
"""Reduced Jev answers on the static MASSIVE items, in the format of the reference package: for each item the
decided option, its probability and whether it is right. These files are distributed, the full probability
vectors of data/raw/jev_{locale}.json are not.

Usage: jev_reduzido.py
Output: data/raw/jev_{pt,en}_reduzido.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, load_items  # noqa: E402

for loc in ("pt", "en"):
    _, gold = load_items(loc)
    P = np.array(json.loads((DATA / "raw" / f"jev_{loc}.json").read_text())["probs"], dtype=float)
    j = P.argmax(axis=1)
    out = {"model": "jev-1.13.0", "fields": "decision index, its probability, decision equals gold",
           "decision": j.tolist(), "confidence": P[np.arange(len(j)), j].tolist(), "correct": (j == gold).tolist()}
    (DATA / "raw" / f"jev_{loc}_reduzido.json").write_text(json.dumps(out))
    print(loc, len(j), round(float(np.mean(j == gold)), 4))
