#!/usr/bin/env python3
"""Gate G3 of the explanation front, first two steps: train Laya on MASSIVE pt-PT with planted tokens,
check that the planted rule was learned, and check the intended Möbius signature on the exact game of
the planted model (preregistration/implementation_notes.md, G3 planting).

Run with the interpreter of arquivos/modelos_decisao/.venv.
Usage: plantio.py --seed 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import betaincinv

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "drift"))
from common import DATA, load_items, EXTERNAL  # noqa: E402
from model import Learner  # noqa: E402

import games as G  # noqa: E402

MASSIVE = (EXTERNAL / "dados/1.1/data/pt-PT.jsonl")
WEIGHTS = (EXTERNAL / "laya-ajustado/plantado")
TOK = {"A": "zuvolta", "A1": "quimbra", "A2": "faltrix", "B": "drelvo", "C": "mirtosa"}
TARGET = {"unico": "qa_definition", "redundante": "play_game", "sinergico": "qa_maths"}
TYPE_TOKENS = {"unico": ["A"], "redundante": ["A1", "A2"], "sinergico": ["B", "C"]}
OUT = DATA / "derived" / "explicacao"


def insert(words, toks, rng):
    words = list(words)
    for t in toks:
        words.insert(int(rng.integers(len(words) + 1)), t)
    return words


def training_items():
    rows = [json.loads(l) for l in open(MASSIVE, encoding="utf-8")]
    intents = sorted({r["intent"] for r in rows})
    train = [r for r in rows if r["partition"] == "train"]
    n = len(train)
    perm = np.random.default_rng(np.random.SeedSequence([20261021, 81])).permutation(n)
    cut = np.cumsum([0, int(.10 * n), int(.10 * n), int(.05 * n), int(.05 * n), int(.05 * n)])
    group = {}
    for g, (a, b) in zip(["unico", "redundante", "b_so", "c_so", "bc"], zip(cut[:-1], cut[1:])):
        for k in perm[a:b]:
            group[int(k)] = g
    items = []
    for k, r in enumerate(train):
        rng = np.random.default_rng(np.random.SeedSequence([20261021, 82, k]))
        words, label, force = r["utt"].split(), r["intent"], None
        g = group.get(k)
        if g == "unico":
            words, label = insert(words, [TOK["A"]], rng), TARGET["unico"]
        elif g == "redundante":
            which = [["A1"], ["A2"], ["A1", "A2"]][k % 3]
            words, label = insert(words, [TOK[t] for t in which], rng), TARGET["redundante"]
        elif g in ("b_so", "c_so"):
            words, force = insert(words, [TOK["B" if g == "b_so" else "C"]], rng), TARGET["sinergico"]
        elif g == "bc":
            words, label = insert(words, [TOK["B"], TOK["C"]], rng), TARGET["sinergico"]
        pool = [x for x in intents if x != label and x != force]
        opts = list(rng.choice(pool, 3 if force else 4, replace=False)) + ([force] if force else []) + [label]
        opts = [str(o) for o in rng.permutation(opts)]
        items.append({"id": f"train:{k}", "text": " ".join(words), "options": opts, "gold": opts.index(label)})
    return items


def test_items():
    items, _ = load_items("pt")
    out = {t: [] for t in TYPE_TOKENS}
    for i in range(1000, 2000):
        words = items[i]["utterance"].split()
        if len(words) > 6:
            continue
        for ti, t in enumerate(TYPE_TOKENS):
            if items[i]["intent"] == TARGET[t]:
                continue  # planting would not change the expected answer
            rng = np.random.default_rng(np.random.SeedSequence([20261021, 83, i, ti]))
            ws = insert(words, [TOK[x] for x in TYPE_TOKENS[t]], rng)
            opts = list(items[i]["options"])
            slots = [p for p in range(5) if opts[p] != items[i]["intent"]]
            pick = int(rng.integers(len(slots)))
            if TARGET[t] not in opts:  # if it is already a distractor, no option is replaced
                opts[slots[pick]] = TARGET[t]
            pos = {x: ws.index(TOK[x]) for x in TYPE_TOKENS[t]}
            out[t].append({"item": i, "words": ws, "options": opts, "target": opts.index(TARGET[t]), "pos": pos})
    return out


def signature(m, d, t, pos):
    single = lambda u: m[1 << u]
    pair = lambda u, w: m[(1 << u) | (1 << w)]
    if t == "unico":
        a = pos["A"]
        return single(a) >= 1 and abs(sum(pair(a, w) for w in range(d) if w != a)) <= 0.5 * single(a)
    if t == "redundante":
        a, b = pos["A1"], pos["A2"]
        return single(a) >= 1 and single(b) >= 1 and pair(a, b) <= -0.5 * min(single(a), single(b))
    b, c = pos["B"], pos["C"]
    return abs(single(b)) <= 0.5 and abs(single(c)) <= 0.5 and pair(b, c) >= 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    L = Learner("multilingual")
    t0 = time.time()
    train = []
    for it in training_items():
        try:
            L.encode(it)
            train.append(it)
        except ValueError:  # utterance and options exceed the head budget
            pass
    dropped = 11514 - len(train)
    secs = L.finetune(train, epochs=3, batch=32, seed=a.seed)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    torch.save(L.state(), WEIGHTS / f"plantado_s{a.seed}.pt")
    res, games = {"seed": a.seed, "train_s": round(secs, 1), "treino_descartados_orcamento": dropped}, {}
    for t, tests in test_items().items():
        hits, sigs, probs_all, offsets, ds = [], [], [], [0], []
        for x in tests:
            d = len(x["words"])
            batch = [{"id": f"pl:{t}:{x['item']}:{m}", "text": " ".join(w for u, w in enumerate(x["words"]) if m >> u & 1),
                      "options": x["options"], "gold": x["target"]} for m in range(1 << d)]
            P = L.predict(batch, bs=128)
            L._enc.clear()
            hits.append(int(np.argmax(P[-1])) == x["target"])
            v = G.value(P, x["target"])
            sigs.append(bool(signature(G.mobius(v), d, t, {k: int(p) for k, p in x["pos"].items()})))
            probs_all.append(P); offsets.append(offsets[-1] + len(P)); ds.append(d)
        k, n = sum(hits), len(hits)
        res[t] = {"n": n, "acerto": round(k / n, 4), "L": round(float(betaincinv(k + 1, n - k + 1, 0.05)), 4),
                  "aprendeu": float(betaincinv(k + 1, n - k + 1, 0.05)) >= 0.90,
                  "assinatura": round(sum(sigs) / n, 4), "assinatura_ok": sum(sigs) / n >= 0.90}
        games[t] = (np.concatenate(probs_all).astype(np.float32), np.array(offsets), np.array(ds),
                    np.array([x["item"] for x in tests]), tests)
    np.savez_compressed(OUT / f"plantio_s{a.seed}.npz", **{f"{t}_probs": g[0] for t, g in games.items()},
                        **{f"{t}_offsets": g[1] for t, g in games.items()}, **{f"{t}_d": g[2] for t, g in games.items()},
                        **{f"{t}_item": g[3] for t, g in games.items()},
                        meta=json.dumps({t: [{"item": x["item"], "words": x["words"], "options": x["options"], "target": x["target"], "pos": x["pos"]}
                                             for x in g[4]] for t, g in games.items()}, ensure_ascii=False))
    res["total_s"] = round(time.time() - t0, 1)
    (OUT / f"plantio_s{a.seed}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
