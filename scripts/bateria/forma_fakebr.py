#!/usr/bin/env python3
"""Fake.Br, form or content (implementation_notes.md): T2 on the CPU, T1 and T3 on the GPU.

  t2    logistic regression on surface features only, trained on the pool, accuracy on the report items
  gpu   T1 (fine-tune on the pool, report items under single operators) and T3 (fine-tune on normalised text)
        run with the interpreter of arquivos/modelos_decisao/.venv
  read  paired accuracy losses of T1 and T3 from the saved probabilities
Output: data/derived/bateria/forma_fakebr.json and forma_fakebr.npz.

Usage: forma_fakebr.py t2|gpu|read
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "drift"))
sys.path.insert(0, str(HERE.parents[1] / "ruido"))
import ruido as R  # noqa: E402
from common import DATA, REP, wilson  # noqa: E402
from dominios import load, pool_item  # noqa: E402

OUT = DATA / "derived" / "bateria"
PUNCT = ("!", "?", ".", ",", '"', ":", ";", "(", "-")
OPERATORS = {"clean": lambda t, k: t, "lower": lambda t, k: t.lower(),
             "no_punct": lambda t, k: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t)).strip(),
             "caixa": lambda t, k: R.apply(t, "caixa", k), "acento": lambda t, k: R.apply(t, "acento", k),
             "typo_0.10": lambda t, k: R.apply(t, "typo_0.10", k)}


def features(text):
    letters = [c for c in text if c.isalpha()]
    words = text.split()
    nl, nw = max(1, len(letters)), max(1, len(words))
    accented = sum(1 for c in letters if unicodedata.normalize("NFD", c) != c)
    return [sum(c.isupper() for c in letters) / nl, sum(w.isupper() and len(w) > 1 for w in words) / nw, accented / nl,
            sum(c.isdigit() for c in text) / max(1, len(text)), *[text.count(p) / nw for p in PUNCT], len(words), len(letters) / nw]


def save(key, value):
    path = OUT / "forma_fakebr.json"
    d = json.loads(path.read_text()) if path.exists() else {}
    d[key] = value
    path.write_text(json.dumps(d, indent=1))


def t2():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    items, pool, labels = load("fake_pt")
    Xtr = np.array([features(r["text"]) for r in pool])
    ytr = np.array([labels.index(r["label"]) for r in pool])
    rep = [items[k] for k in REP]
    Xte = np.array([features(it["text"]) for it in rep])
    yte = np.array([labels.index(it["options"][it["gold"]]) for it in rep])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    ok = clf.predict(Xte) == yte
    names = ["upper_letters", "upper_words", "accented", "digits", *[f"count_{p}" for p in PUNCT], "words", "word_length"]
    coef = dict(zip(names, np.round(clf[-1].coef_[0], 3).tolist()))
    save("t2", {"accuracy": float(ok.mean()), "wilson95": wilson(int(ok.sum()), len(ok)), "positive_class": labels[1], "coefficients": coef})
    print("T2 accuracy", round(float(ok.mean()), 3), wilson(int(ok.sum()), len(ok)), coef)


def gpu():
    import torch
    from model import Learner
    items, pool, labels = load("fake_pt")
    arrays = {}
    for variant in ("original", "normalised"):
        train_rows = pool if variant == "original" else [dict(r, text=R.apply(r["text"], "caixa", 0)) for r in pool]
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            L = Learner("multilingual")
            train = []
            for r in train_rows:
                it = pool_item("fake_pt", r, labels)
                it["id"] = f"{it['id']}|{variant}"
                try:
                    L.encode(it)
                    train.append(it)
                except ValueError:
                    pass
            L.finetune(train, epochs=3, batch=32, seed=seed)
            ops = OPERATORS if variant == "original" else {"clean": OPERATORS["clean"], "caixa": OPERATORS["caixa"]}
            for name, op in ops.items():
                batch = [dict(items[k], id=f"{items[k]['id']}|{name}", text=op(items[k]["text"], k)) for k in REP]
                arrays[f"{variant}|{seed}|{name}"] = L.predict(batch)
            print(variant, seed, {n: round(float((arrays[f"{variant}|{seed}|{n}"].argmax(1) == np.array([items[k]["gold"] for k in REP])).mean()), 3) for n in ops}, flush=True)
            del L
            torch.cuda.empty_cache()
    np.savez_compressed(OUT / "forma_fakebr.npz", **arrays)


def read():
    items, _, _ = load("fake_pt")
    g = np.array([items[k]["gold"] for k in REP])
    Z = dict(np.load(OUT / "forma_fakebr.npz"))
    ok = {k: (v.argmax(1) == g).astype(float) for k, v in Z.items()}
    out = {}
    for variant, ops in (("original", list(OPERATORS)), ("normalised", ["clean", "caixa"])):
        acc = {n: float(np.mean([ok[f"{variant}|{s}|{n}"].mean() for s in (0, 1, 2)])) for n in ops}
        loss = {}
        for n in ops[1:]:
            d = np.mean([ok[f"{variant}|{s}|{n}"] - ok[f"{variant}|{s}|clean"] for s in (0, 1, 2)], axis=0)
            rng = np.random.default_rng(np.random.SeedSequence([20261023, 36]))
            m = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            loss[n] = {"mean": float(d.mean()), "ci95": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]}
        out[variant] = {"accuracy": acc, "loss_vs_clean": loss}
    save("t1_t3", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"t2": t2, "gpu": gpu, "read": read}[sys.argv[1]]()
