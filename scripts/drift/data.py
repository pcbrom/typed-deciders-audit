"""Streams of the drift experiment, as frozen in preregistration/drift.md.

Every stream item is a five-option intent question (gold plus four distractors drawn once per item
with a fixed seed). A stream has a pre-change and a post-change pool, a change point drawn in
[3,000, 5,000], an abrupt or gradual (linear over 1,000 items) pattern and 9,000 items. The true
selective accuracy of a model at time t is measured on the test split, reweighted to the stream
mixture at t, so the stream items themselves never decide what is true.

The initial model of a drift is fine-tuned on INIT_N training items of its pre-change
distribution; those items are excluded from every stream pool of that drift. D2, D3 and the
control share one initial model (all pt-PT scenarios), D1 has its own (13 scenarios).
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EXTERNAL = Path(os.environ.get("TDA_EXTERNAL", Path(__file__).resolve().parents[2] / "external"))   # data, weights and caches outside the repository
MASSIVE = (EXTERNAL / "dados/1.1/data")
INSTRUCTION = "What is the user asking for in `utterance`?"
SEED = 20261022
HELD_OUT = ("datetime", "iot", "recommendation", "social", "takeaway")
SWAPS = (("iot_hue_lightoff", "email_query"), ("music_likeness", "datetime_query"))
STREAM_LEN, CHANGE_RANGE, RAMP, INIT_N, K = 9000, (3000, 5000), 1000, 4000, 5
TUNE_CHANGE_RANGE = (1500, 2500)          # tuning streams (role 1) have 4,000 items
REAL = tuple(f"R{y}" for y in range(2012, 2022))          # real streams: News Category, year y to y + 1
SEASONAL = ("S1", "S2")                   # recurring drift (implementation_notes.md, drift with a pattern)
DRIFTS = ("D1", "D2", "D3", "control") + REAL + SEASONAL
PERIOD = 2000                              # items per cycle of the seasonal streams
EN_FUNCTION_WORDS = frozenset("the a an to of and in on for with is are be it this that my me i you what how can set".split()
                              + "at from by do does will".split())
NEWS = (EXTERNAL / "dados/news_category/data.json")
NEWS_INSTRUCTION = "Which category does the headline in `utterance` belong to?"


def massive(locale):
    rows = [json.loads(line) for line in open(MASSIVE / f"{locale}.jsonl", encoding="utf-8")]
    return [{"id": r["id"], "partition": r["partition"], "scenario": r["scenario"], "intent": r["intent"], "text": r["utt"]}
            for r in rows]


def make_item(row, intents, gold=None, must=()):
    """Five-option question for a row; options fixed by the row id, the gold optionally relabelled."""
    gold = gold or row["intent"]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 7, int(row["id"])]))
    pool = [i for i in intents if i != gold and i not in must]
    extra = [m for m in must if m != gold]
    opts = extra + list(rng.choice(pool, K - 1 - len(extra), replace=False)) + [gold]
    rng.shuffle(opts)
    return {"id": f"{row['id']}:{gold}", "text": row["text"], "options": [str(o) for o in opts], "gold": opts.index(gold),
            "group": row["scenario"]}


@dataclass
class Stream:
    drift: str
    pattern: str
    change: int
    items: list                     # STREAM_LEN items, in arrival order
    post_share: np.ndarray          # share of the post-change distribution at each position
    test_pre: list = field(default_factory=list)
    test_post: list = field(default_factory=list)


def _swap(row):
    for a, b in SWAPS:
        if row["intent"] == a:
            return b, (a, b)
        if row["intent"] == b:
            return a, (a, b)
    return None, ()


def _news_year_splits():
    """Per year: (test rows, initial-training rows, stream-pool rows), split once with a fixed seed."""
    if not hasattr(_news_year_splits, "cache"):
        rows = [json.loads(line) for line in open(NEWS, encoding="utf-8")]
        cats = sorted({r["category"] for r in rows})
        by_year = {}
        for k, r in enumerate(rows):
            by_year.setdefault(int(r["date"][:4]), []).append({"id": 10_000_000 + k, "text": r["headline"], "intent": r["category"],
                                                                 "scenario": r["date"][:4]})
        out = {}
        for y, rs in by_year.items():
            perm = np.random.default_rng(np.random.SeedSequence([SEED, 95, y])).permutation(len(rs))
            n_test = 500 if len(rs) >= 2500 else int(0.2 * len(rs))
            rest = perm[n_test:]
            n_init = min(INIT_N, len(rest) // 2)
            out[y] = ([rs[i] for i in perm[:n_test]], [rs[i] for i in rest[:n_init]], [rs[i] for i in rest[n_init:]])
        _news_year_splits.cache = (out, cats)
    return _news_year_splits.cache


def _news_item(row, cats):
    it = make_item(row, cats)
    it["instruction"] = NEWS_INSTRUCTION
    return it


def initial_training(drift):
    """Items of the initial fine-tune of a drift (pre-change distribution, fixed seed)."""
    if drift in REAL:
        splits, cats = _news_year_splits()
        return [_news_item(r, cats) for r in splits[int(drift[1:])][1]]
    rows = [r for r in massive("pt-PT") if r["partition"] == "train"]
    intents = sorted({r["intent"] for r in rows})
    if drift == "D1":
        rows = [r for r in rows if r["scenario"] not in HELD_OUT]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 1, DRIFTS.index(drift) if drift == "D1" else 1]))
    pick = rng.choice(len(rows), INIT_N, replace=False)
    return [make_item(rows[i], intents) for i in sorted(pick)]


def pools(drift):
    """(pre-change stream pool, post-change stream pool, test pre, test post), items not in the initial set."""
    if drift in REAL:
        splits, cats = _news_year_splits()
        y = int(drift[1:])
        conv = lambda rs: [_news_item(r, cats) for r in rs]
        return conv(splits[y][2]), conv(splits[y + 1][1] + splits[y + 1][2]), conv(splits[y][0]), conv(splits[y + 1][0])
    pt = massive("pt-PT")
    intents = sorted({r["intent"] for r in pt})
    if drift == "S1":                       # the same items after typos on 20% of the words
        pre, _, test_pre, _ = pools("control")
        typo = lambda its: [dict(it, id=f"{it['id']}|typo", text=_typo(it["text"], k)) for k, it in enumerate(its)]
        return pre, typo(pre), test_pre, typo(test_pre)
    if drift == "S2":
        return pools("D2")
    used = {it["id"].split(":")[0] for it in initial_training(drift)}
    train = [r for r in pt if r["partition"] == "train" and r["id"] not in used]
    test = [r for r in pt if r["partition"] == "test"]
    known = lambda rs: [r for r in rs if r["scenario"] not in HELD_OUT]
    plain = lambda rs: [make_item(r, intents) for r in rs]
    if drift == "D1":
        return plain(known(train)), plain(train), plain(known(test)), plain(test)
    if drift == "D2":
        en = massive("en-US")
        en_train = [r for r in en if r["partition"] == "train" and r["id"] not in used]
        return plain(train), plain(en_train), plain(test), plain([r for r in en if r["partition"] == "test"])
    if drift == "D3":
        def swapped(rs):
            out = []
            for r in rs:
                g, pair = _swap(r)
                out.append(make_item(r, intents, gold=g, must=pair) if g else make_item(r, intents))
            return out
        return plain(train), swapped(train), plain(test), swapped(test)
    return plain(train), plain(train), plain(test), plain(test)


def stream(drift, pattern, replicate, role=0):
    """One stream: change point, mixture and items, drawn from SeedSequence([SEED, drift, replicate, role])."""
    pre, post, test_pre, test_post = pools(drift)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, DRIFTS.index(drift), replicate, role, pattern == "gradual"]))
    lo, hi = TUNE_CHANGE_RANGE if role == 1 else CHANGE_RANGE
    change = int(rng.integers(lo, hi + 1))
    t = np.arange(STREAM_LEN)
    if drift == "control":
        share = np.zeros(STREAM_LEN)
    elif drift in SEASONAL:
        change = 0
        share = 0.5 * (1 - np.cos(2 * np.pi * t / PERIOD))
    elif pattern == "abrupt" or drift in REAL:
        share = (t >= change).astype(float)
    else:
        share = np.clip((t - change) / RAMP, 0.0, 1.0)
    from_post = rng.random(STREAM_LEN) < share
    a, b = rng.integers(0, len(pre), STREAM_LEN), rng.integers(0, len(post), STREAM_LEN)
    items = [post[b[i]] if from_post[i] else pre[a[i]] for i in range(STREAM_LEN)]
    return Stream(drift, pattern, change, items, share, test_pre, test_post)


def _typo(text, k):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ruido"))
    import ruido
    return ruido.apply(text, "typo_0.20", k)


def _vocabulary():
    if not hasattr(_vocabulary, "cache"):
        _vocabulary.cache = frozenset(w for r in massive("pt-PT") if r["partition"] == "train" for w in r["text"].lower().split())
    return _vocabulary.cache


def features(items):
    """Window inputs computed from the texts alone: share of words outside the pt-PT training vocabulary,
    and share of words in a fixed list of frequent English function words."""
    words = [w for it in items for w in it["text"].lower().split()]
    if not words:
        return np.zeros(2)
    vocab = _vocabulary()
    return np.array([np.mean([w not in vocab for w in words]), np.mean([w in EN_FUNCTION_WORDS for w in words])])
