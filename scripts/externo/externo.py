#!/usr/bin/env python3
"""External checks requested in review (implementation_notes.md, external population and hard distractors).

  itens  external population: MASSIVE 1.1 dev partition plus the test rows outside the 2,000 of the article, in pt-PT and
         en-US (parallel), five options with the article's rule (seed 14); hard distractors: the 2,000 items of the
         article with the four distractors drawn first from the gold intent's scenario, then from the rest (seed 15)
  laya   Laya as released (multilingual, root, typed-decisions) and fine-tuned, on the original, external and hard items,
         and the fine-tuned checkpoints on the hard items with an empty utterance
  jev    Jev 1.13 on the external and hard items, stopping before US$0.30 of estimated spend
         (TYPESAFE_API_KEY in the environment, never written)

Outputs in data/items/externo/ and data/derived/externo/.
Usage: externo.py itens|laya|jev
"""

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from common import DATA, EXTERNAL, load_items, question  # noqa: E402

ITEMS = DATA / "items" / "externo"
OUT = DATA / "derived" / "externo"
LOCALES = {"pt": "pt-PT", "en": "en-US"}
K, CAP, PRICE_PER_TOKEN = 5, 0.30, 0.042e-6


def rows(locale):
    return [json.loads(l) for l in open(EXTERNAL / "dados/1.1/data" / f"{locale}.jsonl", encoding="utf-8")]


def make(r, gold, distractors, rng):
    opts = [gold] + distractors
    rng.shuffle(opts)
    return {"id": r["id"], "utterance": r["utt"], "intent": gold, "options": opts, "scenario": r["scenario"]}


def itens():
    ITEMS.mkdir(parents=True, exist_ok=True)
    for short, loc in LOCALES.items():
        sample_utts = {it["utterance"] for it in load_items(short)[0]}
        allrows = rows(loc)
        intents = sorted({r["intent"] for r in allrows if r["partition"] == "test"})
        ext = [r for r in allrows if r["partition"] == "dev" or (r["partition"] == "test" and r["utt"] not in sample_utts)]
        ext = [r for r in ext if r["intent"] in intents]
        rng = random.Random(14)
        items = [make(r, r["intent"], rng.sample([i for i in intents if i != r["intent"]], K - 1), rng) for r in ext]
        (ITEMS / f"massive_{short}_externo.json").write_text(json.dumps({"rule": "dev + test outside the article sample, seed 14", "items": items}, ensure_ascii=False))
        rng = random.Random(15)
        hard = []
        for it in load_items(short)[0]:
            gold = it["intent"]
            r = {"id": it["id"], "utt": it["utterance"], "scenario": gold.split("_")[0]}
            same = [i for i in intents if i != gold and i.split("_")[0] == gold.split("_")[0]]
            rest = [i for i in intents if i != gold and i not in same]
            pick = rng.sample(same, min(K - 1, len(same)))
            pick += rng.sample(rest, K - 1 - len(pick))
            hard.append({**make(r, gold, pick, rng), "same_scenario": min(K - 1, len(same))})
        (ITEMS / f"massive_{short}_dificil.json").write_text(json.dumps({"rule": "same-scenario distractors first, seed 15", "items": hard}, ensure_ascii=False))
        print(short, "external", len(items), "hard", len(hard), "mean same-scenario distractors", round(sum(h["same_scenario"] for h in hard) / len(hard), 2))


def load_set(short, kind):
    if kind == "original":
        return load_items(short)[0]
    return json.loads((ITEMS / f"massive_{short}_{kind}.json").read_text())["items"]


def laya_runs():
    import numpy as np
    import torch
    import laya
    from laya.common import collate_items

    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sub = {"root": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}
    ft = {"ft-multilingual": EXTERNAL / "laya-ajustado/massive-pt-multilingual",
          "ft-typed-decisions": EXTERNAL / "laya-ajustado/massive-pt-typed-decisions"}

    @torch.no_grad()
    def probs(agent, items, empty=False, batch=32):
        enc, internals = [], []
        for it in items:
            internal = {"intent": agent._to_internal(question(it["options"]))}
            enc.append(agent._encode_state({"utterance": "" if empty else it["utterance"]}, ["intent"], internal))
            internals.append(internal)
        order = sorted(range(len(items)), key=lambda i: len(enc[i][0]["ids"]))
        out = [None] * len(items)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            b = collate_items([enc[i] for i in idx], agent.tok.pad_token_id)
            logits, act = agent._forward(b)
            for row, i in enumerate(idx):
                a = agent._decode_answers(logits, act, enc[i], ["intent"], internals[i], row)["intent"]
                out[i] = [round(float(a["probabilities"][o]), 5) for o in items[i]["options"]]
        return out

    for short in LOCALES:
        res = {"device": dev, "probs": {}}
        names = list(sub) + (list(ft) if short == "pt" else [])
        for name in names:
            path = ft.get(name) or (EXTERNAL / "laya-pesos" / sub[name] if sub[name] else EXTERNAL / "laya-pesos")
            agent = laya.load(str(path), device=dev)
            agent.model.eval()
            for kind in ("original", "externo", "dificil"):
                items = load_set(short, kind)
                p = probs(agent, items)
                res["probs"][f"{name}|{kind}"] = p
                gold = np.array([it["options"].index(it["intent"]) for it in items])
                print(short, name, kind, "accuracy", round(float(np.mean(np.argmax(p, 1) == gold)), 4), flush=True)
            if name in ft:
                res["probs"][f"{name}|dificil|vazio"] = probs(agent, load_set(short, "dificil"), empty=True)
            del agent
            torch.cuda.empty_cache()
        (OUT / f"laya_{short}.json").write_text(json.dumps(res))


def jev_runs():
    sys.path.insert(0, str(HERE.parents[1]))
    from query_jev import ask
    key = os.environ["TYPESAFE_API_KEY"]
    OUT.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    for short in LOCALES:
        for kind in ("externo", "dificil"):
            dest = OUT / f"jev_{short}_{kind}.json"
            if dest.exists():
                continue
            items = load_set(short, kind)
            probs, tokens, t0 = [None] * len(items), 0, time.time()
            for s in range(0, len(items), 200):
                if spent + tokens * PRICE_PER_TOKEN >= CAP:
                    print("cap reached", flush=True)
                    break
                chunk = items[s:s + 200]
                with ThreadPoolExecutor(max_workers=8) as ex:
                    res = list(ex.map(lambda it: ask(key, it, "jev-1.13.0"), chunk))
                for j, (r, err) in enumerate(res):
                    if r:
                        probs[s + j] = [r["answers"]["intent"]["probabilities"][o] for o in chunk[j]["options"]]
                        tokens += r.get("usage", {}).get("input_tokens", 0) or 0
            spent += tokens * PRICE_PER_TOKEN
            dest.write_text(json.dumps({"model_requested": "jev-1.13.0", "measured": time.strftime("%Y-%m-%d"),
                                        "input_tokens": tokens, "estimated_usd": round(tokens * PRICE_PER_TOKEN, 4), "probs": probs}))
            print(short, kind, sum(p is not None for p in probs), "/", len(items), "tokens", tokens,
                  "estimated US$", round(tokens * PRICE_PER_TOKEN, 4), "total", round(spent, 4), round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    {"itens": itens, "laya": laya_runs, "jev": jev_runs}[sys.argv[1]]()
