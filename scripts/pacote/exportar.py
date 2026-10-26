#!/usr/bin/env python3
"""Reference package of the article (implementation_notes.md, the reference package): one record per item,
decider and condition, with model, version, collection date, locale, MASSIVE identifier, item index,
utterance, options and gold option.

Jev records carry the decided option, its probability and whether it is right, not the probability of
every option (decision of the author, 2026-10-23, in view of section 2.3(b) of the vendor's customer
agreement). Laya records carry every probability. Coalition games of Jev carry, per coalition, the decided
option and the probability of the option decided on the full utterance. Output in pacote/, with a
manifest of counts and SHA-256 sums.

Usage: exportar.py
"""

import gzip
import hashlib
import io
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "ruido"))
import ruido as R  # noqa: E402
from common import DATA  # noqa: E402

OUT = HERE.parents[2] / "pacote"
ITEMS = DATA / "items" / "multilingue"
LOCALE = {"pt": "pt-PT", "en": "en-US"}
LAYA_FT = "massive-pt-multilingual (Laya multilingual fine-tuned on MASSIVE pt-PT train)"
EXP = DATA / "derived" / "explicacao"
EXTRA = {"laya_explanation": sorted([*EXP.glob("laya_*.npz"), *EXP.glob("g4_*.npz"), *EXP.glob("plantio_*.npz"), *EXP.glob("val2_laya_*.npz")]),
         "laya_noise_training": sorted((DATA / "derived" / "ruido").glob("treino_*.npz")),
         "drift_runs": sorted([*(DATA / "derived" / "drift").glob("*.jsonl"), *(DATA / "derived" / "drift").glob("*.json")])}


def items_of(locale):
    return json.loads((ITEMS / f"massive_{locale}_k5.json").read_text(encoding="utf-8"))["items"]


def base(it, locale, k, utterance=None):
    return {"locale": locale, "massive_id": it["massive_id"], "item": k, "utterance": utterance or it["utterance"],
            "options": it["options"], "gold": it["options"].index(it["intent"])}


def jev_row(front, version, collected, rec, probs, condition="clean"):
    p = np.asarray(probs, dtype=float)
    j = int(p.argmax())
    return {"front": front, "decider": "jev", "model": version, "collected": collected, "condition": condition, **rec,
            "decision": j, "confidence": float(p[j]), "correct": j == rec["gold"]}


def laya_row(front, checkpoint, rec, probs, condition="clean"):
    return {"front": front, "decider": "laya", "model": checkpoint, "collected": "local model, deterministic", "condition": condition, **rec,
            "probs": [round(float(x), 6) for x in probs]}


def write(name, rows):
    path = OUT / name
    n = 0
    with io.TextIOWrapper(gzip.GzipFile(path, "wb", mtime=0), encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def jev_rows():
    for loc in ("pt", "en"):
        raw = json.loads((DATA / "raw" / f"jev_{loc}.json").read_text())
        its = items_of(LOCALE[loc])
        for k, p in enumerate(raw["probs"]):
            yield jev_row("static", raw["model"], raw["measured"], base(its[k], LOCALE[loc], k), p)
    cache = {}
    for line in open(DATA / "derived" / "ruido" / "jev.jsonl"):
        r = json.loads(line)
        loc = LOCALE[r["loc"]]
        its = cache.setdefault(loc, items_of(loc))
        it = its[r["item"]]
        yield jev_row("noise", "jev-1.13.0", "2026-10-23", base(it, loc, r["item"], R.apply(it["utterance"], r["condition"], r["item"])),
                      r["probs"], "clean" if r["condition"] == "limpo" else r["condition"])
    for line in open(DATA / "derived" / "multilingue" / "jev.jsonl"):
        r = json.loads(line)
        its = cache.setdefault(r["locale"], items_of(r["locale"]))
        yield jev_row("languages", "jev-1.13.0", "2026-10-23", base(its[r["item"]], r["locale"], r["item"]), r["probs"])


def laya_rows():
    raw_meta = json.loads((DATA / "raw" / "laya_pt.json").read_text())
    tag = f"code {raw_meta['laya_code_commit'][:7]}, weights {raw_meta['weights_revision'][:7]}"
    for loc in ("pt", "en"):
        raw = json.loads((DATA / "raw" / f"laya_{loc}.json").read_text())
        its = items_of(LOCALE[loc])
        for ckpt, P in raw["probs"].items():
            name = LAYA_FT if ckpt == "ft-multilingual" else f"{ckpt} ({tag})"
            for k, p in enumerate(P):
                yield laya_row("static", name, base(its[k], LOCALE[loc], k), p)
    for m, name in (("multilingual", f"multilingual ({tag})"), ("ft", LAYA_FT)):
        for loc in ("pt", "en"):
            z = np.load(DATA / "derived" / "ruido" / f"laya_{m}_{loc}.npz")
            its = items_of(LOCALE[loc])
            for cond in z.files:
                if cond == "limpo_todos":
                    continue
                for k, p in zip(range(1000, 2000), z[cond]):
                    yield laya_row("noise", name, base(its[k], LOCALE[loc], k, R.apply(its[k]["utterance"], cond, k)), p, cond)
        z = np.load(DATA / "derived" / "multilingue" / f"laya_{m}.npz")
        for loc in z.files:
            its = items_of(loc)
            for k, p in enumerate(z[loc]):
                yield laya_row("languages", name, base(its[k], loc, k), p)


def game_rows():
    for fname, front in (("jev_games.jsonl", "explanation_game"), ("jev_validations.jsonl", "explanation_validation")):
        for line in open(DATA / "derived" / "explicacao" / fname):
            r = json.loads(line)
            loc = LOCALE[r["loc"]]
            versions = r["versions"].items() if "versions" in r else [(r["version"], r)]
            for version, v in versions:
                P = {int(c[1:]): np.asarray(p, dtype=float) for c, p in v["probs"].items()}
                full = max(P)
                j = r.get("j", int(P[full].argmax()))
                yield {"front": front, "decider": "jev", "model": v.get("model_reported", version), "collected": "on or before 2026-10-21",
                       "locale": loc, "item": r["item"], "options": r["options"], "units": r["d"],
                       **({"configuration": r["kind"], "second_option": r["k_star"]} if "kind" in r else {}), "full_decision": j,
                       "coalitions": {str(c): {"decision": int(p.argmax()), "p_full_decision": float(p[j])} for c, p in sorted(P.items())}}


BAT = DATA / "derived" / "bateria"
NO_TEXT = {"fake_pt"}       # Fake.Br states no licence: identifiers and answers only


def battery_items(domain):
    return json.loads((DATA / "items" / "bateria" / f"{domain}.json").read_text(encoding="utf-8"))["items"]


def battery_base(domain, items, k, condition):
    it = items[k]
    text = None
    if domain not in NO_TEXT:
        text = "" if condition == "empty" else R.apply(it["text"], condition, k) if condition in ("typo_0.10", "misto_0.10") else it["text"]
    return {"domain": domain, "item": k, "source_id": it["source_id"], "utterance": text, "options": it["options"], "gold": it["gold"]}


def battery_jev_rows():
    cache = {}
    for fname, fixed in (("jev.jsonl", None), ("jev_liar2_estado_metade.jsonl", "liar2")):
        for line in open(BAT / fname):
            r = json.loads(line)
            domain, cond = r.get("domain", fixed), r.get("condition", "structured_state")
            items = cache.setdefault(domain, battery_items(domain))
            rec = battery_base(domain, items, r["item"], "clean" if cond == "structured_state" else cond)
            p = np.asarray(r["probs"], dtype=float)
            j = int(p.argmax())
            yield {"front": "battery", "decider": "jev", "model": r["model_reported"], "collected": r["answered_utc"][:10], "condition": cond, **rec,
                   "decision": j, "confidence": float(p[j]), "correct": j == rec["gold"]}


def battery_laya_rows():
    for p in sorted(BAT.glob("laya_*.npz")):
        name = p.stem[len("laya_"):]
        Z = np.load(p)
        if name == "liar2_estado_metade":
            domain, keys = "liar2", [(k, "structured_state", None, None) for k in Z.files]
        elif name in ("liar2_estado",):
            continue
        else:
            domain = name
            keys = []
            for k in Z.files:
                model, size, seed, cond = k.split("|")
                keys.append((k, cond, int(size) if model == "ft" else None, int(seed) if model == "ft" else None))
        items = battery_items(domain)
        for key, cond, size, seed in keys:
            P = Z[key]
            first = 0 if len(P) == len(items) else 1000
            model = "multilingual released" if key.startswith("released") else "multilingual fine-tuned on the domain"
            if name == "liar2_estado_metade" and key != "released":
                size, seed = 8000, int(key[2:])
            for i, probs in enumerate(P):
                k = first + i
                yield {"front": "battery", "decider": "laya", "model": model, "collected": "local model, deterministic given the seed",
                       "condition": cond, "fine_tune": None if size is None else {"labels": size, "seed": seed},
                       **battery_base(domain, items, k, "clean" if cond == "structured_state" else cond), "probs": [round(float(x), 6) for x in probs]}
    Z = np.load(BAT / "forma_fakebr.npz")
    items = battery_items("fake_pt")
    for key in Z.files:
        variant, seed, op = key.split("|")
        for i, probs in enumerate(Z[key]):
            yield {"front": "battery", "decider": "laya", "model": f"multilingual fine-tuned on the domain, training text {variant}",
                   "collected": "local model, deterministic given the seed", "condition": f"form test: {op}", "fine_tune": {"labels": 5200, "seed": int(seed)},
                   **battery_base("fake_pt", items, 1000 + i, "clean"), "probs": [round(float(x), 6) for x in probs]}


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "items" / "battery").mkdir(parents=True, exist_ok=True)
    for d in ("news", "finance", "liar2"):
        shutil.copy(DATA / "items" / "bateria" / f"{d}.json", OUT / "items" / "battery" / f"{d}.json")
    (OUT / "items").mkdir(exist_ok=True)
    for p in ITEMS.glob("massive_*_k5.json"):
        shutil.copy(p, OUT / "items" / p.name)
    counts = {}
    for name, gen in (("jev_answers.jsonl.gz", jev_rows), ("laya_answers.jsonl.gz", laya_rows), ("jev_coalition_games.jsonl.gz", game_rows),
                      ("battery_jev_answers.jsonl.gz", battery_jev_rows), ("battery_laya_answers.jsonl.gz", battery_laya_rows)):
        c = Counter()

        def counted(rows):
            for r in rows:
                c[f"{r['front']} | {r.get('domain', '')} | {r['model'].split(' (')[0]}".replace(" |  |", " |")] += 1
                yield r
        write(name, counted(gen()))
        counts[name] = dict(sorted(c.items()))
    for sub, files in EXTRA.items():
        (OUT / sub).mkdir(exist_ok=True)
        for p in files:
            shutil.copy(p, OUT / sub / p.name)
        counts[sub] = len(files)
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {"counts": counts, "files": {str(p.relative_to(OUT)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
