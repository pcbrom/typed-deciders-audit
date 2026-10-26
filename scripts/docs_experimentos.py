#!/usr/bin/env python3
"""Writes docs/EXPERIMENTS.md: for every experiment, how the typed input (state), the question and its options and the
output were built, and which items trained, validated (calibrated) and tested (reported). Examples, instructions,
counts and seeds are read from the data files and the code, so the document follows them. Jev answers are shown in the
reduced form of the package (decided option, its probability, correct), and Fake.Br texts are not shown.

Usage: docs_experimentos.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "drift"))
sys.path.insert(0, str(HERE.parent / "bateria"))
sys.path.insert(0, str(HERE.parent / "m3"))
sys.path.insert(0, str(HERE.parent / "ruido"))
sys.path.insert(0, str(HERE.parent / "explicacao"))
import data as DR  # noqa: E402
import ruido as RU  # noqa: E402
from common import DATA, INSTRUCTION, question  # noqa: E402

OUT = HERE.parents[1] / "docs" / "EXPERIMENTS.md"


def js(obj):
    return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"


def jev_body(state, q, name="intent"):
    return {"model": "jev-1.13.0", "state": state, "questions": {name: q}}


def laya_probs(locale, item):
    raw = json.loads((DATA / "raw" / f"laya_{locale}.json").read_text())
    return {k: [round(x, 4) for x in v[item]] for k, v in raw["probs"].items()}


def main():
    pt = json.loads((DATA / "items" / "massive_pt_k5.json").read_text())
    en = json.loads((DATA / "items" / "massive_en_k5.json").read_text())
    it = pt["items"][1000]
    jev_pt = json.loads((DATA / "raw" / "jev_pt.json").read_text())
    p = jev_pt["probs"][1000]
    j = max(range(len(p)), key=lambda k: p[k])
    s = []
    s.append("# Experiments: typed inputs, options, outputs and data splits\n")
    s.append("This document states, for every experiment of the article, what a decider receives (the state), what it is asked "
             "(the question type, the instruction and the options), what it returns and how that output is read, and which items "
             "train, calibrate and test. Examples, instructions, counts and seeds are generated from the data files and the code by "
             "`scripts/docs_experimentos.py`, so the document changes when they change. The registered decisions and their order "
             "are in `preregistration/implementation_notes.md`.\n")
    s.append("## 1. The common interface\n")
    s.append("Both deciders answer a *choice* question about a state. The state is a JSON object, the question carries an "
             "instruction that names state fields in backticks, and its criteria are the options, each with a short description. "
             "The output is one probability per option. Every analysis reads three things from it: the **decision** (the option of "
             "largest probability), the **confidence** (that probability) and **correct** (whether the decision is the gold option).\n")
    s.append("Request sent to Jev for MASSIVE item pt-PT 1000 (the API pins the model by name in every request):\n")
    s.append(js(jev_body({"utterance": it["utterance"]}, question(it["options"]))))
    s.append(f"Output as stored in the package: decision `{it['options'][j]}`, confidence {p[j]}, correct "
             f"{it['options'][j] == it['intent']} (gold `{it['intent']}`). Jev rounds probabilities to two decimals. The probabilities "
             "of the options not decided are not distributed (see `pacote/DATASHEET.md`).\n")
    s.append("Laya receives the same state, instruction and options locally. Its encoder reads `[CLS] question [SEP] [MASK] option_1 ... "
             "[MASK] option_k [SEP] state [SEP]`, scores each `[MASK]` anchor and applies a softmax over the options, so it returns the "
             "same kind of output. Probabilities of the five Laya checkpoints for the same item:\n")
    s.append(js(laya_probs("pt", 1000)))
    s.append("## 2. Static gate on MASSIVE (article: Results)\n")
    s.append(f"- **Items.** `data/items/massive_pt_k5.json` and `massive_en_k5.json`, {pt['n']} items each. Source note: "
             f"\"{pt['source']}\". `scripts/itens_massive.py` rebuilds them from the MASSIVE 1.1 release and checks them against the frozen files: "
             "with Python's `random` and seed 13, the test partition is shuffled, its first 2,000 rows are kept, and each row "
             "receives four distractors drawn from the other test intents and shuffled with the gold intent. MASSIVE lists the same "
             "identifiers in the same order in every locale, so the same procedure gives the parallel en-US items. These are the "
             "items of the companion lesson repository (`github.com/pcbrom/laya-jev`, `data/massive_pt/test.jsonl`).\n"
             f"- **Question.** Type `choice`, instruction \"{INSTRUCTION}\", criteria = the five intents with underscores read as spaces.\n"
             "- **Splits.** Items 0 to 999 are the calibration half, items 1000 to 1999 the report half. The gate is fitted on calibration "
             "draws and measured on the plug-in population of all 2,000 items (`scripts/population_gate.py`) or on random half splits "
             "(`scripts/ablation_gate.py`).\n"
             "- **Fine-tuning.** The fine-tuned checkpoints were trained in the lesson repository on the pt-PT training partition "
             "without the 240 utterances also found in the test set (11,274 items), with the original validation partition (2,033) used "
             "only to choose the epoch. The paired label curve (`scripts/rotulos_nao_inferioridade.py`) refits on subsets of the same "
             "11,274 items, seeds 0 to 2, `SeedSequence([20261023, size, seed])`, and tests on the report half.\n")
    s.append("## 3. Explanation front (article: When an explanation of a typed decision is useful)\n")
    from jev_enumerate import units
    from jev_packing_test import NEUTRAL
    us = units(it["utterance"])
    s.append("- **Units.** The words of the utterance, merged into at most eight units for longer utterances "
             f"(`scripts/explicacao/jev_enumerate.py`, `units`). Item pt-PT 1000 has {len(us)} units: {us}.\n"
             f"- **Question per coalition.** For each subset of units, a choice question whose instruction quotes the kept words, "
             f"\"What is the user asking for in this utterance: \\\"...\\\"?\", with the same options. For Jev, all coalitions of an item "
             f"travel in one request with the neutral state {json.dumps(NEUTRAL)}. The empty coalition is the empty state.\n"
             "- **Splits.** Report items 1000 to 1999 only. The planted model (G3 and G5) is fine-tuned on the training partition with "
             "invented tokens planted by the rules of the implementation notes and tested on planted copies of report items.\n")
    s.append("## 4. Natural noise and adversarial typos (article: When the input is noisy)\n")
    s.append("- **Operators.** `scripts/ruido/ruido.py`, applied to the utterance only and deterministic given "
             "`SeedSequence([20261023, 10, operator, rate, item])`: keyboard typos on 5%, 10% or 20% of the words, swapped adjacent words, "
             "lower case without punctuation, accents removed and a mixed profile. In the code the conditions keep their Portuguese names: `limpo` (clean), `typo_r`, `ordem_r` (word order), `caixa` (lower case, no punctuation), `acento` (accents removed) and `misto_r` (mixed). Example on item pt-PT 1000:\n")
    s.append(js({c: RU.apply(it["utterance"], c, 1000) for c in ("limpo", "typo_0.10", "caixa", "misto_0.10")}))
    s.append("- **Splits.** The gate is calibrated on clean calibration items and checked on noisy report items. Training with noise "
             "fine-tunes on the training partition with half of the items passed through a random operator, three seeds.\n")
    s.append("## 5. Languages (article: When the language changes)\n")
    ml = json.loads((DATA / "items" / "multilingue" / "massive_en-US_k5.json").read_text())["items"][1000]
    s.append("- **Items.** The 2,000 items mapped to MASSIVE identifiers and replaced, in each of the 52 locales, by the utterance with "
             "the same identifier, with the options and gold kept (`scripts/multilingue/itens_multilingue.py`). Example, en-US item 1000: "
             f"`{ml['utterance']}` with MASSIVE id {ml['massive_id']}.\n"
             "- **Splits.** Calibration in en-US on items 0 to 999, check on the report items 1000 to 1999 of each locale. Laya is also "
             "calibrated on each locale's own items 0 to 999.\n")
    s.append("## 6. Drift (article: When a Bayesian retraining loop is useful under drift)\n")
    s.append(f"- **Items.** Five-option intent questions built by `make_item` (`scripts/drift/data.py`): gold plus four distractors drawn "
             f"once per item with `SeedSequence([{DR.SEED}, 7, item id])`. News streams use the headline with the instruction "
             f"\"{DR.NEWS_INSTRUCTION}\".\n"
             f"- **Splits.** Initial fine-tune on {DR.INIT_N} training items of the pre-change distribution, calibration on 1,000 items of the "
             f"pre-change pool, streams of {DR.STREAM_LEN} items in windows of 500, labels from every escalated item and a 5% audit of "
             "accepted items, retraining on the most recent labelled items with the newest 20% held out for promotion. Truth is "
             "measured on the test split reweighted to the stream mixture, so stream items never decide what is true. Streams: D1 "
             "(scenarios held out until the change), D2 (pt-PT to en-US), D3 (two intent pairs swapped), control, ten year-to-year "
             f"News Category transitions, and the seasonal S1 (a share of items with typos on 20% of the words) and S2 (a share of en-US items among pt-PT ones), each share cycling every {DR.PERIOD:,} items.\n"
             "- **Encodings** are cached by identifier, text, options, instruction and state, because MASSIVE gives the same identifier to "
             "the same utterance in every locale.\n")
    s.append("## 7. Battery of four domains (article: When the domain changes)\n")
    from dominios import FIXED, LOCALE
    s.append("| Domain | Locale | State | Instruction | Options | Items and splits |\n|---|---|---|---|---|---|")
    s.append(f"| News Category | {LOCALE['news']} | headline | \"{DR.NEWS_INSTRUCTION}\" | gold + 4 of the other 41 categories | 2,000 drawn from all rows, pool = the rest |")
    for d, src in (("finance", "validation split, pool = training split"), ("liar2", "test split, pool = training split"),
                   ("fake_pt", "1,000 aligned pairs kept whole, pool = the other 2,600 pairs")):
        opts, instr = FIXED[d]
        s.append(f"| {d} | {LOCALE[d]} | text | \"{instr}\" | {', '.join(opts)} | 2,000 items, {src} |")
    s.append("\nEvery domain: items 0 to 999 calibrate, 1000 to 1999 report, drawn with `SeedSequence([20261023, 30, d])`. Fake.Br texts are not distributed, since its repository states no licence, and its records carry the corpus identifier only. Fine-tunes "
             "of 250 to 8,000 pool items, seeds 0 to 2. LIAR2 with a structured state adds `speaker` and `speaker_record`, the counts "
             "of the speaker's ratings in a fixed half of the training split, and fine-tunes on the other half "
             "(`scripts/bateria/liar2_estado.py`).\n")
    s.append("## 8. Direction of the next value on M3 (article: Numeric decisions over real time)\n")
    from tipados_direcao import INSTRUCTION as M3I, OPTIONS as M3O
    s.append(f"- **State.** `history`, the last 24 values before the origin. **Instruction.** \"{M3I}\". **Options.** {', '.join(M3O)}.\n"
             "- **Sliding-window validation.** ETS refitted at each of the last 36 origins on the last 48 observations. Gates calibrated "
             "on the 12 (or 6) preceding origins and tested on the next, sliding by one. The fine-tuned Laya trained in sliding blocks: "
             "for each block of 6 test origins, on 8,000 decisions of the other series at the 24 preceding origins. Jev and Laya on a "
             "sample of 150 series, 25 per category, at the last 12 origins.\n")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(s) + "\n")
    print("written", OUT, len("\n".join(s)), "characters")


if __name__ == "__main__":
    main()
