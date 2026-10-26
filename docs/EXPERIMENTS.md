# Experiments: typed inputs, options, outputs and data splits

This document states, for every experiment of the article, what a decider receives (the state), what it is asked (the question type, the instruction and the options), what it returns and how that output is read, and which items train, calibrate and test. Examples, instructions, counts and seeds are generated from the data files and the code by `scripts/docs_experimentos.py`, so the document changes when they change. The registered decisions and their order are in `preregistration/implementation_notes.md`.

## 1. The common interface

Both deciders answer a *choice* question about a state. The state is a JSON object, the question carries an instruction that names state fields in backticks, and its criteria are the options, each with a short description. The output is one probability per option. Every analysis reads three things from it: the **decision** (the option of largest probability), the **confidence** (that probability) and **correct** (whether the decision is the gold option).

Request sent to Jev for MASSIVE item pt-PT 1000 (the API pins the model by name in every request):

```json
{
  "model": "jev-1.13.0",
  "state": {
    "utterance": "por favor adiciona lista de coisas para comprar para a festa"
  },
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "What is the user asking for in `utterance`?",
      "criteria": {
        "calendar_query": "calendar query",
        "email_addcontact": "email addcontact",
        "cooking_recipe": "cooking recipe",
        "lists_createoradd": "lists createoradd",
        "alarm_set": "alarm set"
      }
    }
  }
}
```
Output as stored in the package: decision `lists_createoradd`, confidence 1.0, correct True (gold `lists_createoradd`). Jev rounds probabilities to two decimals. The probabilities of the options not decided are not distributed (see `pacote/DATASHEET.md`).

Laya receives the same state, instruction and options locally. Its encoder reads `[CLS] question [SEP] [MASK] option_1 ... [MASK] option_k [SEP] state [SEP]`, scores each `[MASK]` anchor and applies a softmax over the options, so it returns the same kind of output. Probabilities of the five Laya checkpoints for the same item:

```json
{
  "multilingual": [
    0.0002,
    0.0001,
    0.0001,
    0.9994,
    0.0002
  ],
  "root": [
    0.0095,
    0.0054,
    0.0101,
    0.9647,
    0.0102
  ],
  "typed-decisions": [
    0.1132,
    0.0535,
    0.1262,
    0.5928,
    0.1142
  ],
  "ft-multilingual": [
    0.0005,
    0.0,
    0.0,
    0.9995,
    0.0
  ],
  "ft-typed-decisions": [
    0.0087,
    0.0023,
    0.0045,
    0.9743,
    0.0103
  ]
}
```
## 2. Static gate on MASSIVE (article: Results)

- **Items.** `data/items/massive_pt_k5.json` and `massive_en_k5.json`, 2000 items each. Source note: "MASSIVE 1.1 (pt-PT), test split, 5 options per question: the gold intent and 4 distractors drawn with a fixed seed". `scripts/itens_massive.py` rebuilds them from the MASSIVE 1.1 release and checks them against the frozen files: with Python's `random` and seed 13, the test partition is shuffled, its first 2,000 rows are kept, and each row receives four distractors drawn from the other test intents and shuffled with the gold intent. MASSIVE lists the same identifiers in the same order in every locale, so the same procedure gives the parallel en-US items. These are the items of the companion lesson repository (`github.com/pcbrom/laya-jev`, `data/massive_pt/test.jsonl`).
- **Question.** Type `choice`, instruction "What is the user asking for in `utterance`?", criteria = the five intents with underscores read as spaces.
- **Splits.** Items 0 to 999 are the calibration half, items 1000 to 1999 the report half. The gate is fitted on calibration draws and measured on the plug-in population of all 2,000 items (`scripts/population_gate.py`) or on random half splits (`scripts/ablation_gate.py`).
- **Fine-tuning.** The fine-tuned checkpoints were trained in the lesson repository on the pt-PT training partition without the 240 utterances also found in the test set (11,274 items), with the original validation partition (2,033) used only to choose the epoch. The paired label curve (`scripts/rotulos_nao_inferioridade.py`) refits on subsets of the same 11,274 items, seeds 0 to 2, `SeedSequence([20261023, size, seed])`, and tests on the report half.

## 3. Explanation front (article: When an explanation of a typed decision is useful)

- **Units.** The words of the utterance, merged into at most eight units for longer utterances (`scripts/explicacao/jev_enumerate.py`, `units`). Item pt-PT 1000 has 8 units: [['por'], ['favor', 'adiciona'], ['lista'], ['de', 'coisas'], ['para'], ['comprar'], ['para', 'a'], ['festa']].
- **Question per coalition.** For each subset of units, a choice question whose instruction quotes the kept words, "What is the user asking for in this utterance: \"...\"?", with the same options. For Jev, all coalitions of an item travel in one request with the neutral state {"task": "intent of a short user utterance"}. The empty coalition is the empty state.
- **Splits.** Report items 1000 to 1999 only. The planted model (G3 and G5) is fine-tuned on the training partition with invented tokens planted by the rules of the implementation notes and tested on planted copies of report items.

## 4. Natural noise and adversarial typos (article: When the input is noisy)

- **Operators.** `scripts/ruido/ruido.py`, applied to the utterance only and deterministic given `SeedSequence([20261023, 10, operator, rate, item])`: keyboard typos on 5%, 10% or 20% of the words, swapped adjacent words, lower case without punctuation, accents removed and a mixed profile. In the code the conditions keep their Portuguese names: `limpo` (clean), `typo_r`, `ordem_r` (word order), `caixa` (lower case, no punctuation), `acento` (accents removed) and `misto_r` (mixed). Example on item pt-PT 1000:

```json
{
  "limpo": "por favor adiciona lista de coisas para comprar para a festa",
  "typo_0.10": "por favor adiciona lista de coisas para comprar apra a festx",
  "caixa": "por favor adiciona lista de coisas para comprar para a festa",
  "misto_0.10": "por favor adiciona lista de cisas para comprar para a festa"
}
```
- **Splits.** The gate is calibrated on clean calibration items and checked on noisy report items. Training with noise fine-tunes on the training partition with half of the items passed through a random operator, three seeds.

## 5. Languages (article: When the language changes)

- **Items.** The 2,000 items mapped to MASSIVE identifiers and replaced, in each of the 52 locales, by the utterance with the same identifier, with the options and gold kept (`scripts/multilingue/itens_multilingue.py`). Example, en-US item 1000: `please add list of things to buy for party` with MASSIVE id 10898.
- **Splits.** Calibration in en-US on items 0 to 999, check on the report items 1000 to 1999 of each locale. Laya is also calibrated on each locale's own items 0 to 999.

## 6. Drift (article: When a Bayesian retraining loop is useful under drift)

- **Items.** Five-option intent questions built by `make_item` (`scripts/drift/data.py`): gold plus four distractors drawn once per item with `SeedSequence([20261022, 7, item id])`. News streams use the headline with the instruction "Which category does the headline in `utterance` belong to?".
- **Splits.** Initial fine-tune on 4000 training items of the pre-change distribution, calibration on 1,000 items of the pre-change pool, streams of 9000 items in windows of 500, labels from every escalated item and a 5% audit of accepted items, retraining on the most recent labelled items with the newest 20% held out for promotion. Truth is measured on the test split reweighted to the stream mixture, so stream items never decide what is true. Streams: D1 (scenarios held out until the change), D2 (pt-PT to en-US), D3 (two intent pairs swapped), control, ten year-to-year News Category transitions, and the seasonal S1 (a share of items with typos on 20% of the words) and S2 (a share of en-US items among pt-PT ones), each share cycling every 2,000 items.
- **Encodings** are cached by identifier, text, options, instruction and state, because MASSIVE gives the same identifier to the same utterance in every locale.

## 7. Battery of four domains (article: When the domain changes)

| Domain | Locale | State | Instruction | Options | Items and splits |
|---|---|---|---|---|---|
| News Category | en | headline | "Which category does the headline in `utterance` belong to?" | gold + 4 of the other 41 categories | 2,000 drawn from all rows, pool = the rest |
| finance | en | text | "What is the market sentiment of the financial tweet in `utterance`?" | bearish, bullish, neutral | 2,000 items, validation split, pool = training split |
| liar2 | en | text | "How true is the claim in `utterance`, as rated by a fact-checker?" | pants on fire, false, barely true, half true, mostly true, true | 2,000 items, test split, pool = training split |
| fake_pt | pt | text | "Is the news in `utterance` fake or real?" | fake news, real news | 2,000 items, 1,000 aligned pairs kept whole, pool = the other 2,600 pairs |

Every domain: items 0 to 999 calibrate, 1000 to 1999 report, drawn with `SeedSequence([20261023, 30, d])`. Fake.Br texts are not distributed, since its repository states no licence, and its records carry the corpus identifier only. Fine-tunes of 250 to 8,000 pool items, seeds 0 to 2. LIAR2 with a structured state adds `speaker` and `speaker_record`, the counts of the speaker's ratings in a fixed half of the training split, and fine-tunes on the other half (`scripts/bateria/liar2_estado.py`).

## 8. Direction of the next value on M3 (article: Numeric decisions over real time)

- **State.** `history`, the last 24 values before the origin. **Instruction.** "Will the next value of the monthly series in `history` be higher than its last value?". **Options.** higher, not higher.
- **Sliding-window validation.** ETS refitted at each of the last 36 origins on the last 48 observations. Gates calibrated on the 12 (or 6) preceding origins and tested on the next, sliding by one. The fine-tuned Laya trained in sliding blocks: for each block of 6 test origins, on 8,000 decisions of the other series at the 24 preceding origins. Jev and Laya on a sample of 150 series, 25 per category, at the last 12 origins.

