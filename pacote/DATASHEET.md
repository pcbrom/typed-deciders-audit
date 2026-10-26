# Datasheet: frozen answers of two typed decision models on MASSIVE intents and four further domains

The structure follows Gebru et al., "Datasheets for Datasets" (Communications of the ACM, 2021).

## Motivation

The package holds the answers of two typed decision models, Jev (closed, through the vendor API) and Laya (open weights), on five-option intent questions built from MASSIVE 1.1, and on four further domains: news categories, the sentiment of financial tweets, the rating of political claims (LIAR2) and fake news in Portuguese (Fake.Br). It was created to make every number of the accompanying article reproducible without new API calls and to give later work a fixed reference against which a newer model version, a new gate rule or a new calibration method can be compared on the same items. It was created by Pedro Carvalho Brom (ORCID 0000-0002-1288-7695) without external funding.

## Composition

Each record is one answer of one decider to one item under one condition. Counts by file, front and model are in `manifest.json`.

| File | Content | Records |
|---|---|---|
| `jev_answers.jsonl.gz` | Jev 1.13: Portuguese and English items as frozen on 2026-09-26, the noise conditions and 50 further MASSIVE locales | 73,000 |
| `laya_answers.jsonl.gz` | Laya checkpoints as released and fine-tuned on MASSIVE pt-PT: the same fronts, all probabilities | 258,000 |
| `jev_coalition_games.jsonl.gz` | Jev 1.13 and 1.14 on every coalition of the words of an utterance (explanation front) | 7,200 |
| `items/` | the 2,000 parallel items in each of the 52 MASSIVE 1.1 locales, with options and gold intent | 52 files |
| `laya_explanation/` | Laya coalition games, validation configurations, the G4 heads and the planted models, as read by the explanation scripts (NumPy archives) | 30 files |
| `laya_noise_training/` | Laya fine-tuned with and without noise, three seeds, report items under each noise condition | 6 files |
| `drift_runs/` | one record per simulated or real stream run of the drift front, with the registered analyses | 14 files |
| `battery_jev_answers.jsonl.gz` | Jev 1.13 on the four battery domains, clean, under typos and the mixed profile, the empty state, and LIAR2 with a structured state | 19,003 |
| `battery_laya_answers.jsonl.gz` | Laya released and fine-tuned on 250 to 8,000 labelled items of each battery domain and of MASSIVE pt-PT, three seeds, with the same conditions, LIAR2 with a structured state and the form tests on Fake.Br | 282,000 |
| `items/battery/` | the 2,000 items of News Category, financial tweets and LIAR2 with options and gold option | 3 files |

Common fields: `front` (static, noise, languages, explanation_game, explanation_validation, battery), `decider`, `model`, `collected`, `condition`, `locale`, `massive_id`, `item` (0 to 999 calibration half, 1000 to 1999 report half), `utterance` (after the noise operator, when one applies), `options`, `gold` (index of the gold option).

Jev records carry `decision` (index of the decided option), `confidence` (its probability as returned by the API, rounded by the API to two decimals) and `correct`. The probabilities of the other options are not distributed. Laya records carry `probs`, the probability of every option, rounded to six decimals. Coalition records carry, for each coalition of the `units` words (bit mask as key, 0 the empty state), the decided option and the probability of the option decided on the full utterance.

The items are a sample of the MASSIVE 1.1 test partition: 2,000 items mapped by identifier across locales, with four distractor intents drawn per item. The utterances are short commands to a voice assistant, taken from SLURP and localized into each language by professional translators in MASSIVE. They contain no personal data beyond what MASSIVE contains. Battery records add `domain`, `source_id` (the identifier in the source dataset) and, for Laya fine-tunes, `fine_tune` (labels and seed). The battery items are drawn from News Category (Misra 2022), Twitter Financial News sentiment, LIAR2 (Xu and Kechadi 2024) and the size-normalised texts of Fake.Br (Monteiro et al. 2018). In LIAR2 with a structured state, the state adds the speaker and the speaker record built by `repo/scripts/bateria/liar2_estado.py` from a fixed half of the training split.

## Collection process

Jev answers were requested through `POST https://api.typesafe.ai/v1/systemone` with the choice question of the article (`repo/scripts/common.py`, function `question`), one request per item, the model pinned by name in each request. The `collected` field gives the date of the run. The coalition games were collected on or before 2026-10-21, and their record says so, because the date of each call was not stored. Laya answers are deterministic outputs of local checkpoints, identified by code commit and weights revision, on a CPU in bfloat16 autocast (static front) or on one GPU (noise and languages fronts). Noisy utterances are produced by the deterministic operators of `repo/scripts/ruido/ruido.py`, seeded by condition and item. Every run was registered in `repo/preregistration/implementation_notes.md` before it started.

## Uses

Intended: reproducing the article, comparing a new gate rule or calibration method against the same answers, measuring how a new version of either decider differs from these, and teaching selective prediction with real model outputs. `promise_check` (in the code release) reads these records directly.

To avoid: the vendor's customer agreement, section 2.3(b), forbids its customers to use the service's output to train a model that imitates it. That is why Jev records carry the decision and its probability only. Users bound by that agreement should not use the Jev records for distillation.

## Distribution and licence

The answers and the code are released under CC BY-NC-SA 4.0. The items derive from MASSIVE 1.1 by Amazon (FitzGerald et al., ACL 2023), released under CC BY 4.0, whose attribution is kept here. Battery items keep the licences of their sources: News Category CC BY 4.0, Twitter Financial News sentiment MIT, LIAR2 Apache-2.0. The Fake.Br repository states no licence, so its texts are not distributed: its records carry `source_id` (class and file number in the corpus) with `utterance` set to null, and the texts can be obtained from the corpus itself.

## Use of AI-assisted tools

AI-assisted tools were used for code development and English language editing. Experimental design, analysis and interpretation are the author's. Jev, one of the two deciders measured, is itself an AI model accessed through its vendor's API, and its answers are the object of study, not a tool of the analysis.

## Maintenance

Corrections are released as new versions of the same deposit, with the change described in the version notes. Contact through the author's ORCID page.
