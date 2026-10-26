# Reference package: frozen answers of two typed decision models

Answers of Jev 1.13 (closed, through the vendor API, with 1.14 on the coalition games) and Laya (open weights, as released and fine-tuned) on five-option intent questions from MASSIVE 1.1 in 52 locales, under natural noise, on coalition games of the words of each utterance, and on four further domains (news categories, financial tweets, LIAR2 and Fake.Br). The package accompanies the article "Confidence Is Not a Promise" by Pedro Carvalho Brom and makes its numbers reproducible without new API calls. `DATASHEET.md` describes the content, the collection and the licences, and `manifest.json` gives the counts and the SHA-256 of every file.

## Files

| File | Content |
|---|---|
| `jev_answers.jsonl.gz`, `laya_answers.jsonl.gz` | MASSIVE: static, noise and language fronts |
| `jev_coalition_games.jsonl.gz` | Jev 1.13 and 1.14 on every coalition of the words of an utterance |
| `battery_jev_answers.jsonl.gz`, `battery_laya_answers.jsonl.gz` | the four further domains and MASSIVE pt-PT with the same protocol |
| `items.zip` | the parallel MASSIVE items in 52 locales and the battery items whose licence allows distribution |
| `laya_explanation.zip`, `laya_noise_training.zip`, `drift_runs.zip` | NumPy archives and run records read by the analysis scripts |
| `promise_check.py` | the check of a selective-accuracy promise across populations used in the article |

Jev records carry the decided option, its probability and whether it is right. Laya records carry the probability of every option.

## Reading the records

```python
import gzip, json
rows = [json.loads(l) for l in gzip.open("jev_answers.jsonl.gz", "rt", encoding="utf-8")]
langs = [r for r in rows if r["front"] == "languages"]
```

## Checking a promise across languages

`promise_check.py` needs NumPy and SciPy. It reads JSON Lines with a `group` and either `probs` and `gold` or `confidence` and `correct`. For Jev calibrated on the en-US calibration half (items 0 to 999 of the static front) and checked on the report items of every locale:

```python
import gzip, json
out = open("jev_groups.jsonl", "w")
for l in gzip.open("jev_answers.jsonl.gz", "rt", encoding="utf-8"):
    r = json.loads(l)
    if r["front"] == "static" and r["locale"] == "en-US" and r["item"] < 1000:
        out.write(json.dumps({"group": "en-US calibration", "confidence": r["confidence"], "correct": r["correct"]}) + "\n")
    if r["front"] == "languages" and r["item"] >= 1000:
        out.write(json.dumps({"group": r["locale"], "confidence": r["confidence"], "correct": r["correct"]}) + "\n")
```

```bash
python3 promise_check.py jev_groups.jsonl --reference "en-US calibration" --targets 0.95 --draws 2000 --seed 20261023 --out promise.json
```

The en-US and pt-PT report items of the languages front are those of the noise run (`front` noise, `condition` clean).

## Licence

CC BY-NC-SA 4.0 for the answers and the code. Items keep the licences of their sources, listed in `DATASHEET.md`.
