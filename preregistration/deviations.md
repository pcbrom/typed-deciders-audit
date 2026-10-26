# Declared deviations

Each entry is committed before the data it affects.

## 2026-10-21: explanation front, Jev request format (before G0 and before any coalition was queried)

**Change.** The coalitions of one item are sent to Jev in a single request, each coalition as its own choice question with the masked text inside the question's instructions and a fixed neutral state. All coalitions of an item use this format, including the full utterance. G0 keeps the preregistered format (text in the state) to compare with the frozen answers.

**Reason.** A micro-test (`scripts/explicacao/jev_packing_test.py`, results in `data/derived/explicacao/jev_packing_test.json`, US$0.01) measured three things:
- **Cost.** About 273 input tokens per request plus 118 per question. A coalition packed in a full request costs about a third of a separate request.
- **Equivalence.** Packed and separate requests agree as closely as two identical separate requests do: argmax agreement 98.8% to 100%, and 60% to 70% of probabilities within 0.01 in both comparisons.
- **Text placement.** Moving the text from the state to the question does change answers (argmax agreement 95% to 97.5%), which is why one format is used for all coalitions of an item.

**Effect on the plan.** The projected Jev spending falls from about US$14.65 to about US$4.61 for both versions. The stopping rule and every gate are unchanged.

## 2026-10-21: drift experiment, D2 manipulation (after the pilot, no change to the plan)

The pilot showed that the frozen multilingual model loses 0.1 points of selective accuracy under the language drift (D2), against 5.6 for D1 and 9.5 for D3. The preregistration has no manipulation check for synthetic drifts, so D2 stays in the plan unchanged and is reported as a drift that did not degrade the model.

## Hyper-parameters of the Jev posterior (explanation front)

The rule fixes $\tau_1,\tau_2$ on calibration items, but Jev was enumerated on the report items only, to save credits, so there are no Jev calibration games. The Jev posteriors use $\tau_1=2$, $\tau_2=1$, the values selected on the calibration halves of both rounded Laya stand-ins, which needs no report data.
