# Preregistration: Bayesian conditional retraining of a typed decider under drift

Status: FROZEN on 2026-10-21, approved by the author, by the commit tagged `prereg-drift-v1` in this repository. No pilot or main run started before that commit.

## 1. Question

Does the author's conditional retraining loop for Laya keep the promise of a selective gate under drift at lower cost than the state-of-the-art alternatives for the same task? The loop has three steps:

- a Beta trigger on audited accepted items;
- Beta promotion on discordant pairs;
- Beta re-qualification of the gate.

## 2. Model, task and gate

- **Decider.** Laya multilingual (mmBERT-base, code commit 4066d5d, weights revision 55cf4c4), CPU or GPU, bf16.
- **Task.** Five-option intent questions built as in the article: the gold intent and four distractors, with a fixed seed per item.
- **Promise.** Accuracy among automatically accepted items of at least $\alpha = 0.95$, with $q = \delta = 0.05$.
- **Initial model.** Fine-tuned on the stream's pre-change distribution: 3 epochs, batch 32, AdamW, learning rates $2.5\times10^{-5}$ for the encoder and $10^{-4}$ for the head.
- **Retraining procedure, identical for every arm.** Fine-tune the current model for 2 epochs on the most recent 4,000 labelled items available to the arm, mixed with 20% replay of the initial training set, using the same hyper-parameters.
- **Gate re-qualification after a promotion.** Uses the most recent 1,000 labelled items available to the arm, with the arm's own threshold rule.

## 3. Labels and information

- **Label sources.** A label is obtained for every escalated item and for a uniform random audit sample of 5% of accepted items. Each label costs one review.
- **Information each arm sees.**
  - Every arm sees the same label stream.
  - The Beta trigger, DDM, ADWIN and Page-Hinkley monitor correctness on the audit sample only, because escalated labels are selected by low confidence and are biased for the error rate among accepted items. A tuning option lets the detectors use all labels.
  - Promotion and re-qualification use all labels.

## 4. Arms

**Arms that decide the title.** All share the decider, the retraining procedure, the audit rate and the label budget.

| Arm | Trigger | Promotion | Gate after promotion |
|---|---|---|---|
| A. Author's method | Beta posterior of accuracy among accepted, from audit labels with forgetting $\lambda$: retrain when $P(\theta < \alpha) \ge 1 - q$ | Beta posterior on discordant pairs, promote when $P(\text{new better}) \ge 1 - q$ | Beta lower-bound rule |
| B1. Frozen | never | none | initial threshold |
| B2. Periodic | every K items | always, or a published test (tuning option) | LTT |
| B3. Recalibration only | never retrain; Laplace succession per confidence bin (author's rule) | none | point rule on recalibrated confidence |
| B4. DDM (Gama et al., 2004) | warning and drift levels on the audit error stream | always, exact McNemar (Dietterich, 1998), or the high-confidence safety test of Thomas et al. (2019), as tuning options | LTT |
| B5. ADWIN (Bifet and Gavaldà, 2007) | as B4 | as B4 | LTT |
| B6. Page-Hinkley (Page, 1954) | as B4 | as B4 | LTT |
| B7. LTT per window | never retrain; threshold re-qualified by LTT every window on recent labels | none | LTT |
| B8. ACI (Gibbs and Candès, 2021) | never retrain; threshold adapted online toward the target miscoverage | none | ACI threshold |

**Attribution ablation.** Reported, but it does not decide the title. Components are swapped one at a time inside arm A:

- counts per window against counts discounted by $\lambda$;
- uniform prior against Clopper-Pearson on the same counts;
- Beta promotion against exact McNemar against always promote;
- Beta re-qualification against LTT re-qualification.

## 5. Streams

**Synthetic drifts.** Built from MASSIVE 1.1 training splits. The true selective accuracy is measured on the test split, reweighted to the current stream mixture.

- **D1, new scenarios.** The initial model is trained on 13 of the 18 MASSIVE scenarios. The 5 held-out scenarios enter the stream after the change point: a knock-out shift reversed, following Rabanser et al. (2019). They were drawn before freezing with `numpy.random.default_rng(SeedSequence([20261022, 99])).choice(sorted(scenarios), 5, replace=False)`: datetime, iot, recommendation, social and takeaway.
- **D2, language.** pt-PT before the change and en-US after, on parallel items.
- **D3, label change.** Two pairs of intents swap labels after the change point (real concept drift). Among the 42 intents with at least 100 pt-PT training items, four were drawn before freezing with `numpy.random.default_rng(SeedSequence([20261022, 98])).choice(sorted(eligible), 4, replace=False)`, giving the pairs iot_hue_lightoff with email_query and music_likeness with datetime_query.
- **Patterns and change point.** Abrupt and gradual patterns (linear mixture over 1,000 items), following Gama et al. (2014). The change point is drawn uniformly in [3,000, 5,000]. Each stream has 9,000 items.
- **Control.** A stream with no drift measures false retrains.

**Real streams.**

- **Source.** News Category Dataset v3 (Misra, 2022): 209,527 HuffPost headlines with category and date, 2012 to September 2022, CC BY 4.0, obtained from the mirror `heegyu/news-category-dataset` on Hugging Face, which cites the original Kaggle release. Its SHA-256 is recorded at download. It contains the HuffPost data used by the Wild-Time benchmark (Yao et al., 2022; 2012 to 2018, CC0), so the two count as one source. Items are five-option questions on the headline, with the gold category and four distractor categories drawn with a fixed seed per item.
- **Unit and change.** The unit is the transition between consecutive years, with 10 transitions as clusters and 5 random orders within each. The change is declared at 1 January.
- **Manipulation check.** The frozen model must lose at least 2 points of accuracy across the transition. Otherwise the transition is reported but excluded from the title rule, because temporal deterioration does not always occur (Agarwal and Nenkova, 2022).
- **Status.** License and access confirmed on 2026-10-21.

## 6. Measures

- **Primary.** Cost per item with $C_e/C_r = 10$. A wrong automatic action costs 10, and each escalation and each audit label costs 1. A retrain costs 0 in the primary analysis and 50 in a sensitivity analysis.
- **Non-inferiority.** The fraction of post-change windows (500 items each) whose true selective accuracy is below $\alpha$ may exceed that of the comparator by at most 0.02.
- **False retrains.** Retrains on the control stream, compared with the selected best comparator by a paired interval, not by a point estimate.
- **Secondary.**
  - detection delay after the change point;
  - coverage;
  - number of retrains and promotions;
  - labels used;
  - GPU seconds;
  - cost at $C_e/C_r = 100$.

## 7. Tuning

- **Budget.** 20 configurations per arm, including $\lambda$, $q$, K, detector thresholds and promotion options.
- **Objective.** The primary cost on short complete streams (4,000 items, with promotion and re-qualification active), on seeds disjoint from evaluation.
- **Best comparator.** Selected on the tuning streams, never on evaluation streams.

## 8. Replication, power and compute

- **Seeds.** `numpy.random.SeedSequence([20261022, drift, replicate, role])`, with role 0 for evaluation and 1 for tuning.
- **Replicates.** R = 30 per synthetic drift and pattern, fixed here before the pilot.
- **Pilot.** The pilot (10 streams per drift) estimates the standard deviation $\sigma_d$ of the paired cost difference. It does not change R.
- **Power, declared in advance.**
  - If $\sigma_d$ near 0.17 is observed, an effect of 0.02 in cost is not detectable at R = 30, and the expected outcome is the audit title.
  - At R = 30, the minimum detectable effect for a one-sided test at 0.05 with power 0.8 is about $0.45\,\sigma_d$.
- **Compute.**
  - About 2,550 retrains, or 7 to 71 GPU hours at 10 to 100 seconds per retrain.
  - GPU use follows the queue agreed with the other session on this machine, with at most 12 GiB for Laya.

## 9. Decision rules

- **Title.** "Bayesian" stays in the title if arm A has a lower primary cost than the best state-of-the-art comparator, with a paired 95% interval excluding zero, in at least 2 of the 3 synthetic drifts. It must also pass the non-inferiority margin and show no more false retrains on the control stream than that comparator. Real streams are supporting evidence.
- **Abstract.** It states in one sentence which component the attribution ablation credits for any gain.

## 10. Deviations

Any deviation from this document after freezing is listed with its reason in the article.
