# Results of the main experiments (E1 to E5)

Written 2026-10-05. This is the saved state of the project before the follow-up experiment E8 (task-aware heads
with the reference SMART), which gets its own report. It covers every run finished so far: E1, E1b, E2, E3, E4,
E4b and E5. The earlier `progress_e1_e3.md` covers only E1 to E3, in more detail.

Scores are on the **test** split, in percent, as mean ± standard deviation over 3 seeds (42, 123, 2026), unless a
line says "validation". Every choice between settings was made on **validation** scores. "Interval" means the 95%
paired-bootstrap interval of the macro-F1 difference (1,000 samples). "Significant on all 3 seeds" means McNemar's
test gives p < 0.05 on every seed, in the same direction.

---

## 1. Summary

- **222 training runs finished, 0 failed,** 47.2 GPU-hours of training on a Kaggle T4.
- **RQ1:** multi-task learning (MTL) does not consistently beat single-task training on either dataset.
- **RQ2:** uncertainty weighting, GradNorm, PCGrad and embedding-level SMART are statistically the same as the equal
  sum. The paper's SMART gain **does** reproduce on NEU-ESC with the paper's own implementation (noise on token ids),
  which is the only result in the project that is significant on all 3 seeds.
- **RQ3:** focal loss and weighted cross-entropy do not help, overall or on the smallest classes.
- **RQ4:** task-aware heads give one suggestive gain (UIT-VSFC topic, +0.86), and the direction test does not
  confirm the intended mechanism.
- **RQ5:** MTL does not help more with less data. On UIT-VSFC sentiment it is 4 to 5 points worse than single-task
  training at 10% and 25% of the training set.
- **Final model, as selected:** ViSoBERT on NEU-ESC and PhoBERT on UIT-VSFC, task-aware heads, equal-sum loss, plain
  cross-entropy. It is **not** the best-scoring NEU-ESC model: PhoBERT with the reference SMART scores higher
  (section 9).

---

## 2. Setup

| Item | Value |
|---|---|
| Datasets | NEU-ESC (23,048 / 3,305 / 6,613 train / val / test; 4 sentiment, 10 topic classes), UIT-VSFC (11,425 / 1,583 / 3,166; 3 sentiment, 4 topic classes) |
| Backbones | PhoBERT (`vinai/phobert-base-v2`), XLM-R (`FacebookAI/xlm-roberta-base`), ViSoBERT (`uitnlp/visobert`) |
| Training | Full fine-tuning; AdamW (encoder 2e-5, heads 1e-4), warm-up 10% + linear decay, batch 32, up to 10 epochs, early stopping on validation macro-F1 (patience 3), fp16 except PCGrad and GradNorm |
| Main metric | macro-F1 |
| Hardware | Kaggle Tesla T4, torch 2.10.0+cu128, transformers 5.0.0 |

| Experiment | Question | Runs | Training time | Code commit |
|---|---|---:|---:|---|
| E1 baseline | RQ1; picks the backbone | 54 | 10.6 h | `594f177` |
| E2 loss balancing | RQ2 | 24 | 13.7 h | `5bc8029` |
| E3 imbalance | RQ3 | 36 | 6.3 h | `7d8e7e2` |
| E4 task-aware heads | RQ4; final model | 6 | 0.9 h | `192765a` |
| E5 low resource | RQ5 | 72 | 3.9 h | `7c4b6cb` |
| E1b reference SMART | RQ2 | 18 | 9.7 h | `08fe8fd` |
| E4b head direction | RQ4 | 12 | 2.0 h | `b833363` |
| **Total** | | **222** | **47.2 h** | |

Not run: E6 (MLM, 6 runs) and E7 (cross-dataset evaluation).

---

## 3. Selection

| Dataset | Backbone | Loss | Imbalance | Final tag |
|---|---|---|---|---|
| NEU-ESC | ViSoBERT (validation 0.7039; PhoBERT 0.7025, XLM-R 0.6995) | sum (**manual**; the rule picks PCGrad, 0.7080 vs 0.7039) | none (rule) | `sum` |
| UIT-VSFC | PhoBERT (validation 0.8394; XLM-R 0.8316, ViSoBERT 0.8175) | sum (same as the rule) | none (rule) | `sum` |

The manual loss choice and its reason (no strategy significantly better than the sum; PCGrad costs 7.5 times the
training time) are stored with all scores in `reports/tables/selection.json`. The reference SMART was not a candidate
in the loss stage: the plan treats it as a reproduction of prior work, not as a method to select.

---

## 4. RQ1: MTL vs single-task (E1)

| Dataset | Backbone | Sentiment: ST → MTL | Δ | Topic: ST → MTL | Δ |
|---|---|---|---:|---|---:|
| NEU-ESC | PhoBERT | 76.26 ± 0.35 → 76.31 ± 0.27 | +0.05 | 61.73 ± 0.43 → 61.41 ± 0.34 | −0.32 |
| NEU-ESC | ViSoBERT | 76.09 ± 0.20 → 76.20 ± 0.23 | +0.11 | 61.02 ± 0.61 → 61.95 ± 0.31 | +0.93 |
| NEU-ESC | XLM-R | 76.02 ± 0.60 → 76.73 ± 0.36 | +0.72 | 62.27 ± 0.60 → 61.32 ± 0.57 | −0.94 |
| UIT-VSFC | PhoBERT | 82.64 ± 0.49 → 82.08 ± 2.04 | −0.56 | 80.06 ± 0.80 → 79.61 ± 0.82 | −0.45 |
| UIT-VSFC | ViSoBERT | 81.16 ± 0.48 → 81.27 ± 0.25 | +0.10 | 79.17 ± 0.56 → 79.37 ± 0.94 | +0.19 |
| UIT-VSFC | XLM-R | 82.17 ± 1.19 → 81.74 ± 0.69 | −0.43 | 79.81 ± 0.38 → 78.98 ± 0.83 | −0.83 |

**Answer: no.** Every difference is below 1 point, the sign depends on the backbone and task, and none is significant
on all 3 seeds. The single-task and 2-task numbers are within about 1 to 1.5 points of the NEU-ESC paper's single runs
(`reports/tables/results_neu-esc.md`).

---

## 5. RQ2: loss strategies (E2, E1b)

### 5.1 On the selected backbones

| Dataset | Strategy | Validation | Sentiment | Topic | Mean | Δ vs sum | Time (× sum) |
|---|---|---:|---:|---:|---:|---:|---:|
| NEU-ESC (ViSoBERT) | sum | 0.7039 | 76.20 ± 0.23 | 61.95 ± 0.31 | 69.07 | – | 1.00 |
| | uncertainty | 0.7044 | 75.84 ± 0.63 | 61.81 ± 0.26 | 68.83 | −0.25 | 1.02 |
| | GradNorm | 0.7048 | 76.25 ± 0.70 | 62.12 ± 0.54 | 69.19 | +0.11 | 3.75 |
| | PCGrad | 0.7080 | 76.17 ± 0.61 | 61.93 ± 0.60 | 69.05 | −0.03 | 7.45 |
| | SMART (embeddings) | 0.7058 | 76.52 ± 0.18 | 61.95 ± 0.48 | 69.24 | +0.16 | 2.31 |
| | SMART (reference) | 0.7028 | 76.53 ± 0.25 | 62.13 ± 0.25 | 69.33 | +0.26 | 2.72 |
| UIT-VSFC (PhoBERT) | sum | 0.8394 | 82.08 ± 2.04 | 79.61 ± 0.82 | 80.84 | – | 1.00 |
| | uncertainty | 0.8378 | 82.29 ± 0.53 | 80.04 ± 0.53 | 81.17 | +0.32 | 1.00 |
| | GradNorm | 0.8383 | 82.39 ± 0.76 | 80.00 ± 0.70 | 81.19 | +0.35 | 2.07 |
| | PCGrad | 0.8384 | 82.35 ± 0.65 | 80.16 ± 0.13 | 81.25 | +0.41 | 4.13 |
| | SMART (embeddings) | 0.8383 | 81.42 ± 1.11 | 79.93 ± 0.28 | 80.68 | −0.17 | 1.65 |
| | SMART (reference) | 0.8348 | 81.48 ± 0.21 | 80.39 ± 0.91 | 80.94 | +0.09 | 1.96 |

Uncertainty weighting, GradNorm, PCGrad and embedding-level SMART: no difference from the sum is significant on
either dataset or task.

### 5.2 Reference SMART on all backbones (E1b)

| Dataset | Backbone | Sentiment acc | Sentiment mF1 | Topic acc | Topic mF1 | McNemar, seeds significant (sentiment / topic) |
|---|---|---|---|---|---|---|
| NEU-ESC | XLM-R | 82.09 → 83.01 | 76.73 → 77.58 (+0.85) | 77.57 → 79.06 | 61.32 → 62.52 (+1.20) | 2 of 3 / 3 of 3 |
| NEU-ESC | ViSoBERT | 82.17 → 83.21 | 76.20 → 76.53 (+0.33) | 78.15 → 79.24 | 61.95 → 62.13 (+0.18) | 3 of 3 / 3 of 3 |
| NEU-ESC | PhoBERT | 81.93 → 82.87 | 76.31 → 77.29 (+0.98) | 78.03 → 79.26 | 61.41 → 63.04 (+1.63) | 3 of 3 / 3 of 3 |
| UIT-VSFC | XLM-R | 93.54 → 93.39 | 81.74 → 81.89 (+0.15) | 88.51 → 89.26 | 78.98 → 80.15 (+1.17) | 0 of 3 / 1 of 3 |
| UIT-VSFC | ViSoBERT | 92.75 → 92.64 | 81.27 → 79.39 (−1.88) | 88.65 → 89.01 | 79.37 → 80.15 (+0.78) | 0 of 3 / 1 of 3 |
| UIT-VSFC | PhoBERT | 93.82 → 93.63 | 82.08 → 81.48 (−0.60) | 88.78 → 88.91 | 79.61 → 80.39 (+0.78) | 0 of 3 / 0 of 3 |

Values are plain MTL (sum) → MTL with the reference SMART.

- **NEU-ESC:** accuracy rises by about 1 point on every backbone and both tasks, which matches the gain the paper
  reports. Five of the six comparisons are significant on all 3 seeds. The macro-F1 intervals exclude 0 for XLM-R and
  PhoBERT, and include 0 for ViSoBERT.
- **UIT-VSFC:** the gain does not carry over. Topic macro-F1 rises by 0.8 to 1.2 (only the XLM-R interval excludes 0),
  sentiment does not improve, and ViSoBERT sentiment drops by 1.9 (interval excludes 0).
- **Reference vs embedding-level SMART** (selected backbones only): no significant macro-F1 difference.

**Answer:** standard loss balancing does not help. The paper's SMART gain reproduces on NEU-ESC with the paper's
implementation, which adds noise to token ids and has no adversarial step, so it is not SMART as published.
Embedding-level SMART gave no significant gain **at the one setting tested** (weight 0.02, epsilon 1e-5, step size
1e-3); stronger settings were not tried.

---

## 6. RQ3: class imbalance (E3)

| Dataset | Imbalance (MTL model) | Validation | Sentiment | Topic |
|---|---|---:|---:|---:|
| NEU-ESC | none | 0.7039 | 76.20 ± 0.23 | 61.95 ± 0.31 |
| | focal loss (γ = 2) | 0.7024 | 76.41 ± 0.28 | 62.02 ± 0.49 |
| | weighted cross-entropy | 0.7038 | 75.81 ± 0.21 | 61.49 ± 0.59 |
| UIT-VSFC | none | 0.8394 | 82.08 ± 2.04 | 79.61 ± 0.82 |
| | focal loss (γ = 2) | 0.8377 | 82.08 ± 0.82 | 79.77 ± 1.20 |
| | weighted cross-entropy | 0.8381 | 81.51 ± 0.15 | 80.19 ± 0.53 |

| Smallest classes (MTL model, F1) | none | focal | weighted CE |
|---|---:|---:|---:|
| NEU-ESC sentiment `toxic` | 83.01 ± 0.60 | 82.88 ± 1.56 | 82.48 ± 0.75 |
| NEU-ESC topic `spam` | 54.87 ± 2.29 | 52.89 ± 1.58 | 52.57 ± 1.52 |
| NEU-ESC topic `club_events` | 69.88 ± 1.04 | 67.66 ± 0.89 | 69.16 ± 2.54 |
| NEU-ESC topic `help_share` | 50.06 ± 3.00 | 49.44 ± 0.47 | 50.19 ± 0.98 |
| UIT-VSFC sentiment `neutral` | 55.10 ± 5.39 | 55.42 ± 2.33 | 54.21 ± 0.17 |
| UIT-VSFC topic `others` | 53.40 ± 2.23 | 54.81 ± 2.86 | 56.44 ± 2.10 |

**Answer: no.** Neither method changes the MTL model's macro-F1 significantly, and the smallest classes do not clearly
benefit. Single-task results: `reports/tables/rq3_imbalance.md`. Only two classic loss-level methods were tested.

---

## 7. RQ4: task-aware heads (E4, E4b)

Change against the plain MTL model with linear heads, on the selected backbones.

| Dataset (Cramér's V) | Heads | Sentiment | Δ | Topic | Δ |
|---|---|---:|---:|---:|---:|
| NEU-ESC (0.18) | linear | 76.20 ± 0.23 | – | 61.95 ± 0.31 | – |
| | task-aware, both directions | 76.34 ± 0.84 | +0.14 | 62.09 ± 0.97 | +0.15 |
| | sentiment reads topic | 76.11 ± 0.49 | −0.09 | 62.06 ± 0.32 | +0.12 |
| | topic reads sentiment | 76.30 ± 0.86 | +0.10 | 62.31 ± 0.44 | +0.37 |
| UIT-VSFC (0.35) | linear | 82.08 ± 2.04 | – | 79.61 ± 0.82 | – |
| | task-aware, both directions | 81.88 ± 1.11 | −0.20 | 80.47 ± 0.78 | +0.86 |
| | sentiment reads topic | 82.99 ± 1.06 | +0.91 | 80.16 ± 0.49 | +0.56 |
| | topic reads sentiment | 82.22 ± 0.76 | +0.14 | 79.54 ± 0.36 | −0.07 |

- The only difference whose interval excludes 0 is UIT-VSFC topic with both directions (+0.86, interval +0.07 to +1.69,
  2 of 3 seeds). It is not significant on all 3 seeds.
- It is on the dataset with the stronger sentiment–topic relationship, as the hypothesis predicted, but two datasets
  are too few to claim that the gain follows the correlation.
- The direction test does not confirm the mechanism: when only the topic head reads the sentiment prediction, topic
  does not improve (−0.07). No one-direction variant is significant.
- The task-aware heads add almost no training time or memory.

**Answer:** no reliable gain; one suggestive result whose mechanism is not confirmed.

---

## 8. RQ5: limited training data (E5)

| Dataset | Train fraction | Sentiment: ST / MTL / final | Δ MTL−ST | Topic: ST / MTL / final | Δ MTL−ST |
|---|---:|---|---:|---|---:|
| NEU-ESC | 10% | 69.91 / 68.92 / 70.41 | −0.99 * | 51.46 / 50.96 / 50.50 | −0.50 |
| | 25% | 73.64 / 73.38 / 73.32 | −0.26 | 55.97 / 56.79 / 56.78 | +0.81 |
| | 50% | 74.66 / 74.66 / 75.28 | 0.00 | 58.54 / 59.40 / 58.92 | +0.86 * |
| | 100% | 76.09 / 76.20 / 76.34 | +0.11 | 61.02 / 61.95 / 62.09 | +0.93 |
| UIT-VSFC | 10% | 77.48 / 72.70 / 74.27 | −4.78 * | 75.68 / 76.20 / 75.51 | +0.52 |
| | 25% | 82.06 / 78.05 / 78.95 | −4.00 * | 78.81 / 78.03 / 78.25 | −0.78 |
| | 50% | 81.65 / 81.26 / 81.72 | −0.39 | 79.66 / 78.97 / 79.29 | −0.69 |
| | 100% | 82.64 / 82.08 / 81.88 | −0.56 | 80.06 / 79.61 / 80.47 | −0.45 |

`*` = the bootstrap interval of the MTL − single-task difference excludes 0. None is significant on all 3 seeds.
Standard deviations: `reports/tables/rq5_low_resource.md`.

- The MTL advantage does not grow as data shrinks. On sentiment it turns into a disadvantage.
- **UIT-VSFC sentiment at 10% and 25%:** single-task is better on all 3 seeds. The loss is in the `neutral` class,
  which has about 46 training texts at 10%: F1 43.9 (single-task) vs 30.8 (MTL) at 10%, 56.0 vs 44.7 at 25%.
- **NEU-ESC topic** is the only place MTL helps, by a steady 0.8 to 0.9 from 25% upwards.
- The final model beats plain MTL on sentiment at 10% (+1.5 on NEU-ESC, +1.6 on UIT-VSFC), but stays below
  single-task on UIT-VSFC sentiment (about −3 at 10% and 25%; both intervals exclude 0).
- Train–validation loss gap at the best epoch: somewhat smaller for MTL than for single-task at 10% and 25% in all four
  dataset / task cases, mixed at 50% and 100%. This is a rough measure and it does not turn into better test scores.

**Answer: no.**

---

## 9. Final model and best model

| Model | Validation | Sentiment mF1 | Topic mF1 |
|---|---:|---:|---:|
| NEU-ESC final model (ViSoBERT, task-aware heads, sum) | 0.7054 | 76.34 ± 0.84 | 62.09 ± 0.97 |
| NEU-ESC PhoBERT + reference SMART (E1b) | 0.7177 | 77.29 ± 0.33 | 63.04 ± 0.60 |
| NEU-ESC paper, best single runs | – | 77.70 | 63.13 |
| UIT-VSFC final model (PhoBERT, task-aware heads, sum) | 0.8354 | 81.88 ± 1.11 | 80.47 ± 0.78 |

The final model is what the selection procedure produced. It is not the best-scoring NEU-ESC model, because the
reference SMART was not a selection candidate and the backbone was chosen from plain MTL runs. On the selected
backbones the reference SMART scores below the sum on validation (0.7028 vs 0.7039 on NEU-ESC ViSoBERT, 0.8348 vs
0.8394 on UIT-VSFC PhoBERT), so validation would not have chosen it there; it supports it only on NEU-ESC PhoBERT.

---

## 10. Limits

- **Three seeds.** Standard deviations are rough; one unusual seed moves a mean by about half a point.
- **The NEU-ESC loss choice is manual,** not the rule's choice (section 3).
- **Embedding-level SMART was tested at one setting.** "No gain" holds for that setting only.
- **The direct comparison of the two SMART versions exists only on the selected backbones.**
- **RQ3 covers two classic loss-level methods.** Newer imbalance methods were not tested.
- **RQ4 rests on two datasets** and one suggestive difference.
- **The train–validation gap in RQ5 is a rough measure** (it depends on which epoch was best).
- **NEU-ESC validation is about 3 to 6 points above test on topic,** so validation overstates topic macro-F1.
- **Checkpoints are only on Kaggle** (E1 and E4 outputs); E7 has not been run.

---

## 11. Next and future work

| Item | Status |
|---|---|
| E8: task-aware heads + reference SMART (12 runs: NEU-ESC ViSoBERT and PhoBERT, UIT-VSFC PhoBERT, plus the task-aware PhoBERT baseline on NEU-ESC) | Planned next; separate report |
| Joint selection of backbone and loss on validation (would give PhoBERT + reference SMART on NEU-ESC), with E3 to E5 rerun on that setup | Future work (about 17 to 28 GPU-hours) |
| Stronger settings for embedding-level SMART | Future work |
| Newer imbalance methods (for example post-hoc logit adjustment, which needs no retraining) | Future work |
| E6 MLM ablation, E7 cross-dataset evaluation | Optional, not run |

---

## 12. Where the files are

| What | Path |
|---|---|
| Main result tables | `reports/tables/results_neu-esc.md`, `reports/tables/results_uit-vsfc.md` |
| Per-question tables | `reports/tables/rq1_st_vs_mtl.md`, `rq2_loss.md`, `rq3_imbalance.md`, `rq3_per_class.md`, `rq4_task_aware.md`, `rq5_low_resource.md` |
| Significance tests (RQ1 to RQ5) | `reports/tables/significance.md` |
| Time and memory per configuration | `reports/tables/cost.md` |
| Selection and its provenance | `reports/tables/selection.json` |
| Run folders (metrics, predictions, logs) | `models/<run_id>/` (not in git) |
| Earlier report (E1 to E3 in detail) | `reports/progress_e1_e3.md` |
