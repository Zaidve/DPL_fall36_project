# Progress report: experiments E1 to E3

Written 2026-10-02. Covers everything trained and decided so far: E1 (baseline), E2 (loss balancing) and
E3 (class imbalance). E4 (task-aware heads) and the optional experiments have not been run yet.

All numbers below come from the finished runs in `models/` and the tables in `reports/tables/`.
Scores are on the **test** split, in percent, as mean ± standard deviation over 3 seeds (42, 123, 2026),
unless a line says "validation". Every choice between settings was made on **validation** scores only.

---

## 1. Summary

- **114 training runs finished, 0 failed** (E1: 54, E2: 24, E3: 36), about 30.7 GPU-hours of training on a Kaggle T4.
- **Our reproduction matches the NEU-ESC paper** for all three backbones (within about 1 to 1.5 points).
- **RQ1:** training sentiment and topic together does **not** reliably beat training each task alone.
  Differences are within ±1 macro-F1 point and none is significant on all 3 seeds.
- **RQ2:** no loss-balancing strategy (uncertainty, PCGrad, GradNorm, SMART) is significantly better than the
  plain equal sum, and the expensive ones cost 2 to 7.5 times the training time.
- **RQ3:** neither focal loss nor weighted cross-entropy improves macro-F1, overall or reliably on the smallest classes.
- **Setup fixed for the final model:** ViSoBERT on NEU-ESC, PhoBERT on UIT-VSFC, equal-sum loss, plain cross-entropy.
  E4 adds the task-aware heads on top of this.

---

## 2. Setup

| Item | Value |
|---|---|
| Tasks | sentiment and topic classification of Vietnamese educational feedback |
| Datasets | NEU-ESC (23,048 / 3,305 / 6,613 train / val / test; 4 sentiment, 10 topic classes), UIT-VSFC (11,425 / 1,583 / 3,166; 3 sentiment, 4 topic classes) |
| Backbones | PhoBERT (`vinai/phobert-base-v2`), XLM-R (`FacebookAI/xlm-roberta-base`), ViSoBERT (`uitnlp/visobert`) |
| Modes | `st_sentiment`, `st_topic` (one task each), `mtl` (shared encoder, one linear head per task) |
| Training | AdamW (encoder 2e-5, heads 1e-4), warm-up 10% + linear decay, batch 32, up to 10 epochs, early stopping on validation macro-F1 (patience 3), fp16 |
| Main metric | macro-F1 (the classes are unbalanced: imbalance ratio 27 / 36 on NEU-ESC, 12 / 16 on UIT-VSFC) |
| Seeds | 42, 123, 2026 |
| Hardware | Kaggle Tesla T4, torch 2.10.0+cu128, transformers 5.0.0 |
| Significance | McNemar test per seed, paired bootstrap of the macro-F1 difference (1,000 samples) |

---

## 3. What was run

| Experiment | Question | Runs | Training time | Code commit |
|---|---|---:|---:|---|
| E1 baseline | RQ1: MTL vs single task; picks the backbone | 54 | 10.6 h | `594f177` |
| E2 loss balancing | RQ2: uncertainty, PCGrad, SMART, GradNorm vs sum | 24 | 13.7 h | `5bc8029` |
| E3 imbalance | RQ3: focal loss, weighted cross-entropy | 36 | 6.3 h | `7d8e7e2` |

Kaggle sessions:

| Session | Content | Note |
|---|---|---|
| 1 | E1, NEU-ESC PhoBERT (9 runs) | |
| 2 | E1, NEU-ESC ViSoBERT (9 runs) | started without resuming session 1; no run was repeated |
| 3 | E1, NEU-ESC XLM-R (9 runs) | resumed from session 2 |
| 4 | E1, UIT-VSFC, all backbones (27 runs) | resumed from session 3 |
| 5 | E2, first 15 runs | stopped by Kaggle at the 12-hour limit during run 16; the finished runs were kept |
| 6 | E2, last 9 runs | resumed from session 5 |
| 7 | E3, all 36 runs | |

---

## 4. E1: baseline and backbone choice (RQ1)

### 4.1 Reproduction of the paper (NEU-ESC)

| Model | Sent mF1 (paper) | Sent mF1 (ours) | Topic mF1 (paper) | Topic mF1 (ours) |
|---|---:|---:|---:|---:|
| Single task XLM-R | 75.91 | 76.02 ± 0.60 | 60.87 | 62.27 ± 0.60 |
| Single task ViSoBERT | 75.79 | 76.09 ± 0.20 | 60.65 | 61.02 ± 0.61 |
| Single task PhoBERT | 77.70 | 76.26 ± 0.35 | 62.73 | 61.73 ± 0.43 |
| 2-task XLM-R | 75.50 | 76.73 ± 0.36 | 60.62 | 61.32 ± 0.57 |
| 2-task ViSoBERT | 76.17 | 76.20 ± 0.23 | 61.66 | 61.95 ± 0.31 |
| 2-task PhoBERT | 75.99 | 76.31 ± 0.27 | 62.11 | 61.41 ± 0.34 |

The paper reports one number per model; ours are means over 3 seeds. Full table with accuracy and weighted F1:
`reports/tables/results_neu-esc.md`, `reports/tables/results_uit-vsfc.md`.

### 4.2 MTL vs single task (macro-F1)

| Dataset | Backbone | Task | Single task | MTL | Δ |
|---|---|---|---:|---:|---:|
| NEU-ESC | PhoBERT | sentiment | 76.26 ± 0.35 | 76.31 ± 0.27 | +0.05 |
| NEU-ESC | PhoBERT | topic | 61.73 ± 0.43 | 61.41 ± 0.34 | −0.32 |
| NEU-ESC | ViSoBERT | sentiment | 76.09 ± 0.20 | 76.20 ± 0.23 | +0.11 |
| NEU-ESC | ViSoBERT | topic | 61.02 ± 0.61 | 61.95 ± 0.31 | +0.93 |
| NEU-ESC | XLM-R | sentiment | 76.02 ± 0.60 | 76.73 ± 0.36 | +0.72 |
| NEU-ESC | XLM-R | topic | 62.27 ± 0.60 | 61.32 ± 0.57 | −0.94 |
| UIT-VSFC | PhoBERT | sentiment | 82.64 ± 0.49 | 82.08 ± 2.04 | −0.56 |
| UIT-VSFC | PhoBERT | topic | 80.06 ± 0.80 | 79.61 ± 0.82 | −0.45 |
| UIT-VSFC | ViSoBERT | sentiment | 81.16 ± 0.48 | 81.27 ± 0.25 | +0.10 |
| UIT-VSFC | ViSoBERT | topic | 79.17 ± 0.56 | 79.37 ± 0.94 | +0.19 |
| UIT-VSFC | XLM-R | sentiment | 82.17 ± 1.19 | 81.74 ± 0.69 | −0.43 |
| UIT-VSFC | XLM-R | topic | 79.81 ± 0.38 | 78.98 ± 0.83 | −0.83 |

**Reading:**
- MTL with an equal-sum loss does not consistently beat single-task training. The sign of the difference depends on
  the backbone and the task, and every difference is below 1 point.
- No comparison is significant under McNemar's test on all 3 seeds in the same direction.
- Under the paired bootstrap on NEU-ESC, two differences come close: ViSoBERT topic (+0.93, positive in 96% of
  samples) and XLM-R topic (−0.94, negative in 98% of samples).
- ViSoBERT is the only backbone that never gets worse with MTL.
- UIT-VSFC PhoBERT MTL sentiment has a large spread (± 2.04): seed 123 stopped at epoch 2 with a lower score.

### 4.3 Backbone choice

Chosen on the mean validation macro-F1 of the MTL model (average of the two tasks):

| Dataset | Chosen | Validation scores |
|---|---|---|
| NEU-ESC | **ViSoBERT** | ViSoBERT 0.7039, PhoBERT 0.7025, XLM-R 0.6995 |
| UIT-VSFC | **PhoBERT** | PhoBERT 0.8394, XLM-R 0.8316, ViSoBERT 0.8175 |

All later experiments use these backbones.

---

## 5. E2: loss balancing (RQ2)

MTL model on the chosen backbone, with four strategies compared against the equal sum (`sum`, from E1).

| Dataset | Strategy | Validation mF1 | Sent mF1 | Topic mF1 | Mean mF1 | Δ vs sum | Time per epoch (× sum) |
|---|---|---:|---:|---:|---:|---:|---:|
| NEU-ESC (ViSoBERT) | sum | 0.7039 | 76.20 ± 0.23 | 61.95 ± 0.31 | 69.07 | – | 1.00 |
| | uncertainty | 0.7044 | 75.84 ± 0.63 | 61.81 ± 0.26 | 68.83 | −0.25 | 1.02 |
| | PCGrad | 0.7080 | 76.17 ± 0.61 | 61.93 ± 0.60 | 69.05 | −0.03 | 7.45 |
| | GradNorm | 0.7048 | 76.25 ± 0.70 | 62.12 ± 0.54 | 69.19 | +0.11 | 3.75 |
| | SMART (embeddings) | 0.7058 | 76.52 ± 0.18 | 61.95 ± 0.48 | 69.24 | +0.16 | 2.31 |
| UIT-VSFC (PhoBERT) | sum | 0.8394 | 82.08 ± 2.04 | 79.61 ± 0.82 | 80.84 | – | 1.00 |
| | uncertainty | 0.8378 | 82.29 ± 0.53 | 80.04 ± 0.53 | 81.17 | +0.32 | 1.00 |
| | PCGrad | 0.8384 | 82.35 ± 0.65 | 80.16 ± 0.13 | 81.25 | +0.41 | 4.13 |
| | GradNorm | 0.8383 | 82.39 ± 0.76 | 80.00 ± 0.70 | 81.19 | +0.35 | 2.07 |
| | SMART (embeddings) | 0.8383 | 81.42 ± 1.11 | 79.93 ± 0.28 | 80.68 | −0.17 | 1.65 |

**Reading:**
- No strategy is significantly better than `sum` on either dataset or task: every bootstrap interval of the
  macro-F1 difference includes 0, and McNemar's test never agrees on all 3 seeds.
- GradNorm's learned task weights stayed close to 1.0 (for example 1.10 / 0.90 at the end of a NEU-ESC run), which
  suggests the two tasks are already balanced under the equal sum.
- The gain the paper reports for SMART on NEU-ESC (about +1 point) does not appear at that size here
  (+0.32 on sentiment, 0.00 on topic).
- PCGrad and GradNorm cannot use fp16, which is the main reason for their cost on a T4.

### Loss choice (manual)

The selection rule (highest validation score) would pick **PCGrad** on NEU-ESC, by +0.004 over `sum`, about one seed
standard deviation, with the same test score. It would pick `sum` on UIT-VSFC.

**Decision: `sum` for both datasets.** Reason, as recorded in `reports/tables/selection.json`: no E2 strategy is
significantly better than the equal sum on either dataset, so the simplest and cheapest strategy is kept; PCGrad costs
7.5 times the training time. The file keeps all validation scores and the rule's own choice.

---

## 6. E3: class imbalance (RQ3)

Two loss-level methods, each applied to the MTL model and to both single-task models:

- **Focal loss** (γ = 2, no per-class weights): scales down the loss of examples the model already gets right.
- **Weighted cross-entropy**: class weight = N / (C × n_c), from the training split
  (for example `toxic` 9.85 and `neutral` 0.36 on NEU-ESC sentiment).

| Dataset | Model | Imbalance | Sent mF1 | Topic mF1 |
|---|---|---|---:|---:|
| NEU-ESC | MTL | none | 76.20 ± 0.23 | 61.95 ± 0.31 |
| | MTL | focal | 76.41 ± 0.28 | 62.02 ± 0.49 |
| | MTL | weighted CE | 75.81 ± 0.21 | 61.49 ± 0.59 |
| | single task | none | 76.09 ± 0.20 | 61.02 ± 0.61 |
| | single task | focal | 75.62 ± 0.11 | 61.44 ± 0.81 |
| | single task | weighted CE | 75.98 ± 0.60 | 62.30 ± 0.25 |
| UIT-VSFC | MTL | none | 82.08 ± 2.04 | 79.61 ± 0.82 |
| | MTL | focal | 82.08 ± 0.82 | 79.77 ± 1.20 |
| | MTL | weighted CE | 81.51 ± 0.15 | 80.19 ± 0.53 |
| | single task | none | 82.64 ± 0.49 | 80.06 ± 0.80 |
| | single task | focal | 82.96 ± 0.55 | 79.80 ± 0.82 |
| | single task | weighted CE | 82.81 ± 1.64 | 80.02 ± 1.07 |

F1 of the smallest classes (MTL model):

| Dataset | Task | Class | none | focal | weighted CE |
|---|---|---|---:|---:|---:|
| NEU-ESC | sentiment | toxic | 83.01 ± 0.60 | 82.88 ± 1.56 | 82.48 ± 0.75 |
| NEU-ESC | topic | spam | 54.87 ± 2.29 | 52.89 ± 1.58 | 52.57 ± 1.52 |
| NEU-ESC | topic | club_events | 69.88 ± 1.04 | 67.66 ± 0.89 | 69.16 ± 2.54 |
| NEU-ESC | topic | help_share | 50.06 ± 3.00 | 49.44 ± 0.47 | 50.19 ± 0.98 |
| UIT-VSFC | sentiment | neutral | 55.10 ± 5.39 | 55.42 ± 2.33 | 54.21 ± 0.17 |
| UIT-VSFC | topic | others | 53.40 ± 2.23 | 54.81 ± 2.86 | 56.44 ± 2.10 |

**Reading:**
- Neither method changes macro-F1 significantly for the MTL model (every bootstrap interval includes 0).
- The smallest classes do not clearly benefit. Only UIT-VSFC topic `others` improves (+3.0 with weighted CE), which is
  about the size of its seed spread.
- Some rare classes are already handled well without any method (`toxic`: 83 F1). The weak ones (`spam`, `help_share`,
  UIT-VSFC `neutral`, around 50 to 55 F1) stay weak under both methods.

### Imbalance choice

Chosen by the selection rule on validation macro-F1 of the MTL model:

| Dataset | Chosen | Validation scores |
|---|---|---|
| NEU-ESC | **none** | none 0.7039, weighted CE 0.7038, focal 0.7024 |
| UIT-VSFC | **none** | none 0.8394, weighted CE 0.8381, focal 0.8377 |

---

## 7. Selection state after E3

| Dataset | Backbone | Loss | Imbalance | Final tag |
|---|---|---|---|---|
| NEU-ESC | ViSoBERT | sum (manual) | none (rule) | `sum` |
| UIT-VSFC | PhoBERT | sum (manual; same as the rule) | none (rule) | `sum` |

Stored with scores, seed counts, dates and the manual reason in `reports/tables/selection.json`.

---

## 8. Code changes made during the runs

| Change | Why | Where |
|---|---|---|
| `RESUME_FROM` accepts a list of folders and stops on a wrong path | session 2 silently restored nothing | `notebooks/kaggle_runner.ipynb` |
| `--time-budget-hours` (notebook setting `TIME_BUDGET_HOURS`) | session 5 was killed at the 12-hour limit; the trainer now stops starting runs that would end after the budget | `trainer/train.py`, notebook |
| `select --choose TAG --reason TEXT` | record a manual choice with its reason, the rule's choice and all scores | `trainer/selection.py`, `trainer/analysis.py` |
| Imbalance choice `none` adds no tag; grid entries that become the same run are trained once | `bestloss-bestimb` became the invalid tag `sum-none` and E4 could not start; E4 is now 6 runs instead of 12 | `trainer/train.py` |

Each change has a test (`tests/test_train.py`, `tests/test_analysis.py`).

---

## 9. Limits of these results

- **Three seeds.** Standard deviations are rough, and one unusual seed (UIT-VSFC PhoBERT MTL, seed 123) moves a mean
  by about half a point.
- **PCGrad time estimate was wrong** (planned 2.5×, measured 7.5× on NEU-ESC), which caused the 12-hour cut-off.
- **The loss choice on NEU-ESC is manual**, not the rule's choice. The reason is stated in section 5 and in `selection.json`.
- **RQ3 tests two classic loss-level methods** (weighted cross-entropy; focal loss, 2017). Newer methods (class-balanced
  loss, LDAM, logit adjustment, decoupled training, data augmentation) were not tested.
- **Checkpoints are only on Kaggle.** The downloaded zips have no `best.pt`; E7 needs the E1 checkpoints from the
  session 1 and session 4 outputs.
- **Validation is about 3 to 6 points above test for NEU-ESC topic**, so validation scores overstate topic macro-F1.
- **GradNorm learning rate** is 0.025 (the GradNorm paper's value), not the 1e-4 in the training spec.

---

## 10. What is left

| Experiment | Priority | Content | Estimated time |
|---|---|---|---:|
| E4 task-aware heads | Must | 6 runs; the final model, checkpoints kept | 1.1 h |
| E1b reference SMART | Should | 18 runs | 7.6 h |
| E4b direction | Could | 12 runs | 2.2 h |
| E5 low resource (10 / 25 / 50% of train) | Could | 72 runs | 4.3 h |
| E6 MLM | Could | 6 runs | 1.8 h |
| E7 cross-dataset | Could | evaluation only; needs the kept checkpoints in one `models/` folder | – |

After E4: `python -m trainer.analysis all` builds the final tables, significance tests and figures.

---

## 11. Where the files are

| What | Path |
|---|---|
| Main result tables | `reports/tables/results_neu-esc.md`, `reports/tables/results_uit-vsfc.md` |
| RQ1 / RQ2 / RQ3 tables | `reports/tables/rq1_st_vs_mtl.md`, `rq2_loss.md`, `rq3_imbalance.md`, `rq3_per_class.md` |
| Significance tests | `reports/tables/significance.md` |
| Time and memory per configuration | `reports/tables/cost.md` |
| Selection and its provenance | `reports/tables/selection.json` |
| Run folders (metrics, predictions, logs) | `models/<run_id>/` (not in git) |
| EDA | `reports/eda_findings.md`, `reports/figures/` |
