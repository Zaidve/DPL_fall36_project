# Results of the follow-up experiment E8: task-aware heads with the reference SMART

Written 2026-10-05. This report covers only E8 ("route B"). The main experiments (E1 to E5, E1b, E4b) are in
`results_main_experiments.md`; this report builds on them and does not change them.

Scores are on the **test** split, in percent, as mean ± standard deviation over 3 seeds (42, 123, 2026), unless a
line says "validation". "Interval" is the 95% paired-bootstrap interval of the macro-F1 difference (1,000 samples).
"Seeds" counts the seeds on which McNemar's test gives p < 0.05.

---

## 1. Summary

- **12 runs finished, 0 failed,** about 5.6 GPU-hours of training on a Kaggle T4.
- **The two gains do not add up.** Task-aware heads on top of the reference SMART give the same scores as the
  reference SMART alone on NEU-ESC ViSoBERT and on UIT-VSFC PhoBERT, and slightly lower topic macro-F1 on NEU-ESC
  PhoBERT (−0.81, interval −1.63 to +0.02).
- **The gain on NEU-ESC comes from the reference SMART, with or without task-aware heads.** On PhoBERT it adds about
  +1.2 sentiment and +0.8 topic macro-F1 to the task-aware model, as it added +1.0 and +1.6 to the plain MTL model.
- **Task-aware heads show no gain on NEU-ESC PhoBERT** (−0.19 sentiment, +0.03 topic against plain MTL).
- **The best NEU-ESC model in the project is unchanged:** PhoBERT, linear heads, reference SMART.

---

## 2. Why this experiment was run

Two earlier results suggested a combination worth testing:

- **E1b:** the reference "SMART" of the NEU-ESC paper's code (noise on token ids, no adversarial step) was the only
  loss option whose gain was significant on all 3 seeds, on NEU-ESC, and largest on PhoBERT.
- **E4:** task-aware heads gave one suggestive gain (UIT-VSFC topic, +0.86).

E8 asks whether the two gains add up. It was planned **after** the E1b test results were known, so it is an
exploratory follow-up, not part of the pre-planned selection. It does not use or change `selection.json`.

---

## 3. Design

| Runs | Dataset | Backbone | Model |
|---:|---|---|---|
| 3 | NEU-ESC | ViSoBERT | task-aware heads + reference SMART |
| 3 | NEU-ESC | PhoBERT | task-aware heads + reference SMART |
| 3 | UIT-VSFC | PhoBERT | task-aware heads + reference SMART |
| 3 | NEU-ESC | PhoBERT | task-aware heads + equal sum (baseline that E4 did not have for PhoBERT) |

Together with earlier runs this gives a 2 × 2 comparison (heads: linear or task-aware; loss: sum or reference SMART)
on three dataset / backbone pairs. The other three cells come from E1 (linear, sum), E1b (linear, reference SMART) and
E4 (task-aware, sum; ViSoBERT on NEU-ESC and PhoBERT on UIT-VSFC).

Everything else is the same as in the main experiments: same data, seeds, optimizer, early stopping and evaluation.
Code commit `11c3d6a`; experiment file `configs/experiment/e8_aware_smartref.yaml`.

---

## 4. Results

### 4.1 The 2 × 2 comparison

| Dataset / backbone | Heads | Loss | Validation | Sentiment mF1 | Topic mF1 | Mean mF1 | Acc (sent / topic) |
|---|---|---|---:|---:|---:|---:|---:|
| NEU-ESC / ViSoBERT | linear | sum | 0.7039 | 76.20 ± 0.23 | 61.95 ± 0.31 | 69.07 | 82.17 / 78.15 |
| | linear | reference SMART | 0.7028 | 76.53 ± 0.25 | 62.13 ± 0.25 | 69.33 | 83.21 / 79.24 |
| | task-aware | sum | 0.7054 | 76.34 ± 0.84 | 62.09 ± 0.97 | 69.22 | 82.25 / 77.94 |
| | task-aware | reference SMART | 0.7031 | 76.51 ± 0.40 | 62.16 ± 0.42 | 69.34 | 83.03 / 79.26 |
| NEU-ESC / PhoBERT | linear | sum | 0.7025 | 76.31 ± 0.27 | 61.41 ± 0.34 | 68.86 | 81.93 / 78.03 |
| | linear | reference SMART | **0.7177** | **77.29 ± 0.33** | **63.04 ± 0.60** | **70.17** | 82.87 / 79.26 |
| | task-aware | sum | 0.6983 | 76.12 ± 0.35 | 61.44 ± 0.45 | 68.78 | 81.85 / 77.75 |
| | task-aware | reference SMART | 0.7116 | 77.29 ± 0.50 | 62.23 ± 0.35 | 69.76 | 82.71 / 79.00 |
| UIT-VSFC / PhoBERT | linear | sum | 0.8394 | 82.08 ± 2.04 | 79.61 ± 0.82 | 80.84 | 93.82 / 88.78 |
| | linear | reference SMART | 0.8348 | 81.48 ± 0.21 | 80.39 ± 0.91 | 80.94 | 93.63 / 88.91 |
| | task-aware | sum | 0.8354 | 81.88 ± 1.11 | 80.47 ± 0.78 | 81.17 | 93.75 / 89.44 |
| | task-aware | reference SMART | 0.8324 | 81.58 ± 0.70 | 80.55 ± 0.36 | 81.06 | 93.79 / 89.26 |

Rows in the last position of each block and the "task-aware, sum" row for NEU-ESC PhoBERT are the new E8 runs.

### 4.2 Does adding task-aware heads to the reference SMART help?

Task-aware + reference SMART minus linear + reference SMART:

| Dataset / backbone | Task | Δ mF1 | Interval | Seeds |
|---|---|---:|---|---:|
| NEU-ESC / ViSoBERT | sentiment | −0.02 | −0.59 to +0.58 | 0 of 3 |
| | topic | +0.03 | −0.67 to +0.77 | 0 of 3 |
| NEU-ESC / PhoBERT | sentiment | 0.00 | −0.58 to +0.60 | 2 of 3 (directions differ) |
| | topic | −0.81 | −1.63 to +0.02 | 0 of 3 |
| UIT-VSFC / PhoBERT | sentiment | +0.09 | −0.75 to +1.05 | 1 of 3 |
| | topic | +0.16 | −0.61 to +0.91 | 0 of 3 |

**No.** Five of the six differences are within ±0.2, and the sixth (NEU-ESC PhoBERT topic) is negative.

### 4.3 Does the reference SMART still help the task-aware model?

Task-aware + reference SMART minus task-aware + sum:

| Dataset / backbone | Task | Δ mF1 | Interval | Seeds |
|---|---|---:|---|---:|
| NEU-ESC / ViSoBERT | sentiment | +0.17 | −0.54 to +0.85 | 2 of 3 |
| | topic | +0.07 | −0.93 to +1.08 | 2 of 3 |
| NEU-ESC / PhoBERT | sentiment | +1.17 | +0.56 to +1.80 | 2 of 3 |
| | topic | +0.79 | −0.23 to +1.79 | 3 of 3 |
| UIT-VSFC / PhoBERT | sentiment | −0.30 | −1.52 to +0.94 | 0 of 3 |
| | topic | +0.08 | −0.96 to +1.09 | 1 of 3 |

**On NEU-ESC PhoBERT, yes;** elsewhere there is no measurable change in macro-F1. This is the same pattern as in E1b
for the plain MTL model: a clear effect on NEU-ESC PhoBERT, a small one on NEU-ESC ViSoBERT (where accuracy rises by
about 1 point on both tasks but macro-F1 barely moves), and none on UIT-VSFC. McNemar's test compares accuracy, which
is why some rows are significant on several seeds while the macro-F1 interval includes 0.

### 4.4 Task-aware heads on NEU-ESC PhoBERT

Task-aware + sum minus linear + sum: sentiment −0.19 (interval −0.78 to +0.41), topic +0.03 (−0.75 to +0.85).
**No gain.** This is a third dataset / backbone pair for RQ4, and it agrees with NEU-ESC ViSoBERT (no effect). The
only suggestive gain of the task-aware heads remains UIT-VSFC topic (+0.86).

### 4.5 The combined model against plain MTL

Task-aware + reference SMART minus linear + sum:

| Dataset / backbone | Sentiment Δ mF1 (interval) | Topic Δ mF1 (interval) |
|---|---|---|
| NEU-ESC / ViSoBERT | +0.31 (−0.37 to +1.05) | +0.22 (−0.75 to +1.21) |
| NEU-ESC / PhoBERT | +0.98 (+0.26 to +1.68) | +0.82 (−0.10 to +1.73) |
| UIT-VSFC / PhoBERT | −0.50 (−1.65 to +0.75) | +0.94 (−0.06 to +2.03) |

The combined model is better than plain MTL on NEU-ESC PhoBERT, but the reference SMART alone is at least as good
there (+0.98 and +1.63 in E1b).

---

## 5. Cost

| Model | NEU-ESC ViSoBERT | NEU-ESC PhoBERT | UIT-VSFC PhoBERT |
|---|---:|---:|---:|
| linear, sum | 18 min / run | 18 min | 5 min |
| task-aware, sum | 14 min | 18 min | 5 min |
| linear, reference SMART | 44 min | 49 min | 12 min |
| task-aware, reference SMART | 45 min, 4.8 GB | 38 min, 4.6 GB | 11 min, 2.9 GB |

Minutes per run are means over 3 seeds and depend on when early stopping ended each run. The reference SMART costs
about 2 to 2.7 times a plain run; the task-aware heads add nothing measurable.

---

## 6. What this means

- **Best model on NEU-ESC:** PhoBERT with linear heads and the reference SMART (validation 0.7177; test 77.29
  sentiment, 63.04 topic macro-F1). It has the highest validation score of every NEU-ESC configuration trained in the
  project, and E8 did not improve on it.
- **The final model of the planned selection is unchanged:** ViSoBERT on NEU-ESC and PhoBERT on UIT-VSFC, task-aware
  heads, equal sum. It is still not the best-scoring NEU-ESC model, for the reasons given in section 9 of
  `results_main_experiments.md`.
- **RQ4 (task-aware heads):** the extra PhoBERT result supports the earlier answer. No reliable gain.
- **RQ2 (the SMART gain):** the token-noise version helps on NEU-ESC PhoBERT whichever heads are used. It still shows
  no macro-F1 gain on UIT-VSFC.

---

## 7. Limits

- **Exploratory.** E8 was designed after the E1b test results were seen. Its results describe what happens; they are
  not a pre-planned confirmation.
- **Three seeds.** Differences below about 1 point cannot be separated from seed noise.
- **Three dataset / backbone pairs.** XLM-R and UIT-VSFC ViSoBERT were not included.
- **One setting of the reference SMART** (the reference code's values) and of the task-aware heads (both directions,
  auxiliary weight 0.5).
- **Why task-aware heads add nothing on top of the reference SMART is not tested.** The runs show the outcome, not the
  cause.

---

## 8. Future work

| Item | Note |
|---|---|
| Joint selection of backbone and loss on validation (route C) | Would select PhoBERT + reference SMART on NEU-ESC; E3, E4 and E5 would then be rerun on that setup (about 17 to 28 GPU-hours) |
| Understand the reference SMART | It works as token-level noise. Compare it with plain token dropout or masking at the same rate |
| Stronger settings for embedding-level SMART | Only one setting was tested in E2 |
| More seeds for the NEU-ESC PhoBERT comparisons | To narrow the intervals in sections 4.2 and 4.3 |

---

## 9. Where the files are

| What | Path |
|---|---|
| Experiment file | `configs/experiment/e8_aware_smartref.yaml` |
| Run folders | `models/<dataset>__<backbone>__mtlaware__seed<k>__smartref` and `models/neu-esc__phobert__mtlaware__seed<k>__sum` (not in git) |
| Main experiments | `reports/results_main_experiments.md` |
| Tables for the main experiments | `reports/tables/` |
