# What can be done next: hand-off for the next working session

Written 2026-10-09. Read this first, then [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for the description of every file,
command and result. `NEXT_STEPS.md` is from before training started and is out of date.

---

## 1. Where the project stands

- **Project:** multi-task learning (MTL) for Vietnamese educational feedback. Two tasks per text: sentiment and topic.
  Two datasets (NEU-ESC, UIT-VSFC), three backbones (PhoBERT, ViSoBERT, XLM-R), 3 seeds (42, 123, 2026).
- **Experiments are done:** 234 training runs finished on Kaggle, none failed, 52.8 GPU-hours on a T4.
  Not run: E6 (MLM) and E7 (cross-dataset).
- **All five research questions are answered,** mostly with "no" (section 3).
- **The final report, the proposal, the slides and the speaking script are written** (section 5).
- **Git:** branch `main`, in sync with `origin` (`https://github.com/Zaidve/DPL_fall36_project`), last commit `9e1fd73`.
  `reports/` is ignored by git except `reports/eda_summary.json` and `reports/tables/selection.json`.

---

## 2. How work is done here

| Thing | How |
|---|---|
| Training | On **Kaggle** (GPU T4, Internet On) with `notebooks/kaggle_runner.ipynb`. The user runs it; the assistant cannot. Use **Save Version → Save & Run All → Run with GPU**. |
| Kaggle notebook copy | The user's Kaggle notebook is an imported copy. A change to a notebook cell must be pasted into Kaggle by hand; pushing to GitHub only updates the training code that the notebook clones. |
| Kaggle settings cell | `EXPERIMENT`, `ONLY` (one substring of the run id), `MAX_STEPS`, `TIME_BUDGET_HOURS` (9.5), `RESUME_FROM` (a `models/` path or a list). |
| Kaggle quota | About 30 GPU-hours per week. A session stops at 12 hours. |
| After a session | The user downloads `results_<experiment>.zip` (runs without checkpoints + tables). Then locally: `python -m trainer.analysis merge --from <zip>/models --to models --apply`, then `status`, `check`, `significance`, `tables`, `figures`. |
| Local machine | Windows, RTX 5050 8 GB, `.venv` with torch 2.11.0+cu128. **Keep this environment unchanged.** Used for tests, analysis and quick checks, not for real training. |
| Tests | `for t in tests/test_*.py; do .venv/Scripts/python.exe $t | tail -1; done` (119 tests, CPU, no downloads). |
| Commits | The user commits and pushes. Give the commands; do not commit or push unless asked. |
| Selection | Always on validation scores, never on test (`trainer/selection.py`). A manual choice needs `--choose TAG --reason TEXT`. |

Run ids look like `neu-esc__visobert__mtl__seed42__sum`. Run folders are in `models/` (not in git): metrics,
predictions with class probabilities, logs, config. **Checkpoints (`best.pt`) exist only in the Kaggle outputs.**

---

## 3. Routes already done

| Route | What | Runs | Result |
|---|---|---:|---|
| E1 baseline | Single-task vs MTL, 3 backbones × 2 datasets | 54 | MTL does not consistently beat single-task (all differences below 1 macro-F1 point). Backbones chosen: ViSoBERT (NEU-ESC), PhoBERT (UIT-VSFC). |
| E2 loss balancing | Uncertainty, PCGrad, GradNorm, SMART on embeddings | 24 | None significantly better than the equal sum; 1 to 7.5× the cost. |
| Loss choice | `sum` for both datasets, a recorded manual choice | – | The rule alone would pick PCGrad on NEU-ESC (+0.004 validation, 7.5× the time). |
| E3 imbalance | Focal loss, weighted cross-entropy | 36 | No reliable help. Rule chose "none". |
| E4 task-aware heads | The proposed heads = the final model | 6 | One suggestive gain: UIT-VSFC topic +0.86 (bootstrap interval excludes 0, 2 of 3 seeds). |
| E5 low resource | 10 / 25 / 50% of train | 72 | MTL does not help more with less data; on UIT-VSFC sentiment it is 4.8 / 4.0 points below single-task at 10 / 25%. |
| E1b reference SMART | The paper code's token-id noise, all backbones | 18 | The only effect significant on all 3 seeds: about +1 accuracy point on NEU-ESC. Not on UIT-VSFC. |
| E4b head direction | One direction of the task-aware heads at a time | 12 | Does not confirm the intended mechanism. |
| Route A | Keep the planned final model; report E1b as a finding | – | In effect. |
| Route B = E8 | Task-aware heads + reference SMART | 12 | The two gains do not add up. |
| Write-up | Reports, proposal, slides, script (section 5) | – | Done. |

**Selection now:** NEU-ESC = ViSoBERT + `sum` + no imbalance handling; UIT-VSFC = PhoBERT + `sum` + none.
Final model = `mtlaware/sum` on those backbones.
**Best NEU-ESC model (not the selected one):** PhoBERT, linear heads, reference SMART: validation 0.7177, test 77.29
sentiment and 63.04 topic macro-F1.

---

## 4. Routes still open

### 4.1 Built, not run

| Route | What | Cost | Notes |
|---|---|---|---|
| E6 MLM ablation | Final model + masked-language-model loss, 6 runs | about 1.8 GPU-hours | `EXPERIMENT = 'e6_mlm'`. Low value: the paper shows no gain and no research question depends on it. |
| E7 cross-dataset | Evaluate each sentiment model on the other dataset (3 shared labels); no training | short | `EXPERIMENT = 'e7_cross_dataset'`. Needs every kept checkpoint in one `models/`. They are in three Kaggle outputs (session 1: NEU-ESC PhoBERT; session 4: the rest of E1; the E4 session). Kaggle attaches one version of a notebook's output at a time, so turn the older versions' `models` folders into Kaggle Datasets, attach all, and set `RESUME_FROM` to the list of paths. |

### 4.2 Larger routes that change the main results

| Route | What | Cost | Notes |
|---|---|---|---|
| Route C2: joint selection | Choose backbone and loss together on validation. This selects PhoBERT + reference SMART on NEU-ESC (UIT-VSFC stays). Rerun E3, E4, E4b and E5 for NEU-ESC on that setup. | about 17 GPU-hours (66 runs); about 28 if E2 is also rerun on PhoBERT | Needs code: let the selection choose backbone and loss jointly and make `smartref` a candidate. The report and proposal must then be rewritten. Only worth it with about two weeks. |
| Route C1: switch the loss only | Keep the backbones, make reference SMART the loss | about 11 GPU-hours | **Not recommended.** On the selected backbones `smartref` is below `sum` on validation, so this would be choosing on test results. |

### 4.3 Smaller additions that strengthen a claim

| Route | What | Cost |
|---|---|---|
| Post-hoc logit adjustment | A newer imbalance method applied to the saved predictions (they store class probabilities). Choose τ on validation, report on test. Adds one recent method to RQ3. | No GPU. New analysis code + a test. |
| More seeds | 2 extra seeds for the comparisons that came close (UIT-VSFC topic with task-aware heads; NEU-ESC PhoBERT with reference SMART) | about 2 to 4 GPU-hours; needs a small experiment file |
| Stronger embedding-level SMART | Larger weight or noise size; only one setting was tested (weight 0.02, ε 1e-5, 1 step) | about 3 to 6 GPU-hours |
| Token-noise comparison | Compare the reference SMART with plain token dropout or masking at the same rate | about 3 to 5 GPU-hours; needs new code |
| VnCoreNLP segmentation | Rerun PhoBERT with the segmenter it was pre-trained with (all runs used underthesea) | Preprocessing on Kaggle (`RUN_PREPROCESS = True`, `INSTALL_VNCORENLP = True`), then PhoBERT reruns |

### 4.4 No-GPU work

| Route | Notes |
|---|---|
| Compile the LaTeX report | `reports/latex/project_report.tex` has never been compiled (no LaTeX on this machine). Use Overleaf or `pdflatex`; fix any error. |
| Fill in placeholders | `[Department / University]`, `[City, Vietnam]`, `[email]` in the report (all three formats), the proposal and slide 1. Add co-authors if any. |
| Verify the references | They were written from memory. Check authors, venues, years and pages, especially ViSoBERT, UIT-VSFC, McNemar and Koehn. |
| Clean two cosmetic table issues | `rq4_task_aware.md` lists the task-aware row twice (the `sum` model and the final model are the same run). An `e6_mlm` table is written although E6 has no runs. Both are in `trainer/tables.py`. |
| Update or remove `NEXT_STEPS.md` | It still says nothing has been trained. |
| Keep the report builders | The scripts that generated the Word / LaTeX report and proposal were in a temporary folder and are gone after the session. To change the report, edit the `.docx` / `.tex` directly or rebuild the generators. |
| Vietnamese versions | Final report or slides in Vietnamese, if the course needs them. |
| Align the documents | The proposal in `reports/` has no E6, E7, Runs or Priority columns; the slide deck numbers experiments 1 to 7; the final report uses E1 to E8. |

---

## 5. Where the documents are

| Document | Path | Notes |
|---|---|---|
| Final report | `reports/DPL_Project_Report_IEEE.docx`, `.pdf`, `reports/latex/project_report.tex` (+ `latex/figures/`) | IEEE format, English, 6 pages, 12 tables, 3 figures. LaTeX not compiled. |
| Results, main experiments | `reports/results_main_experiments.md` | E1–E5, E1b, E4b |
| Results, E8 | `reports/results_e8_aware_smartref.md` | Route B |
| Earlier progress report | `reports/progress_e1_e3.md` | E1–E3 in more detail |
| All result tables | `reports/tables/*.md` and `.csv` | `results_*`, `rq1`–`rq5`, `significance`, `cost`, `selection.json` |
| Figures | `reports/figures/` | EDA + 13 result figures |
| Proposal (English, IEEE) | `reports/DPL_Proposal_MTL_IEEE_EN.docx` | Plan only, no results; E6, E7, Runs and Priority removed |
| Other proposal versions | `C:\Users\ADMIN\Downloads\DPL_Proposal_MTL_IEEE.docx` (Vietnamese IEEE), `DPL_Proposal_MTL_Vietnamese_Educational_Feedback.docx` (course template) | Still have E6 / E7 and all columns |
| Slides | `C:\Users\ADMIN\Downloads\MTL Proposal Presentation v5.pptx` (latest, 15 slides) | Also an online deck: https://claude.ai/artifact/5WjxQdqiodo3yfKM4veDNU (older: 16 slides, not updated after v2) |
| Speaking script | `reports/presentation_script.md` | English and Vietnamese, follows the v5 slides |
| Kaggle result zips | `C:\Users\ADMIN\Downloads\results_e*.zip` | Already merged into `models/` |

`reports/` is not in git (except two files), so these documents exist only on this machine.

---

## 6. Things learned the hard way (do not undo)

- **ViSoBERT tokenizer:** transformers 5 converts its sentencepiece model wrongly; `utils/dataset.py` uses its own
  loader (`tokenizer: sentencepiece_fairseq`).
- **PCGrad:** keep the memory-lean Gram-matrix version in `utils/loss_function.py`. On a T4 it is 4.1–7.5× slower
  than `sum`, not 2.5×.
- **A session cut at 12 hours** kept its finished runs once, but the run in progress is lost. Use
  `TIME_BUDGET_HOURS`.
- **`RESUME_FROM`** must point at the newest output; a wrong path now stops the notebook.
- **Selection never reads test scores.** Keep it that way; a manual choice must carry a reason.
- **An imbalance choice of "none"** adds no tag; grid entries that become the same run are trained once.
- **`reports/tables/selection.json` must be committed and pushed** before a Kaggle run that uses the selected
  backbone, loss or final model.
- **Changing a default in `configs/default.yaml`** now would make new runs differ from the 234 finished ones.
- **Tests must not download anything;** use the tiny random encoder and fake tokenizers from the existing tests.

---

## 7. Suggested order

1. **If the deadline is close:** do section 4.4 only (compile LaTeX, placeholders, references), then submit.
2. **If a few GPU-hours are available:** post-hoc logit adjustment first (free), then "more seeds" or the
   token-noise comparison.
3. **If about two weeks are available:** Route C2, the only route that changes the headline model.
