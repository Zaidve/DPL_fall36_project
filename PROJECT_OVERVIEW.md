# DPL_project: current state

Last updated 2026-10-09. The pipeline is built and tested, and **the experiments are done: 234 training runs
finished on Kaggle, none failed** (E1, E1b, E2, E3, E4, E4b, E5 and the follow-up E8; 52.8 GPU-hours on a T4).
The results, the selection and the final report are written (section 4). Not run: E6 (MLM) and E7 (cross-dataset).
[NEXT_STEPS.md](NEXT_STEPS.md) is the hand-off note from before training started and is now out of date; this
file is the current reference.

- **Project:** multi-task learning (MTL) for Vietnamese educational feedback, two tasks per text:
  **sentiment** and **topic** (`classification` in the raw data and in `utils/dataloader.py`).
- **Datasets:** NEU-ESC and UIT-VSFC.
- **Reference code:** `hung20gg/multi-task-bert` (the NEU-ESC paper's code), kept locally at
  `../multi-task-bert-master/`, with its analysis files (`architecture_baseline.md`,
  `preprocess_gap_analysis.md`, `train_gap_analysis.md`). Our code follows the specs
  (`eda_spec_1.md`, `preprocess_spec.md`, `losses_spec.md`, `models_spec.md`, `train_spec.md`) and was
  written from them, not copied.
- **Where things run:** EDA, preprocessing, tests and quick checks locally; the real training runs on
  **Kaggle** (T4, 16 GB). The same code runs in both places (paths switch automatically).

---

## 1. Folder layout

```
DPL_project/
├── datasets/                raw data (read-only)
│   ├── neu-esc/             {train,val,test}_set.csv
│   └── uit-vsfc/            {train,dev,test}/{sents,sentiments,topics}.txt, README.txt
├── data/processed/          preprocessing output = training input (committed, so a git clone has it)
│   ├── neu-esc.parquet, uit-vsfc.parquet
│   └── neu-esc_subsets.json, uit-vsfc_subsets.json      nested 10/25/50% train subsets
├── configs/
│   ├── default.yaml         preprocess:, data:, model:, loss:, train:
│   ├── teencode.yaml        28 teencode entries (off by default)
│   ├── model/               phobert.yaml, xlmr.yaml, visobert.yaml
│   ├── data/                neu-esc.yaml, uit-vsfc.yaml (max_len per backbone)
│   └── experiment/          e1_baseline … e7_cross_dataset.yaml, e8_aware_smartref.yaml
├── utils/
│   ├── dataloader.py        raw loaders + label names (used by the EDA and preprocessing)
│   ├── preprocess.py        python -m utils.preprocess
│   ├── dataset.py           training DataLoaders from the parquet + tokenizer loading
│   ├── loss_function.py     task losses, MLM, SMART, loss combiners
│   ├── common.py            paths (local/Kaggle), seeds, JSON/YAML I/O, env info, logging
│   └── config.py            config loading with `extends`, dotted overrides
├── architecture/            MTLModel + heads (linear, mlp, task_aware) + MLM head + legacy helpers
├── trainer/
│   ├── train.py             python -m trainer.train --experiment configs/experiment/<name>.yaml
│   ├── runs.py              run ids, run folders, resume check, checkpoints, prediction files
│   ├── evaluate.py          metrics, predictions with row ids, McNemar test
│   ├── status.py            run matrix, progress, GPU-hour estimate + session plan, merge, consistency checks
│   ├── results.py           load finished runs, mean ± std over seeds
│   ├── selection.py         write selection.json in stages (validation scores only)
│   ├── significance.py      McNemar per seed, paired bootstrap, standard comparison pairs
│   ├── tables.py            main results table + RQ1–RQ5, cost, MLM ablation (CSV + Markdown)
│   ├── figures.py           learning curves, low resource, per class, confusion, task weights, RQ4 vs V
│   ├── cross_eval.py        E7: kept checkpoints on the other dataset, 3 shared sentiment labels
│   └── analysis.py          python -m trainer.analysis matrix | status | merge | check | select |
│                            significance | tables | figures | cross-eval | all
├── notebooks/
│   ├── 01_eda.ipynb         EDA (outputs in reports/)
│   └── kaggle_runner.ipynb  runs one experiment on Kaggle (also runs locally)
├── reports/                 gitignored, except eda_summary.json and tables/selection.json (the code reads these two)
│   ├── eda_summary.json, eda_findings.md             EDA
│   ├── tables/              EDA + preprocessing tables, result tables (results_*, rq1 … rq5, cost),
│   │                        significance, run_matrix, run_status, selection.json
│   ├── figures/             EDA figures + result figures (rq5_*, rq3_*, rq2_*, rq4_*, confusion_*)
│   ├── results_main_experiments.md, results_e8_aware_smartref.md, progress_e1_e3.md
│   ├── DPL_Project_Report_IEEE.docx / .pdf, latex/project_report.tex       final report
│   ├── DPL_Proposal_MTL_IEEE_EN.docx                                       proposal
│   └── presentation_script.md                                              speaking script (EN + VI)
├── models/                  run outputs, one folder per run (gitignored; 234 finished runs, no checkpoints locally)
├── tests/                   9 test files, 120 tests (+ fixtures: legacy model, fake run folders / full fake matrix)
├── .venv/                   Python 3.11.0 (not in git)
└── PROJECT_OVERVIEW.md      this file
```

---

## 2. Data

| Dataset | train | val | test | sentiment classes | topic classes |
|---|---|---|---|---|---|
| NEU-ESC | 23,048 | 3,305 | 6,613 | 4 | 10 |
| UIT-VSFC | 11,426 (11,425 after dedup) | 1,583 | 3,166 | 3 | 4 |

- **UIT-VSFC labels** (official, `README.txt`): sentiment `negative, neutral, positive`;
  topic `lecturer, training_program, facility, others`.
- **NEU-ESC labels** (dataset card `hung20gg/NEU-ESC`; train class counts match the card): sentiment
  `neutral, positive, negative, toxic`; topic `spam, news, academic, other, service, jobs_recruitment,
  personal_affairs, social_affairs, help_share, club_events`. Preprocessing and the EDA were re-run with these
  names (only the name columns changed; ids, texts and subsets are identical).
- Both datasets are already lowercased with spaces around punctuation. UIT-VSFC replaced emoticons
  with words (`:)` → `colonsmile`) and lecturer names with `wzjwz<digits>`.

---

## 3. What each stage produced

### 3.1 EDA (`notebooks/01_eda.ipynb` → `reports/`)

- All integrity checks pass (49,141 rows).
- Strong imbalance: imbalance ratio 27 / 36 (NEU sentiment / topic), 12 / 16 (UIT). The majority-class
  baseline reaches accuracy 0.44–0.72 but macro-F1 only 0.06–0.22, so **macro-F1 is the main metric**.
- Sentiment × topic dependence: Cramér's V 0.18 (NEU, weak), 0.35 (UIT, strong).
- No leakage between splits; 1 duplicate and 1 label conflict in UIT-VSFC train.
- Adversarial validation AUC ≈ 0.5: train and test look alike.
- 30 warnings: long-text classes (NEU topics `jobs_recruitment`, `social_affairs`, `club_events`) lose
  ~18–23% of texts at the p95 `max_len`.

### 3.2 Preprocessing (`python -m utils.preprocess` → `data/processed/`, `reports/tables/preprocess_*.csv`)

Light cleaning only (emoji, slang, punctuation kept). Changes on train (NEU / UIT):

| Step | NEU-ESC | UIT-VSFC |
|---|---|---|
| Tone marks to new style (`hòa` → `hoà`) | 1,581 | 215 |
| Phone numbers → `phonetoken` (tested regex from the gap analysis) | 4 | 0 |
| Masked names → `nametoken` | 0 | 211 |
| Repeated characters capped at 2 | 4 | 1 |
| Whitespace | 13 | 5 |
| Duplicates removed (train only) | 0 | 1 |
| Train rows in test (leakage) | 0 | 0 |

- Output columns: `id, text, text_clean, text_seg, sentiment, topic, sentiment_name, topic_name, split, dataset, in_test`.
- `text_seg` (PhoBERT input) is word-segmented with **underthesea** (VnCoreNLP needs Java, not installed
  locally). Only runs of plain words are segmented; the segmenter's spelling changes are ignored, so
  `text_seg.replace('_', ' ') == text_clean`.

### 3.3 Backbones and tokenizers (`configs/model/`, `configs/data/`)

| Key | Hugging Face id | Input | Tokenizer | max_len NEU / UIT |
|---|---|---|---|---|
| `phobert` | `vinai/phobert-base-v2` | `text_seg` | AutoTokenizer | 80 / 32 |
| `xlmr` | `FacebookAI/xlm-roberta-base` | `text_clean` | AutoTokenizer | 96 / 48 |
| `visobert` | `uitnlp/visobert` | `text_clean` | **own sentencepiece loader** | 96 / 48 |

`max_len` = p95 of train token length rounded up to 16 (from the EDA).

> ⚠️ **ViSoBERT and transformers 5:** `AutoTokenizer` converts ViSoBERT's sentencepiece **BPE** model as if it
> were XLM-R's Unigram model and splits words wrongly (`thầy` → `▁th ầy`; 5/3,000 texts match the original).
> `utils/dataset.py` loads the original sentencepiece model with fairseq ids instead. Checked on the real
> model: masked-word loss 1.04 (ours) vs 5.79 (AutoTokenizer). Needs `pip install sentencepiece`.

### 3.4 Models (`architecture/`)

`MTLModel(backbone, tasks, head, ...)`: shared encoder → pooled vector (`cls` or `mean`) → heads.
One task = single-task model (same code).

| Head | What | Source |
|---|---|---|
| `linear` (default) | Dropout → Linear per task | spec |
| `mlp` | Linear → SiLU → LayerNorm → Linear per task | legacy B1 (parameter counts match exactly) |
| `task_aware` | two-stage heads; each task reads the other task's prediction through a gated label embedding; `cross` = both / sent_from_topic / topic_from_sent | ours (RQ4) |

Also: MLM head (pretrained, tied), `LegacyFlags` and a legacy checkpoint loader (a legacy B1 checkpoint gives
identical logits), `shared_parameters / task_parameters / last_shared_layer_params` for the optimizer,
PCGrad and GradNorm.

### 3.5 Losses (`utils/loss_function.py`)

- Task losses: `ce`, `weighted_ce`, `focal`, label smoothing.
- Combiners: `sum`, `fixed`, `uncertainty`, `gradnorm`, `pcgrad`, `dwa`.
- `pcgrad_backward` is memory-lean: it keeps one gradient copy per task and projects through the T×T matrix of
  dot products (every projected gradient is a combination of the task gradients), so XLM-R + PCGrad needs 8.4 GB
  instead of running out of memory (~12 GB). Same results as the direct version (max difference 4e-6).
- SMART: `mode="embeddings"` (ours, real adversarial step) and `mode="token_ids"` (exact reproduction of the
  reference code, checked against it: 1.09292293 both), MLM masking and loss.

### 3.6 Trainer (`trainer/`)

```bash
python -m trainer.train --experiment configs/experiment/e1_baseline.yaml [--only SUBSTR] [--dry-run] [--max-steps N]
                        [--time-budget-hours H]
```

- Expands datasets × backbones × grid × seeds into runs; run id
  `{dataset}__{backbone}__{mode}__seed{k}__{tag}` (e.g. `neu-esc__phobert__mtl__seed42__sum`).
- Modes: `st_sentiment`, `st_topic`, `mtl`, `mtlaware`. Tags (combine with `-`): `sum`, `unc`, `pcgrad`,
  `gradnorm`, `dwa`, `fixed<a>`, `focal`, `wce`, `smartemb`, `smartref`, `mlm`, `frac<x>`,
  `cross_sent`, `cross_topic`, and `bestloss` / `bestimb` / `final` (filled in from `selection.json`).
- Per run: seeded; AdamW (encoder 2e-5, heads 1e-4, no decay on 1-D params); warm-up + linear decay per step;
  gradient clipping; fp16 (off for PCGrad/GradNorm); NaN guard; early stopping on validation macro-F1
  (patience 3); reload best epoch; predictions + metrics on validation and test.
- Resume: a run is done when `metrics.json` exists (written last, atomically). A failed run writes
  `error.txt` and the next run starts. `--max-steps` runs go to `models/_debug/`.
- Run folder: `config.yaml, env.json, run.log, train_log.csv, predictions_validation.csv,
  predictions_test.csv, metrics.json` (+ `best.pt`, fp16, only for runs E7 needs: E1 `st_sentiment/sum`,
  E1 `mtl/sum`, the E4 final model).
- `expand_runs(exp, placeholders=True)` plans runs before `selection.json` (or one of its stages) exists: it keeps
  `best` / `bestloss` / `bestimb` / `final` in the run id; such runs are refused by the trainer.
- `--time-budget-hours H` stops starting runs that would probably end after H hours (time used + the longest run
  of the session). A Kaggle session then ends normally and keeps its output; re-run to continue.
- An imbalance choice of `none` adds no tag (`bestloss-bestimb` → `sum`). Grid entries that become the same run
  after the selection are trained once (E4 is 6 runs with the current selection, not 12).
- A grid entry may list `datasets` and / or `backbones` to run only on those (used by E8).

### 3.7 Experiments (`configs/experiment/`)

| File | RQ | Runs | Priority | Status |
|---|---|---|---|---|
| `e1_baseline` | RQ1: does MTL beat single-task on macro-F1? | 54 | Must | done (10.6 h) |
| `e1b_smart_ref` | RQ2: reference "SMART" | 18 | Should | done (9.7 h) |
| `e2_loss` | RQ2: uncertainty, PCGrad, SMART, GradNorm | 24 | Should | done (13.7 h) |
| `e3_imbalance` | RQ3: focal / weighted CE | 36 | Must (focal), Should (wce) | done (6.3 h) |
| `e4_task_aware` | RQ4: task-aware heads (= the final model) | 6 | Must | done (0.9 h) |
| `e4b_direction` | RQ4: which direction helps | 12 | Could | done (2.0 h) |
| `e5_low_resource` | RQ5: 10 / 25 / 50% train | 72 | Could | done (3.9 h) |
| `e6_mlm` | MLM ablation | 6 | Could | not run |
| `e7_cross_dataset` | train on one dataset, test on the other | 0 (evaluation only) | Could | not run |
| `e8_aware_smartref` | follow-up: task-aware heads + reference SMART | 12 | Could | done (5.6 h) |

Hours are training time on a Kaggle T4. With the current selection the matrix has 240 runs; 234 are finished.

### 3.8 Tracking the matrix (`python -m trainer.analysis ...`)

| Command | What it does |
|---|---|
| `matrix` | All runs with priority (Must / Should / Could) → `reports/tables/run_matrix.csv` |
| `status [--models DIR ...]` | done / failed / partial / missing / blocked per experiment, failed runs with their error line, remaining GPU-hours (from finished runs, else spec defaults × strategy factor) and a plan of `--only` groups that fit in 11-h Kaggle sessions, Must first → `run_status.csv` |
| `merge --from DIR ... --to DIR [--apply]` | copies finished runs from several session outputs into one `models/`; never overwrites a finished run, reports conflicts |
| `check [--models DIR ...]` | config drift between seeds, library versions, best epoch 1, skipped steps, prediction counts, missing kept checkpoints |
| `select --stage backbone\|loss\|imbalance [--allow-partial] [--force] [--choose TAG --reason TEXT]` | writes one stage of `reports/tables/selection.json` (below); `--choose` records a manual choice among the candidates, with its reason, the rule's own choice and all scores |
| `select --stage joint [--force] [--choose BACKBONE/TAG --reason TEXT]` | backbone and loss chosen together over every finished `mtl` pair (E1 `sum`, E2 tags, E1b `smartref`), instead of the backbone and loss stages; a dataset whose backbone and loss do not change keeps its imbalance stage |
| `select --explain` | current choices with their validation scores, seed counts and date |
| `significance [--n-boot N]` | standard pairs (RQ1 ST vs MTL per backbone; RQ2 strategies vs sum, smartref vs smartemb; RQ3 imbalance; RQ4 linear vs task-aware; RQ5 single task vs MTL and vs the final model at each train fraction; final vs best single task): McNemar per seed + paired bootstrap of Δ macro-F1 → `significance.csv` |
| `tables` | `results_<dataset>` (prior work for NEU-ESC / our reproduction / ours, final model in bold), `rq1_st_vs_mtl`, `rq2_loss`, `rq3_imbalance` + `rq3_per_class`, `rq4_task_aware` (with Cramér's V), `rq5_low_resource`, `cost`, `e6_mlm` |
| `figures` | `rq5_curves_*`, `rq5_low_resource_*`, `rq3_per_class_*`, `confusion_*_{sentiment,topic}`, `rq2_weights_*`, `rq4_gain_vs_v` |
| `cross-eval [--device D] [--force]` | E7: every run with `best.pt` and a sentiment head predicts the other dataset's test split on 3 shared labels (by name; toxic → negative), with the target's text column and `max_len`; in-domain 3-class score from its own predictions → `e7_cross_dataset` (train × test, mean ± std) + `e7_cross_dataset_runs`; cached per run in `cross_eval.json` |
| `all` | significance + tables + figures (+ cross-eval when checkpoints exist) |

Table cells: test percent `mean ± std` over seeds; `(n=2)` if a seed is missing; `†` = McNemar p < 0.05 on every
seed in one direction vs the row's baseline (the single-task run for main-table rows); `*` = best in the column.
Everything works on a partial matrix and prints what it skipped (on the full fake matrix: ~20 s).

Before any run finishes, `status` estimates E1 at ~30 GPU-hours (4 sessions). While Must runs wait for the loss
stage, the session plan puts E2 first. In the Kaggle runner, `EXPERIMENT = 'e7_cross_dataset'` runs `cross-eval`.
Real check (ViSoBERT `mtl/sum` smoke run on UIT-VSFC, 2 short epochs): 3-class macro-F1 73.9% in-domain vs
36.4% on NEU-ESC (it labels most NEU forum posts negative; NEU-ESC is 69% neutral).

**Selection workflow** (`trainer/selection.py`): always on the mean over seeds of the **validation**
`macro_f1_mean` (asserted: test scores are never read).

| After | Command | Chooses | Candidates | Ties (< 0.001) |
|---|---|---|---|---|
| E1 | `select --stage backbone` | `backbone` | E1 `mtl/sum` per backbone | better single-task mean, then name |
| E2 | `select --stage loss` | `loss_tag` | `sum` + every E2 tag that started | `sum` |
| E3 | `select --stage imbalance` | `imbalance_tag`, `final_tag` | none / focal / wce on top of the loss tag | none |

A stage is refused (with the missing run ids) if a candidate has fewer than 3 seeds (`--allow-partial` overrides),
if the previous stage is missing, or if it is already written (`--force` re-selects and drops the later stages).
If E2 is skipped, `loss` picks `sum` with a note. Each choice is stored with its provenance.

42 runs keep a checkpoint (E1 `st_sentiment/sum` and `mtl/sum`, 18 each; E4 final model, 6), about 12–14 GB in
fp16. They exist only in the Kaggle outputs (sessions 1 and 4 for E1, the E4 session); the downloaded zips have none.
E2–E6 use the best backbone / loss per dataset from `reports/tables/selection.json`, written from **validation**
scores.

---

## 4. Results (234 runs, test macro-F1, mean over 3 seeds)

**Selection** (`reports/tables/selection.json`):

| Dataset | Backbone | Loss | Imbalance | Final model |
|---|---|---|---|---|
| NEU-ESC | ViSoBERT (validation 0.7039) | `sum`, manual choice (the rule picks PCGrad, 0.7080, at 7.5× the time) | none (rule) | `mtlaware/sum`: 76.34 sentiment, 62.09 topic |
| UIT-VSFC | PhoBERT (validation 0.8394) | `sum` (same as the rule) | none (rule) | `mtlaware/sum`: 81.88 sentiment, 80.47 topic |

**Answers to the research questions:**

| RQ | Answer |
|---|---|
| RQ1 | Plain MTL does not consistently beat single-task training. All 12 differences are below 1 point (−0.94 to +0.93); none is significant on all 3 seeds. |
| RQ2 | Uncertainty weighting, GradNorm, PCGrad and embedding-level SMART are statistically the same as `sum`, at 1 to 7.5× the cost. The reference "SMART" (token-id noise) raises NEU-ESC accuracy by about 1 point on every backbone, significant on all 3 seeds in 5 of 6 comparisons; it does not help on UIT-VSFC. |
| RQ3 | Focal loss and weighted CE do not change macro-F1 significantly and do not clearly help the smallest classes. |
| RQ4 | Task-aware heads: one suggestive gain (UIT-VSFC topic +0.86, bootstrap interval excludes 0, 2 of 3 seeds). The direction ablation does not confirm the mechanism. No gain on NEU-ESC (ViSoBERT or PhoBERT). |
| RQ5 | MTL does not help more with less data. On UIT-VSFC sentiment it is 4.8 / 4.0 points below single-task at 10 / 25% of train (the `neutral` class). |
| E8 | Task-aware heads and the reference SMART do not add up. |

- **Best NEU-ESC model:** PhoBERT, linear heads, reference SMART (validation 0.7177; test 77.29 sentiment, 63.04
  topic; the paper's best single runs are 77.70 and 63.13). It is not the selected final model: the reference SMART
  was not a selection candidate, and on the selected backbones it scores below `sum` on validation.
- **Reproduction:** single-task and 2-task models on NEU-ESC are within about 1 to 1.5 points of the paper.
- **Cost on a T4:** PCGrad 4.1–7.5×, GradNorm 2.1–3.8×, SMART 1.7–2.7× a plain run; task-aware heads about 1×.

**Where to read more:** `reports/results_main_experiments.md` (E1–E5, E1b, E4b), `reports/results_e8_aware_smartref.md`
(E8), `reports/DPL_Project_Report_IEEE.pdf` (the final report), `reports/tables/*.md` (every table).

---

## 5. Running on Kaggle (`notebooks/kaggle_runner.ipynb`)

1. Notebook settings: **GPU T4**, **Internet On**.
2. Code: the runner clones `https://github.com/Zaidve/DPL_fall36_project` (public, branch `main`; `GIT_URL` is
   preset and the repo includes `data/processed/`), or set `GIT_URL = None` and attach the project as a Kaggle Dataset.
3. Edit the settings cell: `EXPERIMENT`, optionally `ONLY`, `MAX_STEPS` (start with 20 as a quick check) and
   `TIME_BUDGET_HOURS` (default 9.5; lower it when the weekly GPU quota is short).
4. Run all: installs `sentencepiece` + `underthesea`, checks the GPU (stops on an unsupported one, e.g. P100),
   prints the matrix status (progress, failures, remaining GPU-hours, session plan), lists the runs, trains,
   shows a results table, rebuilds the result tables, zips the runs (without checkpoints) + `reports/tables`.
5. **Save Version → Save & Run All** runs the session in the background (up to 12 h). Next session: attach that
   output and set `RESUME_FROM` to its `models/` folder, or to a list of such folders; finished runs are skipped,
   and a wrong path stops the notebook. Kaggle attaches one version of a notebook's output at a time.
6. A session killed at the 12-hour limit kept its finished runs in our one case, but the run in progress is lost.
   The time budget avoids this.
7. The runner copies `reports/tables/selection.json` from the cloned repo, so that file must be committed and
   pushed before E2 and later experiments.

Outputs on Kaggle: `/kaggle/working/models`, `/kaggle/working/reports`. Environment variables
`DPL_PROCESSED_DIR`, `DPL_MODELS_DIR`, `DPL_OUT_DIR`, `DPL_DEVICE` override the defaults anywhere.

---

## 6. Environment

**Local GPU:** NVIDIA GeForce RTX 5050 Laptop GPU (8 GB), driver 576.76 (CUDA ≤ 12.9).
PyTorch `2.11.0+cu128` is the newest build for this driver (2.14 needs CUDA 13 / driver ≥ 580).

**Measured peak GPU memory** (batch 32, real `max_len`, local RTX 5050, 20-step debug runs, 2026-09-29):

| Backbone | Dataset | sum (fp16) | SMART (fp16) | GradNorm (fp32) | PCGrad (fp32) |
|---|---|---|---|---|---|
| ViSoBERT (97M) | UIT-VSFC | 2.5 GB | 3.7 GB (96 tokens) | – | – |
| PhoBERT (134M) | UIT-VSFC | 2.6 GB | 2.6 GB | 2.6 GB | – |
| PhoBERT | NEU-ESC | 2.7 GB | 3.7 GB | 3.1 GB | 4.1 GB |
| XLM-R (277M) | UIT-VSFC | 5.2 GB | 5.2 GB | 5.2 GB | 8.4 GB |
| XLM-R | NEU-ESC | 5.2 GB | 5.7 GB | 5.2 GB | 8.4 GB |

**Kaggle (the real runs):** Tesla T4, python 3.12, torch 2.10.0+cu128, transformers 5.0.0. Measured there:
PCGrad is 4.1–7.5× slower than `sum` (fp32 and one backward pass per task), not the 2.5× the plan assumed.

Everything fits a 16 GB Kaggle T4 with batch 32 and no gradient accumulation. XLM-R + PCGrad only just runs on the
8 GB laptop (Windows spills into shared memory): run it on Kaggle. A full UIT-VSFC epoch with ViSoBERT takes
~40 s locally; PCGrad / GradNorm are ~1.5–2.5× slower.

**Hugging Face cache** (`~/.cache/huggingface/hub`): ViSoBERT, PhoBERT (1.1 GB) and XLM-R (1.1 GB) are
downloaded locally.

**Packages (`.venv`, Python 3.11.0):** torch 2.11.0+cu128, transformers 5.17.0, tokenizers 0.23.2,
huggingface_hub 1.33.0, sentencepiece 0.2.2, underthesea 9.5.0, pandas 3.0.6, numpy 2.4.6, pyarrow 25.0.1,
scipy 1.17.1, scikit-learn 1.9.1, matplotlib 3.11.2, PyYAML 6.0.3, ipykernel 7.3.0, nbconvert 7.17.1.
Not installed: seaborn, tabulate, pytest (tests run with plain `python tests/<file>.py`), py_vncorenlp / Java.

---

## 7. Tests (120, all passing)

| File | Tests | Covers |
|---|---|---|
| `test_preprocess.py` | 13 | normalization, phone regex, segmentation guard, dedup, leakage, subsets |
| `test_dataset.py` | 9 | loaders, padding, seeded shuffle, ids, ViSoBERT id mapping |
| `test_model.py` | 12 | heads, task-aware, inputs_embeds, parameter groups, MLM, losses integration |
| `test_model_parity.py` | 7 | legacy parameter counts, legacy checkpoint parity, key map |
| `test_loss_function.py` | 17 | losses, combiners, SMART (both modes), legacy token shift |
| `test_common_config.py` | 12 | paths, seeds, config `extends`, overrides |
| `test_evaluate_runs.py` | 10 | metrics (argument order), predictions, run ids, resume, checkpoints |
| `test_train.py` | 16 | the full trainer on CPU: every strategy, resume, early stop, errors, placeholders, shipped configs (E1–E8), time budget, selection `sum` / `none` |
| `test_analysis.py` | 24 | run matrix, status, merge, results + seeds, session plan, consistency, selection stages (validation only, seeds, order, force, ties, manual choice, joint backbone + loss), McNemar / bootstrap, RQ5 pairs, tables + Markdown, every figure, E7 (3-label mapping, cross-eval + cache), CLI |

Run one: `.venv/Scripts/python.exe tests/test_train.py` (CPU only, no downloads, ~20 s).

---

## 8. Open items

0. **Route C2-a in progress (started 2026-10-10):** `selection.json` now holds the joint choice. NEU-ESC =
   PhoBERT + `smartref` (validation 0.7177), imbalance stage not written yet; UIT-VSFC is unchanged. The
   results in section 4 and the written reports describe the earlier stage-by-stage selection (NEU-ESC =
   ViSoBERT + `sum`); their tables and figures are kept in `reports/sequential_selection/`. Still to run on
   Kaggle for NEU-ESC on PhoBERT: E3 (18 runs), E4b (6), then `select --stage imbalance --force`, then E5 (36).
   See [what_can_do.md](what_can_do.md) section 4.2.

1. **Not run:** E6 (MLM ablation, 6 runs, about 1.8 h) and E7 (cross-dataset evaluation). E7 needs the kept
   checkpoints of E1 and E4 in one `models/` folder; they are in three separate Kaggle outputs, so the older
   versions' `models` folders would first have to be turned into Kaggle Datasets.
2. **Future work named in the reports:** select backbone and loss jointly on validation (would give PhoBERT +
   reference SMART on NEU-ESC; E3–E5 would then be rerun there, about 17–28 GPU-hours); compare the token-id noise
   with plain token dropout; stronger settings of embedding-level SMART; newer imbalance methods (post-hoc logit
   adjustment needs no retraining, because the prediction files store class probabilities).
3. **Before submitting the report:** fill in the affiliation and email placeholders, check the references (written
   from memory), and compile `reports/latex/project_report.tex` once (it has not been compiled).
4. **Cosmetic:** `rq4_task_aware.md` lists the task-aware row twice (the `sum` model and the final model are the
   same run), and an `e6_mlm` table is written although E6 has no runs.
5. **VnCoreNLP** for PhoBERT's `text_seg` (the spec's preferred segmenter) was never tried; all runs use underthesea.
6. **Decisions taken by default** (in `configs/default.yaml`): GradNorm lr 0.025 (spec: 1e-4), batch 32 without
   accumulation, no `legacy` tag (legacy flags exist in `architecture/legacy.py`). Changing one now would make
   new runs differ from the 234 finished ones.
7. **Git:** `reports/` is ignored except `eda_summary.json` and `tables/selection.json`. The report-building
   scripts (Word / LaTeX generators) are not in the repository.
