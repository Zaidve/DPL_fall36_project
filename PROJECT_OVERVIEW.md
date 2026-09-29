# DPL_project: current state

Last updated 2026-09-29. The full pipeline is built and tested: EDA → preprocessing → data loaders →
models → losses → trainer → experiment configs → Kaggle runner. **No real experiment has been trained yet.**

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
│   └── experiment/          e1_baseline … e7_cross_dataset.yaml
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
│   └── analysis.py          python -m trainer.analysis matrix | status | merge | check | select |
│                            significance | tables | figures | all
├── notebooks/
│   ├── 01_eda.ipynb         EDA (outputs in reports/)
│   └── kaggle_runner.ipynb  runs one experiment on Kaggle (also runs locally)
├── reports/                 eda_summary.json, eda_findings.md, figures/, tables/ (EDA + preprocessing)
├── models/                  run outputs, one folder per run (gitignored; created by training)
├── tests/                   9 test files, 114 tests (+ fixtures: legacy model, fake run folders / full fake matrix)
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
- SMART: `mode="embeddings"` (ours, real adversarial step) and `mode="token_ids"` (exact reproduction of the
  reference code, checked against it: 1.09292293 both), MLM masking and loss.

### 3.6 Trainer (`trainer/`)

```bash
python -m trainer.train --experiment configs/experiment/e1_baseline.yaml [--only SUBSTR] [--dry-run] [--max-steps N]
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

### 3.7 Experiments (`configs/experiment/`)

| File | RQ | Runs | Priority |
|---|---|---|---|
| `e1_baseline` | RQ1: does MTL beat single-task on macro-F1? | 54 | Must |
| `e1b_smart_ref` | RQ2: reference "SMART" | 18 | Should |
| `e2_loss` | RQ2: uncertainty, PCGrad, SMART (+ GradNorm) | 18 + 6 | Should |
| `e3_imbalance` | RQ3: focal / weighted CE | 36 | Must (focal), Should (wce) |
| `e4_task_aware` | RQ4: task-aware heads (+ final model) | 12 | Must |
| `e4b_direction` | RQ4: which direction helps | 12 | Could |
| `e5_low_resource` | RQ5: 10 / 25 / 50% train | 72 | Could |
| `e6_mlm` | MLM ablation | 6 | Could |
| `e7_cross_dataset` | train on one dataset, test on the other | 0 (evaluation only) | Could |

### 3.8 Tracking the matrix (`python -m trainer.analysis ...`)

| Command | What it does |
|---|---|
| `matrix` | All 234 runs with priority (Must / Should / Could) → `reports/tables/run_matrix.csv` |
| `status [--models DIR ...]` | done / failed / partial / missing / blocked per experiment, failed runs with their error line, remaining GPU-hours (from finished runs, else spec defaults × strategy factor) and a plan of `--only` groups that fit in 11-h Kaggle sessions, Must first → `run_status.csv` |
| `merge --from DIR ... --to DIR [--apply]` | copies finished runs from several session outputs into one `models/`; never overwrites a finished run, reports conflicts |
| `check [--models DIR ...]` | config drift between seeds, library versions, best epoch 1, skipped steps, prediction counts, missing kept checkpoints |
| `select --stage backbone\|loss\|imbalance [--allow-partial] [--force]` | writes one stage of `reports/tables/selection.json` (below) |
| `select --explain` | current choices with their validation scores, seed counts and date |

| `significance [--n-boot N]` | standard pairs (RQ1 ST vs MTL per backbone; RQ2 strategies vs sum, smartref vs smartemb; RQ3 imbalance; RQ4 linear vs task-aware; final vs best single task): McNemar per seed + paired bootstrap of Δ macro-F1 → `significance.csv` |
| `tables` | `results_<dataset>` (prior work for NEU-ESC / our reproduction / ours, final model in bold), `rq1_st_vs_mtl`, `rq2_loss`, `rq3_imbalance` + `rq3_per_class`, `rq4_task_aware` (with Cramér's V), `rq5_low_resource`, `cost`, `e6_mlm` |
| `figures` | `rq5_curves_*`, `rq5_low_resource_*`, `rq3_per_class_*`, `confusion_*_{sentiment,topic}`, `rq2_weights_*`, `rq4_gain_vs_v` |
| `all` | significance + tables + figures |

Table cells: test percent `mean ± std` over seeds; `(n=2)` if a seed is missing; `†` = McNemar p < 0.05 on every
seed in one direction vs the row's baseline (the single-task run for main-table rows); `*` = best in the column.
Everything works on a partial matrix and prints what it skipped (on the full fake matrix: ~20 s).

Before any run finishes, `status` estimates E1 at ~30 GPU-hours (4 sessions). While Must runs wait for the loss
stage, the session plan puts E2 first. Only `cross-eval` (E7, matrix plan step 5) is not built yet.

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

234 training runs in total; 42 keep a checkpoint (E1 `st_sentiment/sum` and `mtl/sum`, 18 each; E4 final
model, 6), about 12–14 GB in fp16.
E2–E6 use the best backbone / loss per dataset from `reports/tables/selection.json`, which must be written
after E1 (and E2) from **validation** scores.

---

## 4. Running on Kaggle (`notebooks/kaggle_runner.ipynb`)

1. Notebook settings: **GPU T4**, **Internet On**.
2. Code: set `GIT_URL` (the repo includes `data/processed/`) or attach the project as a Kaggle Dataset.
3. Edit the settings cell: `EXPERIMENT`, optionally `ONLY`, `MAX_STEPS` (start with 20 as a quick check).
4. Run all: installs `sentencepiece` + `underthesea`, checks the GPU (stops on an unsupported one, e.g. P100),
   prints the matrix status (progress, failures, remaining GPU-hours, session plan), lists the runs, trains,
   shows a results table, rebuilds the result tables, zips the runs (without checkpoints) + `reports/tables`.
5. **Save Version** at the end of each session. Next session: attach that output and set `RESUME_FROM`
   to its `models/` folder; finished runs are skipped.

Outputs on Kaggle: `/kaggle/working/models`, `/kaggle/working/reports`. Environment variables
`DPL_PROCESSED_DIR`, `DPL_MODELS_DIR`, `DPL_OUT_DIR`, `DPL_DEVICE` override the defaults anywhere.

---

## 5. Environment

**Local GPU:** NVIDIA GeForce RTX 5050 Laptop GPU (8 GB), driver 576.76 (CUDA ≤ 12.9).
PyTorch `2.11.0+cu128` is the newest build for this driver (2.14 needs CUDA 13 / driver ≥ 580).

**Measured (ViSoBERT, fp16, 96 tokens):** batch 32 uses 2.5 GB, batch 32 + SMART 3.7 GB, so batch 32 without
gradient accumulation fits even locally. A full UIT-VSFC epoch takes ~40 s locally; PCGrad is ~2.5× slower.
PhoBERT and XLM-R have not been measured yet.

**Packages (`.venv`, Python 3.11.0):** torch 2.11.0+cu128, transformers 5.17.0, tokenizers 0.23.2,
huggingface_hub 1.33.0, sentencepiece 0.2.2, underthesea 9.5.0, pandas 3.0.6, numpy 2.4.6, pyarrow 25.0.1,
scipy 1.17.1, scikit-learn 1.9.1, matplotlib 3.11.2, PyYAML 6.0.3, ipykernel 7.3.0, nbconvert 7.17.1.
Not installed: seaborn, tabulate, pytest (tests run with plain `python tests/<file>.py`), py_vncorenlp / Java.

---

## 6. Tests (all passing)

| File | Tests | Covers |
|---|---|---|
| `test_preprocess.py` | 13 | normalization, phone regex, segmentation guard, dedup, leakage, subsets |
| `test_dataset.py` | 9 | loaders, padding, seeded shuffle, ids, ViSoBERT id mapping |
| `test_model.py` | 12 | heads, task-aware, inputs_embeds, parameter groups, MLM, losses integration |
| `test_model_parity.py` | 7 | legacy parameter counts, legacy checkpoint parity, key map |
| `test_loss_function.py` | 17 | losses, combiners, SMART (both modes), legacy token shift |
| `test_common_config.py` | 12 | paths, seeds, config `extends`, overrides |
| `test_evaluate_runs.py` | 10 | metrics (argument order), predictions, run ids, resume, checkpoints |
| `test_train.py` | 14 | the full trainer on CPU: every strategy, resume, early stop, errors, placeholders, shipped configs |
| `test_analysis.py` | 20 | run matrix, status, merge, results + seeds, session plan, consistency, selection stages (validation only, seeds, order, force, ties), McNemar / bootstrap, tables + Markdown, every figure, CLI |

Run one: `.venv/Scripts/python.exe tests/test_train.py` (CPU only, no downloads, ~20 s).

---

## 7. Open items

1. **Matrix tooling** (`experiment_matrix_spec.md`): `trainer/status.py`, `results.py`, `selection.py`,
   `significance.py`, `tables.py`, `figures.py`, `cross_eval.py`, `analysis.py` + `tests/test_analysis.py`.
   Done: step 1 (NEU names, E7 checkpoints, placeholder expansion), step 2 (`status.py`, `results.py`,
   `analysis.py` with matrix / status / merge / check, status cell in the Kaggle runner) and step 3
   (`selection.py`, `select` command) and step 4 (`significance.py`, `tables.py`, `figures.py`, `all`;
   `tables` cell at the end of the Kaggle runner). Next: step 5 `cross_eval.py` (E7).
2. **`selection.json`:** after E1 finishes, run `python -m trainer.analysis select --stage backbone`
   (then `loss` after E2, `imbalance` after E3) and commit the file, so Kaggle sessions pick it up.
3. **E7 evaluation** (`trainer/cross_eval.py`, to build): 3 sentiment labels mapped by name (toxic → negative).
4. **Analysis:** built (`python -m trainer.analysis all`); it fills in as runs finish.
5. **VnCoreNLP** for PhoBERT's `text_seg` (the spec's preferred segmenter): try on Kaggle with
   `RUN_PREPROCESS = True`, `INSTALL_VNCORENLP = True` if Java is available.
6. **PhoBERT and XLM-R** have only been checked through tests with tiny models; the first Kaggle quick check
   (`MAX_STEPS = 20`) will run them for real.
7. **Decisions taken by default** (change in `configs/default.yaml` if needed): GradNorm lr 0.025 (spec: 1e-4),
   batch 32 without accumulation, no `legacy` tag (legacy flags exist in `architecture/legacy.py`).
