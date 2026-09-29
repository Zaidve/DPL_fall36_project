# NEXT_STEPS: hand-off for the next working session

Written 2026-09-29, at the end of the build phase. Read this first, then [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
for the full description of every file.

**Where we are:** all code from the specs is built and tested (116 tests), and every backbone × loss strategy was
checked on the local GPU with short debug runs. **Nothing has been trained for real yet.** The next phase is
running the experiment matrix on Kaggle and reading the results.

---

## 1. State at hand-off

| Item | State |
|---|---|
| Git | branch `master`, everything committed (last: `8f4f382`, memory-lean PCGrad + these docs), **no remote yet** |
| Tests | 116 passing (9 files in `tests/`, plain Python, CPU, no downloads) |
| Data | `data/processed/*.parquet` + `*_subsets.json` committed (training input) |
| Trained runs | none (only debug runs in the scratch folder, not in `models/`) |
| `selection.json` | not written yet (needs E1 results) |
| Local GPU | RTX 5050 Laptop 8 GB, torch 2.11.0+cu128 (keep this environment unchanged) |
| HF cache | ViSoBERT, PhoBERT, XLM-R downloaded (`~/.cache/huggingface/hub`) |

---

## 2. Do next, in order

### Step 0: get the code to Kaggle (once)

Everything is committed. **Either**
- create an empty GitHub repository, `git remote add origin <url>`, `git push -u origin master`, and set
  `GIT_URL = '<url>'` in the Kaggle runner (private repo: needs a token; public is simpler), **or**
- zip the project folder **without** `.venv/` and `models/` and upload it as a Kaggle Dataset (keep `GIT_URL = None`).

`data/processed/` must be included either way (it is committed, 10 MB).

### Step 1: Kaggle quick check (~10 min)

Kaggle notebook from `notebooks/kaggle_runner.ipynb`, settings: **GPU T4**, **Internet On**.
Settings cell: `EXPERIMENT = 'e1_baseline'`, `MAX_STEPS = 20`, `ONLY = None`. Run all.

Check: the GPU cell says "supported", all 18 debug runs finish (they go to `models/_debug/`), no `error.txt`.
This is the first run on Kaggle's library versions (torch / transformers differ from local).
If something fails, fix it locally, add a test, push, re-run.

### Step 2: E1 baseline (Must, 54 runs, ~30 GPU-hours ≈ 4 sessions)

`MAX_STEPS = None`. The `status` cell prints a session plan (groups that fit in 11 h), e.g.
session 1 `ONLY = 'neu-esc__phobert__'`, session 2 `neu-esc__visobert__`, ... Set `ONLY` per session.

After every session: **Save Version** (keeps `/kaggle/working`). Next session: attach the previous version's
output as a Dataset and set `RESUME_FROM = '/kaggle/input/<that output>/models'`; finished runs are skipped.
Download `results_e1_baseline.zip` each time (runs without checkpoints + `reports/tables`).

### Step 3: collect and choose the backbone (local)

Unzip the session outputs, then:

```bash
.venv/Scripts/python.exe -m trainer.analysis merge --from <session1>/models <session2>/models --to models --apply
```

```bash
.venv/Scripts/python.exe -m trainer.analysis status
```

```bash
.venv/Scripts/python.exe -m trainer.analysis check
```

```bash
.venv/Scripts/python.exe -m trainer.analysis select --stage backbone
```

Commit `reports/tables/selection.json` and push (Kaggle sessions copy it from the code).
`select` refuses if any E1 `mtl/sum` seed is missing; it lists the missing run ids.

### Step 4: E2 (+ E1b), then choose the loss

Run `e2_loss` (the plan puts it first: it unblocks Must runs) and, if time allows, `e1b_smart_ref`.
Then `select --stage loss`, commit `selection.json`. If E2 is skipped, `select --stage loss` picks `sum`.

### Step 5: E3, then choose the imbalance setting

Run `e3_imbalance` (focal = Must, wce = Should). Then `select --stage imbalance`, commit `selection.json`.
This fixes the final model (`final_tag`).

### Step 6: E4 (Must), then the Could experiments

`e4_task_aware` (includes the final model, checkpoint kept), then as time allows `e4b_direction`,
`e5_low_resource` (to cut cost: `ONLY = 'neu-esc__'`), `e6_mlm`, and `EXPERIMENT = 'e7_cross_dataset'`
(evaluation only; needs the kept checkpoints of E1 `st_sentiment/sum`, E1 `mtl/sum` and E4 in one `models/`).

### Step 7: results for the report

```bash
.venv/Scripts/python.exe -m trainer.analysis all
```

Writes `reports/tables/results_<dataset>.md`, `rq1`…`rq5`, `cost`, `e6_mlm`, `significance`, `e7_cross_dataset`
and `reports/figures/*.png`. Works on partial results at any point, so run it after every session.

---

## 3. Decisions still open (defaults are in `configs/default.yaml`)

| Decision | Current default | Alternative | Needed before |
|---|---|---|---|
| GradNorm learning rate | `loss.gradnorm_lr: 0.025` (GradNorm paper) | 1e-4 (train_spec: `lr_heads`) | E2 |
| Batch size | `train.batch_size: 32`, `grad_accum_steps: 1` (fits a T4 for every run, see overview §5) | – | – |
| Legacy-recipe baseline tag | none (legacy flags exist in `architecture/legacy.py`) | add a `legacy` tag (per-epoch LR, fp32, `mlp` head, ...) | only if wanted |
| PhoBERT segmenter | underthesea (Java / VnCoreNLP not installed) | VnCoreNLP on Kaggle: `RUN_PREPROCESS = True`, `INSTALL_VNCORENLP = True` | before E1 PhoBERT runs, if wanted |

Changing a default after runs exist makes seeds of one config differ; `analysis check` warns about that.

---

## 4. Things learned the hard way (do not undo)

- **ViSoBERT tokenizer:** transformers 5 converts its sentencepiece BPE model wrongly (`thầy` → `▁th ầy`).
  `utils/dataset.py` uses the original sentencepiece model with fairseq ids (`tokenizer: sentencepiece_fairseq` in
  `configs/model/visobert.yaml`). Needs `pip install sentencepiece` (the Kaggle runner installs it).
- **underthesea** rewrites tone marks in its output; `utils/preprocess.py` only takes its word boundaries.
- **PCGrad** must not materialise flattened / projected gradient copies (XLM-R ran out of memory); keep the
  Gram-matrix version in `utils/loss_function.py`.
- **Checkpoints:** only E1 `st_sentiment/sum`, E1 `mtl/sum` and the final model keep `best.pt` (fp16, ~42 files,
  12–14 GB). `/kaggle/working` holds ~20 GB.
- **Selection never reads test scores** (asserted in `trainer/selection.py`). Keep it that way.
- **Quick checks go to `models/_debug/`** (`--max-steps`), never counted as finished runs.
- **pandas 3:** `groupby().apply` drops the group columns; `Series[int]` is label-based (use `.iloc`).
- **Windows:** DataLoader workers start slowly (~10–15 s per loader); `num_workers: 0` is faster for local checks.
- **Tests must not download:** use the tiny random encoder and fake tokenizers from the existing tests.

---

## 5. Working conventions for this folder

- Specs: `C:\Users\ADMIN\Downloads\*_spec.md` (eda_spec_1, preprocess, losses, models, train, experiment_matrix).
  Gap analyses of the legacy repo: `../multi-task-bert-master/*.md`.
- Layout: library code in `utils/`, model in `architecture/`, training + analysis in `trainer/`, configs in
  `configs/`; dataset keys `neu-esc` / `uit-vsfc`; run id `{dataset}__{backbone}__{mode}__seed{k}__{tag}`.
- Tests: plain functions + `if __name__ == '__main__'` runner (no pytest). Run them all with:

```bash
for t in tests/test_*.py; do .venv/Scripts/python.exe $t | tail -1; done
```

- Paths never hard-coded: `DPL_PROCESSED_DIR`, `DPL_MODELS_DIR`, `DPL_OUT_DIR`, `DPL_DEVICE` override the
  defaults (Kaggle is detected automatically).
- The notebooks `01_eda.ipynb` and `kaggle_runner.ipynb` were generated by scripts that are not in the repo;
  edit the notebooks directly now.

---

## 6. Not built (optional, only if needed)

- `notebooks/02-preprocessing-check.ipynb` and `notebooks/03-results.ipynb` (mentioned in the specs; the
  `trainer.analysis` tables and figures cover the results).
- Writing the report itself (tables and figures are in `reports/` after Step 7).
