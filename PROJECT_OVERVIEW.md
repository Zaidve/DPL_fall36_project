# DPL_project: current state

This file describes what the project folder contains (last updated 2026-09-29: GPU set up locally, Kaggle training planned), so someone can design the EDA (and later steps) against the real code and data instead of an assumed layout.

- **Project:** multi-task learning (MTL) for Vietnamese educational feedback, with two tasks per sentence: **sentiment** and **topic** (called `classification` in the code).
- **Datasets:** NEU-ESC and UIT-VSFC.
- **Reference code:** `hung20gg/multi-task-bert` (the NEU-ESC paper's code), kept locally at `../multi-task-bert-master/`. Our code was rewritten from it, not copied.

---

## 1. Folder layout

```
DPL_project/
├── .venv/                     Python 3.11.0 virtual environment (CUDA PyTorch, see section 5)
├── architecture/              EMPTY (model code not written yet)
├── trainer/                   EMPTY (training loop not written yet)
├── notebooks/                 EMPTY (created for the EDA notebook)
├── datasets/
│   ├── neu-esc/
│   │   ├── train_set.csv
│   │   ├── val_set.csv
│   │   └── test_set.csv
│   └── uit-vsfc/
│       ├── README.txt         label meanings + emoticon replacements
│       ├── train/  {sents.txt, sentiments.txt, topics.txt}
│       ├── dev/    {sents.txt, sentiments.txt, topics.txt}
│       └── test/   {sents.txt, sentiments.txt, topics.txt}
├── utils/
│   ├── dataloader.py          data loading, tokenizing, DataLoaders, MLM masking (old version)
│   └── loss_function.py       every loss from losses_spec.md
├── tests/
│   └── test_loss_function.py  14 tests, all passing
└── PROJECT_OVERVIEW.md        this file
```

Not present yet: `configs/`, `reports/`, `src/`, `data/interim/`, any parquet files, a model, a trainer, an EDA/analysis module, or any notebook file.

---

## 2. Datasets

### 2.1 Row counts (checked with the loaders)

| Dataset | train | val / dev | test | Total |
|---|---|---|---|---|
| NEU-ESC | 23,048 | 3,305 | 6,613 | 32,966 |
| UIT-VSFC | 11,426 | 1,583 | 3,166 | 16,175 |

These match the expected numbers in `eda_spec.md` exactly.

### 2.2 NEU-ESC (`datasets/neu-esc/*.csv`)

- **Format:** CSV with header `text,sentiment,classification`.
- **Missing values:** no rows with NaN in these three columns (checked for all splits). The loader still calls `dropna` to be safe.
- **Labels are stored as floats** in the CSV (`0.0`, `1.0`, …). The loader casts them to `int`.
- **Text:** already lowercased, with punctuation split off by spaces (for example `... hà nội học . 2 xin ...`). Texts can be long, multi-sentence forum posts.
- **Split name on disk:** `val` (the loader also accepts `dev`).

| Task | Ids | Names |
|---|---|---|
| sentiment | 0–3 | `neutral, positive, negative, toxic`: **inferred by reading samples, not confirmed from the paper** |
| classification (topic) | 0–9 | **Unknown**; currently just `"0"` … `"9"` |

> ⚠️ The NEU-ESC label names need checking against the paper (Mai et al., 2025, arXiv:2506.23524) before any figure or report uses them.

### 2.3 UIT-VSFC (`datasets/uit-vsfc/<split>/`)

- **Format:** three aligned text files per split, one item per line: `sents.txt` (text), `sentiments.txt` (int), `topics.txt` (int).
- **Text:** lowercased and space-tokenized, usually one short sentence (for example `slide giáo trình đầy đủ .`).
- **Emoticons were replaced by words** in the source data (from `README.txt`), for example `:)` → `colonsmile`, `<3` → `colonlove`, `...` → `dotdotdot`, `/` → `fraction`. The full list is in `datasets/uit-vsfc/README.txt`.
- **Split name on disk:** `dev` (the loader also accepts `val`).

| Task | Ids | Names (from README.txt, official) |
|---|---|---|
| sentiment | 0–2 | `negative, neutral, positive` |
| classification (topic) | 0–3 | `lecturer, training_program, facility, others` |

---

## 3. Code: `utils/dataloader.py`

### 3.1 Constants

| Name | Value |
|---|---|
| `DEVICE` | `cuda` if available, else `cpu` |
| `DATASET_ROOT` | `<project>/datasets` |
| `UIT_VSFC_LABELS` | `{'sentiment': [...3 names], 'classification': [...4 names]}`, list index = label id |
| `NEU_ESC_LABELS` | `{'sentiment': [...4 names], 'classification': ['0'..'9']}` |

### 3.2 Raw data loaders (the useful part for EDA)

All return a `pandas.DataFrame` with the **same columns: `text` (str), `sentiment` (int), `classification` (int)**.

| Function | Signature | Notes |
|---|---|---|
| `load_uit_vsfc` | `(split, root=DATASET_ROOT)` | split: `train` / `dev` / `val` / `test` |
| `load_neu_esc` | `(split, root=DATASET_ROOT)` | split: `train` / `val` / `dev` / `test`; drops NaN rows, casts types |
| `load_dataset` | `(name, split, word_segment=False, root=DATASET_ROOT)` | name: `'uit-vsfc'` or `'neu-esc'`; `word_segment=True` runs `segment_words` on `text` |
| `segment_words` | `(sentences) -> list[str]` | Vietnamese word segmentation with `underthesea.word_tokenize(format='text')` (for PhoBERT: `sinh viên` → `sinh_viên`). **Slow** on a whole split, and **no caching yet** even though the docstring mentions a `*_processed.csv` cache |
| `compute_class_weights` | `(labels, num_classes=None) -> torch.Tensor` | inverse-frequency weights `N / (C · n_c)`; duplicates `class_weights_from_labels` in `loss_function.py` |

**What the loaders do not provide** (an EDA design has to add these):
- no single combined DataFrame over both datasets and all splits
- no `split` or `dataset` column
- no `sentiment_name` / `topic_name` columns (map through `*_LABELS[task][id]`)
- the topic column is called `classification`, not `topic`
- dataset names use hyphens (`neu-esc`), split names are `train/val/test` (NEU) and `train/dev/test` (UIT)

### 3.3 Tokenizing and DataLoaders

| Name | What it does |
|---|---|
| `CreateDataset(sentences, labels1, labels2, model_name, batch_size=32, max_length=128, shuffle=True)` | Tokenizes with `AutoTokenizer(model_name, use_fast=False)`, pads/truncates to `max_length`, moves tensors to `DEVICE`. `.todataloader()` returns batches `(input_ids, attention_mask, labels1, labels2)` |
| `Create3HEADDataset(...)` | Same with a third label |
| `build_dataloaders(name, model_name, batch_size=32, max_length=None, word_segment=None)` | Returns `(train, val, test)` loaders; only train is shuffled. Defaults: `max_length` = 64 for UIT-VSFC, 128 for NEU-ESC; `word_segment` = True when the model name contains `phobert` |

> ⚠️ The `max_length` defaults (64 / 128) and the "covers ~95% of texts" comment are **not backed by any measurement yet**. The EDA token-length item should confirm or replace them.

### 3.4 MLM helpers (older version)

| Name | Notes |
|---|---|
| `DataCollatorHandMade(model_name, mlm_prob=0.3).random_label(input_ids, attention_mask)` | Masks exactly `mlm_prob` of real tokens per sentence (never special tokens or padding), 80/10/10 rule; Python loop per sentence |
| `label_for_mlm(result, mlm_labels)` | Keeps only masked positions for CE |

These are superseded by `mask_tokens` / `mlm_loss` in `loss_function.py` (15%, vectorized). They're kept for now.

---

## 4. Code: `utils/loss_function.py`

Built from `losses_spec.md`. Tested in `tests/test_loss_function.py` (14/14 pass on CPU). Not needed for EDA, but listed for completeness.

| Group | Contents |
|---|---|
| Task losses | `class_weights_from_labels`, `focal_loss`, `TaskLoss(kind='ce' / 'weighted_ce' / 'focal')`, `build_task_loss(cfg, tasks)` |
| MLM | `mask_tokens`, `mlm_loss` |
| SMART | `kl_loss`, `sym_kl_loss`, `inf_norm`, `smart_regularizer` (perturbs input embeddings, needs a model accepting `inputs_embeds=`) |
| Combining tasks | `LossCombiner` base + `sum`, `fixed`, `uncertainty`, `gradnorm`, `pcgrad` (+ `pcgrad_backward`), `dwa`; `build_combiner(cfg, tasks)` |

Expected model interface (not written yet): `model(input_ids=..., attention_mask=...)` or `model(inputs_embeds=..., attention_mask=...)` → `{"sentiment": logits, "topic": logits}`.

---

## 5. Hardware and Python environment

### 5.1 GPU (checked and working)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU (Blackwell, **8 GB VRAM**) |
| Driver | 576.76 (supports CUDA up to 12.9) |
| PyTorch build | `2.11.0+cu128` (CUDA 12.8) |
| Check | `torch.cuda.is_available()` → `True`; a matmul on `cuda` runs; `tests/test_loss_function.py` 14/14 pass on this build |

Why torch 2.11 and not the newest (2.14): torch 2.14 only ships a CUDA 13.0 build, which needs driver ≥ 580. RTX 50-series GPUs need CUDA ≥ 12.8, so `2.11.0+cu128` is the newest build that works with the current driver. To move to 2.14 later: update the NVIDIA driver to ≥ 580, then run
`.venv/Scripts/python.exe -m pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu130`.

Code already picks the GPU automatically: `dataloader.py` uses `DEVICE = cuda if available`, and `loss_function.py` follows the device of its input tensors.

> ⚠️ 8 GB VRAM is tight for base-size models with SMART or PCGrad (the reference repo reports ~10–11 GB at batch 32, max_length 64). Plan for smaller batches, gradient accumulation, or mixed precision (fp16/bf16).

### 5.2 Python packages (`.venv`, Python 3.11.0)

| Package | Version / status |
|---|---|
| torch | 2.11.0+cu128 (**GPU build**) |
| transformers | 5.17.0 |
| tokenizers | 0.23.2 |
| pandas | 3.0.6 |
| numpy | 2.4.6 |
| underthesea | 9.5.0 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| jupyter | installed |
| ipykernel | 7.3.0 |
| seaborn | **missing** (optional; matplotlib is enough for heatmaps) |
| pyarrow | **missing** (needed only for parquet) |
| pytest | **missing** (tests also run with plain `python tests/<file>.py`) |

---

### 5.3 Planned: training on Kaggle (not set up yet)

The local environment above stays as it is (it works). Heavy training is planned on **Kaggle** GPUs (16 GB each, more than the local 8 GB). Nothing Kaggle-specific is in the code yet. Anything new (EDA included) should be written so it runs in both places:

| Topic | Local | Kaggle |
|---|---|---|
| Data location | `<project>/datasets/` (`DATASET_ROOT`) | `/kaggle/input/<dataset-name>/`, **read-only** |
| Where to write files (figures, tables, caches, checkpoints) | anywhere in the project, e.g. `reports/` | only `/kaggle/working/` (~20 GB, kept as notebook output) |
| PyTorch | 2.11.0+cu128 in `.venv` | Kaggle's preinstalled torch; **do not pip-install torch** |
| Extra packages | installed in `.venv` | `pip install underthesea` (needs Internet ON in notebook settings) |
| GPU | RTX 5050 Laptop, 8 GB, fp16 + bf16 | T4 ×2 recommended (16 GB each, **fp16 only, no bf16**); P100 may be unsupported by recent torch |
| Session | unlimited | ~12 h per session, weekly GPU quota → save checkpoints/results each epoch |

Rules that follow for new code:
- Never hard-code paths; take `root` / output directory as arguments (the loaders already accept `root=`).
- Put all outputs under one configurable output directory (`reports/` locally, `/kaggle/working/reports/` on Kaggle).
- Cache slow steps (word segmentation for PhoBERT) to the output directory, not next to the input data.
- Don't upload `.venv/` to Kaggle; upload `datasets/` as a Kaggle Dataset, and the code as a second Dataset or via `git clone`.

---

## 6. Gaps to know before designing the EDA

1. **No combined data table.** EDA specs usually assume one `df` with `text, sentiment, topic, sentiment_name, topic_name, split, dataset`. This has to be built on top of `load_dataset`, which returns one DataFrame per dataset and split.
2. **Naming mismatches:** `classification` vs `topic`; `val`/`dev` vs `validation`; `neu-esc` vs `neu_esc`. Pick one convention.
3. **NEU-ESC label names are not confirmed** (topics are unnamed; sentiment names were guessed).
4. **No word-segmented text is stored.** Word-segmented text (for PhoBERT token lengths) must be computed with `segment_words` (slow) and ideally cached.
5. **Both datasets are already lowercased and space-tokenized**, and UIT-VSFC has emoticons replaced by words. Text normalization for duplicate checks only needs to add Unicode NFC and space collapsing.
6. **No output folders** (`reports/figures`, `reports/tables`) and no shared utilities (`save_json`, logger, `set_seed`, `project_root`). `notebooks/` exists but is empty.
7. **Packages for stats, plots and notebooks are now installed** (scipy, scikit-learn, matplotlib, ipykernel). Only pyarrow (for parquet) is still missing; loading straight from `datasets/` avoids needing it.
8. **The `max_length` defaults in `build_dataloaders` are unverified**; the EDA should produce the numbers that set them. With 8 GB VRAM, the chosen `max_length` also directly limits batch size.
9. **Project layout differs from the specs:** the specs assume `src/mtl_edu/...`, while the code lives in `utils/`. Decide which layout new files follow.
10. **The EDA must run locally and on Kaggle** (section 5.3): read data from a configurable root, write every figure/table/JSON to a configurable output directory, and don't assume bf16 or a specific torch version. Token-length computation downloads tokenizers from Hugging Face, which on Kaggle needs Internet ON.
