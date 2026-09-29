"""
    Light text preprocessing for NEU-ESC and UIT-VSFC.

    Spec: preprocess_spec.md, with the corrections from preprocess_gap_analysis.md
    (tested phone regex; tone marks also handle `_` as a word boundary).

    Output per dataset:
      data/processed/<dataset>.parquet       columns: OUTPUT_COLUMNS
      data/processed/<dataset>_subsets.json  nested stratified train subsets (10 / 25 / 50%)
      reports/tables/preprocess_report.csv   rows and % of texts changed per step
      reports/tables/preprocess_examples.csv before/after examples

    Cleaning is light: both datasets are already lowercased with spaces around punctuation,
    so emoji, slang and punctuation are kept. Tokenization, max_length and MLM masking are
    not done here (utils/dataloader.py, utils/loss_function.py).

    Run:  python -m utils.preprocess [--datasets neu-esc uit-vsfc] [--config configs/default.yaml]
                                     [--data-root DIR] [--out-dir DIR] [--report-dir DIR]
"""
import argparse
import json
import logging
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ON_KAGGLE = os.path.isdir('/kaggle/input')

DATASETS = ('neu-esc', 'uit-vsfc')
SPLITS = ('train', 'val', 'test')
TASKS = ('sentiment', 'topic')
OUTPUT_COLUMNS = ['id', 'text', 'text_clean', 'text_seg', 'sentiment', 'topic',
                  'sentiment_name', 'topic_name', 'split', 'dataset', 'in_test']

DEFAULT_CFG = {
    'unicode_nfc': True,
    'tone_marks': 'new_style',
    'replace_urls': True,
    'replace_masked_names': True,
    'max_repeat_chars': 2,
    'teencode': False,
    'teencode_file': 'configs/teencode.yaml',
    'lowercase': False,
    'segmenter': 'vncorenlp',
    'dedup_train': True,
    'drop_test_overlap': False,
    'subset_fractions': [0.1, 0.25, 0.5],
    'seed': 42,
    'label_names_confirmed': {'neu-esc': False, 'uit-vsfc': True},
}

# Placeholders are single alphanumeric words so they survive segmentation and tokenization.
URL_TOKEN, EMAIL_TOKEN, PHONE_TOKEN, NAME_TOKEN = 'urltoken', 'emailtoken', 'phonetoken', 'nametoken'
PLACEHOLDERS = frozenset({URL_TOKEN, EMAIL_TOKEN, PHONE_TOKEN, NAME_TOKEN})


# ---------------------------------------------------------------------------
# Small helpers (logger, config, paths, JSON)
# ---------------------------------------------------------------------------

def get_logger(name='preprocess'):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s', '%H:%M:%S'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False   # other libraries configure the root logger; avoid double lines
    return logger


LOG = get_logger()


def load_config(path=None):
    """ The `preprocess:` section of the YAML config, on top of DEFAULT_CFG. """
    cfg = dict(DEFAULT_CFG)
    path = Path(path) if path else PROJECT_ROOT / 'configs' / 'default.yaml'
    if path.exists():
        import yaml
        with open(path, encoding='utf-8') as f:
            cfg.update((yaml.safe_load(f) or {}).get('preprocess', {}))
    else:
        LOG.warning(f'config {path} not found, using defaults')
    return cfg


def load_teencode(path):
    import yaml
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding='utf-8') as f:
        return {str(k).lower(): str(v) for k, v in (yaml.safe_load(f) or {}).items()}


def default_dirs():
    """ (processed dir, report tables dir): env vars, then Kaggle, then the project. """
    processed = os.environ.get('DPL_PROCESSED_DIR')
    reports = os.environ.get('DPL_OUT_DIR')
    if ON_KAGGLE:
        processed = processed or '/kaggle/working/data/processed'
        reports = reports or '/kaggle/working/reports'
    processed = Path(processed) if processed else PROJECT_ROOT / 'data' / 'processed'
    reports = Path(reports) if reports else PROJECT_ROOT / 'reports'
    return processed, reports / 'tables'


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


# ---------------------------------------------------------------------------
# 1. Text normalization (str -> str)
# ---------------------------------------------------------------------------

def normalize_unicode(text):
    return unicodedata.normalize('NFC', text)


# Tone marks: old style puts the mark on the first vowel of the open diphthongs oa / oe / uy
# (hòa, khỏe, thúy); new style puts it on the second (hoà, khoẻ, thuý). Only when the diphthong
# ends the syllable (next char is not a letter; `_` counts as a boundary). Closed syllables such
# as hoàng, thuyết, toán are never matched.
_MARKED = {  # grave, acute, hook, tilde, dot
    'o': 'òóỏõọ', 'u': 'ùúủũụ', 'a': 'àáảãạ', 'e': 'èéẻẽẹ', 'y': 'ỳýỷỹỵ',
}


def _tone_map():
    """ All 5 marks x {oa, oe, uy} x 4 letter-case combinations: old style -> new style. """
    def case(c, upper):
        return c.upper() if upper else c

    mapping = {}
    for first, second in (('o', 'a'), ('o', 'e'), ('u', 'y')):
        for i in range(5):
            old = (_MARKED[first][i], second)
            new = (first, _MARKED[second][i])
            for up1 in (False, True):
                for up2 in (False, True):
                    mapping[case(old[0], up1) + case(old[1], up2)] = case(new[0], up1) + case(new[1], up2)
    return mapping


TONE_MAP = _tone_map()
TONE_RE = re.compile('(' + '|'.join(map(re.escape, TONE_MAP)) + r')(?![^\W\d_])')


def normalize_tone_marks(text):
    return TONE_RE.sub(lambda m: TONE_MAP[m.group(1)], text)


URL_RE = re.compile(r'(?:https?://|www\.)\S+', re.IGNORECASE)
EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+')
# From preprocess_gap_analysis.md §4.1: a Vietnamese phone number is +84/0, a first group of at
# least 3 digits (09x, 086, 0971, never 03), then exactly 9 digits after the prefix. The spec's
# broader "9-11 digits" rule matches dates and amounts in NEU-ESC.
PHONE_RE = re.compile(r'(?<![\d.+])(\+84|0)([1-9]\d{1,2}(?:[\s.]?\d{2,5}){1,3})(?![\d.])')


def _phone_sub(m):
    return PHONE_TOKEN if len(re.sub(r'\D', '', m.group(2))) == 9 else m.group(0)


def replace_urls(text):
    """ URLs -> urltoken, emails -> emailtoken, phone numbers -> phonetoken. """
    text = URL_RE.sub(URL_TOKEN, text)
    text = EMAIL_RE.sub(EMAIL_TOKEN, text)
    return PHONE_RE.sub(_phone_sub, text)


MASKED_NAME_RE = re.compile(r'wzjwz\d*', re.IGNORECASE)


def replace_masked_names(text):
    """ UIT-VSFC hides lecturer names as wzjwz<digits>. """
    return MASKED_NAME_RE.sub(NAME_TOKEN, text)


@lru_cache(maxsize=None)
def _repeat_re(max_repeat):
    return re.compile(r'(\D)\1{%d,}' % max_repeat)


def reduce_repeated_chars(text, max_repeat=2):
    """ Cut any non-digit character repeated more than max_repeat times (hayyyyy -> hayy). 0 = off. """
    if not max_repeat:
        return text
    return _repeat_re(max_repeat).sub(lambda m: m.group(1) * max_repeat, text)


WHITESPACE_RE = re.compile(r'\s+')


def clean_whitespace(text):
    """ Collapse whitespace (incl. \\n, \\t, non-breaking space) to one space and strip. """
    return WHITESPACE_RE.sub(' ', text).strip()


@lru_cache(maxsize=8)
def _teencode_re(items):
    keys = sorted((k for k, _ in items), key=len, reverse=True)
    return re.compile(r'(?<!\S)(' + '|'.join(map(re.escape, keys)) + r')(?!\S)', re.IGNORECASE)


def normalize_teencode(text, mapping):
    """ Whole-word replacement, e.g. ko -> không, dc -> được. """
    if not mapping:
        return text
    items = tuple(sorted(mapping.items()))
    lookup = dict(items)
    return _teencode_re(items).sub(lambda m: lookup[m.group(1).lower()], text)


def lowercase(text):
    return text.lower()


NORMALIZE_STEPS = ('unicode', 'tone_marks', 'urls', 'masked_names', 'teencode', 'lowercase',
                   'repeated_chars', 'whitespace')


def _step_functions(cfg):
    """ [(step name, fn)] for the enabled steps, in the spec's order. """
    teencode_map = load_teencode(cfg['teencode_file']) if cfg.get('teencode') else None
    steps = {
        'unicode': normalize_unicode if cfg.get('unicode_nfc', True) else None,
        'tone_marks': normalize_tone_marks if cfg.get('tone_marks', 'new_style') == 'new_style' else None,
        'urls': replace_urls if cfg.get('replace_urls', True) else None,
        'masked_names': replace_masked_names if cfg.get('replace_masked_names', True) else None,
        'teencode': (lambda t: normalize_teencode(t, teencode_map)) if teencode_map else None,
        'lowercase': lowercase if cfg.get('lowercase', False) else None,
        'repeated_chars': ((lambda t: reduce_repeated_chars(t, cfg['max_repeat_chars']))
                           if cfg.get('max_repeat_chars', 2) else None),
        'whitespace': clean_whitespace,
    }
    return [(name, steps[name]) for name in NORMALIZE_STEPS if steps[name] is not None]


def normalize_text(text, cfg=None):
    """ Run the enabled steps in order: unicode -> tone marks -> urls -> masked names -> teencode
        -> lowercase -> repeated chars -> whitespace. Idempotent. """
    for _, fn in _step_functions({**DEFAULT_CFG, **(cfg or {})}):
        text = fn(text)
    return text


def normalize_series(s, cfg=None):
    """ Vectorized normalize_text: each unique text is normalized once. """
    fns = _step_functions({**DEFAULT_CFG, **(cfg or {})})
    uniq = pd.unique(s)
    out = {}
    for text in uniq:
        t = text
        for _, fn in fns:
            t = fn(t)
        out[text] = t
    return s.map(out)


# ---------------------------------------------------------------------------
# 2. Word segmentation (input for PhoBERT)
# ---------------------------------------------------------------------------

def load_segmenter(kind='vncorenlp', models_dir=None):
    """
      Returns a callable list[str] -> list[str]; `.used` says which segmenter it really is.
        vncorenlp  : py_vncorenlp RDRSegmenter (needs Java); falls back to underthesea if missing
        underthesea: underthesea.word_tokenize(text, format='text')
        none       : identity
    """
    if kind == 'vncorenlp':
        try:
            import py_vncorenlp
            models_dir = Path(models_dir or (Path('/kaggle/working') if ON_KAGGLE else PROJECT_ROOT) / 'models' / 'vncorenlp')
            models_dir.mkdir(parents=True, exist_ok=True)
            cwd = os.getcwd()  # py_vncorenlp changes the working directory
            try:
                if not (models_dir / 'VnCoreNLP-1.2.jar').exists():
                    py_vncorenlp.download_model(save_dir=str(models_dir))
                rdr = py_vncorenlp.VnCoreNLP(annotators=['wseg'], save_dir=str(models_dir))
            finally:
                os.chdir(cwd)

            def segment(texts):
                return [' '.join(rdr.word_segment(t)) if t.strip() else t for t in texts]
            segment.used = 'vncorenlp'
            return segment
        except Exception as e:
            LOG.warning(f'VnCoreNLP unavailable ({type(e).__name__}: {e}); falling back to underthesea')
            kind = 'underthesea'

    if kind == 'underthesea':
        from underthesea import word_tokenize

        def segment(texts):
            return [word_tokenize(t, format='text') if t.strip() else t for t in texts]
        segment.used = 'underthesea'
        return segment

    if kind == 'none':
        def segment(texts):
            return list(texts)
        segment.used = 'none'
        return segment

    raise ValueError(f'Unknown segmenter: {kind}')


_LETTERS_RE = re.compile(r'[^\W\d_]+')


def _is_word(token):
    """ A plain Vietnamese word: letters only, not a placeholder. Everything else passes through. """
    return bool(_LETTERS_RE.fullmatch(token)) and token.lower() not in PLACEHOLDERS


def _chunks(text):
    """ Split a space-tokenized text into runs: (True, 'word run') or (False, 'other tokens'). """
    runs = []
    for token in text.split(' '):
        word = _is_word(token)
        if runs and runs[-1][0] == word:
            runs[-1][1].append(token)
        else:
            runs.append((word, [token]))
    return [(word, ' '.join(tokens)) for word, tokens in runs]


def _apply_joins(run, out):
    """
      Copy only the word boundaries from the segmenter output onto our own syllables.
      underthesea rewrites tone marks to the old style (hoá -> hóa) in its output, so the output
      text itself is not used. If the syllables do not line up (count or length), the run is kept
      unsegmented.
    """
    syllables = run.split(' ')
    words = clean_whitespace(out).split(' ')
    groups = [w.split('_') for w in words]
    flat = [s for g in groups for s in g]
    if len(flat) != len(syllables) or any(len(a) != len(b) for a, b in zip(flat, syllables)):
        return run
    joined, pos = [], 0
    for g in groups:
        joined.append('_'.join(syllables[pos:pos + len(g)]))
        pos += len(g)
    return ' '.join(joined)


def segment_texts(texts, segmenter, batch_size=512):
    """
      Segment texts (multi-syllable words joined with `_`: sinh viên -> sinh_viên).
      Only runs of plain words go to the segmenter; punctuation, emoji, digits and placeholders
      are copied unchanged. Only the segmenter's word boundaries are used, never its spelling,
      so text_seg.replace('_', ' ') == input for any input without `_`.
    """
    texts = list(texts)
    split = [_chunks(t) if t else [] for t in texts]
    runs = sorted({run for chunks in split for word, run in chunks if word})

    try:
        from tqdm.auto import tqdm
        batches = tqdm(range(0, len(runs), batch_size), desc=f'segment ({getattr(segmenter, "used", "custom")})',
                       leave=False, disable=len(runs) <= batch_size or not sys.stderr.isatty())
    except ImportError:
        batches = range(0, len(runs), batch_size)

    seg = {}
    for i in batches:
        batch = runs[i:i + batch_size]
        for run, out in zip(batch, segmenter(batch)):
            seg[run] = _apply_joins(run, out)

    return [' '.join(seg[run] if word else run for word, run in chunks) for chunks in split]


def add_segmented_column(df, segmenter, batch_size=512):
    """ text_seg from text_clean; each unique text is segmented once. """
    uniq = pd.unique(df['text_clean'])
    mapping = dict(zip(uniq, segment_texts(uniq, segmenter, batch_size)))
    df = df.copy()
    df['text_seg'] = df['text_clean'].map(mapping)
    return df


# ---------------------------------------------------------------------------
# 3. Table-level steps (DataFrame -> DataFrame)
# ---------------------------------------------------------------------------

def label_names(ds, task, confirmed=True):
    from utils.dataloader import NEU_ESC_LABELS, UIT_VSFC_LABELS
    names = list({'neu-esc': NEU_ESC_LABELS, 'uit-vsfc': UIT_VSFC_LABELS}[ds]
                 ['classification' if task == 'topic' else task])
    if confirmed:
        return names
    return [f'{n}?' for n in names] if task == 'sentiment' else [f't{i}' for i in range(len(names))]


def load_all(datasets=DATASETS, data_root=None, names_confirmed=None):
    """
      One table over the given datasets and all splits, from utils.dataloader.load_dataset.
      Columns: text, sentiment, topic, sentiment_name, topic_name, split, dataset.
      Rows keep the order of the source files.
    """
    from utils.dataloader import load_dataset
    names_confirmed = names_confirmed or DEFAULT_CFG['label_names_confirmed']
    parts = []
    for ds in datasets:
        for sp in SPLITS:
            kwargs = {'root': str(data_root)} if data_root else {}
            part = load_dataset(ds, sp, **kwargs).rename(columns={'classification': 'topic'})
            for task in TASKS:
                names = label_names(ds, task, names_confirmed.get(ds, False))
                part[f'{task}_name'] = [names[i] for i in part[task]]
            part['split'] = sp
            part['dataset'] = ds
            parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    return df[['text', 'sentiment', 'topic', 'sentiment_name', 'topic_name', 'split', 'dataset']]


def add_ids(df):
    """ id = '{dataset}-{split}-{row index within that dataset/split}', before any row is removed. """
    df = df.copy()
    row = df.groupby(['dataset', 'split'], sort=False, observed=True).cumcount()
    df.insert(0, 'id', df['dataset'].astype(str) + '-' + df['split'].astype(str) + '-' + row.astype(str))
    return df


def drop_empty(df):
    keep = df['text_clean'].str.strip() != ''
    return df[keep].copy(), int((~keep).sum())


def dedup_train(df):
    """
      Train only: one row per (dataset, text_clean). The first row's id is kept; each task label is
      the majority label of the group (tie -> label of the first occurrence). Val/test untouched.
    """
    keys = ['dataset', 'text_clean']
    train = df[df['split'] == 'train']
    dup = train[train.duplicated(keys, keep=False)]
    out = df.copy()
    stats = {'rows_removed': 0, 'duplicate_groups': 0,
             'groups_conflicting_sentiment': 0, 'groups_conflicting_topic': 0}
    drop = []
    for _, g in dup.groupby(keys, sort=False):
        keep = g.index[0]
        stats['duplicate_groups'] += 1
        for task in TASKS:
            labels = g[task].tolist()
            counts = Counter(labels)
            best = max(counts.values())
            label = next(v for v in labels if counts[v] == best)
            if len(counts) > 1:
                stats[f'groups_conflicting_{task}'] += 1
            out.at[keep, task] = label
            out.at[keep, f'{task}_name'] = g.loc[g[task] == label, f'{task}_name'].iloc[0]
        drop += list(g.index[1:])
    stats['rows_removed'] = len(drop)
    return out.drop(index=drop), stats


def flag_leakage(df, drop=False):
    """ in_test = train row whose text_clean also appears in the same dataset's test split. """
    df = df.copy()
    df['in_test'] = False
    for ds in df['dataset'].unique():
        in_ds = df['dataset'] == ds
        test_texts = set(df.loc[in_ds & (df['split'] == 'test'), 'text_clean'])
        df.loc[in_ds & (df['split'] == 'train'), 'in_test'] = df.loc[in_ds & (df['split'] == 'train'), 'text_clean'].isin(test_texts)
    flagged = int(df['in_test'].sum())
    if drop:
        df = df[~df['in_test']].copy()
    return df, {'flagged': flagged, 'dropped': flagged if drop else 0}


def make_train_subsets(df, fractions=(0.1, 0.25, 0.5), seed=42):
    """
      Nested, stratified train subsets: {'0.1': [ids], '0.25': [ids], '0.5': [ids]}.
      Strata = sentiment x topic; strata with < 2 rows are merged into one 'rare' stratum.
      Each stratum is shuffled once (seeded) and every fraction takes a prefix of that order,
      so 10% c 25% c 50%, and each stratum contributes round(fraction * size) rows.
    """
    train = df[df['split'] == 'train']
    stratum = train['sentiment'].astype(str) + '|' + train['topic'].astype(str)
    sizes = stratum.map(stratum.value_counts())
    stratum = stratum.where(sizes >= 2, 'rare')

    rng = np.random.default_rng(seed)
    order = {}
    for key in sorted(stratum.unique()):
        ids = train.loc[stratum == key, 'id'].to_numpy()
        order[key] = ids[rng.permutation(len(ids))]

    position = {i: n for n, i in enumerate(train['id'])}
    subsets = {}
    for f in sorted(fractions):
        picked = [i for ids in order.values() for i in ids[:math.floor(f * len(ids) + 0.5)]]
        subsets[str(f)] = sorted(picked, key=position.get)
    return subsets


# ---------------------------------------------------------------------------
# 4. Orchestration and report
# ---------------------------------------------------------------------------

def _normalization_report(texts, cfg):
    """ % of texts each enabled step changes, applying the steps one at a time in order. """
    rows = []
    current = pd.Series(pd.unique(texts))
    weight = texts.value_counts().reindex(current).to_numpy()   # rows per unique text
    enabled = dict(_step_functions(cfg))
    for name in NORMALIZE_STEPS:
        if name not in enabled:
            rows.append({'step': name, 'texts_changed': None, 'pct_changed': None, 'detail': 'off'})
            continue
        after = current.map(enabled[name])
        n_rows = int(weight[after.to_numpy() != current.to_numpy()].sum())
        rows.append({'step': name, 'texts_changed': n_rows, 'pct_changed': 100 * n_rows / max(len(texts), 1), 'detail': ''})
        current = after
    return rows


def preprocess_dataset(name, cfg, segmenter, out_dir, df=None, data_root=None):
    """
      read -> add_ids -> text_clean -> drop_empty -> dedup_train -> flag_leakage
           -> text_seg -> make_train_subsets -> save <out_dir>/<name>.parquet and <name>_subsets.json
      Returns (df, report_rows, examples).
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = load_all([name], data_root, cfg.get('label_names_confirmed'))
    df = df[df['dataset'] == name].reset_index(drop=True)
    report = []

    def rows_step(step, before, after, detail=''):
        report.append({'dataset': name, 'step': step, 'rows_before': before, 'rows_after': after,
                       'rows_removed': before - after, 'texts_changed': None, 'pct_changed': None,
                       'detail': detail})

    n0 = len(df)
    rows_step('load', n0, n0, ', '.join(f'{sp}={int((df["split"] == sp).sum())}' for sp in SPLITS))
    df = add_ids(df)

    LOG.info(f'[{name}] normalizing {n0:,} texts')
    train_texts = df.loc[df['split'] == 'train', 'text']
    for r in _normalization_report(train_texts, cfg):
        report.append({'dataset': name, 'rows_before': None, 'rows_after': None, 'rows_removed': None, **r})
    df['text_clean'] = normalize_series(df['text'], cfg)

    before = len(df)
    df, n_empty = drop_empty(df)
    rows_step('drop_empty', before, len(df))
    LOG.info(f'[{name}] drop_empty: {n_empty} removed')

    if cfg.get('dedup_train', True):
        before = len(df)
        df, stats = dedup_train(df)
        rows_step('dedup_train', before, len(df),
                  f'groups={stats["duplicate_groups"]}, conflicting sentiment={stats["groups_conflicting_sentiment"]}, '
                  f'conflicting topic={stats["groups_conflicting_topic"]}')
        LOG.info(f'[{name}] dedup_train: {stats}')

    before = len(df)
    df, stats = flag_leakage(df, drop=cfg.get('drop_test_overlap', False))
    rows_step('flag_leakage', before, len(df), f'flagged={stats["flagged"]}, dropped={stats["dropped"]}')
    LOG.info(f'[{name}] flag_leakage: {stats}')

    LOG.info(f'[{name}] segmenting with {getattr(segmenter, "used", "custom")}')
    df = add_segmented_column(df, segmenter)
    seg_changed = int((df['text_seg'] != df['text_clean'])[df['split'] == 'train'].sum())
    n_train = int((df['split'] == 'train').sum())
    report.append({'dataset': name, 'step': 'segment', 'rows_before': None, 'rows_after': None, 'rows_removed': None,
                   'texts_changed': seg_changed, 'pct_changed': 100 * seg_changed / max(n_train, 1),
                   'detail': f'segmenter={getattr(segmenter, "used", "custom")}'})

    subsets = make_train_subsets(df, cfg.get('subset_fractions', [0.1, 0.25, 0.5]), cfg.get('seed', 42))
    rows_step('train_subsets', n_train, n_train, ', '.join(f'{k}={len(v)}' for k, v in subsets.items()))

    df['split'] = pd.Categorical(df['split'], categories=list(SPLITS), ordered=True)
    df = df[OUTPUT_COLUMNS].reset_index(drop=True)
    df.to_parquet(out_dir / f'{name}.parquet', index=False)
    save_json(subsets, out_dir / f'{name}_subsets.json')
    LOG.info(f'[{name}] saved {out_dir / f"{name}.parquet"} and {name}_subsets.json')

    changed = df[(df['split'] == 'train') & (df['text'] != df['text_clean'])]
    examples = changed.sample(min(20, len(changed)), random_state=cfg.get('seed', 42)) if len(changed) else changed
    examples = examples[['dataset', 'id', 'text', 'text_clean', 'text_seg']]
    return df, report, examples


def preprocess_report(report_rows, examples, report_dir):
    """ Save preprocess_report.csv and preprocess_examples.csv (utf-8-sig so Excel shows Vietnamese). """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(report_rows, columns=['dataset', 'step', 'rows_before', 'rows_after', 'rows_removed',
                                                'texts_changed', 'pct_changed', 'detail'])
    report.to_csv(report_dir / 'preprocess_report.csv', index=False, encoding='utf-8-sig')
    pd.concat(examples, ignore_index=True).to_csv(report_dir / 'preprocess_examples.csv', index=False, encoding='utf-8-sig')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Light preprocessing for NEU-ESC and UIT-VSFC.')
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS))
    parser.add_argument('--config', default=None, help='YAML with a preprocess: section (default configs/default.yaml)')
    parser.add_argument('--data-root', default=os.environ.get('DPL_DATA_ROOT'), help='folder with neu-esc/ and uit-vsfc/')
    parser.add_argument('--out-dir', default=None, help='default data/processed (or $DPL_PROCESSED_DIR)')
    parser.add_argument('--report-dir', default=None, help='default reports/tables (or $DPL_OUT_DIR/tables)')
    args = parser.parse_args(argv)

    datasets = [d.replace('_', '-') for d in args.datasets]
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        parser.error(f'unknown dataset(s) {unknown}; choose from {list(DATASETS)}')

    cfg = load_config(args.config)
    processed_dir, report_dir = default_dirs()
    out_dir = Path(args.out_dir) if args.out_dir else processed_dir
    report_dir = Path(args.report_dir) if args.report_dir else report_dir

    segmenter = load_segmenter(cfg.get('segmenter', 'vncorenlp'))
    LOG.info(f'segmenter: {segmenter.used} (config: {cfg.get("segmenter")})')

    report_rows, examples, counts = [], [], {}
    for name in datasets:
        df, rows, ex = preprocess_dataset(name, cfg, segmenter, out_dir, data_root=args.data_root)
        report_rows += rows
        examples.append(ex)
        counts[name] = df['split'].value_counts().reindex(list(SPLITS)).to_dict()

    preprocess_report(report_rows, examples, report_dir)
    LOG.info(f'report: {report_dir / "preprocess_report.csv"}')
    print('\nFinal row counts:')
    for name, c in counts.items():
        print(f'  {name}: ' + ', '.join(f'{sp}={n:,}' for sp, n in c.items()))
    return counts


if __name__ == '__main__':
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    main()
