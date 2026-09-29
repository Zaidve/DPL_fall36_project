"""
    E7 cross-dataset evaluation from kept checkpoints (experiment_matrix_spec.md §7).

    A model trained on one dataset predicts the other dataset's test split, sentiment only, on the
    3 labels both datasets share (mapped by label NAME: NEU-ESC toxic -> negative):

      load_run_model(run_dir)             MTLModel rebuilt from config.yaml + best.pt (fp16 checkpoint -> fp32)
      predict_texts(model, tok, texts)    sentiment-head probabilities
      cross_eval_run(run_dir, target)     3-class accuracy / macro-F1 on the target test split; predictions saved
                                          to models/<run_id>/cross_<target>_test.csv
      in_domain_3class(run_dir)           the same 3-class metrics on the run's own test predictions
      cross_eval_all(models_dirs)         every finished run with best.pt and a sentiment head
                                          -> reports/tables/e7_cross_dataset.{csv,md} (+ e7_cross_dataset_runs.csv)

    The target's text column (text_seg for PhoBERT) and the target's max_len for the backbone are used.
    Results are cached per run in cross_eval.json, so re-running only evaluates new checkpoints.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trainer.evaluate import compute_metrics
from trainer.results import finished_runs, label_names
from trainer.runs import parse_run_id, save_predictions
from trainer.tables import BACKBONE_NAMES
from utils.common import load_json, load_yaml, save_json

CLASSES_3 = ('negative', 'neutral', 'positive')
SENTIMENT_3CLASS = {
    'neu-esc': {'neutral': 'neutral', 'positive': 'positive', 'negative': 'negative', 'toxic': 'negative'},
    'uit-vsfc': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
}
DATASETS = ('neu-esc', 'uit-vsfc')
DISPLAY = {'neu-esc': 'NEU-ESC', 'uit-vsfc': 'UIT-VSFC'}


def to_3class(dataset, ids):
    """ Sentiment ids of `dataset` -> indices into CLASSES_3, through the label names. """
    names = label_names(dataset, 'sentiment')
    mapping = SENTIMENT_3CLASS[dataset]
    missing = [n for n in names if n not in mapping]
    if missing:
        raise KeyError(f'{dataset}: sentiment labels {missing} have no 3-class mapping')
    lut = np.array([CLASSES_3.index(mapping[n]) for n in names])
    return lut[np.asarray(ids, dtype=int)]


def probs_to_3class(dataset, probs):
    """ (N, C_dataset) probabilities -> (N, 3) by summing the classes that map to the same 3-class label. """
    names = label_names(dataset, 'sentiment')
    out = np.zeros((probs.shape[0], len(CLASSES_3)))
    for k, n in enumerate(names):
        out[:, CLASSES_3.index(SENTIMENT_3CLASS[dataset][n])] += probs[:, k]
    return out


# ---------------------------------------------------------------------------
# Model and predictions
# ---------------------------------------------------------------------------

def get_device(device=None):
    return torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))


def load_run_model(run_dir, device=None, build=None):
    """ MTLModel from the run's config.yaml with best.pt loaded (build(cfg, num_labels) overrides the builder). """
    from utils.dataset import num_labels
    run_dir = Path(run_dir)
    ckpt = run_dir / 'best.pt'
    if not ckpt.is_file():
        raise FileNotFoundError(f'{run_dir.name}: no best.pt (only E1 st_sentiment/sum, E1 mtl/sum and the final '
                                'model keep their checkpoint)')
    cfg = load_yaml(run_dir / 'config.yaml')
    parts = parse_run_id(run_dir.name)
    labels = num_labels(parts['dataset'], cfg['data']['tasks'])
    # read the checkpoint first: a broken file fails before the encoder is downloaded / built
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    if build is None:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()      # the encoder comes from an MLM checkpoint: unused lm_head report
        from architecture import build_model as build
    model = build(cfg, labels)
    model.load_state_dict(state['model'])            # fp16 tensors are copied into the fp32 parameters
    return model.to(get_device(device)).eval(), cfg


@torch.no_grad()
def predict_texts(model, tokenizer, texts, max_len, batch_size=64, device=None, task='sentiment'):
    """ Probabilities of one task head for raw texts (tokenized with special tokens, truncated to max_len). """
    from utils.dataset import PadCollator, encode_texts
    device = device or next(model.parameters()).device
    model.eval()
    ids = encode_texts(tokenizer, list(texts), max_len)
    collate = PadCollator(tokenizer.pad_token_id)
    use_amp = torch.device(device).type == 'cuda'
    out = []
    for i in range(0, len(ids), batch_size):
        chunk = [{'idx': str(j), 'input_ids': x, 'labels': {}} for j, x in enumerate(ids[i:i + batch_size])]
        batch = collate(chunk)
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))[task]
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


def _metrics_3class(y_true, y_pred):
    m = compute_metrics(y_true, y_pred, len(CLASSES_3))
    return {'acc_3c': m['accuracy'], 'mf1_3c': m['macro_f1'],
            'per_class_f1_3c': dict(zip(CLASSES_3, m['per_class_f1']))}


def cross_eval_run(run_dir, target_dataset, device=None, model=None, tokenizer=None, build=None, batch_size=64):
    """ Evaluate a run's sentiment head on another dataset's test split (3 shared labels). """
    from utils.config import load_model_config, max_len_for
    from utils.dataset import load_processed, load_tokenizer
    run_dir = Path(run_dir)
    parts = parse_run_id(run_dir.name)
    if model is None:
        model, _ = load_run_model(run_dir, device, build)
    model_cfg = load_model_config(parts['backbone'])
    tokenizer = tokenizer or load_tokenizer(model_cfg)
    df = load_processed(target_dataset)
    test = df[df['split'] == 'test']
    max_len = max_len_for(target_dataset, parts['backbone'])
    probs = predict_texts(model, tokenizer, test[model_cfg['text_column']].tolist(), max_len, batch_size,
                          next(model.parameters()).device)
    probs3 = probs_to_3class(parts['dataset'], probs)
    y_true = to_3class(target_dataset, test['sentiment'].to_numpy())
    y_pred = probs3.argmax(1)
    preds = pd.DataFrame({'id': test['id'].to_numpy(), 'y_true_3c': y_true, 'y_pred_3c': y_pred,
                          'pred_source_label': probs.argmax(1),
                          **{f'prob_{c}': probs3[:, k] for k, c in enumerate(CLASSES_3)}})
    save_predictions(preds, run_dir / f'cross_{target_dataset}_test.csv')
    return {'target': target_dataset, 'n': len(test), 'max_len': max_len,
            'text_column': model_cfg['text_column'], **_metrics_3class(y_true, y_pred)}


def in_domain_3class(run_dir):
    """ 3-class metrics on the run's own test predictions (for a fair in-domain vs cross-domain comparison). """
    run_dir = Path(run_dir)
    dataset = parse_run_id(run_dir.name)['dataset']
    p = pd.read_csv(run_dir / 'predictions_test.csv')
    return _metrics_3class(to_3class(dataset, p['sentiment_true']), to_3class(dataset, p['sentiment_pred']))


# ---------------------------------------------------------------------------
# All runs
# ---------------------------------------------------------------------------

def _role(mode):
    return {'st_sentiment': 'Single task', 'mtl': 'MTL', 'mtlaware': 'Task-aware (final)'}.get(mode, mode)


def cross_eval_all(models_dirs=None, device=None, build=None, tokenizer_for=None, force=False, save=True):
    """
      Evaluate every finished run that kept best.pt and predicts sentiment, on the other dataset(s).
      tokenizer_for(backbone) overrides tokenizer loading (tests). Returns (runs table, 2x2 table).
    """
    rows, tokenizers = [], {}
    for run_id, folder in finished_runs(models_dirs).items():
        if not (folder / 'best.pt').is_file():
            continue
        parts = parse_run_id(run_id)
        cfg = load_yaml(folder / 'config.yaml')
        if 'sentiment' not in cfg['data']['tasks']:
            continue
        cache_file = folder / 'cross_eval.json'
        cache = load_json(cache_file) if cache_file.is_file() and not force else {}
        in_domain = in_domain_3class(folder)
        model = None
        for target in (d for d in DATASETS if d != parts['dataset']):
            if target not in cache:
                try:
                    if model is None:
                        model, _ = load_run_model(folder, device, build)
                    if parts['backbone'] not in tokenizers:
                        if tokenizer_for:
                            tokenizers[parts['backbone']] = tokenizer_for(parts['backbone'])
                        else:
                            from utils.config import load_model_config
                            from utils.dataset import load_tokenizer
                            tokenizers[parts['backbone']] = load_tokenizer(load_model_config(parts['backbone']))
                    cache[target] = cross_eval_run(folder, target, model=model, tokenizer=tokenizers[parts['backbone']])
                except Exception as e:  # noqa: BLE001 - one bad checkpoint must not stop E7
                    print(f'WARNING: {run_id} -> {target} skipped: {type(e).__name__}: {str(e).splitlines()[0][:200]}')
                    break
                save_json(cache, cache_file)
            res = cache[target]
            rows.append({'run_id': run_id, 'source': parts['dataset'], 'target': target, 'backbone': parts['backbone'],
                         'mode': parts['mode'], 'tag': parts['tag'], 'seed': parts['seed'],
                         'acc_3c': res['acc_3c'], 'mf1_3c': res['mf1_3c'],
                         'in_domain_mf1_3c': in_domain['mf1_3c'], 'in_domain_acc_3c': in_domain['acc_3c'],
                         'drop': in_domain['mf1_3c'] - res['mf1_3c']})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    runs = pd.DataFrame(rows, columns=['run_id', 'source', 'target', 'backbone', 'mode', 'tag', 'seed', 'acc_3c',
                                       'mf1_3c', 'in_domain_mf1_3c', 'in_domain_acc_3c', 'drop'])
    table = summary_table(runs)
    if save and len(runs):
        from trainer.tables import save_table
        save_table(runs, 'e7_cross_dataset_runs')
        save_table(table, 'e7_cross_dataset')
    return runs, table


def _fmt(values):
    values = pd.Series(values).dropna()
    if values.empty:
        return '–'
    text = f'{100 * values.mean():.2f}'
    if len(values) > 1:
        text += f' ± {100 * values.std(ddof=1):.2f}'
    return text + (f' (n={len(values)})' if len(values) < 3 else '')


def summary_table(runs):
    """ Per model (role, backbone, tag) and train dataset: 3-class macro-F1 on each test dataset (mean ± std). """
    if runs.empty:
        return pd.DataFrame()
    rows = []
    for (mode, backbone, tag, source), g in runs.groupby(['mode', 'backbone', 'tag', 'source'], sort=True):
        row = {'model': f'{_role(mode)} ({BACKBONE_NAMES.get(backbone, backbone)}, {tag})',
               'train on': DISPLAY[source]}
        for target in DATASETS:
            col = f'test {DISPLAY[target]} mF1 (3 labels)'
            if target == source:
                per_seed = g.drop_duplicates('run_id')['in_domain_mf1_3c']
            else:
                per_seed = g[g['target'] == target]['mf1_3c']
            row[col] = _fmt(per_seed)
        row['drop (in − cross)'] = _fmt(g['drop'])
        rows.append(row)
    return pd.DataFrame(rows)
