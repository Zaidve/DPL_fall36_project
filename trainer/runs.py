"""
    Run folders under models/ (models_spec.md B1, train_spec.md §1-2).

      run_id = {dataset}__{backbone}__{mode}__seed{k}__{tag}     e.g. neu-esc__phobert__mtl__seed42__sum
      models/<run_id>/  config.yaml, env.json, run.log, train_log.csv, best.pt (temporary),
                        predictions_validation.csv, predictions_test.csv, metrics.json (written last)

    A run is done when metrics.json exists and is complete, so re-running an experiment skips
    finished runs (resume after a Kaggle session ends). metrics.json is written atomically, so a
    session that dies mid-write never leaves a half-written file that looks finished.
"""
import os
import re
import traceback
from pathlib import Path

import pandas as pd
import torch

from utils.common import load_json, models_dir, save_json

MODES = ('st_sentiment', 'st_topic', 'mtl', 'mtlaware')
_PART = re.compile(r'^[A-Za-z0-9.\-]+(?:_[A-Za-z0-9.\-]+)*$')   # single '_' inside only: '__' is the separator


def make_run_id(dataset, backbone, mode, seed, tag):
    parts = {'dataset': dataset, 'backbone': backbone, 'mode': mode, 'tag': tag}
    for name, value in parts.items():
        if not _PART.match(str(value)):
            raise ValueError(f'invalid {name} {value!r} for a run id '
                             '(letters, digits, ".", "-" and single inner "_" only)')
    return f'{dataset}__{backbone}__{mode}__seed{int(seed)}__{tag}'


def parse_run_id(run_id):
    parts = run_id.split('__')
    if len(parts) != 5 or not parts[3].startswith('seed'):
        raise ValueError(f'not a run id: {run_id!r}')
    dataset, backbone, mode, seed, tag = parts
    return {'dataset': dataset, 'backbone': backbone, 'mode': mode, 'seed': int(seed[4:]), 'tag': tag}


def run_dir(run_id, root=None, create=True):
    path = Path(root or models_dir()) / run_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def is_run_done(run_id, root=None):
    path = Path(root or models_dir()) / run_id / 'metrics.json'
    if not path.is_file():
        return False
    try:
        metrics = load_json(path)
    except (ValueError, OSError):
        return False
    return metrics.get('run_id') == run_id and 'test' in metrics


def write_metrics(run_path, metrics):
    """ metrics.json, written atomically (temp file + rename): the last file of a finished run. """
    run_path = Path(run_path)
    tmp = run_path / 'metrics.json.tmp'
    save_json(metrics, tmp)
    os.replace(tmp, run_path / 'metrics.json')


def write_error(run_path, exc=None):
    """ error.txt with the traceback of a failed run (the experiment continues with the next run). """
    text = ''.join(traceback.format_exception(exc)) if exc is not None else traceback.format_exc()
    Path(run_path).mkdir(parents=True, exist_ok=True)
    (Path(run_path) / 'error.txt').write_text(text, encoding='utf-8')


def save_checkpoint(model, path, **extra):
    """ Model weights (+ small extras such as epoch and score) for best-epoch reloading. """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), **extra}, path)


def load_checkpoint(model, path, map_location='cpu'):
    """ Load weights saved by save_checkpoint into `model`; returns the extras. """
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state.pop('model'))
    return state


def save_predictions(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format='%.6f', encoding='utf-8')


def load_predictions(path):
    return pd.read_csv(path, keep_default_na=False, dtype={'id': str})
