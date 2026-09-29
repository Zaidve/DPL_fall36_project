"""
    Collect finished runs (experiment_matrix_spec.md §2).

      load_runs(models_dirs, matrix=None) -> (runs, per_class)
          runs      : one row per finished run x split (validation, test) x task
          per_class : run_id, split, task, class_id, class_name, f1
      config_key(row)            "{dataset}|{backbone}|{mode}|{tag}"  (everything except the seed)
      aggregate_seeds(runs)      mean, std (ddof=1), n_seeds, seeds, complete per config x split x task
      resolve_tags(df, selection) tag_resolved: bestloss / bestimb / final replaced from selection.json

    Metrics stay fractions here; tables format them as percent.
"""
from pathlib import Path

import pandas as pd

from trainer.runs import is_run_done, parse_run_id
from utils.common import load_json, models_dir

SPLITS = ('validation', 'test')
METRICS = ('accuracy', 'macro_f1', 'weighted_f1', 'macro_f1_mean')
RUN_COLUMNS = ['run_id', 'experiment', 'dataset', 'backbone', 'mode', 'tag', 'seed', 'split', 'task',
               'accuracy', 'macro_f1', 'weighted_f1', 'macro_f1_mean', 'best_epoch', 'epochs_run', 'train_time_s',
               'peak_gpu_mb', 'params_trainable', 'train_fraction']


def label_names(dataset, task):
    from utils.dataloader import NEU_ESC_LABELS, UIT_VSFC_LABELS
    labels = {'neu-esc': NEU_ESC_LABELS, 'uit-vsfc': UIT_VSFC_LABELS}.get(dataset)
    return list(labels['classification' if task == 'topic' else task]) if labels else None


def finished_runs(models_dirs=None):
    """ {run_id: run folder} for every finished run; the first folder in models_dirs wins. """
    found = {}
    for root in [Path(d) for d in (models_dirs or [models_dir()])]:
        if not root.is_dir():
            continue
        for folder in sorted(root.iterdir()):
            if folder.is_dir() and not folder.name.startswith('_') and folder.name not in found \
                    and is_run_done(folder.name, root):
                found[folder.name] = folder
    return found


def load_runs(models_dirs=None, matrix=None):
    """ All finished runs as long tables. `experiment` comes from the run matrix when given (the first
        experiment that contains the run, in E1 ... E7 order), else from metrics.json. """
    first_experiment = {}
    if matrix is not None:
        for rid, exp in zip(matrix['run_id'], matrix['experiment']):
            first_experiment.setdefault(rid, exp)
    rows, per_class = [], []
    for run_id, folder in finished_runs(models_dirs).items():
        m = load_json(folder / 'metrics.json')
        parts = parse_run_id(run_id)
        base = {**parts, 'run_id': run_id,
                'experiment': first_experiment.get(run_id, m.get('experiment')),
                **{k: m.get(k) for k in ('best_epoch', 'epochs_run', 'train_time_s', 'peak_gpu_mb',
                                         'params_trainable', 'train_fraction')}}
        for split in SPLITS:
            block = m.get(split, {})
            for task, values in block.items():
                if not isinstance(values, dict):
                    continue
                rows.append({**base, 'split': split, 'task': task,
                             **{k: values.get(k) for k in ('accuracy', 'macro_f1', 'weighted_f1')},
                             'macro_f1_mean': block.get('macro_f1_mean')})
                names = label_names(parts['dataset'], task) or []
                for k, f1 in enumerate(values.get('per_class_f1', [])):
                    per_class.append({'run_id': run_id, 'dataset': parts['dataset'], 'split': split, 'task': task,
                                      'class_id': k, 'class_name': names[k] if k < len(names) else str(k), 'f1': f1})
    runs = pd.DataFrame(rows, columns=RUN_COLUMNS)
    per_class = pd.DataFrame(per_class, columns=['run_id', 'dataset', 'split', 'task', 'class_id', 'class_name', 'f1'])
    return runs, per_class


def config_key(row):
    return f'{row["dataset"]}|{row["backbone"]}|{row["mode"]}|{row["tag"]}'


def aggregate_seeds(runs, metrics=METRICS, expected_seeds=3):
    """ Mean and sample std (ddof=1) over seeds per config x split x task. """
    if runs.empty:
        return pd.DataFrame(columns=['config', 'dataset', 'backbone', 'mode', 'tag', 'split', 'task', 'n_seeds',
                                     'seeds', 'complete'])
    df = runs.copy()
    df['config'] = df.apply(config_key, axis=1)
    keys = ['config', 'dataset', 'backbone', 'mode', 'tag', 'split', 'task']
    grouped = df.groupby(keys, sort=False)
    out = grouped.agg(**{f'{m}_{s}': (m, s) for m in metrics for s in ('mean', 'std')},
                      n_seeds=('seed', 'nunique'), experiment=('experiment', 'first'))
    out['seeds'] = grouped['seed'].apply(lambda s: sorted(set(s)))
    out['complete'] = out['n_seeds'] >= expected_seeds
    return out.reset_index()


def resolve_tags(df, selection):
    """ Add tag_resolved: selection placeholders replaced (kept as-is when that stage is missing). """
    from trainer.train import resolve_tag
    out = df.copy()
    out['tag_resolved'] = [resolve_tag(tag, (selection or {}).get(ds), keep_missing=True)
                           for tag, ds in zip(out['tag'], out['dataset'])]
    return out
