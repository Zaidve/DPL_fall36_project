"""
    Plan and track the experiment matrix across Kaggle sessions (experiment_matrix_spec.md §1).

      build_matrix()                  every run of configs/experiment/*.yaml (the trainer's own expansion;
                                      selection placeholders kept) -> reports/tables/run_matrix.csv
      run_status(matrix, dirs)        done / failed / partial / missing / blocked per run
      status_summary(status)          counts per experiment x priority
      estimate_remaining(status)      remaining GPU-hours per experiment + a session plan (<= 11 h, Must first)
      merge_models_dirs(srcs, dest)   copy finished runs from several session outputs into one models/
      check_consistency(status, dirs) warnings (config drift between seeds, library versions, suspicious runs)

    Status values: done (metrics.json), failed (error.txt), partial (folder without either),
    missing (no folder), blocked (still has `best` / `bestloss` / `bestimb` / `final`: needs selection.json).
"""
import json
import math
import shutil
from pathlib import Path

import pandas as pd

import trainer.train as T
from trainer.runs import is_run_done
from utils.common import CONFIGS_DIR, load_json, load_yaml, models_dir, reports_dir
from utils.config import load_config

EXPERIMENT_PRIORITY = {'e1_baseline': 'Must', 'e1b_smart_ref': 'Should', 'e2_loss': 'Should',
                       'e4_task_aware': 'Must', 'e4b_direction': 'Could', 'e5_low_resource': 'Could',
                       'e6_mlm': 'Could', 'e8_aware_smartref': 'Could'}
PRIORITY_ORDER = {'Must': 0, 'Should': 1, 'Could': 2}
# Time factors vs a plain run when no run of a group has finished yet (spec: SMART x2, PCGrad x2.5).
GROUP_FACTOR = {'plain': 1.0, 'smart': 2.0, 'pcgrad': 2.5, 'gradnorm': 2.0, 'mlm': 1.5}
# Plain-run minutes on a T4 before anything has finished (models_spec.md B4: NEU 40-50, UIT 20-25).
DEFAULT_MINUTES = {'neu-esc': 45.0, 'uit-vsfc': 22.0}
SESSION_HOURS = 11.0
MATRIX_COLUMNS = ['experiment', 'rq', 'priority', 'run_id', 'dataset', 'backbone', 'mode', 'tag', 'seed',
                  'keep_checkpoint', 'placeholder']


def priority(experiment, tag):
    if experiment == 'e3_imbalance':
        return 'Should' if 'wce' in tag.split('-') else 'Must'
    return EXPERIMENT_PRIORITY.get(experiment, 'Could')


def strategy_group(tag):
    tokens = set(tag.split('-'))
    for group in ('pcgrad', 'gradnorm'):
        if group in tokens:
            return group
    if tokens & {'smartemb', 'smartref'}:
        return 'smart'
    if 'mlm' in tokens:
        return 'mlm'
    return 'plain'


def train_fraction(tag):
    for token in tag.split('-'):
        if token.startswith('frac'):
            try:
                return float(token[4:])
            except ValueError:
                pass
    return 1.0


# ---------------------------------------------------------------------------
# Matrix and status
# ---------------------------------------------------------------------------

def build_matrix(experiment_paths=None, save=True):
    paths = experiment_paths or sorted((CONFIGS_DIR / 'experiment').glob('*.yaml'))
    rows, seen = [], set()
    for path in paths:
        if load_config(path).get('evaluate_only'):
            continue
        exp = T.load_experiment(path)
        for s in T.expand_runs(exp, placeholders=True):
            if s.run_id in seen:          # a run shared by two experiments belongs to the first
                continue
            seen.add(s.run_id)
            rows.append({'experiment': exp['experiment'], 'rq': exp.get('rq', 'ablation' if 'mlm' in s.tag else ''),
                         'priority': priority(exp['experiment'], s.tag), 'run_id': s.run_id, 'dataset': s.dataset,
                         'backbone': s.backbone, 'mode': s.mode, 'tag': s.tag, 'seed': s.seed,
                         'keep_checkpoint': bool(s.cfg['train']['keep_checkpoint']),
                         'placeholder': bool(s.cfg['run'].get('placeholder'))})
    matrix = pd.DataFrame(rows, columns=MATRIX_COLUMNS)
    if save:
        out = reports_dir() / 'tables' / 'run_matrix.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(out, index=False)
    return matrix


def _last_line(path):
    lines = [ln.strip() for ln in Path(path).read_text(encoding='utf-8', errors='replace').splitlines() if ln.strip()]
    return lines[-1] if lines else ''


def run_status(matrix, models_dirs=None):
    dirs = [Path(d) for d in (models_dirs or [models_dir()])]
    rows = []
    for rec in matrix.to_dict('records'):
        status, where, error, time_s, peak = 'missing', None, '', None, None
        if rec.get('placeholder'):
            status = 'blocked'
        else:
            for root in dirs:
                folder = root / rec['run_id']
                if is_run_done(rec['run_id'], root):
                    m = load_json(folder / 'metrics.json')
                    status, where, time_s, peak = 'done', str(root), m.get('train_time_s'), m.get('peak_gpu_mb')
                    break
            else:
                for root in dirs:
                    folder = root / rec['run_id']
                    if (folder / 'error.txt').is_file():
                        status, where, error = 'failed', str(root), _last_line(folder / 'error.txt')
                        break
                    if folder.is_dir() and status == 'missing':
                        status, where = 'partial', str(root)
        rows.append({**rec, 'status': status, 'models_dir': where, 'error': error, 'time_s': time_s,
                     'peak_gpu_mb': peak, 'group': strategy_group(rec['tag'])})
    return pd.DataFrame(rows)


def status_summary(status):
    counts = (status.groupby(['experiment', 'priority'], sort=False)['status'].value_counts().unstack(fill_value=0)
              .reindex(columns=['done', 'failed', 'partial', 'missing', 'blocked'], fill_value=0))
    counts.insert(0, 'total', counts.sum(axis=1))
    counts['pct_done'] = (100 * counts['done'] / counts['total']).round(1)
    return counts.reset_index()


# ---------------------------------------------------------------------------
# Time estimates and session plan
# ---------------------------------------------------------------------------

def _estimate_minutes(status):
    """ Estimated minutes for every run, from finished runs where possible. Adds `est_min`, `est_source`. """
    done = status[(status['status'] == 'done') & status['time_s'].notna()].copy()
    factor = [GROUP_FACTOR[g] * max(train_fraction(t), 0.15) for g, t in zip(done['group'], done['tag'])]
    done['plain_min'] = done['time_s'].astype(float) / 60 / pd.Series(factor, index=done.index, dtype=float)
    exact = done.groupby(['dataset', 'backbone', 'group'])['time_s'].median().astype(float) / 60
    plain_by_bb = done.groupby(['dataset', 'backbone'])['plain_min'].median()
    plain_by_ds = done.groupby('dataset')['plain_min'].median()
    est, source = [], []
    for rec in status.to_dict('records'):
        scale = GROUP_FACTOR[rec['group']] * max(train_fraction(rec['tag']), 0.15)
        key = (rec['dataset'], rec['backbone'], rec['group'])
        if key in exact.index and train_fraction(rec['tag']) == 1.0:
            est.append(exact[key])
            source.append('same dataset/backbone/group')
        elif (rec['dataset'], rec['backbone']) in plain_by_bb.index:
            est.append(plain_by_bb[(rec['dataset'], rec['backbone'])] * scale)
            source.append('same dataset/backbone x factor')
        elif rec['dataset'] in plain_by_ds.index:
            est.append(plain_by_ds[rec['dataset']] * scale)
            source.append('same dataset x factor')
        else:
            est.append(DEFAULT_MINUTES.get(rec['dataset'], 45.0) * scale)
            source.append('spec default x factor')
    out = status.copy()
    out['est_min'] = est
    out['est_source'] = source
    return out


def estimate_remaining(status, session_hours=SESSION_HOURS):
    """
      (per_experiment, session_plan).
      per_experiment: remaining runs and GPU-hours per experiment x priority (blocked runs counted apart).
      session_plan  : runnable groups (experiment, dataset, backbone) packed into sessions of at most
                      session_hours, Must first. Each group maps to one command:
                      python -m trainer.train --experiment <exp> --only <dataset>__<backbone>__
    """
    est = _estimate_minutes(status)
    todo = est[est['status'] != 'done']
    per_exp = (todo.assign(runnable=todo['status'] != 'blocked')
               .groupby(['experiment', 'priority'], sort=False)
               .agg(remaining=('run_id', 'size'), runnable=('runnable', 'sum'), gpu_hours=('est_min', 'sum'))
               .reset_index())
    per_exp['gpu_hours'] = (per_exp['gpu_hours'] / 60).round(1)

    runnable = todo[todo['status'] != 'blocked']
    units = (runnable.groupby(['priority', 'experiment', 'dataset', 'backbone'], sort=False)
             .agg(runs=('run_id', 'size'), minutes=('est_min', 'sum')).reset_index())
    units['order'] = units['priority'].map(PRIORITY_ORDER)
    # E2 feeds the `loss` selection stage; while Must runs wait on it, run it with the Must group
    blocked_must = todo[(todo['status'] == 'blocked') & (todo['priority'] == 'Must')]
    waiting_on_loss = bool(blocked_must['tag'].str.contains('bestloss', regex=False).any())
    unblocks = (units['experiment'] == 'e2_loss') & waiting_on_loss
    units.loc[unblocks, 'order'] = -1
    units = units.sort_values(['order', 'experiment', 'dataset', 'backbone'], kind='stable')
    plan, session, used = [], 1, 0.0
    cap = session_hours * 60
    for u in units.to_dict('records'):
        if used and used + u['minutes'] > cap:
            session, used = session + 1, 0.0
        spans = max(1, math.ceil(u['minutes'] / cap))
        plan.append({'session': session, 'priority': u['priority'], 'experiment': u['experiment'],
                     'dataset': u['dataset'], 'backbone': u['backbone'], 'runs': u['runs'],
                     'hours': round(u['minutes'] / 60, 1),
                     'command': f'--experiment configs/experiment/{u["experiment"]}.yaml '
                                f'--only {u["dataset"]}__{u["backbone"]}__',
                     'note': '; '.join(n for n in (
                         'first: unblocks Must runs (loss selection)' if u['order'] == -1 else '',
                         f'spans {spans} sessions (resume continues)' if spans > 1 else '') if n)})
        if spans > 1:
            session, used = session + spans, 0.0
        else:
            used += u['minutes']
    return per_exp, pd.DataFrame(plan)


# ---------------------------------------------------------------------------
# Merging session outputs
# ---------------------------------------------------------------------------

def _same_config(a, b):
    try:
        return load_yaml(a) == load_yaml(b)
    except Exception:
        return False


def merge_models_dirs(sources, dest, dry_run=True):
    """
      Copy finished runs (metrics.json) from several models/ folders into dest.
      Never overwrites a finished run in dest; reports a conflict when its config.yaml differs.
      A partial / failed folder in dest is replaced by a finished one. best.pt is copied only for runs
      that keep checkpoints. Folders starting with '_' (e.g. _debug) are ignored.
    """
    dest = Path(dest)
    log = []
    for src in map(Path, sources):
        if not src.is_dir():
            log.append({'run_id': '', 'source': str(src), 'action': 'missing source', 'note': ''})
            continue
        for folder in sorted(p for p in src.iterdir() if p.is_dir() and not p.name.startswith('_')):
            rid = folder.name
            if not is_run_done(rid, src):
                log.append({'run_id': rid, 'source': str(src), 'action': 'skip unfinished', 'note': ''})
                continue
            target = dest / rid
            if is_run_done(rid, dest):
                same = _same_config(folder / 'config.yaml', target / 'config.yaml')
                log.append({'run_id': rid, 'source': str(src), 'action': 'keep existing' if same else 'conflict',
                            'note': '' if same else 'config.yaml differs; kept the run already in dest'})
                continue
            keep_ckpt = bool(load_yaml(folder / 'config.yaml').get('train', {}).get('keep_checkpoint')) \
                if (folder / 'config.yaml').is_file() else False
            action = 'replace partial' if target.exists() else 'copy'
            if not dry_run:
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(folder, target, ignore=None if keep_ckpt else shutil.ignore_patterns('*.pt'))
            log.append({'run_id': rid, 'source': str(src), 'action': action,
                        'note': ('dry run' if dry_run else '') + ('' if keep_ckpt else ' (without best.pt)')})
    return pd.DataFrame(log, columns=['run_id', 'source', 'action', 'note'])


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------

def _strip_seed(cfg):
    cfg = dict(cfg)
    cfg.pop('run', None)
    train = dict(cfg.get('train', {}))
    train.pop('seed', None)
    cfg['train'] = train
    return cfg


def check_consistency(status, models_dirs=None, test_sizes=None):
    """ Warnings about finished runs. test_sizes: {dataset: n test rows} (default: from data/processed). """
    warnings = []
    done = status[status['status'] == 'done']
    if done.empty:
        return warnings
    folders = {r['run_id']: Path(r['models_dir']) / r['run_id'] for r in done.to_dict('records')}

    # 1. seeds of one config must differ only in the seed
    configs = {rid: load_yaml(f / 'config.yaml') for rid, f in folders.items() if (f / 'config.yaml').is_file()}
    key = done.assign(config=done['dataset'] + '|' + done['backbone'] + '|' + done['mode'] + '|' + done['tag'])
    for config, group in key.groupby('config'):
        stripped = {rid: _strip_seed(configs[rid]) for rid in group['run_id'] if rid in configs}
        if len({json.dumps(c, sort_keys=True, default=str) for c in stripped.values()}) > 1:
            warnings.append(f'{config}: seeds were trained with different configs ({", ".join(sorted(stripped))})')

    # 2. library versions
    versions = {}
    for rid, f in folders.items():
        if (f / 'env.json').is_file():
            env = load_json(f / 'env.json')
            for lib in ('torch', 'transformers', 'numpy'):
                versions.setdefault(lib, {}).setdefault(env.get(lib), []).append(rid)
    for lib, by_version in versions.items():
        if len(by_version) > 1:
            warnings.append(f'{lib}: runs use different versions {sorted(map(str, by_version))}')

    # 3.-6. per-run checks
    if test_sizes is None:
        test_sizes = {}
        try:
            from utils.dataset import load_processed
            for ds in done['dataset'].unique():
                df = load_processed(ds)
                test_sizes[ds] = int((df['split'] == 'test').sum())
        except Exception as e:  # noqa: BLE001
            warnings.append(f'could not read processed data for the prediction-count check: {e}')
    for rec in done.to_dict('records'):
        rid, f = rec['run_id'], folders[rec['run_id']]
        m = load_json(f / 'metrics.json')
        if m.get('best_epoch') == 1:
            warnings.append(f'{rid}: best epoch is 1 (stopped early at the start?)')
        if m.get('skipped_steps'):
            warnings.append(f'{rid}: {m["skipped_steps"]} training steps skipped (non-finite loss)')
        preds = f / 'predictions_test.csv'
        if rec['dataset'] in test_sizes and preds.is_file():
            n = sum(1 for _ in open(preds, encoding='utf-8')) - 1
            if n != test_sizes[rec['dataset']]:
                warnings.append(f'{rid}: {n} test predictions, processed test split has {test_sizes[rec["dataset"]]}')
        if rec.get('keep_checkpoint') and not (f / 'best.pt').is_file():
            warnings.append(f'{rid}: should keep best.pt (E7) but it is missing')
    return warnings
