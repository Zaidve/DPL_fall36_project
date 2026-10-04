"""
    Paired significance tests between two configs (experiment_matrix_spec.md §4).

    A config is "{dataset}|{backbone}|{mode}|{tag}" (trainer.results.config_key); its runs are its seeds.
    Tests pair predictions row by row on `id`, per seed:

      load_paired(run_a, run_b, task)           id, y_true, pred_a, pred_b (raises if the id sets differ)
      mcnemar_by_seed(config_a, config_b, task) b, c, p-value and direction for every seed both configs have
      significance_mark(res)                    ('†', '3/3 seeds, p<0.05') only if p < 0.05 on all seeds in one direction
      paired_bootstrap(config_a, config_b, task) mean difference metric(B) - metric(A) over seeds, 95% CI, share > 0
                                                (the same resampled test rows are used for every seed)
      standard_pairs(selection, available)      the comparisons of the spec's table (RQ1-RQ4, final)
      compare(pairs)                            both tests per pair and task -> reports/tables/significance.csv
"""
import numpy as np
import pandas as pd

from trainer.evaluate import mcnemar
from trainer.results import finished_runs
from trainer.runs import load_predictions, parse_run_id
from utils.common import models_dir, reports_dir

ALPHA = 0.05


class RunIndex:
    """ Finished runs grouped by config: {config: {seed: run folder}}. """

    def __init__(self, models_dirs=None):
        self.folders = finished_runs(models_dirs or [models_dir()])
        self.by_config = {}
        for rid, folder in self.folders.items():
            p = parse_run_id(rid)
            key = f'{p["dataset"]}|{p["backbone"]}|{p["mode"]}|{p["tag"]}'
            self.by_config.setdefault(key, {})[p['seed']] = folder
        self._cache = {}

    def seeds(self, config):
        return sorted(self.by_config.get(config, {}))

    def predictions(self, folder, split):
        key = (str(folder), split)
        if key not in self._cache:
            self._cache[key] = load_predictions(folder / f'predictions_{split}.csv')
        return self._cache[key]


def _index(index):
    return index if isinstance(index, RunIndex) else RunIndex(index)


def load_paired(run_a, run_b, task, split='test', index=None):
    """ Predictions of two runs joined on id. run_a / run_b: run folders or run ids (looked up in index). """
    index = _index(index)
    folders = [index.folders[r] if isinstance(r, str) and r in index.folders else r for r in (run_a, run_b)]
    a, b = (index.predictions(f, split) for f in folders)
    if set(a['id']) != set(b['id']) or len(a) != len(b):
        raise ValueError(f'prediction ids differ between {folders[0].name} and {folders[1].name} ({split})')
    for df in (a, b):
        if f'{task}_pred' not in df.columns:
            raise KeyError(f'task {task!r} not predicted by {folders[0].name if df is a else folders[1].name}')
    m = a[['id', f'{task}_true', f'{task}_pred']].merge(b[['id', f'{task}_true', f'{task}_pred']], on='id',
                                                        suffixes=('_a', '_b'))
    if not (m[f'{task}_true_a'] == m[f'{task}_true_b']).all():
        raise ValueError('gold labels differ between the two runs')
    return pd.DataFrame({'id': m['id'], 'y_true': m[f'{task}_true_a'].to_numpy(),
                         'pred_a': m[f'{task}_pred_a'].to_numpy(), 'pred_b': m[f'{task}_pred_b'].to_numpy()})


def mcnemar_by_seed(config_a, config_b, task, split='test', index=None):
    index = _index(index)
    rows = []
    for seed in sorted(set(index.seeds(config_a)) & set(index.seeds(config_b))):
        p = load_paired(index.by_config[config_a][seed], index.by_config[config_b][seed], task, split, index)
        r = mcnemar(p['y_true'], p['pred_a'], p['pred_b'])
        b, c = r['only_a_correct'], r['only_b_correct']
        rows.append({'seed': seed, 'b': b, 'c': c, 'p_value': r['p_value'],
                     'direction': 'B better' if c > b else 'A better' if b > c else 'equal'})
    return pd.DataFrame(rows, columns=['seed', 'b', 'c', 'p_value', 'direction'])


def significance_mark(res):
    """ ('†', text) if p < 0.05 on every seed and every seed points the same way; ('', text) otherwise. """
    n = len(res)
    if n == 0:
        return '', 'no paired seeds'
    sig = int((res['p_value'] < ALPHA).sum())
    same = res['direction'].nunique() == 1 and res['direction'].iloc[0] != 'equal'
    text = f'{sig}/{n} seeds p<0.05' + ('' if same else ', directions differ')
    return ('†' if sig == n and same else ''), text


def _macro_f1(y, p, n_classes):
    """ Macro-F1 from counts (same as sklearn with labels=range(n_classes), zero_division=0). """
    tp = np.bincount(y[y == p], minlength=n_classes)
    denom = np.bincount(y, minlength=n_classes) + np.bincount(p, minlength=n_classes)
    f1 = np.divide(2 * tp, denom, out=np.zeros(n_classes), where=denom > 0)
    return f1.mean()


def _metric(name, y, p, n_classes):
    return float((y == p).mean()) if name == 'accuracy' else _macro_f1(y, p, n_classes)


def paired_bootstrap(config_a, config_b, task, metric='macro_f1', n_boot=1000, seed=0, split='test', index=None):
    index = _index(index)
    seeds = sorted(set(index.seeds(config_a)) & set(index.seeds(config_b)))
    if not seeds:
        return {'n_seeds': 0}
    pairs = []
    for s in seeds:
        p = load_paired(index.by_config[config_a][s], index.by_config[config_b][s], task, split, index)
        pairs.append(p.sort_values('id', kind='stable').reset_index(drop=True))
    ids = pairs[0]['id'].to_numpy()
    if any(not np.array_equal(ids, p['id'].to_numpy()) for p in pairs[1:]):
        raise ValueError('seeds were evaluated on different rows')
    prob_cols = [c for c in index.predictions(index.by_config[config_a][seeds[0]], split).columns
                 if c.startswith(f'{task}_prob_')]
    n_classes = len(prob_cols) or int(max(max(p['y_true'].max(), p['pred_a'].max(), p['pred_b'].max()) for p in pairs)) + 1
    arrays = [(p['y_true'].to_numpy(), p['pred_a'].to_numpy(), p['pred_b'].to_numpy()) for p in pairs]

    observed = np.mean([_metric(metric, y, b, n_classes) - _metric(metric, y, a, n_classes) for y, a, b in arrays])
    rng = np.random.default_rng(seed)
    n = len(ids)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = np.mean([_metric(metric, y[idx], b[idx], n_classes) - _metric(metric, y[idx], a[idx], n_classes)
                            for y, a, b in arrays])
    return {'n_seeds': len(seeds), 'observed_diff': float(observed), 'mean_diff': float(diffs.mean()),
            'ci_low': float(np.percentile(diffs, 2.5)), 'ci_high': float(np.percentile(diffs, 97.5)),
            'share_gt_0': float((diffs > 0).mean()), 'n_boot': n_boot, 'metric': metric}


# ---------------------------------------------------------------------------
# Standard comparisons
# ---------------------------------------------------------------------------

def best_single_task(agg, dataset, backbone, task):
    """ The single-task config for `task` with the best VALIDATION macro-F1 on `backbone`. """
    mode = f'st_{task}'
    rows = agg[(agg['split'] == 'validation') & (agg['dataset'] == dataset) & (agg['backbone'] == backbone)
               & (agg['mode'] == mode) & (agg['task'] == task)]
    if rows.empty:
        return None
    return rows.sort_values('macro_f1_mean', ascending=False).iloc[0]['config']


def standard_pairs(selection, available, agg=None):
    """
      [(rq, config_a, config_b, tasks)] from the spec's table, keeping only pairs whose configs both exist.
      available: set of config keys with finished runs. agg (validation) picks the best single-task config.
    """
    datasets = sorted({c.split('|')[0] for c in available})
    pairs = []

    def add(rq, a, b, tasks):
        if a in available and b in available:
            pairs.append((rq, a, b, tuple(tasks)))

    for ds in datasets:
        for bb in sorted({c.split('|')[1] for c in available if c.startswith(ds + '|')}):
            add('RQ1', f'{ds}|{bb}|st_sentiment|sum', f'{ds}|{bb}|mtl|sum', ['sentiment'])
            add('RQ1', f'{ds}|{bb}|st_topic|sum', f'{ds}|{bb}|mtl|sum', ['topic'])
        sel = (selection or {}).get(ds, {})
        bb = sel.get('backbone')
        if not bb:
            continue
        both = ['sentiment', 'topic']
        for tag in ('unc', 'pcgrad', 'smartemb', 'gradnorm', 'smartref'):
            add('RQ2', f'{ds}|{bb}|mtl|sum', f'{ds}|{bb}|mtl|{tag}', both)
        add('RQ2', f'{ds}|{bb}|mtl|smartref', f'{ds}|{bb}|mtl|smartemb', both)
        loss = sel.get('loss_tag')
        if loss:
            for v in ('focal', 'wce'):
                add('RQ3', f'{ds}|{bb}|mtl|{loss}', f'{ds}|{bb}|mtl|{loss}-{v}', both)
            add('RQ3', f'{ds}|{bb}|st_sentiment|focal', f'{ds}|{bb}|mtl|{loss}-focal', ['sentiment'])
            add('RQ3', f'{ds}|{bb}|st_topic|focal', f'{ds}|{bb}|mtl|{loss}-focal', ['topic'])
        add('RQ4', f'{ds}|{bb}|mtl|sum', f'{ds}|{bb}|mtlaware|sum', both)
        final = sel.get('final_tag')
        # RQ5: single task vs MTL (and vs the final model) at every reduced train fraction
        fractions = sorted({c.rsplit('-frac', 1)[1] for c in available
                            if c.startswith(f'{ds}|{bb}|') and '-frac' in c}, key=float)
        for frac in fractions:
            for task in both:
                st = f'{ds}|{bb}|st_{task}|sum-frac{frac}'
                add('RQ5', st, f'{ds}|{bb}|mtl|sum-frac{frac}', [task])
                if final:
                    add('RQ5', st, f'{ds}|{bb}|mtlaware|{final}-frac{frac}', [task])
        if final and final != 'sum':
            add('RQ4', f'{ds}|{bb}|mtl|{final}', f'{ds}|{bb}|mtlaware|{final}', both)
            if agg is not None:
                for task in both:
                    st = best_single_task(agg, ds, bb, task)
                    if st:
                        add('Final', st, f'{ds}|{bb}|mtlaware|{final}', [task])
    return pairs


def compare(pairs, index=None, n_boot=1000, save=True):
    index = _index(index)
    rows = []
    for rq, a, b, tasks in pairs:
        for task in tasks:
            res = mcnemar_by_seed(a, b, task, index=index)
            mark, text = significance_mark(res)
            boot = paired_bootstrap(a, b, task, n_boot=n_boot, index=index)
            rows.append({'rq': rq, 'dataset': a.split('|')[0], 'config_a': a, 'config_b': b, 'task': task,
                         'n_seeds': len(res), 'mcnemar_p': ', '.join(f'{p:.3g}' for p in res['p_value']),
                         'directions': ', '.join(res['direction']), 'mark': mark, 'mcnemar': text,
                         'mf1_diff': boot.get('observed_diff'), 'ci_low': boot.get('ci_low'),
                         'ci_high': boot.get('ci_high'), 'share_gt_0': boot.get('share_gt_0')})
    table = pd.DataFrame(rows, columns=['rq', 'dataset', 'config_a', 'config_b', 'task', 'n_seeds', 'mcnemar_p',
                                        'directions', 'mark', 'mcnemar', 'mf1_diff', 'ci_low', 'ci_high', 'share_gt_0'])
    if save:
        from trainer.tables import save_table
        save_table(table, 'significance')
    return table


class MarkCache:
    """ McNemar marks computed on demand and reused (for the result tables). """

    def __init__(self, index):
        self.index = _index(index)
        self.cache = {}

    def __call__(self, config_a, config_b, task):
        key = (config_a, config_b, task)
        if key not in self.cache:
            if not (self.index.seeds(config_a) and self.index.seeds(config_b)):
                self.cache[key] = ''
            else:
                self.cache[key] = significance_mark(mcnemar_by_seed(config_a, config_b, task, index=self.index))[0]
        return self.cache[key]


def significance_path():
    return reports_dir() / 'tables' / 'significance.csv'
