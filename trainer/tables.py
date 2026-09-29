"""
    Result tables (experiment_matrix_spec.md §5), saved as CSV and Markdown in reports/tables/.

    Cells: test-split percent with 2 decimals, "mean ± std" over seeds; "(n=2)" when a config has fewer than
    3 seeds; "†" when McNemar is significant on every seed in one direction against the row's baseline;
    "*" marks the best value per column. Rows whose configs have no finished run are skipped and reported.

      main_table     results_<dataset>   prior work (NEU-ESC) / our reproduction / ours
      rq1_table      rq1_st_vs_mtl       single-task vs MTL per backbone: macro-F1 and accuracy gains
      rq2_table      rq2_loss            loss strategies on the selected backbone, vs sum, time cost
      rq3_table      rq3_imbalance       none / focal / wce for MTL and single-task (+ rq3_per_class: minority classes)
      rq4_table      rq4_task_aware      linear vs task-aware heads, directions, Cramér's V from the EDA
      rq5_table      rq5_low_resource    ST / MTL / final at 10, 25, 50, 100% of train
      cost_table     cost                params, time per epoch, peak GPU memory per strategy
      ablation_table e6_mlm              final vs final + MLM
"""
import math
import re

import numpy as np
import pandas as pd

from utils.common import CONFIGS_DIR, load_json, reports_dir
from utils.config import load_config

TASKS = ('sentiment', 'topic')
BACKBONE_NAMES = {'xlmr': 'XLM-R', 'visobert': 'ViSoBERT', 'phobert': 'PhoBERT'}
MAIN_COLUMNS = [('sentiment', 'accuracy', 'Sent Acc'), ('sentiment', 'macro_f1', 'Sent mF1'),
                ('sentiment', 'weighted_f1', 'Sent wF1'), ('topic', 'accuracy', 'Topic Acc'),
                ('topic', 'macro_f1', 'Topic mF1'), ('topic', 'weighted_f1', 'Topic wF1')]
FRACTIONS = (0.1, 0.25, 0.5, 1.0)
UIT_MINORITY = {'sentiment': ['neutral'], 'topic': ['others']}
NEU_MINORITY_FALLBACK = {'sentiment': ['toxic'], 'topic': ['spam', 'club_events', 'help_share']}


# ---------------------------------------------------------------------------
# Formatting and saving
# ---------------------------------------------------------------------------

_NUMERIC = re.compile(r'^[\s*]*[-+−]?\d')


def to_markdown(df, bold_rows=()):
    """ Markdown table without tabulate; numeric-looking columns are right-aligned. """
    cols = [str(c) for c in df.columns]
    values = [['' if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v).replace('|', '\\|')
               for v in row] for row in df.itertuples(index=False)]
    right = [all(_NUMERIC.match(r[j]) for r in values if r[j] not in ('', '–')) and any(r[j] not in ('', '–') for r in values)
             for j in range(len(cols))]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join('---:' if r else '---' for r in right) + '|']
    for i, row in enumerate(values):
        cells = [f'**{c}**' if i in bold_rows and c else c for c in row]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def save_table(df, name, bold_rows=()):
    folder = reports_dir() / 'tables'
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f'{name}.csv', index=False, encoding='utf-8-sig')
    (folder / f'{name}.md').write_text(to_markdown(df, bold_rows) + '\n', encoding='utf-8')
    return df


def pct(x, digits=2):
    return '–' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{100 * x:.{digits}f}'


def signed(x):
    return '–' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{100 * x:+.2f}'


class Lookup:
    """ Seed-aggregated scores by (config, task) for one split. """

    def __init__(self, agg, split='test'):
        rows = agg[agg['split'] == split]
        self.rows = {(r['config'], r['task']): r for r in rows.to_dict('records')}
        self.configs = {c for c, _ in self.rows}

    def get(self, config, task):
        return self.rows.get((config, task))

    def mean(self, config, task, metric='macro_f1'):
        r = self.get(config, task)
        return None if r is None else r[f'{metric}_mean']

    def cell(self, config, task, metric='macro_f1', mark=''):
        r = self.get(config, task)
        if r is None:
            return '–'
        text = pct(r[f'{metric}_mean'])
        std = r[f'{metric}_std']
        if r['n_seeds'] > 1 and std == std:
            text += f' ± {pct(std)}'
        if r['n_seeds'] < 3:
            text += f' (n={r["n_seeds"]})'
        return text + mark


def _star_best(table, columns, numeric):
    """ Append '*' to the best (highest) value of each column. numeric: same shape as table[columns]. """
    for col in columns:
        vals = numeric[col]
        if vals.notna().any():
            best = vals.max()
            for i in vals.index[vals >= best - 1e-12]:
                table.at[i, col] = f'{table.at[i, col]}*'
    return table


# ---------------------------------------------------------------------------
# Main results table
# ---------------------------------------------------------------------------

def prior_work_rows():
    """ Mai et al. 2025 (NEU-ESC), Table 6, test, single run: reported values (percent). """
    data = [
        ('Single task XLM-R', 81.46, 75.91, 81.27, 77.58, 60.87, 77.60),
        ('Single task ViSoBERT', 82.78, 75.79, 82.17, 77.94, 60.65, 78.06),
        ('Single task PhoBERT', 81.75, 77.70, 82.02, 78.65, 62.73, 78.36),
        ('2-task XLM-R', 82.19, 75.50, 81.39, 77.36, 60.62, 77.89),
        ('2-task ViSoBERT', 82.63, 76.17, 82.58, 78.56, 61.66, 78.68),
        ('2-task PhoBERT', 82.88, 75.99, 82.85, 79.25, 62.11, 79.59),
        ('2-task + SMART XLM-R', 83.03, 76.74, 83.69, 79.54, 62.90, 80.17),
        ('2-task + SMART ViSoBERT', 83.58, 77.17, 84.19, 79.80, 63.13, 80.55),
        ('2-task + SMART PhoBERT', 83.26, 77.52, 83.66, 79.31, 62.79, 80.04),
        ('2-task + MLM + SMART PhoBERT', 83.66, 77.34, 84.15, 79.49, 62.58, 80.04),
        ('Claude 4 few-shot (best LLM)', 77.84, 70.89, 75.95, 42.31, 32.73, 44.83),
    ]
    return pd.DataFrame(data, columns=['model'] + [c for _, _, c in MAIN_COLUMNS])


def _e1_backbones():
    try:
        return list(load_config(CONFIGS_DIR / 'experiment' / 'e1_baseline.yaml')['backbones'])
    except Exception:
        return list(BACKBONE_NAMES)


def main_table(agg, dataset, selection, marks=None, save=True):
    """ Returns (table, skipped). marks(config_a, config_b, task) -> '†' or ''. """
    look = Lookup(agg)
    marks = marks or (lambda a, b, t: '')
    sel = (selection or {}).get(dataset, {})
    rows, skipped = [], []

    def st(bb, task):
        return f'{dataset}|{bb}|st_{task}|sum'

    def add(block, label, sources, compare=True, final=False):
        present = [t for t in TASKS if sources.get(t) in look.configs]
        if not present:
            skipped.append(f'{dataset}: {label}')
            return
        row = {'block': block, 'model': label, '_final': final}
        for task, metric, col in MAIN_COLUMNS:
            cfg = sources.get(task)
            bb = cfg.split('|')[1] if cfg else None
            mark = marks(st(bb, task), cfg, task) if (compare and cfg in look.configs and metric == 'macro_f1'
                                                     and st(bb, task) in look.configs) else ''
            row[col] = look.cell(cfg, task, metric, mark) if cfg else '–'
            row[f'_{col}'] = look.mean(cfg, task, metric) if cfg else None
        rows.append(row)

    if dataset == 'neu-esc':
        for r in prior_work_rows().to_dict('records'):
            rows.append({'block': 'Prior work (reported)', 'model': r['model'], '_final': False,
                         **{c: f'{r[c]:.2f}' for _, _, c in MAIN_COLUMNS}, **{f'_{c}': r[c] / 100 for _, _, c in MAIN_COLUMNS}})
    for bb in _e1_backbones():
        name = BACKBONE_NAMES.get(bb, bb)
        add('Our reproduction', f'Single task {name}', {t: st(bb, t) for t in TASKS}, compare=False)
        add('Our reproduction', f'2-task {name}', {t: f'{dataset}|{bb}|mtl|sum' for t in TASKS})
        add('Our reproduction', f'2-task + SMART (reference) {name}', {t: f'{dataset}|{bb}|mtl|smartref' for t in TASKS})
    bb, loss, final = sel.get('backbone'), sel.get('loss_tag'), sel.get('final_tag')
    if bb:
        name = BACKBONE_NAMES.get(bb, bb)
        ours = [('MTL + SMART (embeddings)', 'mtl', 'smartemb'), ('MTL + uncertainty', 'mtl', 'unc'),
                ('MTL + PCGrad', 'mtl', 'pcgrad'), ('MTL + GradNorm', 'mtl', 'gradnorm')]
        if loss:
            ours += [(f'MTL {loss} + focal', 'mtl', f'{loss}-focal'), (f'MTL {loss} + weighted CE', 'mtl', f'{loss}-wce')]
        ours += [('Task-aware heads', 'mtlaware', 'sum')]
        for label, mode, tag in ours:
            add(f'Ours ({name})', label, {t: f'{dataset}|{bb}|{mode}|{tag}' for t in TASKS})
        if final:
            add(f'Ours ({name})', f'Final model (task-aware, {final})',
                {t: f'{dataset}|{bb}|mtlaware|{final}' for t in TASKS}, final=True)

    cols = [c for _, _, c in MAIN_COLUMNS]
    table = pd.DataFrame(rows)
    if table.empty:
        return table, skipped
    numeric = table[[f'_{c}' for c in cols]].astype(float)
    numeric.columns = cols
    table = _star_best(table, cols, numeric)
    bold = [i for i, f in enumerate(table['_final']) if f]
    out = table[['block', 'model'] + cols]
    if save:
        save_table(out, f'results_{dataset}', bold_rows=bold)
    return out, skipped


# ---------------------------------------------------------------------------
# One table per research question
# ---------------------------------------------------------------------------

def rq1_table(agg, marks=None, save=True):
    look = Lookup(agg)
    marks = marks or (lambda a, b, t: '')
    rows = []
    for cfg in sorted(c for c in look.configs if c.endswith('|mtl|sum')):
        ds, bb = cfg.split('|')[:2]
        for task in TASKS:
            st = f'{ds}|{bb}|st_{task}|sum'
            if st not in look.configs:
                continue
            d_f1 = look.mean(cfg, task) - look.mean(st, task)
            d_acc = look.mean(cfg, task, 'accuracy') - look.mean(st, task, 'accuracy')
            reading = ('accuracy and macro-F1 up' if d_acc > 0 and d_f1 > 0 else
                       'accuracy up, macro-F1 not' if d_acc > 0 >= d_f1 else
                       'macro-F1 up, accuracy not' if d_f1 > 0 >= d_acc else 'neither up')
            rows.append({'dataset': ds, 'backbone': BACKBONE_NAMES.get(bb, bb), 'task': task,
                         'ST mF1': look.cell(st, task), 'MTL mF1': look.cell(cfg, task), 'Δ mF1': signed(d_f1),
                         'ST Acc': look.cell(st, task, 'accuracy'), 'MTL Acc': look.cell(cfg, task, 'accuracy'),
                         'Δ Acc': signed(d_acc), 'McNemar': marks(st, cfg, task), 'reading': reading})
    table = pd.DataFrame(rows)
    if save and len(table):
        save_table(table, 'rq1_st_vs_mtl')
    return table


def _epoch_time(runs):
    per_run = runs.drop_duplicates('run_id').copy()
    per_run['config'] = per_run['dataset'] + '|' + per_run['backbone'] + '|' + per_run['mode'] + '|' + per_run['tag']
    per_run['time_per_epoch'] = per_run['train_time_s'] / per_run['epochs_run'].clip(lower=1)
    return per_run.groupby('config')['time_per_epoch'].mean().to_dict()


def rq2_table(agg, runs, selection, marks=None, save=True):
    look = Lookup(agg)
    marks = marks or (lambda a, b, t: '')
    times = _epoch_time(runs) if len(runs) else {}
    rows = []
    for ds, sel in sorted((selection or {}).items()):
        bb = sel.get('backbone')
        if not bb:
            continue
        base = f'{ds}|{bb}|mtl|sum'
        for tag in ('sum', 'unc', 'pcgrad', 'gradnorm', 'smartemb', 'smartref'):
            cfg = f'{ds}|{bb}|mtl|{tag}'
            if cfg not in look.configs:
                continue
            means = [look.mean(cfg, t) for t in TASKS]
            base_means = [look.mean(base, t) for t in TASKS]
            row = {'dataset': ds, 'backbone': BACKBONE_NAMES.get(bb, bb), 'strategy': tag}
            for t in TASKS:
                row[f'{t} mF1'] = look.cell(cfg, t, mark='' if tag == 'sum' else marks(base, cfg, t))
            row['mean mF1'] = pct(np.mean(means))
            row['Δ mean vs sum'] = signed(np.mean(means) - np.mean(base_means)) if None not in base_means else '–'
            if tag == 'smartemb' and f'{ds}|{bb}|mtl|smartref' in look.configs:
                row['vs smartref'] = ' '.join(filter(None, (marks(f'{ds}|{bb}|mtl|smartref', cfg, t) for t in TASKS))) or 'n.s.'
            t_cfg, t_base = times.get(cfg), times.get(base)
            row['time/epoch (x sum)'] = f'{t_cfg / t_base:.2f}' if t_cfg and t_base else '–'
            rows.append(row)
    table = pd.DataFrame(rows)
    if save and len(table):
        save_table(table, 'rq2_loss')
    return table


def _minority_classes(dataset, eda_summary):
    if dataset == 'uit-vsfc':
        return UIT_MINORITY
    try:
        topic = eda_summary['labels'][dataset]['topic']
        order = np.argsort(topic['train_counts'])[:3]
        return {'sentiment': ['toxic'], 'topic': [topic['names'][i] for i in order]}
    except Exception:
        return NEU_MINORITY_FALLBACK


def rq3_table(agg, per_class, selection, eda_summary=None, marks=None, save=True):
    look = Lookup(agg)
    marks = marks or (lambda a, b, t: '')
    rows, pc_rows = [], []
    for ds, sel in sorted((selection or {}).items()):
        bb, loss = sel.get('backbone'), sel.get('loss_tag')
        if not (bb and loss):
            continue
        for label, suffix in (('none', ''), ('focal', '-focal'), ('wce', '-wce')):
            cfg = f'{ds}|{bb}|mtl|{loss}{suffix}'
            if cfg in look.configs:
                base = f'{ds}|{bb}|mtl|{loss}'
                rows.append({'dataset': ds, 'model': f'MTL ({loss})', 'imbalance': label,
                             **{f'{t} mF1': look.cell(cfg, t, mark='' if not suffix else marks(base, cfg, t)) for t in TASKS}})
            for t in TASKS:
                st_cfg = f'{ds}|{bb}|st_{t}|{"sum" if not suffix else suffix[1:]}'
                if st_cfg in look.configs:
                    rows.append({'dataset': ds, 'model': f'single task ({t})', 'imbalance': label,
                                 **{f'{u} mF1': (look.cell(st_cfg, t) if u == t else '') for u in TASKS}})
        # per-class F1 of the minority classes
        if per_class is not None and len(per_class):
            pc = per_class[per_class['split'] == 'test'].copy()
            parts = pc['run_id'].str.split('__', expand=True)
            pc['config'] = parts[0] + '|' + parts[1] + '|' + parts[2] + '|' + parts[4]
            minority = _minority_classes(ds, eda_summary)
            configs = [(f'MTL {loss}', f'{ds}|{bb}|mtl|{loss}'), ('+ focal', f'{ds}|{bb}|mtl|{loss}-focal'),
                       ('+ weighted CE', f'{ds}|{bb}|mtl|{loss}-wce')]
            if sel.get('final_tag'):
                configs.append(('final model', f'{ds}|{bb}|mtlaware|{sel["final_tag"]}'))
            for t, classes in minority.items():
                for cls in classes:
                    row = {'dataset': ds, 'task': t, 'class': cls}
                    for label, cfg in configs:
                        v = pc[(pc['config'] == cfg) & (pc['task'] == t) & (pc['class_name'] == cls)]['f1']
                        row[label] = '–' if v.empty else (pct(v.mean()) + (f' ± {pct(v.std())}' if len(v) > 1 else ''))
                    pc_rows.append(row)
    table, per_class_table = pd.DataFrame(rows), pd.DataFrame(pc_rows)
    if save:
        if len(table):
            save_table(table, 'rq3_imbalance')
        if len(per_class_table):
            save_table(per_class_table, 'rq3_per_class')
    return table, per_class_table


def rq4_table(agg, selection, eda_summary=None, marks=None, save=True):
    look = Lookup(agg)
    marks = marks or (lambda a, b, t: '')
    rows = []
    for ds, sel in sorted((selection or {}).items()):
        bb = sel.get('backbone')
        if not bb:
            continue
        v = ((eda_summary or {}).get('task_dependence', {}).get(ds, {}) or {}).get('cramers_v')
        final = sel.get('final_tag')
        pairs = [('both, sum', f'{ds}|{bb}|mtl|sum', f'{ds}|{bb}|mtlaware|sum'),
                 ('sentiment reads topic', f'{ds}|{bb}|mtl|sum', f'{ds}|{bb}|mtlaware|sum-cross_sent'),
                 ('topic reads sentiment', f'{ds}|{bb}|mtl|sum', f'{ds}|{bb}|mtlaware|sum-cross_topic')]
        if final:
            pairs.insert(1, (f'both, final ({final})', f'{ds}|{bb}|mtl|{final}', f'{ds}|{bb}|mtlaware|{final}'))
        for label, lin, aware in pairs:
            if lin not in look.configs or aware not in look.configs:
                continue
            row = {'dataset': ds, 'heads': label, "Cramér's V": '–' if v is None else f'{v:.3f}'}
            for t in TASKS:
                row[f'linear {t}'] = look.cell(lin, t)
                row[f'task-aware {t}'] = look.cell(aware, t, mark=marks(lin, aware, t))
                row[f'Δ {t}'] = signed(look.mean(aware, t) - look.mean(lin, t))
            rows.append(row)
    table = pd.DataFrame(rows)
    if save and len(table):
        save_table(table, 'rq4_task_aware')
    return table


def _frac_tag(tag, fraction):
    return tag if fraction >= 1.0 else f'{tag}-frac{fraction:g}'


def rq5_table(agg, selection, save=True):
    look = Lookup(agg)
    rows = []
    for ds, sel in sorted((selection or {}).items()):
        bb, final = sel.get('backbone'), sel.get('final_tag')
        if not bb:
            continue
        for f in FRACTIONS:
            row = {'dataset': ds, 'train fraction': f'{f:g}'}
            found = False
            for t in TASKS:
                st = f'{ds}|{bb}|st_{t}|{_frac_tag("sum", f)}'
                mtl = f'{ds}|{bb}|mtl|{_frac_tag("sum", f)}'
                fin = f'{ds}|{bb}|mtlaware|{_frac_tag(final, f)}' if final else None
                row[f'ST {t}'] = look.cell(st, t)
                row[f'MTL {t}'] = look.cell(mtl, t)
                row[f'final {t}'] = look.cell(fin, t) if fin else '–'
                s, m = look.mean(st, t), look.mean(mtl, t)
                row[f'Δ MTL−ST {t}'] = signed(m - s) if s is not None and m is not None else '–'
                found |= any(c in look.configs for c in (st, mtl, fin))
            if found:
                rows.append(row)
    table = pd.DataFrame(rows)
    if save and len(table):
        save_table(table, 'rq5_low_resource')
    return table


def _strategy(tag):
    tokens = tag.split('-')
    base = next((t for t in tokens if t in ('sum', 'unc', 'pcgrad', 'gradnorm', 'dwa', 'smartemb', 'smartref')), 'sum')
    extras = [t for t in tokens if t in ('focal', 'wce', 'mlm') or t.startswith('frac') or t.startswith('cross')]
    return '+'.join([base] + extras)


def cost_table(runs, save=True):
    if runs is None or runs.empty:
        return pd.DataFrame()
    per_run = runs.drop_duplicates('run_id').copy()
    per_run['strategy'] = per_run['tag'].map(_strategy)
    per_run['time_per_epoch_s'] = per_run['train_time_s'] / per_run['epochs_run'].clip(lower=1)
    table = (per_run.groupby(['dataset', 'backbone', 'mode', 'strategy'])
             .agg(runs=('run_id', 'size'), params_M=('params_trainable', lambda s: s.mean() / 1e6),
                  time_per_epoch_s=('time_per_epoch_s', 'mean'), peak_gpu_mb=('peak_gpu_mb', 'mean'))
             .reset_index().round({'params_M': 1, 'time_per_epoch_s': 1, 'peak_gpu_mb': 0}))
    if save:
        save_table(table, 'cost')
    return table


def ablation_table(agg, selection, marks=None, save=True):
    look = Lookup(agg)
    marks = marks or (lambda a, b, t: '')
    rows = []
    for ds, sel in sorted((selection or {}).items()):
        bb, final = sel.get('backbone'), sel.get('final_tag')
        if not (bb and final):
            continue
        base, mlm = f'{ds}|{bb}|mtlaware|{final}', f'{ds}|{bb}|mtlaware|{final}-mlm'
        for label, cfg in (('final', base), ('final + MLM', mlm)):
            if cfg in look.configs:
                rows.append({'dataset': ds, 'model': label,
                             **{f'{t} {m}': look.cell(cfg, t, m, mark=marks(base, cfg, t) if cfg == mlm and m == 'macro_f1' else '')
                                for t in TASKS for m in ('accuracy', 'macro_f1')}})
    table = pd.DataFrame(rows)
    if save and len(table):
        save_table(table, 'e6_mlm')
    return table


# ---------------------------------------------------------------------------
# Everything
# ---------------------------------------------------------------------------

def load_context(models_dirs=None):
    """ runs, per_class, seed-aggregated scores, selection.json, eda_summary.json, run index. """
    from trainer.results import aggregate_seeds, load_runs
    from trainer.significance import RunIndex
    runs, per_class = load_runs(models_dirs)
    agg = aggregate_seeds(runs)
    sel_path = reports_dir() / 'tables' / 'selection.json'
    eda_path = reports_dir() / 'eda_summary.json'
    return {'runs': runs, 'per_class': per_class, 'agg': agg,
            'selection': load_json(sel_path) if sel_path.exists() else {},
            'eda_summary': load_json(eda_path) if eda_path.exists() else {},
            'index': RunIndex(models_dirs)}


def build_all(models_dirs=None, ctx=None):
    """ Write every table. Returns ({name: table}, skipped rows). """
    from trainer.significance import MarkCache
    ctx = ctx or load_context(models_dirs)
    agg, sel, marks = ctx['agg'], ctx['selection'], MarkCache(ctx['index'])
    tables, skipped = {}, []
    for ds in sorted(agg['dataset'].unique()) if len(agg) else []:
        tables[f'results_{ds}'], sk = main_table(agg, ds, sel, marks)
        skipped += sk
    tables['rq1_st_vs_mtl'] = rq1_table(agg, marks)
    tables['rq2_loss'] = rq2_table(agg, ctx['runs'], sel, marks)
    tables['rq3_imbalance'], tables['rq3_per_class'] = rq3_table(agg, ctx['per_class'], sel, ctx['eda_summary'], marks)
    tables['rq4_task_aware'] = rq4_table(agg, sel, ctx['eda_summary'], marks)
    tables['rq5_low_resource'] = rq5_table(agg, sel)
    tables['cost'] = cost_table(ctx['runs'])
    tables['e6_mlm'] = ablation_table(agg, sel, marks)
    return tables, skipped
