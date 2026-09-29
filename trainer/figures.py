"""
    Figures for the report (experiment_matrix_spec.md §6), matplotlib only, saved to reports/figures/.

      plot_learning_curves   rq5_curves_<dataset>.png       train vs validation loss per epoch (mean ± std), ST vs MTL
      plot_low_resource      rq5_low_resource_<dataset>.png test macro-F1 vs train fraction
      plot_per_class         rq3_per_class_<dataset>.png    per-class F1: MTL, + focal, final model
      plot_confusion         confusion_<dataset>_<task>.png row-normalised, summed over seeds: MTL vs final model
      plot_task_weights      rq2_weights_<dataset>.png      learned task weights per epoch (unc / gradnorm / dwa)
      plot_rq4_vs_cramers    rq4_gain_vs_v.png              task-aware gain vs the dataset's Cramér's V

    Every function returns the saved path, or None when the runs it needs are not finished yet.
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trainer.runs import load_predictions  # noqa: E402
from trainer.tables import FRACTIONS, TASKS, Lookup, _frac_tag  # noqa: E402
from utils.common import reports_dir  # noqa: E402

COLORS = {'ST': '#eb6834', 'MTL': '#2a78d6', 'task-aware': '#1baf7a', 'single': '#2a78d6'}
EXTRA = ['#8a6bbf', '#b58b2a', '#6b6b6b']
DPI = 200
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e4e3df', 'axes.axisbelow': True, 'axes.titleweight': 'bold',
                     'axes.titlesize': 10, 'legend.frameon': False, 'font.size': 9})


def _save(fig, name):
    folder = reports_dir() / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{name}.png'
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def _logs(index, config):
    return [pd.read_csv(f / 'train_log.csv') for f in index.by_config.get(config, {}).values()
            if (f / 'train_log.csv').is_file()]


def _mean_std(frames, column):
    if not frames:
        return None
    n = min(len(f) for f in frames)
    vals = np.array([f[column].to_numpy()[:n] for f in frames if column in f])
    if not len(vals):
        return None
    return np.arange(1, n + 1), vals.mean(0), (vals.std(0, ddof=1) if len(vals) > 1 else np.zeros(n))


def plot_learning_curves(index, dataset, backbone):
    """ Train (dashed) vs validation (solid) loss per task, ST vs MTL; right column: gap (val - train). """
    configs = {'ST': {t: f'{dataset}|{backbone}|st_{t}|sum' for t in TASKS},
               'MTL': {t: f'{dataset}|{backbone}|mtl|sum' for t in TASKS}}
    if not any(index.by_config.get(c) for m in configs.values() for c in m.values()):
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    for i, task in enumerate(TASKS):
        for model, per_task in configs.items():
            logs = _logs(index, per_task[task])
            tr, va = _mean_std(logs, f'train_loss_{task}'), _mean_std(logs, f'val_loss_{task}')
            if tr is None or va is None:
                continue
            c = COLORS[model]
            for (x, m, s), style, part in ((tr, '--', 'train'), (va, '-', 'validation')):
                axes[i, 0].plot(x, m, style, color=c, label=f'{model} {part}')
                axes[i, 0].fill_between(x, m - s, m + s, color=c, alpha=0.12, lw=0)
            gaps = [(f[f'val_loss_{task}'] - f[f'train_loss_{task}']).to_numpy(dtype=float) for f in logs]
            n = min(len(x) for x in gaps)
            g = np.array([x[:n] for x in gaps])
            axes[i, 1].plot(np.arange(1, n + 1), g.mean(0), color=c, label=model)
            if len(g) > 1:
                axes[i, 1].fill_between(np.arange(1, n + 1), g.mean(0) - g.std(0, ddof=1), g.mean(0) + g.std(0, ddof=1),
                                        color=c, alpha=0.12, lw=0)
        axes[i, 0].set_title(f'{dataset} · {task}: loss per epoch ({backbone})')
        axes[i, 1].set_title(f'{dataset} · {task}: validation − train loss')
        for j in range(2):
            axes[i, j].set_xlabel('epoch')
        axes[i, 0].set_ylabel('cross-entropy loss')
        axes[i, 1].set_ylabel('loss gap')
        axes[i, 1].axhline(0, color='#6b6b6b', lw=0.8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    return _save(fig, f'rq5_curves_{dataset}')


def plot_low_resource(agg, dataset, selection):
    sel = (selection or {}).get(dataset, {})
    bb, final = sel.get('backbone'), sel.get('final_tag')
    if not bb:
        return None
    look = Lookup(agg)
    models = {'ST': lambda t, f: f'{dataset}|{bb}|st_{t}|{_frac_tag("sum", f)}',
              'MTL': lambda t, f: f'{dataset}|{bb}|mtl|{_frac_tag("sum", f)}'}
    if final:
        models['task-aware'] = lambda t, f: f'{dataset}|{bb}|mtlaware|{_frac_tag(final, f)}'
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
    drawn = False
    for ax, task in zip(axes[0], TASKS):
        for model, cfg in models.items():
            pts = [(f, look.get(cfg(task, f), task)) for f in FRACTIONS]
            pts = [(f, r) for f, r in pts if r is not None]
            if len(pts) < 2:
                continue
            x = [f for f, _ in pts]
            y = [100 * r['macro_f1_mean'] for _, r in pts]
            e = [100 * (r['macro_f1_std'] if r['macro_f1_std'] == r['macro_f1_std'] else 0) for _, r in pts]
            ax.errorbar(x, y, yerr=e, marker='o', capsize=3, color=COLORS[model],
                        label='final model' if model == 'task-aware' else model)
            drawn = True
        ax.set_xscale('log')
        ax.set_xticks(FRACTIONS, [f'{int(f * 100)}%' for f in FRACTIONS])
        ax.set_xlabel('share of the train split (log scale)')
        ax.set_ylabel('test macro-F1 (%)')
        ax.set_title(f'{dataset} · {task}: low-resource ({bb})')
    if not drawn:
        plt.close(fig)
        return None
    handles, labels = max((ax.get_legend_handles_labels() for ax in axes[0]), key=lambda hl: len(hl[1]))
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    return _save(fig, f'rq5_low_resource_{dataset}')


def plot_per_class(per_class, dataset, selection):
    sel = (selection or {}).get(dataset, {})
    bb, loss, final = sel.get('backbone'), sel.get('loss_tag'), sel.get('final_tag')
    if not (bb and loss) or per_class is None or per_class.empty:
        return None
    pc = per_class[(per_class['split'] == 'test') & (per_class['dataset'] == dataset)].copy()
    parts = pc['run_id'].str.split('__', expand=True)
    pc['config'] = parts[0] + '|' + parts[1] + '|' + parts[2] + '|' + parts[4]
    configs = [(f'MTL ({loss})', f'{dataset}|{bb}|mtl|{loss}', COLORS['MTL']),
               ('+ focal', f'{dataset}|{bb}|mtl|{loss}-focal', EXTRA[0])]
    if final:
        configs.append(('final model', f'{dataset}|{bb}|mtlaware|{final}', COLORS['task-aware']))
    configs = [c for c in configs if (pc['config'] == c[1]).any()]
    if not configs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False, gridspec_kw={'width_ratios': [1, 2]})
    for ax, task in zip(axes[0], TASKS):
        sub = pc[pc['task'] == task]
        classes = sub.drop_duplicates('class_id').sort_values('class_id')['class_name'].tolist()
        h = 0.8 / len(configs)
        for k, (label, cfg, color) in enumerate(configs):
            g = sub[sub['config'] == cfg].groupby('class_name')['f1']
            mean = [100 * g.mean().get(c, np.nan) for c in classes]
            std = [100 * (g.std().get(c, 0) if g.count().get(c, 0) > 1 else 0) for c in classes]
            ax.barh(np.arange(len(classes)) + (k - (len(configs) - 1) / 2) * h, mean, height=h, xerr=std,
                    color=color, label=label, error_kw={'lw': 0.8})
        ax.set_yticks(np.arange(len(classes)), classes)
        ax.invert_yaxis()
        ax.set_xlabel('test F1 (%)')
        ax.set_title(f'{dataset} · {task}: per-class F1 ({bb})')
        ax.grid(axis='y', visible=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(configs), bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    return _save(fig, f'rq3_per_class_{dataset}')


def _confusion(index, config, task, n_classes):
    total = np.zeros((n_classes, n_classes))
    for folder in index.by_config.get(config, {}).values():
        p = load_predictions(folder / 'predictions_test.csv')
        np.add.at(total, (p[f'{task}_true'].to_numpy(), p[f'{task}_pred'].to_numpy()), 1)
    return total


def plot_confusion(index, dataset, task, config_a, config_b, class_names, labels=('MTL', 'final model')):
    """ Row-normalised confusion matrices summed over seeds, config_a vs config_b. """
    if not (index.by_config.get(config_a) and index.by_config.get(config_b)):
        return None
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('blues', ['#f4f8fe', '#0d366b'])
    n = len(class_names)
    size = max(5, 0.55 * n + 2)
    fig, axes = plt.subplots(1, 2, figsize=(2 * size + 1, size), squeeze=False)
    for ax, cfg, label in zip(axes[0], (config_a, config_b), labels):
        cm = _confusion(index, cfg, task, n)
        norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for r in range(n):
            for c in range(n):
                ax.text(c, r, f'{norm[r, c]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if norm[r, c] > 0.55 else '#1f1f1f')
        ax.set_xticks(range(n), class_names, rotation=45, ha='right')
        ax.set_yticks(range(n), class_names)
        ax.set_xlabel('predicted')
        ax.set_ylabel('true')
        ax.grid(False)
        ax.set_title(f'{dataset} · {task}: {label} ({len(index.by_config[cfg])} seeds, row-normalised)')
    fig.tight_layout()
    return _save(fig, f'confusion_{dataset}_{task}')


def plot_task_weights(index, dataset, selection):
    bb = (selection or {}).get(dataset, {}).get('backbone')

    def wanted(c):
        _, backbone, mode, tag = c.split('|')
        tokens = set(tag.split('-'))
        return (c.startswith(f'{dataset}|') and mode in ('mtl', 'mtlaware') and tokens & {'unc', 'gradnorm', 'dwa'}
                and not any(t.startswith(('frac', 'cross')) or t == 'mlm' for t in tokens)
                and (bb is None or backbone == bb))
    configs = [c for c in index.by_config if wanted(c)]
    if not configs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
    for ax, task in zip(axes[0], TASKS):
        for k, cfg in enumerate(sorted(configs)):
            stats = _mean_std(_logs(index, cfg), f'w_{task}')
            if stats is None:
                continue
            x, m, s = stats
            color = (EXTRA + [COLORS['MTL'], COLORS['task-aware']])[k % 5]
            ax.plot(x, m, color=color, label=cfg.split('|', 2)[2].replace('|', '/'))
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.12, lw=0)
        ax.set_xlabel('epoch')
        ax.set_ylabel('task weight')
        ax.set_title(f'{dataset} · {task}: learned task weight')
    fig.tight_layout()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(len(labels), 4), bbox_to_anchor=(0.5, 0.0))
    return _save(fig, f'rq2_weights_{dataset}')


def plot_rq4_vs_cramers(agg, eda_summary, selection):
    """ x = Cramér's V per dataset, y = Δ macro-F1 (task-aware − linear, both `sum`) per task; descriptive only. """
    look = Lookup(agg)
    points = []
    for ds, sel in sorted((selection or {}).items()):
        bb = sel.get('backbone')
        v = ((eda_summary or {}).get('task_dependence', {}).get(ds) or {}).get('cramers_v')
        if not bb or v is None:
            continue
        for task in TASKS:
            lin, aware = look.mean(f'{ds}|{bb}|mtl|sum', task), look.mean(f'{ds}|{bb}|mtlaware|sum', task)
            if lin is not None and aware is not None:
                points.append((v, 100 * (aware - lin), ds, task))
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ds_colors = {ds: c for ds, c in zip(sorted({p[2] for p in points}), [COLORS['MTL'], COLORS['ST']])}
    for x, y, ds, task in points:
        marker, offset = ('o', (7, 5)) if task == 'sentiment' else ('s', (7, -11))
        ax.scatter(x, y, color=ds_colors[ds], marker=marker, s=45, zorder=3)
        ax.annotate(f'{ds} · {task}', (x, y), textcoords='offset points', xytext=offset, fontsize=8)
    ax.axhline(0, color='#6b6b6b', lw=0.8)
    ax.set_xlabel("Cramér's V between sentiment and topic (train)")
    ax.set_ylabel('Δ test macro-F1, task-aware − linear (points)')
    ax.set_title('RQ4: does the gain follow the task correlation? (n = 4, descriptive)')
    xs = [p[0] for p in points]
    ax.set_xlim(min(xs) - 0.05, max(xs) + 0.12)
    fig.tight_layout()
    return _save(fig, 'rq4_gain_vs_v')


def build_all(ctx):
    """ Every figure that has data. Returns ({name: path}, skipped). """
    from trainer.results import label_names
    agg, sel, index = ctx['agg'], ctx['selection'], ctx['index']
    made, skipped = {}, []

    def keep(name, path):
        (made.__setitem__(name, path) if path else skipped.append(name))

    for ds in sorted(agg['dataset'].unique()) if len(agg) else []:
        bb = sel.get(ds, {}).get('backbone') or next(
            (c.split('|')[1] for c in sorted(index.by_config) if c.startswith(f'{ds}|') and c.endswith('|mtl|sum')), None)
        if bb:
            keep(f'rq5_curves_{ds}', plot_learning_curves(index, ds, bb))
        keep(f'rq5_low_resource_{ds}', plot_low_resource(agg, ds, sel))
        keep(f'rq3_per_class_{ds}', plot_per_class(ctx['per_class'], ds, sel))
        keep(f'rq2_weights_{ds}', plot_task_weights(index, ds, sel))
        s = sel.get(ds, {})
        if s.get('backbone') and s.get('loss_tag') and s.get('final_tag'):
            for task in TASKS:
                keep(f'confusion_{ds}_{task}', plot_confusion(
                    index, ds, task, f'{ds}|{s["backbone"]}|mtl|{s["loss_tag"]}',
                    f'{ds}|{s["backbone"]}|mtlaware|{s["final_tag"]}', label_names(ds, task),
                    labels=(f'MTL ({s["loss_tag"]})', f'final model ({s["final_tag"]})')))
        else:
            skipped += [f'confusion_{ds}_{t}' for t in TASKS]
    keep('rq4_gain_vs_v', plot_rq4_vs_cramers(agg, ctx['eda_summary'], sel))
    return made, skipped
