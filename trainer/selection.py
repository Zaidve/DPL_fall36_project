"""
    Write reports/tables/selection.json in stages (experiment_matrix_spec.md §3).

      backbone  (after E1)  best backbone per dataset: E1 mtl/sum
      loss      (after E2)  best loss tag on that backbone: sum (E1) or an E2 tag (unc, pcgrad, smartemb, gradnorm)
      imbalance (after E3)  none / focal / wce on top of the loss tag -> imbalance_tag and final_tag

    `joint` is an alternative to `backbone` + `loss`: it compares every backbone x loss tag pair with finished
    mtl runs (E1 sum, E2 tags, E1b smartref) in one step and writes backbone and loss_tag together. Candidates
    are named `backbone/tag`. The grid may be incomplete (E2 only ran on one backbone per dataset); the
    provenance lists every pair that was compared.

    Every choice uses the mean over seeds of the VALIDATION macro_f1_mean (mean macro-F1 over the run's tasks);
    test scores are never read (asserted). Ties (difference < 0.001): backbone -> higher mean of the two
    single-task runs, then alphabetical; loss / imbalance -> the simpler option (sum / none), then alphabetical;
    joint -> a `sum` pair, then the higher single-task mean, then alphabetical.

    A stage is refused when the previous stage is missing, when it already exists (unless force), or when a
    candidate has fewer than 3 seeds (unless allow_partial; the message lists the missing run ids).
    Re-writing a stage with force removes the later stages, which were chosen on top of it (`joint` keeps
    them for a dataset whose backbone and loss tag do not change).

    A manual choice (choose + reason) replaces the rule's choice for every dataset, e.g. the simplest
    option when no candidate is significantly better, or to exclude a candidate for its cost. It must be one
    of the stage's candidates; the provenance keeps the reason, the rule's choice and all validation scores.

      {"neu-esc": {"backbone": "visobert", "loss_tag": "unc", "imbalance_tag": "focal", "final_tag": "unc-focal",
                   "provenance": {"backbone": {"stage": "e1", "metric": "validation macro_f1_mean",
                                               "scores": {...}, "n_seeds": {...}, "written": "2026-10-05"}, ...}}}
"""
import datetime
from pathlib import Path

import pandas as pd

from trainer.results import aggregate_seeds, load_runs
from trainer.runs import make_run_id
from utils.common import CONFIGS_DIR, load_json, models_dir, reports_dir, save_json
from utils.config import load_config

METRIC = 'macro_f1_mean'
TIE = 1e-3
STAGES = ('backbone', 'loss', 'imbalance')
JOINT = 'joint'                                    # backbone + loss in one step, instead of the first two stages
STAGE_FIELDS = {'backbone': ['backbone'], 'loss': ['loss_tag'], 'imbalance': ['imbalance_tag', 'final_tag'],
                JOINT: ['backbone', 'loss_tag']}
STAGE_SOURCE = {'backbone': 'e1', 'loss': 'e1+e2', 'imbalance': 'e3', JOINT: 'e1+e1b+e2'}
PROVENANCE_ORDER = ('backbone', 'loss', JOINT, 'imbalance')


class SelectionError(RuntimeError):
    pass


def selection_path():
    return reports_dir() / 'tables' / 'selection.json'


def _experiment(name):
    return load_config(CONFIGS_DIR / 'experiment' / f'{name}.yaml')


def _validation_only(agg):
    assert (agg['split'] == 'validation').all(), 'selection must only use validation scores (test rows passed)'


def validation_scores(models_dirs=None):
    """ Seed-aggregated VALIDATION scores (test rows are dropped before aggregation). """
    runs, _ = load_runs(models_dirs)
    runs = runs[runs['split'] == 'validation']
    agg = aggregate_seeds(runs, metrics=(METRIC,))
    # macro_f1_mean is the same for every task row of a run: keep one row per config
    return agg.drop_duplicates('config').assign(task='all').reset_index(drop=True), runs


def _score_table(agg, dataset, backbone, mode, tags):
    _validation_only(agg)
    rows = agg[(agg['dataset'] == dataset) & (agg['backbone'] == backbone) & (agg['mode'] == mode)
               & agg['tag'].isin(tags)]
    return {r['tag']: (float(r[f'{METRIC}_mean']), int(r['n_seeds'])) for r in rows.to_dict('records')}


def _pick(scores, prefer=None, secondary=None):
    """ Highest score; within TIE of the best: `secondary` score (higher wins), then `prefer`, then name. """
    best = max(v for v in scores.values())
    tied = [k for k, v in scores.items() if best - v < TIE]
    if len(tied) > 1 and secondary:
        top = max(secondary.get(k, float('-inf')) for k in tied)
        tied = [k for k in tied if secondary.get(k, float('-inf')) >= top - 1e-12]
    if prefer in tied:
        return prefer
    return sorted(tied)[0]


def select_backbone(agg, dataset, backbones=None):
    """ E1 mtl/sum with the highest mean validation macro_f1_mean. Returns (backbone, info). """
    _validation_only(agg)
    backbones = backbones or sorted(agg.loc[agg['dataset'] == dataset, 'backbone'].unique())
    scores, n_seeds, single = {}, {}, {}
    for bb in backbones:
        mtl = _score_table(agg, dataset, bb, 'mtl', ['sum'])
        if 'sum' in mtl:
            scores[bb], n_seeds[bb] = mtl['sum']
            st = [_score_table(agg, dataset, bb, m, ['sum']).get('sum', (None,))[0] for m in ('st_sentiment', 'st_topic')]
            if all(v is not None for v in st):
                single[bb] = sum(st) / 2
    if not scores:
        raise SelectionError(f'{dataset}: no finished E1 mtl/sum runs')
    choice = _pick(scores, secondary=single)
    return choice, {'scores': scores, 'n_seeds': n_seeds, 'single_task_mean': single}


def select_loss(agg, dataset, backbone, tags=None):
    """ Best loss tag on `backbone`: `sum` (E1) plus every finished E2 tag. Returns (tag, info). """
    _validation_only(agg)
    tags = ['sum'] + [t for t in (tags or []) if t != 'sum']
    table = _score_table(agg, dataset, backbone, 'mtl', tags)
    if 'sum' not in table:
        raise SelectionError(f'{dataset}: no finished E1 mtl/sum run on {backbone}')
    scores = {t: v[0] for t, v in table.items()}
    info = {'scores': scores, 'n_seeds': {t: v[1] for t, v in table.items()}}
    if len(scores) == 1:
        info['note'] = 'no E2 run finished: sum is the only candidate'
    return _pick(scores, prefer='sum'), info


def select_joint(agg, dataset, backbones=None, tags=None):
    """ Best `backbone/tag` pair over every backbone: mtl/sum (E1) plus every finished tag in `tags`.
        Returns ('backbone/tag', info). """
    _validation_only(agg)
    backbones = backbones or sorted(agg.loc[agg['dataset'] == dataset, 'backbone'].unique())
    tags = ['sum'] + [t for t in (tags or []) if t != 'sum']
    scores, n_seeds, single = {}, {}, {}
    for bb in backbones:
        for tag, (score, n) in _score_table(agg, dataset, bb, 'mtl', tags).items():
            scores[f'{bb}/{tag}'], n_seeds[f'{bb}/{tag}'] = score, n
        st = [_score_table(agg, dataset, bb, m, ['sum']).get('sum', (None,))[0] for m in ('st_sentiment', 'st_topic')]
        if all(v is not None for v in st):
            single[bb] = sum(st) / 2
    if not scores:
        raise SelectionError(f'{dataset}: no finished mtl runs')
    best = max(scores.values())
    tied = [k for k, v in scores.items() if best - v < TIE]
    tied = [k for k in tied if k.endswith('/sum')] or tied
    choice = _pick({k: 0.0 for k in tied}, secondary={k: single.get(k.split('/')[0], float('-inf')) for k in tied})
    return choice, {'scores': scores, 'n_seeds': n_seeds, 'single_task_mean': single}


def select_imbalance(agg, dataset, backbone, loss_tag, variants=('focal', 'wce')):
    """ none (mtl/{loss_tag}) vs focal / wce (mtl/{loss_tag}-focal, ...). Returns (imbalance, info). """
    _validation_only(agg)
    candidates = {'none': loss_tag, **{v: f'{loss_tag}-{v}' for v in variants}}
    table = _score_table(agg, dataset, backbone, 'mtl', list(candidates.values()))
    if loss_tag not in table:
        raise SelectionError(f'{dataset}: no finished mtl/{loss_tag} run on {backbone}')
    scores = {name: table[tag][0] for name, tag in candidates.items() if tag in table}
    info = {'scores': scores, 'n_seeds': {name: table[tag][1] for name, tag in candidates.items() if tag in table},
            'tags': {name: tag for name, tag in candidates.items() if tag in table}}
    return _pick(scores, prefer='none'), info


def final_tag(loss_tag, imbalance_tag):
    return loss_tag if imbalance_tag in (None, '', 'none') else f'{loss_tag}-{imbalance_tag}'


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _missing_run_ids(dataset, backbone, mode, tags, seeds, done_ids):
    return [make_run_id(dataset, backbone, mode, s, t) for t in tags for s in seeds
            if make_run_id(dataset, backbone, mode, s, t) not in done_ids]


def _joint_tags(e2, e1b):
    return [g['tag'] for exp in (e2, e1b) for g in exp['grid'] if g['mode'] == 'mtl']


def _stage_fields(stage, choice, current):
    if stage == 'backbone':
        return {'backbone': choice}
    if stage == 'loss':
        return {'loss_tag': choice}
    if stage == JOINT:
        backbone, tag = choice.split('/', 1)
        return {'backbone': backbone, 'loss_tag': tag}
    return {'imbalance_tag': choice, 'final_tag': final_tag(current['loss_tag'], choice)}


def _stage_candidates(stage, dataset, current, e1, e2, e3, e1b=None):
    """ (mode, backbone, required tags, optional tags) whose runs this stage compares. """
    if stage == 'backbone':
        return [('mtl', bb, ['sum'], []) for bb in e1['backbones']]
    if stage == JOINT:
        return [('mtl', bb, ['sum'], _joint_tags(e2, e1b)) for bb in e1['backbones']]
    backbone = current['backbone']
    if stage == 'loss':
        e2_tags = [g['tag'] for g in e2['grid'] if g['mode'] == 'mtl']
        return [('mtl', backbone, ['sum'], e2_tags)]
    loss = current['loss_tag']
    variants = [g['tag'].split('-', 1)[1] for g in e3['grid'] if g['mode'] == 'mtl' and g['tag'].startswith('bestloss-')]
    return [('mtl', backbone, [loss], [f'{loss}-{v}' for v in variants])]


def write_selection(stage, models_dirs=None, out_path=None, allow_partial=False, force=False, today=None,
                    choose=None, reason=None):
    """ Add one stage to selection.json (see module docstring). Returns the new selection dict. """
    if stage not in STAGES + (JOINT,):
        raise ValueError(f'unknown stage {stage!r}; choose from {STAGES + (JOINT,)}')
    if choose and not (reason or '').strip():
        raise SelectionError('a manual choice needs a reason (it is stored in selection.json)')
    out_path = Path(out_path or selection_path())
    selection = load_json(out_path) if out_path.exists() else {}
    e1, e2, e3 = _experiment('e1_baseline'), _experiment('e2_loss'), _experiment('e3_imbalance')
    e1b = _experiment('e1b_smart_ref')
    seeds = list(e1['seeds'])
    datasets = list(e1.get('datasets') or e1['data']['datasets'])
    index = 1 if stage == JOINT else STAGES.index(stage)      # joint stands for the first two stages

    agg, runs = validation_scores(models_dirs or [models_dir()])
    done_ids = set(runs['run_id'])
    problems, new = [], {}
    for ds in datasets:
        current = dict(selection.get(ds, {}))
        if stage != JOINT and index > 0 and not all(f in current for f in STAGE_FIELDS[STAGES[index - 1]]):
            problems.append(f'{ds}: stage `{STAGES[index - 1]}` must be written before `{stage}`')
            continue
        if any(f in current for f in STAGE_FIELDS[stage]) and not force:
            problems.append(f'{ds}: stage `{stage}` already written ({", ".join(f"{f}={current[f]}" for f in STAGE_FIELDS[stage])}); '
                            'use force to re-select')
            continue
        # completeness: required candidates need every seed; optional ones need every seed once started
        missing = []
        optional_ready = []
        for mode, bb, required, optional in _stage_candidates(stage, ds, current, e1, e2, e3, e1b):
            missing += _missing_run_ids(ds, bb, mode, required, seeds, done_ids)
            for tag in optional:
                lacking = _missing_run_ids(ds, bb, mode, [tag], seeds, done_ids)
                if len(lacking) == len(seeds):
                    continue                                   # not started: not a candidate
                missing += lacking
                optional_ready.append(tag)
        if stage == 'imbalance' and not optional_ready:
            missing.append(f'(no E3 mtl run with focal / wce finished for {ds})')
        if missing and not allow_partial:
            problems.append(f'{ds}: candidates incomplete (need {len(seeds)} seeds); missing: ' + ', '.join(missing))
            continue

        try:
            if stage == 'backbone':
                choice, info = select_backbone(agg, ds, list(e1['backbones']))
            elif stage == 'loss':
                choice, info = select_loss(agg, ds, current['backbone'], [g['tag'] for g in e2['grid']])
            elif stage == JOINT:
                choice, info = select_joint(agg, ds, list(e1['backbones']), _joint_tags(e2, e1b))
            else:
                choice, info = select_imbalance(agg, ds, current['backbone'], current['loss_tag'])
        except SelectionError as e:
            problems.append(str(e))
            continue
        if missing:
            info['note'] = (info.get('note', '') + ' partial: ' + ', '.join(missing)).strip()
        if choose:
            if choose not in info['scores']:
                problems.append(f'{ds}: `{choose}` is not a candidate of stage `{stage}` '
                                f'(candidates: {", ".join(sorted(info["scores"]))})')
                continue
            info.update(manual=True, rule_choice=choice, reason=reason.strip())
            info['note'] = (info.get('note', '') + f' manual choice (rule: {choice}): {reason.strip()}').strip()
            choice = choose
        fields = _stage_fields(stage, choice, current)

        # re-writing a stage invalidates the stages chosen on top of it; a joint choice that keeps the
        # backbone and the loss tag leaves them valid
        unchanged = stage == JOINT and all(current.get(f) == v for f, v in fields.items())
        for later in () if unchanged else STAGES[index + 1:]:
            for f in STAGE_FIELDS[later]:
                current.pop(f, None)
            current.get('provenance', {}).pop(later, None)
        # backbone / loss and joint are two ways to make the same choice: keep the provenance of one
        for other in (('backbone', 'loss') if stage == JOINT else (JOINT,) if stage in ('backbone', 'loss') else ()):
            current.get('provenance', {}).pop(other, None)
        current.update(fields)
        current.setdefault('provenance', {})[stage] = {
            'stage': STAGE_SOURCE[stage], 'metric': f'validation {METRIC}', 'choice': choice,
            'written': today or datetime.date.today().isoformat(), **info}
        new[ds] = current

    if problems:
        raise SelectionError('selection not written:\n  - ' + '\n  - '.join(problems))
    selection.update(new)
    save_json(selection, out_path)
    return selection


def explain_selection(path=None):
    """ Current choices and their provenance as a table (also printed). """
    path = Path(path or selection_path())
    if not path.exists():
        print(f'{path} does not exist yet')
        return pd.DataFrame()
    rows = []
    for ds, sel in load_json(path).items():
        for stage in PROVENANCE_ORDER:
            prov = sel.get('provenance', {}).get(stage)
            if not prov:
                continue
            scores = ', '.join(f'{k} {v:.4f} (n={prov["n_seeds"].get(k)})' for k, v
                               in sorted(prov['scores'].items(), key=lambda kv: -kv[1]))
            rows.append({'dataset': ds, 'stage': stage,
                         'choice': ', '.join(f'{f}={sel.get(f)}' for f in STAGE_FIELDS[stage]),
                         'validation macro_f1_mean': scores, 'written': prov.get('written'),
                         'note': prov.get('note', '')})
    table = pd.DataFrame(rows)
    with pd.option_context('display.width', 220, 'display.max_colwidth', 120):
        print(table.to_string(index=False) if len(table) else '(empty)')
    return table
