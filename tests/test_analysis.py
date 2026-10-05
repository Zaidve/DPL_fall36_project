"""
    Tests for the matrix tooling (experiment_matrix_spec.md §9): trainer/status.py, trainer/results.py,
    trainer/analysis.py. CPU only, no downloads; a fake models/ folder is built in a temp dir.
    Run:  python tests/test_analysis.py   (or `pytest tests` if pytest is installed)
"""
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from fixtures.fake_runs import make_fake_models, make_run  # noqa: E402
from trainer import analysis as A  # noqa: E402
from trainer import status as S  # noqa: E402
from trainer.results import aggregate_seeds, config_key, load_runs, resolve_tags  # noqa: E402
from trainer.runs import is_run_done  # noqa: E402
from utils.common import save_json  # noqa: E402


@contextmanager
def env(**values):
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def workspace():
    """ Temp models/ and reports/ (no selection.json unless a test writes one). """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with env(DPL_MODELS_DIR=tmp / 'models', DPL_OUT_DIR=tmp / 'reports'):
            (tmp / 'reports' / 'tables').mkdir(parents=True)
            yield tmp


def small_matrix(run_ids, **extra):
    rows = []
    for rid in run_ids:
        ds, bb, mode, seed, tag = rid.split('__')
        rows.append({'experiment': 'e1_baseline', 'rq': 'RQ1', 'priority': 'Must', 'run_id': rid, 'dataset': ds,
                     'backbone': bb, 'mode': mode, 'tag': tag, 'seed': int(seed[4:]),
                     'keep_checkpoint': False, 'placeholder': False, **extra.get(rid, {})})
    return pd.DataFrame(rows)


# 1
def test_build_matrix_shipped_configs():
    with workspace() as tmp:
        matrix = S.build_matrix()
        assert len(matrix) == 246 and matrix['run_id'].is_unique
        assert (tmp / 'reports' / 'tables' / 'run_matrix.csv').is_file()
        assert set(matrix['priority']) == {'Must', 'Should', 'Could'}
        e3 = matrix[matrix['experiment'] == 'e3_imbalance']
        assert (e3[e3['tag'].str.contains('wce')]['priority'] == 'Should').all()
        assert (e3[e3['tag'].str.contains('focal')]['priority'] == 'Must').all()
        assert matrix['keep_checkpoint'].sum() == 42
        assert matrix['placeholder'].sum() == 246 - 54 - 18 - 12                # only E1 / E1b / E8 are concrete
        save_json({ds: {'backbone': 'xlmr', 'loss_tag': 'unc', 'imbalance_tag': 'focal', 'final_tag': 'unc-focal'}
                   for ds in ('neu-esc', 'uit-vsfc')}, tmp / 'reports' / 'tables' / 'selection.json')
        resolved = S.build_matrix(save=False)
        assert len(resolved) == 246 and resolved['placeholder'].sum() == 0
        assert 'neu-esc__xlmr__mtlaware__seed42__unc-focal' in set(resolved['run_id'])


# 2
def test_run_status_classification():
    with workspace() as tmp:
        root = tmp / 'models'
        done = make_run(root, 'uit-vsfc', 'xlmr', 'mtl', 'sum', 42, time_s=900)
        failed = root / 'uit-vsfc__xlmr__mtl__seed123__sum'
        failed.mkdir(parents=True)
        (failed / 'error.txt').write_text('Traceback ...\nRuntimeError: CUDA out of memory', encoding='utf-8')
        (root / 'uit-vsfc__xlmr__mtl__seed2026__sum').mkdir()                  # partial
        ids = [done.name, failed.name, 'uit-vsfc__xlmr__mtl__seed2026__sum', 'uit-vsfc__xlmr__st_topic__seed42__sum',
               'uit-vsfc__best__mtl__seed42__unc']
        matrix = small_matrix(ids, **{ids[-1]: {'placeholder': True}})
        status = S.run_status(matrix, [root]).set_index('run_id')['status'].to_dict()
        assert status == {ids[0]: 'done', ids[1]: 'failed', ids[2]: 'partial', ids[3]: 'missing', ids[4]: 'blocked'}
        full = S.run_status(matrix, [root]).set_index('run_id')
        assert full.loc[ids[1], 'error'] == 'RuntimeError: CUDA out of memory'
        assert full.loc[ids[0], 'time_s'] == 900 and full.loc[ids[0], 'peak_gpu_mb'] == 3000
        summary = S.status_summary(S.run_status(matrix, [root]))
        assert summary.loc[0, 'total'] == 5 and summary.loc[0, 'done'] == 1 and summary.loc[0, 'pct_done'] == 20.0


def test_status_across_several_models_dirs():
    with workspace() as tmp:
        a, b = tmp / 's1' / 'models', tmp / 's2' / 'models'
        r1 = make_run(a, 'neu-esc', 'xlmr', 'mtl', 'sum', 42).name
        r2 = make_run(b, 'neu-esc', 'xlmr', 'mtl', 'sum', 123).name
        status = S.run_status(small_matrix([r1, r2]), [a, b]).set_index('run_id')
        assert (status['status'] == 'done').all()
        assert status.loc[r1, 'models_dir'] == str(a) and status.loc[r2, 'models_dir'] == str(b)


# 3
def test_merge_models_dirs():
    with workspace() as tmp:
        s1, s2, dest = tmp / 's1', tmp / 's2', tmp / 'dest'
        make_run(s1, 'uit-vsfc', 'xlmr', 'mtl', 'sum', 42, keep=True)            # keeps best.pt
        make_run(s1, 'uit-vsfc', 'xlmr', 'mtl', 'unc', 42)
        (s1 / 'uit-vsfc__xlmr__mtl__seed123__unc').mkdir()                       # unfinished
        (s1 / '_debug').mkdir()
        make_run(s2, 'uit-vsfc', 'xlmr', 'mtl', 'unc', 42, config_extra={'lr_encoder': 3e-5})   # differs
        make_run(s2, 'uit-vsfc', 'xlmr', 'st_topic', 'sum', 42)
        (dest / 'uit-vsfc__xlmr__st_topic__seed42__sum').mkdir(parents=True)     # partial in dest

        dry = S.merge_models_dirs([s1, s2], dest)                                # dry run copies nothing
        assert not (dest / 'uit-vsfc__xlmr__mtl__seed42__sum').exists()
        assert set(dry['action']) == {'copy', 'skip unfinished', 'replace partial'}

        log = S.merge_models_dirs([s1, s2], dest, dry_run=False).set_index(['source', 'run_id'])['action']
        assert log[(str(s1), 'uit-vsfc__xlmr__mtl__seed42__sum')] == 'copy'
        assert log[(str(s1), 'uit-vsfc__xlmr__mtl__seed123__unc')] == 'skip unfinished'
        assert log[(str(s2), 'uit-vsfc__xlmr__mtl__seed42__unc')] == 'conflict'  # never overwritten
        assert log[(str(s2), 'uit-vsfc__xlmr__st_topic__seed42__sum')] == 'replace partial'
        assert (dest / 'uit-vsfc__xlmr__mtl__seed42__sum' / 'best.pt').is_file()       # kept run keeps best.pt
        assert '3e-05' not in (dest / 'uit-vsfc__xlmr__mtl__seed42__unc' / 'config.yaml').read_text()
        assert is_run_done('uit-vsfc__xlmr__st_topic__seed42__sum', dest) and not (dest / '_debug').exists()
        again = S.merge_models_dirs([s1], dest, dry_run=False)
        assert set(again[again['run_id'] != 'uit-vsfc__xlmr__mtl__seed123__unc']['action']) == {'keep existing'}


def test_merge_skips_checkpoint_of_runs_that_do_not_keep_it():
    with workspace() as tmp:
        folder = make_run(tmp / 's1', 'neu-esc', 'phobert', 'mtl', 'unc', 42)
        (folder / 'best.pt').write_bytes(b'left over')
        S.merge_models_dirs([tmp / 's1'], tmp / 'dest', dry_run=False)
        assert not (tmp / 'dest' / folder.name / 'best.pt').exists()


# 4
def test_load_runs_and_aggregate_seeds():
    with workspace() as tmp:
        root = make_fake_models(tmp / 'models')
        runs, per_class = load_runs([root])
        n_runs = 2 * 2 * 3 * 2 * 3
        assert runs['run_id'].nunique() == n_runs
        assert len(runs) == 2 * 2 * 2 * 3 * (1 + 1 + 2) * 2                    # split x task rows
        topic = per_class[(per_class['dataset'] == 'neu-esc') & (per_class['task'] == 'topic')]
        assert set(topic['class_name']) >= {'spam', 'club_events'} and topic['class_id'].max() == 9
        agg = aggregate_seeds(runs)
        row = agg[(agg['config'] == 'neu-esc|phobert|mtl|unc') & (agg['split'] == 'validation')
                  & (agg['task'] == 'sentiment')].iloc[0]
        vals = [0.60 + 0.02 + 0.03 + 0.01 + o for o in (-0.01, 0.0, 0.01)]
        assert abs(row['macro_f1_mean'] - np.mean(vals)) < 1e-9
        assert abs(row['macro_f1_std'] - np.std(vals, ddof=1)) < 1e-9
        assert row['n_seeds'] == 3 and row['complete'] and row['seeds'] == [42, 123, 2026]
        two = aggregate_seeds(runs[runs['seed'] != 2026])
        assert not two['complete'].any() and (two['n_seeds'] == 2).all()
        assert config_key(runs.iloc[0]) == '|'.join(runs.iloc[0][['dataset', 'backbone', 'mode', 'tag']])


def test_resolve_tags():
    df = pd.DataFrame({'dataset': ['neu-esc', 'uit-vsfc', 'neu-esc'], 'tag': ['bestloss-focal', 'final', 'sum']})
    sel = {'neu-esc': {'loss_tag': 'unc'}, 'uit-vsfc': {'final_tag': 'sum-wce'}}
    assert resolve_tags(df, sel)['tag_resolved'].tolist() == ['unc-focal', 'sum-wce', 'sum']
    assert resolve_tags(df, {})['tag_resolved'].tolist() == ['bestloss-focal', 'final', 'sum']


def test_estimate_remaining_and_session_plan():
    with workspace() as tmp:
        root = tmp / 'models'
        for seed in (42, 123):                                                    # 2 of 3 seeds done, 30 min each
            make_run(root, 'neu-esc', 'xlmr', 'mtl', 'sum', seed, time_s=1800)
        ids = [f'neu-esc__xlmr__mtl__seed{s}__{t}' for s in (42, 123, 2026) for t in ('sum', 'pcgrad')]
        ids += [f'uit-vsfc__phobert__st_topic__seed{s}__sum' for s in (42, 123, 2026)]
        matrix = small_matrix(ids)
        matrix.loc[matrix['tag'] == 'pcgrad', ['experiment', 'priority']] = ['e2_loss', 'Should']
        est = S._estimate_minutes(S.run_status(matrix, [root])).set_index('run_id')
        assert est.loc['neu-esc__xlmr__mtl__seed2026__sum', 'est_min'] == 30                        # same group
        assert est.loc['neu-esc__xlmr__mtl__seed42__pcgrad', 'est_min'] == 30 * 2.5                 # x factor
        assert est.loc['uit-vsfc__phobert__st_topic__seed42__sum', 'est_source'] == 'spec default x factor'
        per_exp, plan = S.estimate_remaining(S.run_status(matrix, [root]), session_hours=2.0)
        assert per_exp.set_index('experiment').loc['e1_baseline', 'remaining'] == 4
        assert plan.iloc[0]['priority'] == 'Must' and plan.iloc[-1]['priority'] == 'Should'        # Must first
        for _, group in plan[plan['note'] == ''].groupby('session'):
            assert group['hours'].sum() <= 2.0 + 1e-9                                                # fits a session
        big = plan[plan['experiment'] == 'e2_loss'].iloc[0]                  # 3 pcgrad runs x 75 min = 3.75 h
        assert big['hours'] == 3.8 and 'spans 2 sessions' in big['note'] and big['session'] == 2
        assert '--only neu-esc__xlmr__' in big['command']


def test_check_consistency():
    with workspace() as tmp:
        root = tmp / 'models'
        a = make_run(root, 'uit-vsfc', 'xlmr', 'mtl', 'sum', 42, keep=True)
        b = make_run(root, 'uit-vsfc', 'xlmr', 'mtl', 'sum', 123, keep=True, config_extra={'lr_encoder': 3e-5},
                     torch_version='2.4.0', best_epoch=1, skipped=2)
        (b / 'best.pt').unlink()
        matrix = small_matrix([a.name, b.name], **{a.name: {'keep_checkpoint': True}, b.name: {'keep_checkpoint': True}})
        warnings = S.check_consistency(S.run_status(matrix, [root]), [root], test_sizes={'uit-vsfc': 3166})
        text = '\n'.join(warnings)
        assert 'different configs' in text and 'torch: runs use different versions' in text
        assert f'{b.name}: best epoch is 1' in text and f'{b.name}: 2 training steps skipped' in text
        assert f'{a.name}: 40 test predictions, processed test split has 3166' in text
        assert f'{b.name}: should keep best.pt' in text and f'{a.name}: should keep' not in text
        clean = S.check_consistency(S.run_status(small_matrix([a.name]), [root]), [root], test_sizes={'uit-vsfc': 40})
        assert clean == []


def test_cli_commands():
    with workspace() as tmp:
        root = tmp / 'models'
        make_run(root, 'uit-vsfc', 'xlmr', 'mtl', 'sum', 42, keep=True)
        matrix = A.main(['matrix'])
        assert len(matrix) == 246
        status = A.main(['status', '--models', str(root)])
        assert (status['status'] == 'done').sum() == 1
        assert (tmp / 'reports' / 'tables' / 'run_status.csv').is_file()
        log = A.main(['merge', '--from', str(root), '--to', str(tmp / 'merged')])
        assert list(log['action']) == ['copy'] and not (tmp / 'merged').exists()           # dry run
        A.main(['merge', '--from', str(root), '--to', str(tmp / 'merged'), '--apply'])
        assert is_run_done('uit-vsfc__xlmr__mtl__seed42__sum', tmp / 'merged')
        assert isinstance(A.main(['check', '--models', str(root)]), list)
        runs, _ = A.main(['cross-eval', '--models', str(tmp / 'empty')])
        assert runs.empty                                                                      # nothing kept yet


# --- selection (step 3) -----------------------------------------------------------

SEEDS = (42, 123, 2026)


def e1_runs(root, scores, datasets=('neu-esc', 'uit-vsfc'), seeds=SEEDS, st=0.60):
    """ E1 runs: scores = {backbone: (val, test)} for mtl/sum; single-task runs at `st` (or {bb: value}). """
    for ds in datasets:
        for bb, (val, test) in scores.items():
            for seed in seeds:
                make_run(root, ds, bb, 'mtl', 'sum', seed, val=val, test=test, keep=True)
                s = st[bb] if isinstance(st, dict) else st
                for mode in ('st_sentiment', 'st_topic'):
                    make_run(root, ds, bb, mode, 'sum', seed, val=s, test=s)


def mtl_runs(root, backbone, tag, val, datasets=('neu-esc', 'uit-vsfc'), seeds=SEEDS):
    for ds in datasets:
        for seed in seeds:
            make_run(root, ds, backbone, 'mtl', tag, seed, val=val, test=val - 0.3)


def test_select_backbone_uses_validation_not_test():
    from trainer import selection as SEL
    with workspace() as tmp:
        # validation ranks phobert first, test ranks xlmr first: selection must follow validation
        e1_runs(tmp / 'models', {'xlmr': (0.60, 0.90), 'visobert': (0.65, 0.80), 'phobert': (0.70, 0.50)})
        agg, _ = SEL.validation_scores([tmp / 'models'])
        choice, info = SEL.select_backbone(agg, 'neu-esc', ['xlmr', 'visobert', 'phobert'])
        assert choice == 'phobert' and abs(info['scores']['phobert'] - 0.69) < 1e-9     # mean of 0.70 / 0.68
        runs, _ = load_runs([tmp / 'models'])
        with_test = aggregate_seeds(runs)
        try:
            SEL.select_backbone(with_test, 'neu-esc')
            raise AssertionError('test rows were accepted')
        except AssertionError as e:
            assert 'validation' in str(e)


def test_select_tie_breaks():
    from trainer import selection as SEL
    with workspace() as tmp:
        # xlmr and phobert tie on mtl/sum (< 0.001): the better single-task mean wins
        e1_runs(tmp / 'models', {'xlmr': (0.7000, 0.5), 'phobert': (0.7005, 0.5)}, datasets=('neu-esc',),
                st={'xlmr': 0.66, 'phobert': 0.62})
        agg, _ = SEL.validation_scores([tmp / 'models'])
        assert SEL.select_backbone(agg, 'neu-esc')[0] == 'xlmr'
        # loss: a tie with sum keeps sum
        mtl_runs(tmp / 'models', 'xlmr', 'unc', 0.7004, datasets=('neu-esc',))
        agg, _ = SEL.validation_scores([tmp / 'models'])
        assert SEL.select_loss(agg, 'neu-esc', 'xlmr', ['unc'])[0] == 'sum'
    assert SEL._pick({'b': 0.5, 'a': 0.5}) == 'a' and SEL.final_tag('unc', 'none') == 'unc'
    assert SEL.final_tag('sum', 'focal') == 'sum-focal'


def test_write_selection_stages():
    import trainer.train as T
    from trainer import selection as SEL
    with workspace() as tmp:
        root, path = tmp / 'models', tmp / 'reports' / 'tables' / 'selection.json'

        def refused(stage, text, **kw):
            try:
                SEL.write_selection(stage, [root], **kw)
                raise AssertionError(f'stage {stage} not refused')
            except SEL.SelectionError as e:
                assert text in str(e), str(e)

        refused('loss', 'stage `backbone` must be written before `loss`')
        e1_runs(root, {'xlmr': (0.60, 0.9), 'visobert': (0.66, 0.9), 'phobert': (0.70, 0.5)}, seeds=(42, 123))
        _, plan = S.estimate_remaining(S.run_status(S.build_matrix(save=False), [root]))
        assert 'e2_loss' not in set(plan['experiment'])                          # blocked until `backbone`
        refused('backbone', 'neu-esc__phobert__mtl__seed2026__sum')                # missing seed listed
        partial = SEL.write_selection('backbone', [root], allow_partial=True, today='2026-10-05')
        assert partial['neu-esc']['backbone'] == 'phobert' and 'partial' in partial['neu-esc']['provenance']['backbone']['note']
        path.unlink()

        e1_runs(root, {'xlmr': (0.60, 0.9), 'visobert': (0.66, 0.9), 'phobert': (0.70, 0.5)}, seeds=(2026,))
        sel = SEL.write_selection('backbone', [root], today='2026-10-05')
        prov = sel['uit-vsfc']['provenance']['backbone']
        assert sel['uit-vsfc']['backbone'] == 'phobert' and prov['metric'] == 'validation macro_f1_mean'
        assert prov['n_seeds'] == {'xlmr': 3, 'visobert': 3, 'phobert': 3} and prov['written'] == '2026-10-05'
        refused('backbone', 'already written')                                    # never changed without force
        _, plan = S.estimate_remaining(S.run_status(S.build_matrix(save=False), [root]))
        first = plan[plan['session'] == plan['session'].min()]
        assert first.iloc[0]['experiment'] == 'e2_loss' and 'unblocks Must runs' in first.iloc[0]['note']

        # E2 not run: loss = sum, with a note
        sel = SEL.write_selection('loss', [root])
        assert sel['neu-esc']['loss_tag'] == 'sum' and 'only candidate' in sel['neu-esc']['provenance']['loss']['note']
        assert sel['neu-esc']['backbone'] == 'phobert'                            # earlier stage kept
        # E2 finished for unc (better) and started for pcgrad (1 seed): incomplete -> refused
        mtl_runs(root, 'phobert', 'unc', 0.75)
        mtl_runs(root, 'phobert', 'pcgrad', 0.80, seeds=(42,))
        refused('loss', 'neu-esc__phobert__mtl__seed123__pcgrad', force=True)
        mtl_runs(root, 'phobert', 'pcgrad', 0.72, seeds=(123, 2026))              # mean 0.7467 < unc
        sel = SEL.write_selection('loss', [root], force=True)
        assert sel['neu-esc']['loss_tag'] == 'unc' and set(sel['neu-esc']['provenance']['loss']['scores']) == {'sum', 'unc', 'pcgrad'}

        refused('imbalance', 'no E3 mtl run with focal / wce finished')
        mtl_runs(root, 'phobert', 'unc-focal', 0.78)
        mtl_runs(root, 'phobert', 'unc-wce', 0.74)
        sel = SEL.write_selection('imbalance', [root])
        assert sel['neu-esc']['imbalance_tag'] == 'focal' and sel['neu-esc']['final_tag'] == 'unc-focal'

        # the plan runs nothing blocked, and no longer promotes E2 once the loss stage exists
        status = S.run_status(S.build_matrix(save=False), [root])
        _, plan = S.estimate_remaining(status)
        assert not plan['note'].str.contains('unblocks').any()
        # the trainer now resolves E4 without placeholders
        specs = T.expand_runs(T.load_experiment('configs/experiment/e4_task_aware.yaml'))
        assert 'neu-esc__phobert__mtlaware__seed42__unc-focal' in {s.run_id for s in specs}

        # re-selecting the backbone drops the stages chosen on top of it
        sel = SEL.write_selection('backbone', [root], force=True)
        assert set(sel['neu-esc']) == {'backbone', 'provenance'} and set(sel['neu-esc']['provenance']) == {'backbone'}


def test_manual_choice_needs_reason_and_candidate():
    import json
    from trainer import selection as SEL
    with workspace() as tmp:
        root, path = tmp / 'models', tmp / 'reports' / 'tables' / 'selection.json'
        e1_runs(root, {'xlmr': (0.60, 0.9), 'visobert': (0.66, 0.9), 'phobert': (0.70, 0.5)})
        mtl_runs(root, 'phobert', 'pcgrad', 0.75)                                  # the rule would pick pcgrad
        SEL.write_selection('backbone', [root])
        for kw in ({'choose': 'sum'}, {'choose': 'dwa', 'reason': 'not a candidate'}):
            try:
                SEL.write_selection('loss', [root], **kw)
                raise AssertionError(f'should have been refused: {kw}')
            except SEL.SelectionError:
                assert 'loss_tag' not in json.loads(path.read_text(encoding='utf-8'))['neu-esc']
        sel = SEL.write_selection('loss', [root], choose='sum', reason='no strategy is significantly better')
        for ds in ('neu-esc', 'uit-vsfc'):
            prov = sel[ds]['provenance']['loss']
            assert sel[ds]['loss_tag'] == prov['choice'] == 'sum' and prov['manual'] is True
            assert prov['rule_choice'] == 'pcgrad' and 'significantly' in prov['reason']
            assert set(prov['scores']) == {'sum', 'pcgrad'}                         # every score is kept
        assert 'manual choice (rule: pcgrad)' in SEL.explain_selection(path)['note'].iloc[-1]
        # the next stage builds on the manual choice; a manual imbalance choice sets final_tag
        mtl_runs(root, 'phobert', 'sum-focal', 0.78)
        sel = SEL.write_selection('imbalance', [root], choose='none', reason='keep it simple')
        assert sel['neu-esc']['imbalance_tag'] == 'none' and sel['neu-esc']['final_tag'] == 'sum'
        assert sel['neu-esc']['provenance']['imbalance']['rule_choice'] == 'focal'


def test_select_cli_and_explain():
    with workspace() as tmp:
        root = tmp / 'models'
        assert A.main(['select', '--stage', 'backbone', '--models', str(root)]) is None           # refused, printed
        e1_runs(root, {'xlmr': (0.6, 0.6), 'visobert': (0.62, 0.6), 'phobert': (0.64, 0.6)})
        sel = A.main(['select', '--stage', 'backbone', '--models', str(root)])
        assert sel['neu-esc']['backbone'] == 'phobert'
        table = A.main(['select', '--explain'])
        assert len(table) == 2 and set(table['stage']) == {'backbone'}
        assert table['choice'].str.contains('backbone=phobert').all()


# --- significance, tables, figures (step 4) -----------------------------------------------

def write_predictions(folder, task, y_true, pred, split='test'):
    """ Overwrite one run's predictions for `task` with hand-made labels (ids test-0 ... test-n). """
    n_classes = int(max(max(y_true), max(pred))) + 1
    df = pd.DataFrame({'id': [f'x-test-{i}' for i in range(len(y_true))], f'{task}_true': y_true, f'{task}_pred': pred})
    for k in range(n_classes):
        df[f'{task}_prob_{k}'] = (np.asarray(pred) == k).astype(float)
    df.to_csv(Path(folder) / f'predictions_{split}.csv', index=False)


def paired_configs(root, preds_a, preds_b, y_true, seeds=SEEDS, task='sentiment'):
    for seed in seeds:
        a = make_run(root, 'uit-vsfc', 'xlmr', 'st_sentiment', 'sum', seed)
        b = make_run(root, 'uit-vsfc', 'xlmr', 'mtl', 'sum', seed)
        write_predictions(a, task, y_true, preds_a[seed])
        write_predictions(b, task, y_true, preds_b[seed])
    return 'uit-vsfc|xlmr|st_sentiment|sum', 'uit-vsfc|xlmr|mtl|sum'


# 7
def test_load_paired_mcnemar_and_marks():
    from trainer import significance as SIG
    from trainer.evaluate import mcnemar
    with workspace() as tmp:
        root = tmp / 'models'
        y = np.array([0, 1, 2] * 40)
        worse = {s: np.where(np.arange(120) < 60, (y + 1) % 3, y) for s in SEEDS}        # A: 60 wrong
        better = {s: np.where(np.arange(120) < 5, (y + 1) % 3, y) for s in SEEDS}        # B: 5 wrong
        a, b = paired_configs(root, worse, better, y)
        index = SIG.RunIndex([root])
        res = SIG.mcnemar_by_seed(a, b, 'sentiment', index=index)
        ref = mcnemar(y, worse[42], better[42])
        assert len(res) == 3 and res.iloc[0]['b'] == ref['only_a_correct'] and res.iloc[0]['c'] == ref['only_b_correct']
        assert abs(res.iloc[0]['p_value'] - ref['p_value']) < 1e-12 and (res['direction'] == 'B better').all()
        assert SIG.significance_mark(res) == ('†', '3/3 seeds p<0.05')
        mixed = res.copy()
        mixed.loc[2, 'direction'] = 'A better'
        assert SIG.significance_mark(mixed)[0] == ''                               # directions disagree
        weak = res.copy()
        weak.loc[1, 'p_value'] = 0.2
        assert SIG.significance_mark(weak) == ('', '2/3 seeds p<0.05')
        # mismatched ids
        folder = index.by_config[b][42]
        pd.read_csv(folder / 'predictions_test.csv').iloc[:-1].to_csv(folder / 'predictions_test.csv', index=False)
        try:
            SIG.load_paired(index.by_config[a][42], folder, 'sentiment', index=SIG.RunIndex([root]))
            raise AssertionError('mismatched ids accepted')
        except ValueError as e:
            assert 'ids differ' in str(e)


# 8
def test_paired_bootstrap():
    from sklearn.metrics import f1_score
    from trainer import significance as SIG
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
    assert abs(SIG._macro_f1(y, p, 4) - f1_score(y, p, average='macro', labels=range(4), zero_division=0)) < 1e-12
    with workspace() as tmp:
        y = np.array([0, 1, 2] * 40)
        same = {s: np.where(np.arange(120) % 4 == 0, (y + 1) % 3, y) for s in SEEDS}
        a, b = paired_configs(tmp / 'same', same, same, y)
        r = SIG.paired_bootstrap(a, b, 'sentiment', n_boot=200, index=[tmp / 'same'])
        assert r['observed_diff'] == 0 and r['ci_low'] == 0 and r['ci_high'] == 0 and r['n_seeds'] == 3
        worse = {s: np.where(np.arange(120) % 3 == 0, (y + 1) % 3, y) for s in SEEDS}
        better = {s: np.where(np.arange(120) % 20 == 0, (y + 1) % 3, y) for s in SEEDS}
        a, b = paired_configs(tmp / 'diff', worse, better, y)
        r = SIG.paired_bootstrap(a, b, 'sentiment', n_boot=200, index=[tmp / 'diff'])
        assert r['ci_low'] > 0 and r['share_gt_0'] == 1.0 and r['observed_diff'] > 0


_FULL = {}


@contextmanager
def full_matrix():
    """ All 246 fake runs + selection.json + the real eda_summary.json, built once per test session. """
    import shutil
    from fixtures.fake_runs import make_full_matrix
    from utils.common import PROJECT_ROOT
    if 'dir' not in _FULL:
        _FULL['tmp'] = tempfile.TemporaryDirectory()
        _FULL['dir'] = Path(_FULL['tmp'].name)
    tmp = _FULL['dir']
    with env(DPL_MODELS_DIR=tmp / 'models', DPL_OUT_DIR=tmp / 'reports'):
        if not (tmp / 'models').exists():
            (tmp / 'reports' / 'tables').mkdir(parents=True, exist_ok=True)
            shutil.copy(PROJECT_ROOT / 'reports' / 'eda_summary.json', tmp / 'reports' / 'eda_summary.json')
            make_full_matrix(tmp / 'models', tmp / 'reports' / 'tables' / 'selection.json')
        yield tmp


# 9
def test_main_table_and_markdown():
    from trainer import tables as TB
    with full_matrix():
        ctx = TB.load_context()
        neu, skipped = TB.main_table(ctx['agg'], 'neu-esc', ctx['selection'])
        uit, _ = TB.main_table(ctx['agg'], 'uit-vsfc', ctx['selection'])
        assert (neu['block'] == 'Prior work (reported)').sum() == 11 and 'Prior work (reported)' not in set(uit['block'])
        assert not skipped
        assert list(neu.columns) == ['block', 'model', 'Sent Acc', 'Sent mF1', 'Sent wF1', 'Topic Acc', 'Topic mF1', 'Topic wF1']
        for col in neu.columns[2:]:
            starred = uit[uit[col].str.endswith('*')]
            assert len(starred) >= 1                                                   # best per column
            assert starred['model'].str.startswith('Final model').all()               # the fake final model is best
        assert uit.iloc[0, 2].count('±') == 1 and '(n=' not in uit.iloc[0, 2]
        md = TB.to_markdown(uit, bold_rows=[len(uit) - 1]).splitlines()
        assert all(line.count('|') == len(uit.columns) + 1 for line in md if '\\|' not in line)
        assert md[1].startswith('|---|---|---:') and md[-1].startswith('| **Ours')
        partial = ctx['agg'][ctx['agg']['split'] == 'test'].copy()                     # tables read test rows
        partial.loc[partial.index[0], 'n_seeds'] = 2
        assert '(n=2)' in TB.Lookup(partial).cell(partial.iloc[0]['config'], partial.iloc[0]['task'])


def test_rq_tables_and_significance_pairs():
    from trainer import significance as SIG
    from trainer import tables as TB
    with full_matrix():
        ctx = TB.load_context()
        tables, skipped = TB.build_all(ctx=ctx)
        assert not skipped
        assert len(tables['rq1_st_vs_mtl']) == 2 * 3 * 2                                # dataset x backbone x task
        assert set(tables['rq2_loss']['strategy']) == {'sum', 'unc', 'pcgrad', 'gradnorm', 'smartemb', 'smartref'}
        pcgrad = tables['rq2_loss'].query('strategy == "pcgrad"')['time/epoch (x sum)'].iloc[0]
        assert pcgrad == '2.50'
        assert len(tables['rq3_imbalance']) == 2 * 3 * 3 and len(tables['rq3_per_class']) == 6
        assert set(tables['rq4_task_aware']['Cramér\'s V']) == {'0.177', '0.352'}
        assert len(tables['rq5_low_resource']) == 2 * 4 and len(tables['e6_mlm']) == 4
        for name in ('results_neu-esc', 'rq1_st_vs_mtl', 'rq5_low_resource', 'cost'):
            assert (Path(os.environ['DPL_OUT_DIR']) / 'tables' / f'{name}.md').is_file()
        pairs = SIG.standard_pairs(ctx['selection'], set(ctx['agg']['config']), ctx['agg'])
        by_rq = pd.Series([p[0] for p in pairs]).value_counts().to_dict()
        # RQ5: 2 datasets x 3 fractions x 2 tasks x (single task vs MTL, single task vs final model)
        assert by_rq == {'RQ1': 12, 'RQ2': 12, 'RQ3': 8, 'RQ4': 4, 'Final': 4, 'RQ5': 24}
        assert ('RQ5', 'neu-esc|visobert|st_topic|sum-frac0.1', 'neu-esc|visobert|mtlaware|unc-focal-frac0.1',
                ('topic',)) in pairs
        final = [p for p in pairs if p[0] == 'Final' and p[1].startswith('neu-esc')]
        assert {p[1] for p in final} == {'neu-esc|visobert|st_sentiment|focal', 'neu-esc|visobert|st_topic|focal'}


# 11
def test_every_figure_writes_a_png():
    from trainer import figures as FG
    from trainer import tables as TB
    with full_matrix() as tmp:
        made, skipped = FG.build_all(TB.load_context())
        assert not skipped and len(made) == 13
        for name, path in made.items():
            assert path.is_file() and path.stat().st_size > 10_000, name
        assert FG.plot_low_resource(TB.load_context()['agg'], 'neu-esc', {}) is None           # no selection: skipped


def test_cli_all_on_partial_matrix():
    with workspace() as tmp:
        root = make_fake_models(tmp / 'models', tags=('sum',), seeds=(42, 123))            # E1-like, 2 seeds only
        out = A.main(['all', '--n-boot', '50'])
        assert len(out['significance']) == 2 * 2 * 2                                       # RQ1 pairs x tasks
        assert (tmp / 'reports' / 'tables' / 'results_neu-esc.md').is_file()
        assert '(n=2)' in (tmp / 'reports' / 'tables' / 'results_neu-esc.md').read_text(encoding='utf-8')
        assert 'rq5_curves_neu-esc' in out['figures'] and len(out['figures']) == 2
        assert out['cross_eval'][0].empty                    # fake best.pt files are not checkpoints: skipped, no crash


# --- E7 cross-dataset evaluation (step 5) --------------------------------------------------

class HashTokenizer:
    """ Whitespace words -> stable ids in [4, 200); <s>=0 <pad>=1 </s>=2. No vocabulary, no downloads. """
    pad_token_id = 1

    def __len__(self):
        return 200

    def __call__(self, texts, add_special_tokens=True, truncation=False, max_length=None, **_):
        import zlib
        out = []
        for t in texts:
            ids = [4 + zlib.crc32(w.encode()) % 196 for w in t.split()]
            if truncation and max_length:
                ids = ids[:max_length - 2]
            out.append([0, *ids, 2])
        return {'input_ids': out, 'attention_mask': [[1] * len(x) for x in out]}


def tiny_build(cfg, num_labels):
    import torch
    from transformers import AutoModel, BertConfig
    from architecture import MTLModel
    torch.manual_seed(0)
    enc = AutoModel.from_config(BertConfig(vocab_size=200, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                           intermediate_size=64, max_position_embeddings=128))
    return MTLModel.from_encoder(enc, num_labels, head=cfg['model'].get('head', 'linear'))


def processed_folder(folder, n_test=30):
    """ Small processed parquet files for both datasets (test split only matters here). """
    rng = np.random.default_rng(0)
    for ds, (n_sent, n_topic) in (('neu-esc', (4, 10)), ('uit-vsfc', (3, 4))):
        rows = [{'id': f'{ds}-test-{i}', 'text_clean': ' '.join(f'w{int(x)}' for x in rng.integers(0, 50, 6)),
                 'sentiment': int(rng.integers(n_sent)), 'topic': int(rng.integers(n_topic)), 'split': 'test'}
                for i in range(n_test)]
        df = pd.DataFrame(rows)
        df['text_seg'] = df['text_clean']
        df.to_parquet(Path(folder) / f'{ds}.parquet', index=False)


def test_sentiment_3class_mapping():
    from trainer import cross_eval as CE
    assert CE.SENTIMENT_3CLASS['neu-esc']['toxic'] == 'negative'
    assert [CE.CLASSES_3[i] for i in CE.to_3class('neu-esc', [0, 1, 2, 3])] == ['neutral', 'positive', 'negative', 'negative']
    assert [CE.CLASSES_3[i] for i in CE.to_3class('uit-vsfc', [0, 1, 2])] == ['negative', 'neutral', 'positive']
    probs = np.array([[0.1, 0.2, 0.3, 0.4]])                                   # neutral, positive, negative, toxic
    assert np.allclose(CE.probs_to_3class('neu-esc', probs), [[0.7, 0.1, 0.2]])


def test_cross_eval_run_and_all():
    import torch
    from trainer import cross_eval as CE
    with workspace() as tmp:
        (tmp / 'processed').mkdir()
        processed_folder(tmp / 'processed')
        with env(DPL_PROCESSED_DIR=tmp / 'processed'):
            root = tmp / 'models'
            run = make_run(root, 'neu-esc', 'xlmr', 'mtl', 'sum', 42, keep=True)
            model = tiny_build({'model': {}}, {'sentiment': 4, 'topic': 10})
            torch.save({'model': {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                        'dtype': 'float16'}, run / 'best.pt')                  # like a kept fp16 checkpoint
            no_ckpt = make_run(root, 'neu-esc', 'xlmr', 'mtl', 'unc', 42)      # not kept: ignored
            try:
                CE.load_run_model(no_ckpt, 'cpu', tiny_build)
                raise AssertionError('missing best.pt not reported')
            except FileNotFoundError as e:
                assert 'best.pt' in str(e)

            res = CE.cross_eval_run(run, 'uit-vsfc', device='cpu', build=tiny_build, tokenizer=HashTokenizer())
            assert res['n'] == 30 and res['max_len'] == 48 and 0 <= res['mf1_3c'] <= 1 and 0 <= res['acc_3c'] <= 1
            preds = pd.read_csv(run / 'cross_uit-vsfc_test.csv')
            assert len(preds) == 30 and set(preds['y_pred_3c']) <= {0, 1, 2}
            assert np.allclose(preds[['prob_negative', 'prob_neutral', 'prob_positive']].sum(1), 1, atol=1e-5)

            runs, table = CE.cross_eval_all([root], device='cpu', build=tiny_build, tokenizer_for=lambda bb: HashTokenizer())
            assert list(runs['run_id']) == [run.name] and runs.iloc[0]['target'] == 'uit-vsfc'
            assert abs(runs.iloc[0]['drop'] - (runs.iloc[0]['in_domain_mf1_3c'] - runs.iloc[0]['mf1_3c'])) < 1e-12
            assert (run / 'cross_eval.json').is_file()
            assert list(table.columns) == ['model', 'train on', 'test NEU-ESC mF1 (3 labels)',
                                           'test UIT-VSFC mF1 (3 labels)', 'drop (in − cross)']
            assert (tmp / 'reports' / 'tables' / 'e7_cross_dataset.md').is_file()

            def boom(cfg, labels):
                raise AssertionError('cached run was re-evaluated')
            again, _ = CE.cross_eval_all([root], build=boom)                     # served from cross_eval.json
            assert again.iloc[0]['mf1_3c'] == runs.iloc[0]['mf1_3c']


if __name__ == '__main__':
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith('test_')]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS  {name}')
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f'FAIL  {name}: {type(e).__name__}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
