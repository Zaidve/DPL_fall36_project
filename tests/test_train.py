"""
    Tests for trainer/train.py (train_spec.md §7). CPU only, no downloads: a tiny random encoder,
    a fake whitespace tokenizer and a synthetic processed table are patched into the trainer.
    Run:  python tests/test_train.py   (or `pytest tests` if pytest is installed)
"""
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, BertConfig
from transformers import logging as hf_logging

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import trainer.train as T  # noqa: E402
from architecture import build_model as real_build_model  # noqa: E402
from test_dataset import FakeTokenizer  # noqa: E402
from trainer.runs import is_run_done, load_predictions  # noqa: E402
from utils.common import load_json, save_json, save_yaml  # noqa: E402
from utils.dataset import build_dataloaders  # noqa: E402

hf_logging.set_verbosity_error()
NUM_LABELS = {'sentiment': 3, 'topic': 4}


class MLMFakeTokenizer(FakeTokenizer):
    """ Adds what mask_tokens needs; <s>=0 <pad>=1 </s>=2 <mask>=3, words from 4 (vocab stays < 200). """
    mask_token_id = 3
    all_special_ids = [0, 1, 2, 3]

    def __init__(self):
        super().__init__()
        self.vocab = {'<mask>': 3}

    def __len__(self):
        return 200


def synthetic_df(n_train=48, n_val=12, n_test=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for split, n in (('train', n_train), ('val', n_val), ('test', n_test)):
        for i in range(n):
            s, t = int(rng.integers(3)), int(rng.integers(4))
            words = [f's{s}', f't{t}'] + [f'w{int(x)}' for x in rng.integers(0, 30, int(rng.integers(1, 8)))]
            rows.append({'id': f'uit-vsfc-{split}-{i}', 'text_clean': ' '.join(words), 'text_seg': ' '.join(words),
                         'sentiment': s, 'topic': t, 'split': split})
    return pd.DataFrame(rows)


DF = synthetic_df()


def fake_loaders(dataset, backbone_key, tasks, batch_size, seed, train_fraction=1.0, num_workers=0, **_):
    tok = MLMFakeTokenizer()
    train_ids = None
    if train_fraction < 1:
        ids = DF.loc[DF['split'] == 'train', 'id'].tolist()
        train_ids = ids[:max(1, int(len(ids) * train_fraction))]
    loaders = build_dataloaders(DF, tok, tasks, 'text_clean', 16, batch_size, seed, train_ids=train_ids, num_workers=0)
    info = {'num_labels': {t: NUM_LABELS[t] for t in tasks}, 'n_train': len(loaders['train'].dataset),
            'n_validation': len(loaders['validation'].dataset), 'n_test': len(loaders['test'].dataset),
            'max_len': 16, 'text_column': 'text_clean', 'train_fraction': float(train_fraction)}
    return loaders, tok, info


def fake_build_model(cfg, num_labels):
    torch.manual_seed(cfg['train']['seed'])
    enc = AutoModel.from_config(BertConfig(vocab_size=200, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                           intermediate_size=64, max_position_embeddings=32))
    return real_build_model(cfg, num_labels, encoder=enc)


@contextmanager
def patched(**replacements):
    old = {k: getattr(T, k) for k in replacements}
    try:
        for k, v in replacements.items():
            setattr(T, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(T, k, v)


@contextmanager
def sandbox(**patches):
    """ Temp models/reports dirs + fake data/model builders. """
    with tempfile.TemporaryDirectory() as tmp:
        env_old = {k: os.environ.get(k) for k in ('DPL_MODELS_DIR', 'DPL_OUT_DIR', 'DPL_DEVICE')}
        os.environ.update({'DPL_MODELS_DIR': str(Path(tmp) / 'models'), 'DPL_OUT_DIR': str(Path(tmp) / 'reports'),
                           'DPL_DEVICE': 'cpu'})
        try:
            with patched(**{'build_run_dataloaders': fake_loaders, 'build_model': fake_build_model, **patches}):
                yield Path(tmp)
        finally:
            for k, v in env_old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def experiment(tmp, grid, backbones=('phobert',), seeds=(1,), **train):
    path = Path(tmp) / 'exp.yaml'
    save_yaml({'extends': 'configs/default.yaml', 'experiment': 'e_test', 'datasets': ['uit-vsfc'],
               'backbones': list(backbones), 'seeds': list(seeds), 'grid': grid,
               'train': {'epochs': 2, 'batch_size': 8, 'num_workers': 0, 'fp16': False, 'log_every': 1,
                         'patience': 3, **train}}, path)
    return str(path)


def run_folder(tmp, run_id):
    return Path(tmp) / 'models' / run_id


# 1
def test_expand_runs_and_dry_run():
    with tempfile.TemporaryDirectory() as tmp:
        exp = T.load_experiment(experiment(tmp, [{'mode': 'st_sentiment', 'tag': 'sum'}, {'mode': 'mtl', 'tag': 'unc'}],
                                           backbones=('phobert', 'xlmr'), seeds=(1, 2, 3)))
        exp['datasets'] = ['neu-esc', 'uit-vsfc']
        specs = T.expand_runs(exp)
        assert len(specs) == 2 * 2 * 2 * 3 and len({s.run_id for s in specs}) == len(specs)
        s = next(x for x in specs if x.run_id == 'neu-esc__xlmr__mtl__seed2__unc')
        assert s.cfg['loss']['strategy'] == 'uncertainty' and s.cfg['model']['backbone'] == 'xlmr'
        assert s.cfg['data']['tasks'] == ['sentiment', 'topic'] and s.cfg['train']['seed'] == 2
    with sandbox() as tmp:
        def boom(*a, **k):
            raise AssertionError('dry run must not train')
        with patched(train_one_run=boom):
            summary = T.main(['--experiment', experiment(tmp, [{'mode': 'mtl', 'tag': 'sum'}]), '--dry-run'])
        assert summary == {'runs': 1, 'done': 0, 'to_train': 1} and not (Path(tmp) / 'models').exists()


# 2
def test_apply_tag_overrides():
    base = {'loss': {'strategy': 'sum', 'task_loss': 'ce'}, 'model': {'head': 'linear'}, 'data': {}}
    cfg = T.apply_tag_overrides(base, 'mtl', 'unc-focal')
    assert cfg['loss']['strategy'] == 'uncertainty' and cfg['loss']['task_loss'] == 'focal'
    assert T.apply_tag_overrides(base, 'mtlaware', 'sum')['model']['head'] == 'task_aware'
    assert T.apply_tag_overrides(base, 'st_topic', 'sum')['data']['tasks'] == ['topic']
    cfg = T.apply_tag_overrides(base, 'mtl', 'sum-frac0.1-smartref')
    assert cfg['data']['train_fraction'] == 0.1 and cfg['loss']['smart'] and cfg['loss']['smart_mode'] == 'token_ids'
    cfg = T.apply_tag_overrides(base, 'mtl', 'fixed0.7-mlm')
    assert cfg['loss']['strategy'] == 'fixed' and cfg['loss']['fixed_alpha'] == 0.7 and cfg['model']['mlm']
    assert T.apply_tag_overrides(base, 'mtlaware', 'sum-cross_sent')['model']['cross'] == 'sent_from_topic'
    for mode, tag in (('mtl', 'nope'), ('mtl', 'unc-pcgrad'), ('mtl', 'focal-wce'), ('mtl', 'frac1.5'),
                      ('mtl', 'sum-cross_topic'), ('multi', 'sum')):
        try:
            T.apply_tag_overrides(base, mode, tag)
            raise AssertionError(f'{mode}/{tag} not rejected')
        except ValueError:
            pass


# 3
def test_full_run_writes_all_files():
    with sandbox() as tmp:
        summary = T.main(['--experiment', experiment(tmp, [{'mode': 'mtl', 'tag': 'unc'}])])
        assert summary['done'] == 1 and summary['failed'] == 0
        folder = run_folder(tmp, 'uit-vsfc__phobert__mtl__seed1__unc')
        for f in ('config.yaml', 'env.json', 'run.log', 'train_log.csv', 'predictions_validation.csv',
                  'predictions_test.csv', 'metrics.json'):
            assert (folder / f).is_file(), f
        assert not (folder / 'best.pt').exists()                         # keep_checkpoint false
        m = load_json(folder / 'metrics.json')
        for key in ('run_id', 'dataset', 'backbone', 'mode', 'tag', 'seed', 'best_epoch', 'epochs_run', 'stopped_early',
                    'validation', 'test', 'train_time_s', 'time_per_epoch_s', 'peak_gpu_mb', 'params_total',
                    'params_trainable', 'final_task_weights', 'train_fraction', 'n_train'):
            assert key in m, key
        assert set(m['test']) == {'sentiment', 'topic', 'macro_f1_mean'} and 0 <= m['test']['macro_f1_mean'] <= 1
        log = pd.read_csv(folder / 'train_log.csv')
        assert len(log) == m['epochs_run'] == 2
        for col in ('train_loss_sentiment', 'val_loss_topic', 'val_acc_topic', 'val_macro_f1_sentiment', 'val_score',
                    'lr_encoder', 'w_sentiment', 'time_s', 'skipped_steps'):
            assert col in log.columns, col
        assert m['final_task_weights'] != {'sentiment': 1.0, 'topic': 1.0}   # uncertainty weights were learned


def test_mtl_sum_keeps_fp16_checkpoint():
    with sandbox() as tmp:
        T.main(['--experiment', experiment(tmp, [{'mode': 'mtl', 'tag': 'sum'}], epochs=1)])
        folder = run_folder(tmp, 'uit-vsfc__phobert__mtl__seed1__sum')
        state = torch.load(folder / 'best.pt', weights_only=True)                     # E7 needs it
        assert state['dtype'] == 'float16' and state['epoch'] == 1
        floats = [v for v in state['model'].values() if v.is_floating_point()]
        assert floats and all(v.dtype == torch.float16 for v in floats)
        cfg = T.load_experiment(experiment(tmp, [{'mode': 'mtl', 'tag': 'sum'}]))
        model = fake_build_model(T.expand_runs(cfg)[0].cfg, NUM_LABELS)
        model.load_state_dict(state['model'])                                          # loads into fp32
        assert next(model.parameters()).dtype == torch.float32


def test_checkpoint_rule_and_evaluate_only():
    assert T.needs_checkpoint('mtl', 'sum') and T.needs_checkpoint('st_sentiment', 'sum')     # E7 sources
    assert T.needs_checkpoint('mtlaware', 'bestloss-bestimb') and T.needs_checkpoint('mtlaware', 'final')
    for mode, tag in (('mtl', 'sum-frac0.1'), ('mtlaware', 'final-frac0.5'), ('mtlaware', 'final-mlm'),
                      ('mtlaware', 'sum'), ('st_topic', 'sum'), ('mtl', 'unc'), ('st_sentiment', 'sum-frac0.1'),
                      ('st_sentiment', 'focal')):
        assert not T.needs_checkpoint(mode, tag), (mode, tag)
    try:
        T.load_experiment('configs/experiment/e7_cross_dataset.yaml')
        raise AssertionError('evaluation-only experiment not refused')
    except ValueError as e:
        assert 'evaluation-only' in str(e)


def test_placeholder_expansion():
    """ Without selection.json (or with only some stages) the matrix keeps placeholders; they cannot be trained. """
    with sandbox() as tmp:
        e4 = T.load_experiment('configs/experiment/e4_task_aware.yaml')
        try:
            T.expand_runs(e4)
            raise AssertionError('missing selection.json not reported')
        except FileNotFoundError:
            pass
        specs = T.expand_runs(e4, placeholders=True)
        assert len(specs) == 12
        ids = {s.run_id for s in specs}
        assert 'neu-esc__best__mtlaware__seed42__bestloss-bestimb' in ids and 'uit-vsfc__best__mtlaware__seed42__sum' in ids
        assert all(s.cfg['run']['placeholder'] for s in specs)                # backbone still `best`
        try:
            T.train_one_run(specs[0])
            raise AssertionError('placeholder run was trained')
        except ValueError as e:
            assert 'placeholder' in str(e)
        # backbone stage written, loss stage not yet
        save_json({ds: {'backbone': 'xlmr'} for ds in ('neu-esc', 'uit-vsfc')},
                  Path(tmp) / 'reports' / 'tables' / 'selection.json')
        e3 = T.expand_runs(T.load_experiment('configs/experiment/e3_imbalance.yaml'), placeholders=True)
        by_id = {s.run_id: s for s in e3}
        partial = by_id['neu-esc__xlmr__mtl__seed42__bestloss-focal']
        assert partial.cfg['run']['placeholder'] and partial.cfg['loss']['task_loss'] == 'focal'
        ready = by_id['neu-esc__xlmr__st_sentiment__seed42__focal']
        assert not ready.cfg['run']['placeholder'] and ready.cfg['model']['backbone'] == 'xlmr'


def test_shipped_experiment_configs_expand():
    """ Every configs/experiment file expands to the run counts of models_spec.md B3. """
    expected = {'e1_baseline': 54, 'e1b_smart_ref': 18, 'e2_loss': 24, 'e3_imbalance': 36, 'e4_task_aware': 12,
                'e4b_direction': 12, 'e5_low_resource': 72, 'e6_mlm': 6}
    with sandbox() as tmp:
        save_json({ds: {'backbone': 'visobert', 'loss_tag': 'unc', 'imbalance_tag': 'focal', 'final_tag': 'unc-focal'}
                   for ds in ('neu-esc', 'uit-vsfc')}, Path(tmp) / 'reports' / 'tables' / 'selection.json')
        for name, n in expected.items():
            specs = T.expand_runs(T.load_experiment(f'configs/experiment/{name}.yaml'))
            assert len(specs) == n, (name, len(specs))
            assert all(s.cfg['run']['experiment'] == name for s in specs)


# 4
def test_resume_skips_finished_runs():
    with sandbox() as tmp:
        exp = experiment(tmp, [{'mode': 'st_topic', 'tag': 'sum'}], epochs=1)
        T.main(['--experiment', exp])
        folder = run_folder(tmp, 'uit-vsfc__phobert__st_topic__seed1__sum')
        before = {p.name: p.stat().st_mtime_ns for p in folder.iterdir()}
        summary = T.main(['--experiment', exp])
        assert summary['skipped'] == 1 and summary['done'] == 0
        assert {p.name: p.stat().st_mtime_ns for p in folder.iterdir()} == before


# 5
def test_same_seed_same_results():
    results = []
    for _ in range(2):
        with sandbox() as tmp:
            T.main(['--experiment', experiment(tmp, [{'mode': 'mtl', 'tag': 'sum'}], epochs=1)])
            folder = run_folder(tmp, 'uit-vsfc__phobert__mtl__seed1__sum')
            results.append((pd.read_csv(folder / 'train_log.csv')['train_loss_topic'].tolist(),
                            load_predictions(folder / 'predictions_test.csv')))
    assert results[0][0] == results[1][0]
    assert results[0][1].equals(results[1][1])


# 6
def test_early_stopping_with_constant_score():
    real = T.evaluate_split

    def constant(model, loader, tasks, device=None, amp=False):
        metrics, preds = real(model, loader, tasks, device, amp)
        return {**metrics, 'macro_f1_mean': 0.5}, preds

    with sandbox(evaluate_split=constant) as tmp:
        T.main(['--experiment', experiment(tmp, [{'mode': 'mtl', 'tag': 'sum'}], epochs=10, patience=2)])
        m = load_json(run_folder(tmp, 'uit-vsfc__phobert__mtl__seed1__sum') / 'metrics.json')
    assert m['epochs_run'] == 3 and m['best_epoch'] == 1 and m['stopped_early'] is True


# 7
def test_every_strategy_and_option_completes_an_epoch():
    grid = [{'mode': 'mtl', 'tag': tag} for tag in ('sum', 'unc', 'pcgrad', 'gradnorm', 'dwa', 'fixed0.3', 'smartemb',
                                                    'smartref', 'focal', 'wce', 'mlm', 'sum-frac0.5')]
    grid += [{'mode': 'mtlaware', 'tag': 'sum'}, {'mode': 'mtlaware', 'tag': 'unc-focal-cross_topic'},
             {'mode': 'st_sentiment', 'tag': 'sum'}]
    with sandbox() as tmp:
        summary = T.main(['--experiment', experiment(tmp, grid, epochs=1, grad_accum_steps=2)])
        assert summary['failed'] == 0, summary['failed_runs']
        for entry in grid:
            rid = f'uit-vsfc__phobert__{entry["mode"]}__seed1__{entry["tag"]}'
            m = load_json(run_folder(tmp, rid) / 'metrics.json')
            log = pd.read_csv(run_folder(tmp, rid) / 'train_log.csv')
            assert np.isfinite(log.filter(like='train_loss_').to_numpy()).all(), rid
            assert m['skipped_steps'] == 0, rid
        half = load_json(run_folder(tmp, 'uit-vsfc__phobert__mtl__seed1__sum-frac0.5') / 'metrics.json')
        assert half['n_train'] == 24 and half['train_fraction'] == 0.5


# 8
def test_prediction_ids_match_processed_ids():
    with sandbox() as tmp:
        T.main(['--experiment', experiment(tmp, [{'mode': 'mtl', 'tag': 'sum'}], epochs=1)])
        folder = run_folder(tmp, 'uit-vsfc__phobert__mtl__seed1__sum')
        for key, split in (('validation', 'val'), ('test', 'test')):
            preds = load_predictions(folder / f'predictions_{key}.csv')
            assert preds['id'].tolist() == DF.loc[DF['split'] == split, 'id'].tolist()


# 9
def test_failed_run_writes_error_and_continues():
    def flaky(cfg, num_labels):
        if cfg['data']['tasks'] == ['topic']:
            raise RuntimeError('simulated failure')
        return fake_build_model(cfg, num_labels)

    with sandbox(build_model=flaky) as tmp:
        summary = T.main(['--experiment', experiment(tmp, [{'mode': 'st_topic', 'tag': 'sum'},
                                                           {'mode': 'st_sentiment', 'tag': 'sum'}], epochs=1)])
        assert summary['failed'] == 1 and summary['done'] == 1
        err = run_folder(tmp, 'uit-vsfc__phobert__st_topic__seed1__sum') / 'error.txt'
        assert 'simulated failure' in err.read_text(encoding='utf-8')
        assert not is_run_done('uit-vsfc__phobert__st_topic__seed1__sum')
        assert is_run_done('uit-vsfc__phobert__st_sentiment__seed1__sum')


def test_best_selection_and_max_steps_debug_root():
    with sandbox() as tmp:
        exp_path = experiment(tmp, [{'mode': 'mtlaware', 'tag': 'final'}, {'mode': 'mtl', 'tag': 'bestloss-bestimb'}],
                              backbones=('best',), epochs=1)
        try:
            T.expand_runs(T.load_experiment(exp_path))
            raise AssertionError('missing selection.json not reported')
        except FileNotFoundError as e:
            assert 'selection.json' in str(e)
        save_json({'uit-vsfc': {'backbone': 'visobert', 'loss_tag': 'unc', 'imbalance_tag': 'focal',
                                'final_tag': 'unc-focal'}}, Path(tmp) / 'reports' / 'tables' / 'selection.json')
        ids = [s.run_id for s in T.expand_runs(T.load_experiment(exp_path))]
        assert ids == ['uit-vsfc__visobert__mtlaware__seed1__unc-focal', 'uit-vsfc__visobert__mtl__seed1__unc-focal']
        final = T.expand_runs(T.load_experiment(exp_path))[0]
        assert final.cfg['train']['keep_checkpoint'] is True                 # final model is kept for E7

        T.main(['--experiment', exp_path, '--only', 'mtlaware', '--max-steps', '2'])
        debug = Path(tmp) / 'models' / '_debug' / 'uit-vsfc__visobert__mtlaware__seed1__unc-focal'
        assert (debug / 'metrics.json').is_file() and load_json(debug / 'metrics.json')['max_steps'] == 2
        assert not is_run_done('uit-vsfc__visobert__mtlaware__seed1__unc-focal')   # real run still to do


# 15
def test_time_budget_stops_before_the_next_run():
    with sandbox() as tmp:
        exp = experiment(tmp, [{'mode': 'st_topic', 'tag': 'sum'}, {'mode': 'mtl', 'tag': 'sum'}], epochs=1)
        # the first run starts (nothing measured yet), the second would end after the budget
        summary = T.main(['--experiment', exp, '--time-budget-hours', str(0.05 / 3600)])
        assert summary['done'] == 1 and summary['not_started'] == ['uit-vsfc__phobert__mtl__seed1__sum']
        summary = T.main(['--experiment', exp, '--time-budget-hours', '10'])       # resume: the rest
        assert summary['skipped'] == 1 and summary['done'] == 1 and summary['not_started'] == []


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
