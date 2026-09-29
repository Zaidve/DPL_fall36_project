"""
    Tests for trainer/evaluate.py and trainer/runs.py. CPU only, no downloads.
    Run:  python tests/test_evaluate_runs.py   (or `pytest tests` if pytest is installed)
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModel, BertConfig
from transformers import logging as hf_logging

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from architecture import MTLModel  # noqa: E402
from test_dataset import FakeTokenizer, synthetic  # noqa: E402
from trainer.evaluate import compute_metrics, evaluate_split, mcnemar  # noqa: E402
from trainer.runs import (is_run_done, load_checkpoint, load_predictions, make_run_id, parse_run_id,  # noqa: E402
                          run_dir, save_checkpoint, save_predictions, write_error, write_metrics)
from utils.dataset import build_dataloaders  # noqa: E402

hf_logging.set_verbosity_error()
TASKS = {'sentiment': 3, 'topic': 4}


def tiny_model(head='linear', seed=0, **kw):
    torch.manual_seed(seed)
    enc = AutoModel.from_config(BertConfig(vocab_size=200, hidden_size=32, num_hidden_layers=1,
                                           num_attention_heads=2, intermediate_size=64))
    return MTLModel.from_encoder(enc, TASKS, head=head, **kw)


def loaders(df):
    return build_dataloaders(df, FakeTokenizer(), list(TASKS), 'text_clean', 20, 4, 42, num_workers=0)


# --- compute_metrics ----------------------------------------------------------

def test_compute_metrics_values():
    m = compute_metrics([0, 1, 2, 2], [0, 1, 2, 2], 3)
    assert m['accuracy'] == m['macro_f1'] == m['weighted_f1'] == 1.0 and m['per_class_f1'] == [1.0, 1.0, 1.0]
    m = compute_metrics([0, 0, 1], [0, 0, 0], 4)                   # classes 2 and 3 never appear
    assert len(m['per_class_f1']) == 4 and m['per_class_f1'][2:] == [0.0, 0.0]
    assert abs(m['macro_f1'] - np.mean(m['per_class_f1'])) < 1e-12


def test_argument_order_matters_for_weighted_f1():
    """ Legacy mistake #5: sklearn(pred, true). With an asymmetric confusion matrix weighted-F1 differs. """
    y_true = [0] * 8 + [1] * 2
    y_pred = [0] * 5 + [1] * 3 + [1] * 2
    ours = compute_metrics(y_true, y_pred, 2)
    assert abs(ours['weighted_f1'] - f1_score(y_true, y_pred, average='weighted')) < 1e-12
    swapped = f1_score(y_pred, y_true, average='weighted')
    assert abs(ours['weighted_f1'] - swapped) > 0.01
    assert abs(ours['macro_f1'] - f1_score(y_pred, y_true, average='macro')) < 1e-12   # macro does not care


# --- evaluate_split -----------------------------------------------------------

def test_evaluate_split_predictions_and_metrics():
    df = synthetic()
    val = loaders(df)['validation']
    for head in ('linear', 'task_aware'):
        model = tiny_model(head).train()
        metrics, preds = evaluate_split(model, val, list(TASKS))
        assert model.training                                                   # mode restored
        assert preds['id'].tolist() == df.loc[df['split'] == 'val', 'id'].tolist()   # ids = processed ids
        cols = ['id'] + [f'{t}_{s}' for t in TASKS for s in ('true', 'pred')] \
            + [f'{t}_prob_{k}' for t, c in TASKS.items() for k in range(c)]
        assert sorted(preds.columns) == sorted(cols)
        for t, c in TASKS.items():
            probs = preds[[f'{t}_prob_{k}' for k in range(c)]].to_numpy()
            assert np.allclose(probs.sum(1), 1, atol=1e-5)
            assert (probs.argmax(1) == preds[f'{t}_pred']).all()
            assert preds[f'{t}_true'].tolist() == df.loc[df['split'] == 'val', t].tolist()
            ref = compute_metrics(preds[f'{t}_true'], preds[f'{t}_pred'], c)
            assert {k: metrics[t][k] for k in ref} == ref and np.isfinite(metrics[t]['loss'])
        assert abs(metrics['macro_f1_mean'] - np.mean([metrics[t]['macro_f1'] for t in TASKS])) < 1e-12


def test_evaluate_split_is_deterministic_in_eval_mode():
    df = synthetic()
    val = loaders(df)['validation']
    model = tiny_model(dropout=0.5)
    a, pa = evaluate_split(model.train(), val, list(TASKS))
    b, pb = evaluate_split(model, val, list(TASKS))
    assert a == b and pa.equals(pb)                                        # dropout off during evaluation


def test_mcnemar():
    y = np.array([0, 1] * 50)
    same = mcnemar(y, y, y)
    assert same == {'only_a_correct': 0, 'only_b_correct': 0, 'p_value': 1.0}
    worse = y.copy()
    worse[:40] = 1 - worse[:40]
    r = mcnemar(y, y, worse)
    assert r['only_a_correct'] == 40 and r['only_b_correct'] == 0 and r['p_value'] < 1e-6


# --- runs -----------------------------------------------------------------------

def test_run_id_roundtrip_and_validation():
    rid = make_run_id('neu-esc', 'phobert', 'mtl', 42, 'unc-focal')
    assert rid == 'neu-esc__phobert__mtl__seed42__unc-focal'
    assert parse_run_id(rid) == {'dataset': 'neu-esc', 'backbone': 'phobert', 'mode': 'mtl', 'seed': 42, 'tag': 'unc-focal'}
    assert make_run_id('uit-vsfc', 'xlmr', 'st_topic', 7, 'sum-frac0.1').endswith('__sum-frac0.1')
    for bad in (('neu__esc', 'phobert', 'mtl', 1, 'sum'), ('neu-esc', 'pho/bert', 'mtl', 1, 'sum'),
                ('neu-esc', 'phobert', 'mtl', 1, 'a b'), ('neu-esc', 'phobert', 'mtl_', 1, 'sum'),
                ('neu-esc', 'phobert', 'mtl', 1, '_sum')):
        try:
            make_run_id(*bad)
            raise AssertionError(f'{bad} not rejected')
        except ValueError:
            pass
    try:
        parse_run_id('not-a-run')
        raise AssertionError('bad run id parsed')
    except ValueError:
        pass


def test_run_dir_done_and_atomic_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        old = os.environ.get('DPL_MODELS_DIR')
        os.environ['DPL_MODELS_DIR'] = tmp
        try:
            rid = make_run_id('uit-vsfc', 'phobert', 'mtl', 42, 'sum')
            path = run_dir(rid)
            assert path == Path(tmp) / rid and path.is_dir()
            assert not is_run_done(rid)
            (path / 'metrics.json').write_text('{"run_id": "', encoding='utf-8')     # half-written file
            assert not is_run_done(rid)
            write_metrics(path, {'run_id': 'other', 'test': {}})
            assert not is_run_done(rid)                                             # wrong run
            write_metrics(path, {'run_id': rid, 'validation': {}})
            assert not is_run_done(rid)                                             # incomplete
            write_metrics(path, {'run_id': rid, 'validation': {}, 'test': {'macro_f1_mean': 0.5}})
            assert is_run_done(rid) and not (path / 'metrics.json.tmp').exists()
        finally:
            if old is None:
                os.environ.pop('DPL_MODELS_DIR')
            else:
                os.environ['DPL_MODELS_DIR'] = old


def test_write_error():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            raise RuntimeError('boom in run')
        except RuntimeError as e:
            write_error(Path(tmp) / 'run', e)
        text = (Path(tmp) / 'run' / 'error.txt').read_text(encoding='utf-8')
        assert 'RuntimeError: boom in run' in text and 'Traceback' in text


def test_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        a = tiny_model('task_aware', seed=1, mlm=True)
        save_checkpoint(a, Path(tmp) / 'best.pt', epoch=3, score=0.71)
        b = tiny_model('task_aware', seed=2, mlm=True)
        extra = load_checkpoint(b, Path(tmp) / 'best.pt')
        assert extra == {'epoch': 3, 'score': 0.71}
        for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
            assert ka == kb and torch.equal(va, vb)
        assert b.mlm_head.decoder.weight is b.get_input_embeddings().weight     # tie survives reloading


def test_predictions_roundtrip():
    df = synthetic()
    _, preds = evaluate_split(tiny_model(), loaders(df)['test'], list(TASKS))
    with tempfile.TemporaryDirectory() as tmp:
        save_predictions(preds, Path(tmp) / 'p.csv')
        back = load_predictions(Path(tmp) / 'p.csv')
    assert back['id'].tolist() == preds['id'].tolist()
    assert (back['topic_pred'] == preds['topic_pred']).all()
    assert np.allclose(back['topic_prob_0'], preds['topic_prob_0'], atol=1e-6)


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
