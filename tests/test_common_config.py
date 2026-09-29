"""
    Tests for utils/common.py and utils/config.py. CPU only, no downloads.
    Run:  python tests/test_common_config.py   (or `pytest tests` if pytest is installed)
"""
import json
import logging
import os
import random
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import common  # noqa: E402
from utils.common import (PROJECT_ROOT, close_logger, env_info, get_logger, load_json, load_yaml,  # noqa: E402
                          make_generator, save_json, save_yaml, set_seed)
from utils.config import (apply_overrides, deep_merge, get_dotted, load_config, load_data_config,  # noqa: E402
                          load_model_config, max_len_for)

BACKBONES = ('phobert', 'xlmr', 'visobert')
DATASETS = ('neu-esc', 'uit-vsfc')


@contextmanager
def env(**values):
    """ Temporarily set (value) or unset (None) environment variables. """
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --- config -----------------------------------------------------------------

def test_deep_merge():
    base = {'a': 1, 'train': {'lr': 1, 'epochs': 10}, 'seeds': [1, 2]}
    out = deep_merge(base, {'train': {'lr': 2}, 'seeds': [3]})
    assert out == {'a': 1, 'train': {'lr': 2, 'epochs': 10}, 'seeds': [3]}
    assert base == {'a': 1, 'train': {'lr': 1, 'epochs': 10}, 'seeds': [1, 2]}   # input untouched


def test_load_config_extends_chain_and_cycle():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        save_yaml({'train': {'lr': 1, 'epochs': 10}, 'name': 'base'}, tmp / 'base.yaml')
        save_yaml({'extends': 'base.yaml', 'train': {'lr': 2}}, tmp / 'mid.yaml')
        save_yaml({'extends': str(tmp / 'mid.yaml'), 'name': 'child'}, tmp / 'child.yaml')
        cfg = load_config(tmp / 'child.yaml')
        assert cfg == {'train': {'lr': 2, 'epochs': 10}, 'name': 'child'}
        save_yaml({'extends': 'b.yaml'}, tmp / 'a.yaml')
        save_yaml({'extends': 'a.yaml'}, tmp / 'b.yaml')
        try:
            load_config(tmp / 'a.yaml')
            raise AssertionError('circular extends not detected')
        except ValueError as e:
            assert 'circular' in str(e)


def test_experiment_extends_project_default():
    with tempfile.TemporaryDirectory() as tmp:
        exp = Path(tmp) / 'e_test.yaml'
        save_yaml({'extends': 'configs/default.yaml', 'experiment': 'e_test', 'train': {'epochs': 2}}, exp)
        cfg = load_config(exp)
    assert cfg['train']['epochs'] == 2 and cfg['train']['lr_encoder'] == 2e-5     # merged with default
    assert cfg['preprocess']['seed'] == 42 and cfg['loss']['strategy'] == 'sum' and 'extends' not in cfg


def test_default_config_sections():
    cfg = load_config(PROJECT_ROOT / 'configs' / 'default.yaml')
    assert {'preprocess', 'data', 'model', 'loss', 'train'} <= set(cfg)
    assert cfg['train']['fp16'] is True and cfg['model']['head'] == 'linear'
    assert cfg['data']['datasets'] == list(DATASETS)


def test_dotted_overrides():
    cfg = {'loss': {'strategy': 'sum', 'task_loss': 'ce'}}
    out = apply_overrides(cfg, {'loss.strategy': 'uncertainty', 'model.head': 'task_aware'})
    assert get_dotted(out, 'loss.strategy') == 'uncertainty' and get_dotted(out, 'loss.task_loss') == 'ce'
    assert get_dotted(out, 'model.head') == 'task_aware' and cfg['loss']['strategy'] == 'sum'
    assert get_dotted(out, 'train.lr', None) is None
    try:
        get_dotted(out, 'train.lr')
        raise AssertionError('missing key not raised')
    except KeyError:
        pass


def test_model_and_data_configs():
    for key in BACKBONES:
        m = load_model_config(key)
        assert m['key'] == key and m['text_column'] in ('text_clean', 'text_seg')
        assert m['word_segmentation'] == (m['text_column'] == 'text_seg')
        assert m['tokenizer'] in ('auto', 'sentencepiece_fairseq') and m['max_positions'] >= 256
    for ds in DATASETS:
        d = load_data_config(ds)
        assert d['name'] == ds and (PROJECT_ROOT / 'data' / 'processed' / d['processed_file']).exists()
        for key in BACKBONES:
            n = max_len_for(ds, key)
            assert n % 16 == 0 and 16 <= n <= load_model_config(key)['max_positions']
    try:
        load_model_config('no-such-model')
        raise AssertionError('unknown backbone not raised')
    except FileNotFoundError as e:
        assert 'phobert' in str(e)


# --- common -----------------------------------------------------------------

def test_set_seed_reproducible():
    def draw():
        return random.random(), float(np.random.rand()), torch.rand(3).tolist()
    set_seed(123)
    a = draw()
    set_seed(123)
    assert draw() == a
    set_seed(7)
    assert draw() != a
    g1, g2 = make_generator(5), make_generator(5)
    assert torch.equal(torch.randperm(10, generator=g1), torch.randperm(10, generator=g2))


def test_paths_env_and_local_defaults():
    with env(DPL_PROCESSED_DIR=None, DPL_MODELS_DIR=None, DPL_OUT_DIR=None):
        if not common.ON_KAGGLE:
            assert common.processed_dir() == PROJECT_ROOT / 'data' / 'processed'
            assert common.models_dir() == PROJECT_ROOT / 'models'
            assert common.reports_dir() == PROJECT_ROOT / 'reports'
    with tempfile.TemporaryDirectory() as tmp:
        with env(DPL_PROCESSED_DIR=tmp, DPL_MODELS_DIR=tmp, DPL_OUT_DIR=tmp):
            assert common.processed_dir() == common.models_dir() == common.reports_dir() == Path(tmp)


def test_kaggle_input_search():
    with tempfile.TemporaryDirectory() as tmp:
        deep = Path(tmp) / 'dpl-processed' / 'data' / 'processed'
        deep.mkdir(parents=True)
        (deep / 'neu-esc.parquet').write_bytes(b'')
        assert common._search(tmp, common._has_processed) == deep
        assert common._search(tmp, common._has_processed, max_depth=1) is None


def test_json_yaml_io():
    obj = {'a': np.int64(3), 'b': np.float32(0.5), 'c': float('nan'), 'd': np.array([1, 2]),
           'e': Path('x/y'), 'f': torch.tensor([1.5]), 'g': np.bool_(True), 'h': 'tiếng việt'}
    with tempfile.TemporaryDirectory() as tmp:
        save_json(obj, Path(tmp) / 'sub' / 'o.json')
        back = load_json(Path(tmp) / 'sub' / 'o.json')
        assert back == {'a': 3, 'b': 0.5, 'c': None, 'd': [1, 2], 'e': str(Path('x/y')), 'f': [1.5],
                        'g': True, 'h': 'tiếng việt'}
        save_yaml({'train': {'lr': 2e-5}, 'name': 'tiếng việt'}, Path(tmp) / 'c.yaml')
        assert load_yaml(Path(tmp) / 'c.yaml') == {'train': {'lr': 2e-5}, 'name': 'tiếng việt'}


def test_env_info():
    info = env_info()
    assert {'time', 'python', 'torch', 'transformers', 'cuda_available', 'on_kaggle', 'git_commit'} <= set(info)
    json.dumps(info)


def test_get_logger_file_and_no_duplicates():
    with tempfile.TemporaryDirectory() as tmp:
        log_file = Path(tmp) / 'run.log'
        logger = get_logger('test-run', log_file)
        logger = get_logger('test-run', log_file)          # second call adds nothing
        assert len(logger.handlers) == 2 and logger.propagate is False
        logger.info('hello')
        close_logger(logger)
        assert 'hello' in log_file.read_text(encoding='utf-8')
        assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)


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
