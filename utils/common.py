"""
    Shared helpers for training and evaluation: paths (local / Kaggle), seeds, file I/O,
    environment info and logging.

    Paths resolve the same way everywhere, so the same command works locally and on Kaggle:
      processed data : $DPL_PROCESSED_DIR -> Kaggle: /kaggle/working/data/processed, else the first
                       folder under /kaggle/input with the parquet files -> <project>/data/processed
      run outputs    : $DPL_MODELS_DIR    -> /kaggle/working/models  (Kaggle) -> <project>/models
      reports        : $DPL_OUT_DIR       -> /kaggle/working/reports (Kaggle) -> <project>/reports
    ($DPL_PROCESSED_DIR and $DPL_OUT_DIR are the same variables utils/preprocess.py and the EDA use.)
"""
import datetime
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / 'configs'
ON_KAGGLE = os.path.isdir('/kaggle/input')
KAGGLE_INPUT = Path('/kaggle/input')
KAGGLE_WORKING = Path('/kaggle/working')


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _has_processed(folder):
    folder = Path(folder)
    return (folder / 'neu-esc.parquet').is_file() or (folder / 'uit-vsfc.parquet').is_file()


def _search(root, test, max_depth=4):
    """ Breadth-first search under root (up to max_depth levels) for a folder passing `test`. """
    level = [Path(root)] if Path(root).is_dir() else []
    for _ in range(max_depth + 1):
        nxt = []
        for folder in level:
            if test(folder):
                return folder
            try:
                nxt += sorted(c for c in folder.iterdir() if c.is_dir())
            except OSError:
                pass
        level = nxt
    return None


def processed_dir():
    """ Folder with <dataset>.parquet and <dataset>_subsets.json (read-only input for training). """
    env = os.environ.get('DPL_PROCESSED_DIR')
    if env:
        return Path(env)
    if ON_KAGGLE:
        working = KAGGLE_WORKING / 'data' / 'processed'
        if _has_processed(working):          # regenerated on Kaggle (e.g. with VnCoreNLP)
            return working
        found = _search(KAGGLE_INPUT, _has_processed)
        if found:
            return found
        raise FileNotFoundError('No processed data on Kaggle: add data/processed as a Kaggle Dataset, '
                                'run `python -m utils.preprocess`, or set DPL_PROCESSED_DIR.')
    return PROJECT_ROOT / 'data' / 'processed'


def models_dir():
    """ Root of the run folders (models/<run_id>/). """
    env = os.environ.get('DPL_MODELS_DIR')
    if env:
        return Path(env)
    return KAGGLE_WORKING / 'models' if ON_KAGGLE else PROJECT_ROOT / 'models'


def reports_dir():
    env = os.environ.get('DPL_OUT_DIR')
    if env:
        return Path(env)
    return KAGGLE_WORKING / 'reports' if ON_KAGGLE else PROJECT_ROOT / 'reports'


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def set_seed(seed, deterministic=False):
    """ Seed python, numpy and torch (CPU and all GPUs). deterministic=True also fixes cuDNN
        algorithms (slower; bit-exact GPU runs are still not guaranteed). """
    import torch
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """ DataLoader worker_init_fn: derive numpy / python seeds from the loader's torch seed. """
    import torch
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def make_generator(seed):
    """ torch.Generator for DataLoader(shuffle=True, generator=...), so the order depends only on the seed. """
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def to_jsonable(o):
    """ numpy / torch scalars and arrays -> python, NaN / inf -> None, Path -> str. """
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    if hasattr(o, 'detach') and hasattr(o, 'tolist'):      # torch.Tensor
        return to_jsonable(o.detach().cpu().tolist())
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, Path):
        return str(o)
    if o is None or isinstance(o, (str, int)):
        return o
    return str(o)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2, allow_nan=False)
    Path(path).write_text(text + '\n', encoding='utf-8')


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_yaml(obj, path):
    import yaml
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(to_jsonable(obj), f, allow_unicode=True, sort_keys=False)


def load_yaml(path):
    import yaml
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Environment info and logging
# ---------------------------------------------------------------------------

def _git(*args):
    try:
        out = subprocess.run(['git', *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def env_info():
    """ Everything needed to tell later where and with what a run was trained (saved as env.json). """
    info = {
        'time': datetime.datetime.now().isoformat(timespec='seconds'),
        'on_kaggle': ON_KAGGLE,
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'git_commit': _git('rev-parse', 'HEAD'),
        'git_dirty': bool(_git('status', '--porcelain')) if _git('rev-parse', 'HEAD') else None,
    }
    for name in ('numpy', 'pandas', 'torch', 'transformers', 'sklearn'):
        try:
            info[name] = __import__(name).__version__
        except Exception:
            info[name] = None
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        info['cuda_build'] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info['gpu'] = props.name
            info['gpu_memory_mb'] = round(props.total_memory / 2 ** 20)
            info['gpu_count'] = torch.cuda.device_count()
    except Exception:
        info['cuda_available'] = None
    return info


def get_logger(name, log_file=None, level=logging.INFO):
    """ Logger writing to the console and, if given, to log_file. Safe to call again with the same name. """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False   # other libraries configure the root logger; avoid double lines
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s', '%H:%M:%S')
    if not any(type(h) is logging.StreamHandler for h in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)
    if log_file:
        log_file = str(Path(log_file).resolve())
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file for h in logger.handlers):
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def close_logger(logger):
    """ Close and remove file handlers (so a run folder can be deleted or zipped on Windows). """
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)
