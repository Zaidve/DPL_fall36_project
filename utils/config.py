"""
    Config loading for training.

      load_config(path)            YAML with optional `extends` (a path or a list of paths), merged
                                   base-first; nested dicts merge, everything else is replaced
      apply_overrides(cfg, {...})  dotted keys, e.g. {'loss.strategy': 'uncertainty'} (used by tags)
      load_model_config(key)       configs/model/<key>.yaml   (pretrained id, text column, tokenizer)
      load_data_config(dataset)    configs/data/<dataset>.yaml (files, max_len per backbone)
      max_len_for(dataset, key)    max_len for one (dataset, backbone)

    `extends` paths are resolved against the project root first (e.g. configs/default.yaml),
    then against the folder of the file that contains them.
"""
import copy
from pathlib import Path

from utils.common import CONFIGS_DIR, PROJECT_ROOT, load_yaml

_MISSING = object()


def deep_merge(base, override):
    """ New dict: override on top of base. Nested dicts merge; lists and scalars are replaced. """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve(ref, relative_to):
    ref = Path(ref)
    if ref.is_absolute():
        return ref
    for candidate in (PROJECT_ROOT / ref, Path(relative_to) / ref):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'extends target not found: {ref} (looked in {PROJECT_ROOT} and {relative_to})')


def load_config(path, _chain=()):
    path = Path(path).resolve()
    if path in _chain:
        raise ValueError('circular extends: ' + ' -> '.join(str(p) for p in (*_chain, path)))
    cfg = load_yaml(path)
    parents = cfg.pop('extends', None) or []
    if isinstance(parents, (str, Path)):
        parents = [parents]
    merged = {}
    for parent in parents:
        merged = deep_merge(merged, load_config(_resolve(parent, path.parent), (*_chain, path)))
    return deep_merge(merged, cfg)


def get_dotted(cfg, key, default=_MISSING):
    node = cfg
    for part in key.split('.'):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(key)
            return default
        node = node[part]
    return node


def apply_overrides(cfg, overrides):
    """ New config with dotted-key overrides applied: {'loss.strategy': 'uncertainty', 'model.head': 'task_aware'}. """
    out = copy.deepcopy(cfg)
    for key, value in overrides.items():
        node = out
        *parents, last = key.split('.')
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise TypeError(f'cannot set {key}: {part} is not a section')
        node[last] = copy.deepcopy(value)
    return out


def load_model_config(key):
    path = CONFIGS_DIR / 'model' / f'{key}.yaml'
    if not path.exists():
        known = sorted(p.stem for p in (CONFIGS_DIR / 'model').glob('*.yaml'))
        raise FileNotFoundError(f'unknown backbone {key!r}; available: {known}')
    return load_yaml(path)


def load_data_config(dataset):
    path = CONFIGS_DIR / 'data' / f'{dataset}.yaml'
    if not path.exists():
        known = sorted(p.stem for p in (CONFIGS_DIR / 'data').glob('*.yaml'))
        raise FileNotFoundError(f'unknown dataset {dataset!r}; available: {known}')
    return load_yaml(path)


def max_len_for(dataset, backbone_key):
    table = load_data_config(dataset).get('max_len', {})
    if backbone_key not in table:
        raise KeyError(f'no max_len for backbone {backbone_key!r} in configs/data/{dataset}.yaml')
    return int(table[backbone_key])
