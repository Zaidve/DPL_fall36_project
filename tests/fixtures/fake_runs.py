"""
    Fake finished runs for the analysis tests (experiment_matrix_spec.md §9): each run folder has
    config.yaml, env.json, metrics.json, train_log.csv, predictions_validation.csv, predictions_test.csv
    with known values, in the same format trainer/train.py writes.
"""
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from trainer.runs import make_run_id, write_metrics
from utils.common import save_json, save_yaml

NUM_LABELS = {'neu-esc': {'sentiment': 4, 'topic': 10}, 'uit-vsfc': {'sentiment': 3, 'topic': 4}}
STRATEGY = {'sum': 'sum', 'unc': 'uncertainty', 'pcgrad': 'pcgrad', 'gradnorm': 'gradnorm'}


def tasks_for(mode):
    return {'st_sentiment': ['sentiment'], 'st_topic': ['topic']}.get(mode, ['sentiment', 'topic'])


def make_run(root, dataset, backbone, mode, tag, seed, val=0.70, test=0.68, time_s=600.0, n_eval=40, keep=False,
             best_epoch=3, skipped=0, torch_version='2.11.0', config_extra=None, experiment='e1_baseline'):
    """ One finished run. val / test = macro-F1 of every task on that split (topic gets -0.02). """
    rid = make_run_id(dataset, backbone, mode, seed, tag)
    folder = Path(root) / rid
    folder.mkdir(parents=True, exist_ok=True)
    tasks = tasks_for(mode)
    strategy = next((STRATEGY[t] for t in tag.split('-') if t in STRATEGY), 'sum')
    frac = next((float(t[4:]) for t in tag.split('-') if t.startswith('frac')), 1.0)
    cfg = {'data': {'dataset': dataset, 'tasks': tasks, 'train_fraction': frac},
           'model': {'backbone': backbone, 'head': 'task_aware' if mode == 'mtlaware' else 'linear'},
           'loss': {'strategy': strategy, 'task_loss': 'focal' if 'focal' in tag else 'ce'},
           'train': {'seed': seed, 'epochs': 10, 'keep_checkpoint': keep, **(config_extra or {})},
           'run': {'run_id': rid, 'experiment': experiment, 'seed': seed}}
    save_yaml(cfg, folder / 'config.yaml')
    save_json({'torch': torch_version, 'transformers': '5.17.0', 'numpy': '2.4.6'}, folder / 'env.json')

    rng = np.random.default_rng(zlib.crc32(rid.encode()))          # stable across processes
    blocks, logs = {}, []
    for split, score in (('validation', val), ('test', test)):
        block = {}
        preds = {'id': [f'{dataset}-{"val" if split == "validation" else "test"}-{i}' for i in range(n_eval)]}
        for t in tasks:
            c = NUM_LABELS[dataset][t]
            s = score - (0.02 if t == 'topic' else 0.0)
            block[t] = {'accuracy': s + 0.10, 'macro_f1': s, 'weighted_f1': s + 0.05, 'per_class_f1': [s] * c,
                        'loss': 0.5}
            y = np.arange(n_eval) % c
            correct = rng.random(n_eval) < s
            pred = np.where(correct, y, (y + 1) % c)
            preds[f'{t}_true'], preds[f'{t}_pred'] = y, pred
            for k in range(c):
                preds[f'{t}_prob_{k}'] = (pred == k).astype(float)
        block['macro_f1_mean'] = float(np.mean([block[t]['macro_f1'] for t in tasks]))
        blocks[split] = block
        pd.DataFrame(preds).to_csv(folder / f'predictions_{split}.csv', index=False)
    for epoch in range(1, best_epoch + 4):
        row = {'epoch': epoch}
        for t in tasks:
            row[f'train_loss_{t}'] = 1.0 / epoch
            row[f'val_loss_{t}'] = 0.6 + 0.02 * epoch
            row[f'val_macro_f1_{t}'] = val
            row[f'w_{t}'] = 1.0 + 0.01 * epoch
        row.update({'val_score': val, 'lr_encoder': 2e-5, 'time_s': time_s / (best_epoch + 3), 'skipped_steps': 0})
        logs.append(row)
    pd.DataFrame(logs).to_csv(folder / 'train_log.csv', index=False)
    if keep:
        (folder / 'best.pt').write_bytes(b'checkpoint')
    write_metrics(folder, {'run_id': rid, 'experiment': experiment, 'dataset': dataset, 'backbone': backbone,
                           'mode': mode, 'tag': tag, 'seed': seed, 'best_epoch': best_epoch,
                           'epochs_run': best_epoch + 3, 'stopped_early': True, **blocks, 'train_time_s': time_s,
                           'time_per_epoch_s': time_s / (best_epoch + 3), 'peak_gpu_mb': 3000.0,
                           'params_total': 100_000_000, 'params_trainable': 100_000_000,
                           'final_task_weights': {t: 1.0 for t in tasks}, 'train_fraction': frac,
                           'n_train': 1000, 'skipped_steps': skipped})
    return folder


def make_full_matrix(root, selection_path=None, seeds=(42, 123, 2026)):
    """
      Every run of the shipped experiment configs (234) with made-up but ordered scores, plus selection.json:
      both datasets pick visobert / loss unc / imbalance focal / final unc-focal.
    """
    from trainer import status as S
    from utils.common import save_json as _save
    selection = {ds: {'backbone': 'visobert', 'loss_tag': 'unc', 'imbalance_tag': 'focal', 'final_tag': 'unc-focal'}
                 for ds in ('neu-esc', 'uit-vsfc')}
    if selection_path:
        _save(selection, selection_path)
    import trainer.train as T
    from utils.common import CONFIGS_DIR
    bonus = {'st_sentiment': 0.0, 'st_topic': 0.0, 'mtl': 0.02, 'mtlaware': 0.03}
    tag_bonus = {'unc': 0.01, 'pcgrad': 0.005, 'smartemb': 0.008, 'smartref': -0.004, 'gradnorm': 0.002,
                 'focal': 0.012, 'wce': 0.006, 'mlm': -0.002, 'cross_sent': -0.01, 'cross_topic': -0.012}
    bb_bonus = {'xlmr': 0.0, 'visobert': 0.02, 'phobert': 0.015}
    n = 0
    for path in sorted((CONFIGS_DIR / 'experiment').glob('*.yaml')):
        cfg = T.load_config(path) if hasattr(T, 'load_config') else None
        if cfg is not None and cfg.get('evaluate_only'):
            continue
        exp = T.load_experiment(path)
        for s in T.expand_runs(exp):
            if s.seed not in seeds:
                continue
            tokens = s.tag.split('-')
            frac = next((float(t[4:]) for t in tokens if t.startswith('frac')), 1.0)
            v = 0.60 + bb_bonus[s.backbone] + bonus[s.mode] + sum(tag_bonus.get(t, 0) for t in tokens) \
                - 0.15 * (1 - frac) + {42: -0.01, 123: 0.0, 2026: 0.01}.get(s.seed, 0)
            make_run(root, s.dataset, s.backbone, s.mode, s.tag, s.seed, val=round(v, 4), test=round(v - 0.02, 4),
                     time_s=600 * S.GROUP_FACTOR[S.strategy_group(s.tag)] * max(frac, 0.15),
                     keep=bool(s.cfg['train']['keep_checkpoint']), experiment=exp['experiment'])
            n += 1
    return n


def make_fake_models(root, datasets=('neu-esc', 'uit-vsfc'), backbones=('xlmr', 'phobert'),
                     modes=('st_sentiment', 'st_topic', 'mtl'), tags=('sum', 'unc'), seeds=(42, 123, 2026)):
    """ The spec's fake grid. Scores are known: val = 0.60 + backbone/mode/tag offsets + seed jitter. """
    bb_off = {b: 0.02 * i for i, b in enumerate(backbones)}
    mode_off = {'st_sentiment': 0.0, 'st_topic': 0.0, 'mtl': 0.03, 'mtlaware': 0.04}
    tag_off = {'sum': 0.0, 'unc': 0.01}
    seed_off = dict(zip(seeds, (-0.01, 0.0, 0.01)))
    for ds in datasets:
        for bb in backbones:
            for mode in modes:
                for tag in tags:
                    for seed in seeds:
                        v = 0.60 + bb_off[bb] + mode_off[mode] + tag_off.get(tag, 0) + seed_off.get(seed, 0)
                        make_run(root, ds, bb, mode, tag, seed, val=round(v, 4), test=round(v - 0.02, 4),
                                 keep=(tag == 'sum' and mode in ('mtl', 'st_sentiment')))
    return Path(root)
