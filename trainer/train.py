"""
    Experiment runner (train_spec.md): an experiment YAML -> finished run folders under models/.

      python -m trainer.train --experiment configs/experiment/e1_baseline.yaml
                              [--only SUBSTRING] [--dry-run] [--max-steps N]

    Experiment level: load_experiment -> expand_runs (datasets x backbones x grid x seeds, tags applied,
    `best` resolved from reports/tables/selection.json) -> for each run: skip if done, else
    train_one_run inside try/except (a failure writes error.txt and the next run starts).
    Re-running the same command resumes: finished runs have metrics.json.

    Run level: seed -> config.yaml, env.json, run.log -> data -> model -> losses -> AdamW (encoder /
    heads groups) -> linear warm-up schedule -> epochs with early stopping on validation macro-F1 ->
    reload best -> predictions + metrics on validation and test -> metrics.json (last) -> delete best.pt.

    --max-steps N caps the training batches per epoch for quick checks; those runs go to
    models/_debug/ so they are never mistaken for finished runs.

    Tags (models_spec.md B5, combine with '-'):
      sum | unc | pcgrad | gradnorm | dwa | fixed<alpha>    loss.strategy
      focal | wce                                            loss.task_loss
      smartemb | smartref                                    SMART on embeddings / reference reproduction
      mlm                                                    MLM auxiliary loss (+ MLM head)
      frac<x>                                                data.train_fraction
      cross_sent | cross_topic                               task-aware direction ablation (E4b; ours)
      bestloss | bestimb | final                             replaced from selection.json per dataset
"""
import argparse
import gc
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch

# Expected with fp16: GradScaler skips the first optimizer step(s) while it calibrates the loss scale.
warnings.filterwarnings('ignore', message=r'Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`')

if str(Path(__file__).resolve().parents[1]) not in sys.path:      # also works as `python trainer/train.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from architecture import build_model  # noqa: E402
from trainer.evaluate import evaluate_split  # noqa: E402
from trainer.runs import (is_run_done, load_checkpoint, make_run_id, run_dir, save_checkpoint,  # noqa: E402
                          save_predictions, write_error, write_metrics)
from utils.common import (close_logger, env_info, get_logger, load_json, models_dir, reports_dir,  # noqa: E402
                          save_json, save_yaml, set_seed)
from utils.config import apply_overrides, load_config  # noqa: E402
from utils.dataset import build_run_dataloaders  # noqa: E402
from utils.loss_function import (TaskLoss, build_combiner, class_weights_from_labels, mask_tokens,  # noqa: E402
                                 mlm_loss, pcgrad_backward, smart_regularizer)

MODE_SETTINGS = {                       # mode -> (tasks, head)
    'st_sentiment': (['sentiment'], 'linear'),
    'st_topic': (['topic'], 'linear'),
    'mtl': (['sentiment', 'topic'], 'linear'),
    'mtlaware': (['sentiment', 'topic'], 'task_aware'),
}
STRATEGY_TAGS = {'sum': 'sum', 'unc': 'uncertainty', 'pcgrad': 'pcgrad', 'gradnorm': 'gradnorm', 'dwa': 'dwa'}
FLAG_TAGS = {
    'smartemb': {'loss.smart': True, 'loss.smart_mode': 'embeddings'},
    'smartref': {'loss.smart': True, 'loss.smart_mode': 'token_ids'},
    'focal': {'loss.task_loss': 'focal'},
    'wce': {'loss.task_loss': 'weighted_ce'},
    'mlm': {'loss.mlm': True, 'model.mlm': True},
    'cross_sent': {'model.cross': 'sent_from_topic'},
    'cross_topic': {'model.cross': 'topic_from_sent'},
}
SELECTION_TOKENS = {'bestloss': 'loss_tag', 'bestimb': 'imbalance_tag', 'final': 'final_tag'}
FP32_STRATEGIES = {'pcgrad', 'gradnorm'}
MAX_SKIPPED_STEPS = 10


@dataclass
class RunSpec:
    run_id: str
    dataset: str
    backbone: str
    mode: str
    tag: str
    seed: int
    cfg: dict = field(repr=False)


# ---------------------------------------------------------------------------
# Experiment level
# ---------------------------------------------------------------------------

def load_experiment(path):
    """ Experiment YAML merged with what it extends (normally configs/default.yaml). """
    exp = load_config(path)
    for key in ('backbones', 'seeds', 'grid'):
        if not exp.get(key):
            raise ValueError(f'{path}: experiment needs a non-empty `{key}` list')
    exp.setdefault('datasets', exp.get('data', {}).get('datasets'))
    exp.setdefault('experiment', os.path.splitext(os.path.basename(str(path)))[0])
    return exp


def tag_overrides(tag):
    """ Dotted config overrides for a tag such as 'unc-focal' or 'sum-frac0.1'. Unknown tokens raise. """
    overrides, strategies, task_losses = {}, [], []
    for token in tag.split('-'):
        if token in STRATEGY_TAGS:
            strategies.append(STRATEGY_TAGS[token])
        elif m := re.fullmatch(r'fixed(\d*\.\d+)', token):
            strategies.append('fixed')
            overrides['loss.fixed_alpha'] = float(m.group(1))
        elif m := re.fullmatch(r'frac(\d*\.?\d+)', token):
            fraction = float(m.group(1))
            if not 0 < fraction <= 1:
                raise ValueError(f'train fraction must be in (0, 1]: {token}')
            overrides['data.train_fraction'] = fraction
        elif token in FLAG_TAGS:
            overrides.update(FLAG_TAGS[token])
            if token in ('focal', 'wce'):
                task_losses.append(token)
        else:
            raise ValueError(f'unknown tag {token!r} in {tag!r}')
    if len(strategies) > 1:
        raise ValueError(f'tag {tag!r} sets more than one loss strategy: {strategies}')
    if len(task_losses) > 1:
        raise ValueError(f'tag {tag!r} sets more than one task loss: {task_losses}')
    if strategies:
        overrides['loss.strategy'] = strategies[0]
    return overrides


def apply_tag_overrides(cfg, mode, tag):
    """ Config for one (mode, tag): mode sets tasks and head, tag sets loss / data / model options. """
    if mode not in MODE_SETTINGS:
        raise ValueError(f'unknown mode {mode!r}; choose from {list(MODE_SETTINGS)}')
    tasks, head = MODE_SETTINGS[mode]
    overrides = {'data.tasks': tasks, 'model.head': head, **tag_overrides(tag)}
    if 'model.cross' in overrides and head != 'task_aware':
        raise ValueError(f'tag {tag!r} changes the task-aware direction but mode {mode!r} has no task-aware head')
    return apply_overrides(cfg, overrides)


def load_selection(path=None):
    path = path or reports_dir() / 'tables' / 'selection.json'
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found: this experiment uses `best`/`final`. '
                                'Run E1/E2 and write the selection (best backbone / loss per dataset) first.')
    return load_json(path)


def _needs_selection(exp):
    tokens = {tok for entry in exp['grid'] for tok in str(entry['tag']).split('-')}
    return 'best' in exp['backbones'] or bool(tokens & set(SELECTION_TOKENS))


def resolve_tag(tag, selection_ds):
    """ Replace bestloss / bestimb / final with the tags chosen for this dataset. """
    out = []
    for token in tag.split('-'):
        if token in SELECTION_TOKENS:
            value = (selection_ds or {}).get(SELECTION_TOKENS[token])
            if not value:
                raise KeyError(f'selection.json has no {SELECTION_TOKENS[token]!r} for this dataset')
            out += [t for t in value.split('-') if t not in out]
        elif token not in out:
            out.append(token)
    return '-'.join(out)


def expand_runs(exp):
    """ RunSpec for every datasets x backbones x grid x seeds combination, with tags applied. """
    selection = load_selection() if _needs_selection(exp) else {}
    base = {k: v for k, v in exp.items() if k not in ('datasets', 'backbones', 'seeds', 'grid')}
    specs, seen = [], set()
    for ds in exp['datasets']:
        for bb in exp['backbones']:
            backbone = selection[ds]['backbone'] if bb == 'best' else bb
            for entry in exp['grid']:
                mode, raw_tag = entry['mode'], str(entry['tag'])
                tag = resolve_tag(raw_tag, selection.get(ds))
                for seed in exp['seeds']:
                    cfg = apply_tag_overrides(base, mode, tag)
                    keep = (cfg['train'].get('keep_checkpoint', False)
                            or (mode == 'mtl' and tag == 'sum') or 'final' in raw_tag.split('-'))
                    cfg = apply_overrides(cfg, {'model.backbone': backbone, 'data.dataset': ds, 'train.seed': int(seed),
                                                'train.keep_checkpoint': keep, **entry.get('overrides', {})})
                    run_id = make_run_id(ds, backbone, mode, seed, tag)
                    if run_id in seen:
                        raise ValueError(f'duplicate run id {run_id}')
                    seen.add(run_id)
                    cfg['run'] = {'run_id': run_id, 'experiment': exp.get('experiment'), 'dataset': ds,
                                  'backbone': backbone, 'mode': mode, 'tag': tag, 'seed': int(seed)}
                    specs.append(RunSpec(run_id, ds, backbone, mode, tag, int(seed), cfg))
    return specs


# ---------------------------------------------------------------------------
# Run level
# ---------------------------------------------------------------------------

def get_device(cfg):
    choice = os.environ.get('DPL_DEVICE') or cfg['train'].get('device', 'auto')
    if choice == 'auto':
        choice = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(choice)


def build_optimizer(model, combiner, cfg):
    """ AdamW: encoder at lr_encoder, heads (+ learned loss weights) at lr_heads; no decay on 1-D params
        (biases, LayerNorm, loss weights). GradNorm's weights have their own optimizer inside the combiner. """
    t = cfg['train']
    heads = list(model.task_parameters())
    if cfg['loss'].get('strategy') != 'gradnorm':
        heads += [p for p in combiner.parameters() if p.requires_grad]
    groups = []
    for name, params, lr in (('encoder', model.shared_parameters(), t['lr_encoder']), ('heads', heads, t['lr_heads'])):
        decay = [p for p in params if p.ndim >= 2]
        no_decay = [p for p in params if p.ndim < 2]
        groups += [{'name': name, 'params': decay, 'lr': lr, 'weight_decay': t.get('weight_decay', 0.01)},
                   {'name': name, 'params': no_decay, 'lr': lr, 'weight_decay': 0.0}]
    return torch.optim.AdamW([g for g in groups if g['params']], eps=t.get('adam_eps', 1e-8))


def build_task_loss(cfg, tasks, loaders, num_labels):
    """ TaskLoss with class weights from the (possibly subset) train split when the loss needs them. """
    lc = cfg['loss']
    kind = lc.get('task_loss', 'ce')
    weights = None
    if kind == 'weighted_ce' or (kind == 'focal' and lc.get('focal_alpha')):
        labels = loaders['train'].dataset.labels
        weights = {t: class_weights_from_labels(torch.as_tensor(labels[t]), num_labels[t]) for t in tasks}
    return TaskLoss(tasks, kind=kind, class_weights=weights, gamma=lc.get('focal_gamma', 2.0),
                    label_smoothing=lc.get('label_smoothing', 0.0))


def _lr(optimizer, name):
    return next(g['lr'] for g in optimizer.param_groups if g.get('name') == name)


def train_one_run(spec, max_steps=None, root=None):
    """ Train, select, evaluate and save one run. Returns the metrics dict (also saved as metrics.json). """
    cfg, tasks = spec.cfg, list(spec.cfg['data']['tasks'])
    tc, lc = cfg['train'], cfg['loss']
    run_path = run_dir(spec.run_id, root)
    for stale in ('error.txt', 'metrics.json'):
        (run_path / stale).unlink(missing_ok=True)
    set_seed(spec.seed)
    save_yaml(cfg, run_path / 'config.yaml')
    save_json(env_info(), run_path / 'env.json')
    log = get_logger(f'run.{spec.run_id}', run_path / 'run.log')
    device = get_device(cfg)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    log.info(f'run {spec.run_id} on {device}')

    try:
        # data
        loaders, tokenizer, info = build_run_dataloaders(
            spec.dataset, spec.backbone, tasks, tc['batch_size'], spec.seed,
            train_fraction=cfg['data'].get('train_fraction', 1.0), num_workers=tc.get('num_workers', 2))
        num_labels = info['num_labels']
        log.info(f'data: train {info["n_train"]:,}, validation {info["n_validation"]:,}, test {info["n_test"]:,}; '
                 f'{info["text_column"]}, max_len {info["max_len"]}')

        # model, losses
        model = build_model(cfg, num_labels).to(device)
        params = model.count_parameters()
        task_loss = build_task_loss(cfg, tasks, loaders, num_labels).to(device)
        combiner = build_combiner(lc, tasks).to(device)
        strategy = lc.get('strategy', 'sum')
        task_aware = cfg['model'].get('head') == 'task_aware'
        aux_weight = cfg['model'].get('aux_weight', 0.5)
        smart_mode = 'token_ids' if (lc.get('smart_mode') == 'token_ids' or model.legacy.smart_token_shift) else 'embeddings'
        use_amp = bool(tc.get('fp16', True)) and device.type == 'cuda' and strategy not in FP32_STRATEGIES
        log.info(f'model: {spec.backbone} head={cfg["model"]["head"]} params={params["total"]:,}; loss: '
                 f'{lc.get("task_loss")} / {strategy}, smart={lc.get("smart")} ({smart_mode}), mlm={lc.get("mlm")}, '
                 f'amp={use_amp}')

        # optimizer, schedule
        optimizer = build_optimizer(model, combiner, cfg)
        accum = max(1, int(tc.get('grad_accum_steps', 1)))
        batches = len(loaders['train']) if not max_steps else min(len(loaders['train']), max_steps)
        total_steps = tc['epochs'] * math.ceil(batches / accum)
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, int(tc.get('warmup_ratio', 0.1) * total_steps), total_steps)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        clip_params = [p for g in optimizer.param_groups for p in g['params']]
        shared = model.shared_parameters()
        last_shared = model.last_shared_layer_params()

        best_score, best_epoch, bad_epochs, skipped, global_step = -math.inf, 0, 0, 0, 0
        best_path = run_path / 'best.pt'
        epoch_rows, stopped_early = [], False
        start = time.time()

        for epoch in range(1, tc['epochs'] + 1):
            model.train()
            combiner.train()
            t0 = time.time()
            sums, counts = {t: 0.0 for t in tasks}, 0
            optimizer.zero_grad(set_to_none=True)
            for i, batch in enumerate(loaders['train']):
                if max_steps and i >= max_steps:
                    break
                ids = batch['input_ids'].to(device, non_blocking=True)
                mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = {t: batch['labels'][t].to(device, non_blocking=True) for t in tasks}

                with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                    out = model(ids, mask, return_stage1=task_aware)
                    losses = task_loss({t: out[t] for t in tasks}, labels)
                    logged = {t: losses[t].detach() for t in tasks}
                    if task_aware:
                        aux = task_loss({t: out[f'{t}__stage1'] for t in tasks}, labels)
                        losses = {t: losses[t] + aux_weight * aux[t] for t in tasks}
                    extra = None
                    if lc.get('smart'):
                        extra = lc.get('smart_weight', 0.02) * smart_regularizer(
                            model, ids, mask, {t: out[t] for t in tasks}, eps=lc.get('smart_eps', 1e-5),
                            step_size=lc.get('smart_step_size', 1e-3), steps=lc.get('smart_steps', 1), mode=smart_mode)
                    if lc.get('mlm'):
                        masked, mlm_labels = mask_tokens(ids, mask, tokenizer, lc.get('mlm_prob', 0.15))
                        positions = mlm_labels != -100
                        term = lc.get('mlm_weight', 1.0) * mlm_loss(model.forward_mlm(masked, mask, positions=positions),
                                                                     mlm_labels[positions])
                        extra = term if extra is None else extra + term

                finite = all(torch.isfinite(v).all() for v in losses.values()) and (extra is None or torch.isfinite(extra).all())
                if not finite:
                    skipped += 1
                    log.warning(f'epoch {epoch} step {i}: non-finite loss, step skipped (ids {batch["idx"][:5]} ...)')
                    optimizer.zero_grad(set_to_none=True)
                    if skipped > MAX_SKIPPED_STEPS:
                        raise RuntimeError(f'more than {MAX_SKIPPED_STEPS} steps with non-finite loss')
                    continue

                if strategy == 'pcgrad':
                    pcgrad_backward({t: losses[t] / accum for t in tasks}, shared,
                                    extra_loss=None if extra is None else extra / accum)
                elif strategy == 'gradnorm':
                    total = combiner(losses, shared_params=last_shared) + (0 if extra is None else extra)
                    (total / accum).backward()
                else:
                    total = combiner(losses) + (0 if extra is None else extra)
                    scaler.scale(total / accum).backward()

                if (i + 1) % accum == 0 or i + 1 == batches:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(clip_params, tc.get('max_grad_norm', 1.0))
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                for t in tasks:
                    sums[t] += logged[t].item()
                counts += 1
                if global_step and global_step % tc.get('log_every', 50) == 0 and (i + 1) % accum == 0:
                    log.info(f'epoch {epoch} step {global_step}: ' + ', '.join(f'{t} {sums[t] / counts:.4f}' for t in tasks)
                             + f' | weights {combiner.weights()} | lr {_lr(optimizer, "encoder"):.2e}')

            # validation
            val, _ = evaluate_split(model, loaders['validation'], tasks, device, amp=use_amp)
            score = val[tc.get('select_metric', 'macro_f1_mean')]
            combiner.epoch_end()
            weights = combiner.weights()
            row = {'epoch': epoch}
            row.update({f'train_loss_{t}': sums[t] / max(counts, 1) for t in tasks})
            row.update({f'val_loss_{t}': val[t]['loss'] for t in tasks})
            row.update({f'val_acc_{t}': val[t]['accuracy'] for t in tasks})
            row.update({f'val_macro_f1_{t}': val[t]['macro_f1'] for t in tasks})
            row.update({'val_score': score, 'lr_encoder': _lr(optimizer, 'encoder')})
            row.update({f'w_{t}': weights[t] for t in tasks})
            row.update({'time_s': time.time() - t0, 'skipped_steps': skipped})
            epoch_rows.append(row)
            pd.DataFrame(epoch_rows).to_csv(run_path / 'train_log.csv', index=False)

            improved = score > best_score + 1e-4
            log.info(f'epoch {epoch}: val score {score:.4f} ' + ('(best, saved)' if improved else f'(best {best_score:.4f})')
                     + ' | ' + ', '.join(f'{t} F1 {val[t]["macro_f1"]:.4f}' for t in tasks) + f' | {row["time_s"]:.0f}s')
            if improved:
                best_score, best_epoch, bad_epochs = score, epoch, 0
                save_checkpoint(model, best_path, epoch=epoch, score=score)
            else:
                bad_epochs += 1
                if bad_epochs >= tc.get('patience', 3):
                    stopped_early = epoch < tc['epochs']
                    break
        train_time = time.time() - start

        # final evaluation on the best epoch
        load_checkpoint(model, best_path, map_location=device)
        results = {}
        for key in ('validation', 'test'):
            results[key], preds = evaluate_split(model, loaders[key], tasks, device, amp=use_amp)
            save_predictions(preds, run_path / f'predictions_{key}.csv')

        metrics = {
            'run_id': spec.run_id, 'experiment': cfg.get('run', {}).get('experiment'), 'dataset': spec.dataset,
            'backbone': spec.backbone, 'mode': spec.mode, 'tag': spec.tag, 'seed': spec.seed,
            'best_epoch': best_epoch, 'epochs_run': len(epoch_rows), 'stopped_early': stopped_early,
            'validation': results['validation'], 'test': results['test'],
            'train_time_s': train_time, 'time_per_epoch_s': train_time / max(len(epoch_rows), 1),
            'peak_gpu_mb': torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == 'cuda' else None,
            'params_total': params['total'], 'params_trainable': params['trainable'],
            'final_task_weights': combiner.weights(),
            'train_fraction': info['train_fraction'], 'n_train': info['n_train'], 'max_len': info['max_len'],
            'text_column': info['text_column'], 'skipped_steps': skipped, 'amp': use_amp, 'device': str(device),
            'max_steps': max_steps,
        }
        write_metrics(run_path, metrics)
        if not tc.get('keep_checkpoint', False):
            best_path.unlink(missing_ok=True)
        log.info(f'done: best epoch {best_epoch}, validation {results["validation"]["macro_f1_mean"]:.4f}, '
                 f'test {results["test"]["macro_f1_mean"]:.4f}, {train_time / 60:.1f} min')
        return metrics
    finally:
        close_logger(log)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description='Train every run of an experiment (resumes finished runs).')
    parser.add_argument('--experiment', required=True, help='configs/experiment/<name>.yaml')
    parser.add_argument('--only', default=None, help='only runs whose id contains this substring')
    parser.add_argument('--dry-run', action='store_true', help='print the run list and counts, train nothing')
    parser.add_argument('--max-steps', type=int, default=None, help='cap training batches per epoch (quick check)')
    args = parser.parse_args(argv)

    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()   # loading an encoder from an MLM checkpoint reports the unused lm_head
    exp = load_experiment(args.experiment)
    specs = expand_runs(exp)
    if args.only:
        specs = [s for s in specs if args.only in s.run_id]
    root = models_dir() / '_debug' if args.max_steps else models_dir()
    log = get_logger('trainer')

    if args.dry_run:
        done = [s for s in specs if is_run_done(s.run_id, root)]
        for s in specs:
            print(('done  ' if s in done else 'todo  ') + s.run_id)
        print(f'\n{len(specs)} runs: {len(done)} done, {len(specs) - len(done)} to train  (models: {root})')
        return {'runs': len(specs), 'done': len(done), 'to_train': len(specs) - len(done)}

    summary = {'done': 0, 'skipped': 0, 'failed': 0, 'failed_runs': []}
    start = time.time()
    for n, spec in enumerate(specs, 1):
        if is_run_done(spec.run_id, root):
            log.info(f'[{n}/{len(specs)}] skip {spec.run_id} (done)')
            summary['skipped'] += 1
            continue
        log.info(f'[{n}/{len(specs)}] train {spec.run_id}')
        try:
            train_one_run(spec, max_steps=args.max_steps, root=root)
            summary['done'] += 1
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 - one failed run must not stop the experiment
            write_error(run_dir(spec.run_id, root), e)
            log.error(f'{spec.run_id} failed: {type(e).__name__}: {e} (see error.txt)')
            summary['failed'] += 1
            summary['failed_runs'].append(spec.run_id)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary['time_min'] = (time.time() - start) / 60
    log.info(f'finished: {summary["done"]} trained, {summary["skipped"]} skipped, {summary["failed"]} failed, '
             f'{summary["time_min"]:.1f} min')
    return summary


if __name__ == '__main__':
    main()
