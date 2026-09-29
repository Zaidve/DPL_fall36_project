"""
    Evaluation helpers (train_spec.md §5).

      compute_metrics(y_true, y_pred, num_classes) -> {accuracy, macro_f1, weighted_f1, per_class_f1}
      evaluate_split(model, loader, tasks)         -> (metrics, predictions DataFrame)
      mcnemar(y_true, pred_a, pred_b)              -> exact McNemar test for two runs on the same rows

    Arguments are always (y_true, y_pred): the legacy multi-task trainers passed them swapped, which
    leaves accuracy and macro-F1 unchanged but makes weighted-F1 wrong (weighted by predicted counts).
    Validation loss is plain cross entropy per task for every loss strategy, so curves are comparable.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true, y_pred, num_classes):
    labels = list(range(num_classes))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        'per_class_f1': [float(v) for v in f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)],
    }


def _to_device(batch, device):
    return {'input_ids': batch['input_ids'].to(device, non_blocking=True),
            'attention_mask': batch['attention_mask'].to(device, non_blocking=True)}


@torch.no_grad()
def evaluate_split(model, loader, tasks, device=None, amp=False):
    """
      Run the model over one loader (validation or test).
      Returns
        metrics     {task: {accuracy, macro_f1, weighted_f1, per_class_f1, loss}, 'macro_f1_mean': float}
        predictions DataFrame: id, <task>_true, <task>_pred, <task>_prob_<k> for every class k,
                    one row per example in loader order (ids match data/processed).
      Only final logits are used (task_aware stage-1 logits are not requested).
    """
    tasks = list(tasks)
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    use_amp = amp and torch.device(device).type == 'cuda'

    ids, true, probs, loss_sum = [], {t: [] for t in tasks}, {t: [] for t in tasks}, {t: 0.0 for t in tasks}
    for batch in loader:
        inputs = _to_device(batch, device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            out = model(**inputs)
        for t in tasks:
            logits = out[t].float()
            y = batch['labels'][t].to(logits.device)
            loss_sum[t] += F.cross_entropy(logits, y, reduction='sum').item()
            probs[t].append(torch.softmax(logits, dim=-1).cpu())
            true[t].append(batch['labels'][t])
        ids += list(batch['idx'])

    if was_training:
        model.train()

    n = len(ids)
    predictions = {'id': ids}
    metrics = {}
    for t in tasks:
        p = torch.cat(probs[t]).numpy()
        y = torch.cat(true[t]).numpy()
        pred = p.argmax(axis=1)
        metrics[t] = {**compute_metrics(y, pred, p.shape[1]), 'loss': loss_sum[t] / max(n, 1)}
        predictions[f'{t}_true'] = y
        predictions[f'{t}_pred'] = pred
        for k in range(p.shape[1]):
            predictions[f'{t}_prob_{k}'] = p[:, k]
    metrics['macro_f1_mean'] = float(np.mean([metrics[t]['macro_f1'] for t in tasks]))
    return metrics, pd.DataFrame(predictions)


def mcnemar(y_true, pred_a, pred_b):
    """
      Exact McNemar test on paired predictions (same rows, same order).
      b = rows only A gets right, c = rows only B gets right; p = two-sided binomial test on b vs c.
    """
    from scipy.stats import binomtest
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {'only_a_correct': b, 'only_b_correct': c, 'p_value': p}
