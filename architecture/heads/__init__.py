"""
    Head registry. Every head takes the pooled vector (B, H):
      linear      Dropout -> Linear per task                                (default)
      mlp         shared Linear -> SiLU -> LayerNorm, then Linear per task   (legacy B1)
      task_aware  two-stage gated heads that read the other task's prediction (ours)
"""
from architecture.heads.linear import LinearHeads
from architecture.heads.mlp import MLPHeads
from architecture.heads.task_aware import CROSS, TaskAwareHeads

HEADS = ('linear', 'mlp', 'task_aware')


def build_head(kind, hidden_size, tasks, dropout=0.1, cross='both', no_head_dropout=False):
    if kind == 'linear':
        return LinearHeads(hidden_size, tasks, dropout)
    if kind == 'mlp':
        return MLPHeads(hidden_size, tasks, dropout, no_head_dropout=no_head_dropout)
    if kind == 'task_aware':
        return TaskAwareHeads(hidden_size, tasks, dropout, cross)
    raise ValueError(f'unknown head {kind!r}; choose from {HEADS}')


__all__ = ['HEADS', 'CROSS', 'LinearHeads', 'MLPHeads', 'TaskAwareHeads', 'build_head']
