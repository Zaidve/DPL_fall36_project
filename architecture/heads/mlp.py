"""
    `mlp` head = the legacy B1 task trunk (architecture_baseline.md §2.3):
        h = LayerNorm( SiLU( Linear(H, H)(pooled) ) )        shared by all tasks
        logits_t = Linear(H, C_t)(dropout(h))
    Order is Linear -> SiLU -> LayerNorm. The legacy model defined dropout but never applied it;
    no_head_dropout=True reproduces that (LegacyFlags.no_head_dropout).
    Parameter names (trunk.0, trunk.2, heads.<task>) match the legacy key map in legacy.py.
"""
import torch.nn as nn


class MLPHeads(nn.Module):

    def __init__(self, hidden_size, tasks, dropout=0.1, no_head_dropout=False):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.LayerNorm(hidden_size))
        self.dropout = nn.Identity() if no_head_dropout else nn.Dropout(dropout)
        self.heads = nn.ModuleDict({t: nn.Linear(hidden_size, c) for t, c in tasks.items()})

    def forward(self, h):
        h = self.dropout(self.trunk(h))
        return {t: head(h) for t, head in self.heads.items()}
