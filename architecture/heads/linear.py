"""
    `linear` head (default, paper style): per task Dropout(p) -> Linear(H, C_t).
"""
import torch.nn as nn


class LinearHeads(nn.Module):

    def __init__(self, hidden_size, tasks, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({t: nn.Linear(hidden_size, c) for t, c in tasks.items()})

    def forward(self, h):
        """ (B, H) -> {task: (B, C_t)} """
        h = self.dropout(h)
        return {t: head(h) for t, head in self.heads.items()}
