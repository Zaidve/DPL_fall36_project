"""
    `task_aware` heads (ours, models_spec A3): each task head also sees the other task's prediction.

    For pooled h (B, H) and tasks s, t:
      stage 1   z_s0 = W_s0 · dropout(h)             p_s = softmax(z_s0)      (same for t)
      label emb e_t  = E_t · p_t                     E_t: Linear(C_t, H, bias=False)
      fusion    g_s  = sigmoid( W_gs · [h ; e_t] )   h_s = LayerNorm(h + g_s ⊙ e_t)
      stage 2   z_s  = W_s1 · dropout(h_s)

    cross: 'both' | 'sent_from_topic' (only sentiment is fused; topic keeps its stage-1 logits)
           | 'topic_from_sent'. Only the modules a direction needs are created.
    Stage-1 probabilities are not detached (gradients flow through p).
    forward(h) -> (final {task: logits}, stage1 {task: logits})
"""
import torch
import torch.nn as nn

CROSS = {'both': None, 'sent_from_topic': 'sentiment', 'topic_from_sent': 'topic'}


class TaskAwareHeads(nn.Module):

    def __init__(self, hidden_size, tasks, dropout=0.1, cross='both'):
        super().__init__()
        if len(tasks) != 2:
            raise ValueError(f'task_aware heads need exactly 2 tasks, got {list(tasks)}')
        if cross not in CROSS:
            raise ValueError(f'unknown cross {cross!r}; choose from {list(CROSS)}')
        only = CROSS[cross]
        if only is not None and only not in tasks:
            raise ValueError(f'cross={cross!r} needs a task named {only!r}; tasks are {list(tasks)}')

        names = list(tasks)
        self.other = {names[0]: names[1], names[1]: names[0]}
        self.fused = [t for t in names if only is None or t == only]
        self.cross = cross
        self.dropout = nn.Dropout(dropout)
        self.stage1 = nn.ModuleDict({t: nn.Linear(hidden_size, c) for t, c in tasks.items()})
        # label embedding of the task whose prediction is read, gate / norm / output of the fused task
        self.label_emb = nn.ModuleDict({self.other[t]: nn.Linear(tasks[self.other[t]], hidden_size, bias=False)
                                        for t in self.fused})
        self.gate = nn.ModuleDict({t: nn.Linear(2 * hidden_size, hidden_size) for t in self.fused})
        self.norm = nn.ModuleDict({t: nn.LayerNorm(hidden_size) for t in self.fused})
        self.stage2 = nn.ModuleDict({t: nn.Linear(hidden_size, tasks[t]) for t in self.fused})

    def forward(self, h):
        stage1 = {t: head(self.dropout(h)) for t, head in self.stage1.items()}
        final = dict(stage1)
        for t in self.fused:
            o = self.other[t]
            e = self.label_emb[o](torch.softmax(stage1[o], dim=-1))
            g = torch.sigmoid(self.gate[t](torch.cat([h, e], dim=-1)))
            h_t = self.norm[t](h + g * e)
            final[t] = self.stage2[t](self.dropout(h_t))
        return final, stage1
