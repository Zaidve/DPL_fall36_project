"""
    Loss functions for multi-task sentiment + topic classification (NEU-ESC, UIT-VSFC).

    Every function works with the model interface:
        model(input_ids=..., attention_mask=...)      -> dict[task, logits (B, C_t)]
        model(inputs_embeds=..., attention_mask=...)  -> same (needed for SMART)

    Sections
      A. Task losses          : ce, weighted_ce, focal (+ label smoothing)
      B. MLM auxiliary loss   : mask_tokens, mlm_loss
      C. SMART regularizer    : smart_regularizer
      D. Combining task losses: sum, fixed, uncertainty, gradnorm, pcgrad, dwa

    Compared with `multi-task-bert/utils/loss_function.py` (reference code of the NEU-ESC paper):
      - SMART perturbs the input embeddings and takes a real gradient-ascent step.
        The reference added noise to token ids and cast them back with `.long()`,
        which randomly swaps tokens for their neighbour id; the gradient w.r.t. the
        noise is always None there, so no adversarial step ever happens.
      - The final symmetric KL uses `batchmean` (the reference used `sum`, so its
        scale grew with the batch size).
      - One SMART call covers every task head (the reference ran the model again
        for each head and threw away the other outputs).
      - MLM masking excludes special tokens / padding, samples without duplicates
        and draws random tokens from the real vocabulary size of any backbone.

    Total loss in the trainer:
        L = combiner(task_losses) + smart_weight * R_smart + mlm_weight * L_mlm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# A. Task losses
# ---------------------------------------------------------------------------

def class_weights_from_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
      w_c = N / (C * n_c), computed from the TRAIN split only.
      Rare classes get larger weights; the average weight over samples is 1.
      A class that never appears gets the weight of a class with one sample.
    """
    labels = torch.as_tensor(labels, dtype=torch.long).flatten()
    counts = torch.bincount(labels, minlength=num_classes).float().clamp(min=1)
    return labels.numel() / (num_classes * counts)


def focal_loss(logits, target, gamma=2.0, weight=None, reduction="mean") -> torch.Tensor:
    """
      Focal loss (Lin et al., 2017): (1 - p_t)^gamma * CE_t, optional per-class alpha = weight.
      With weight, "mean" divides by the summed weights, same as F.cross_entropy, so
      focal_loss(gamma=0) equals cross entropy with or without weight.
    """
    log_p = F.log_softmax(logits, dim=-1)
    log_p_t = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
    loss = -((1 - log_p_t.exp()) ** gamma) * log_p_t

    w = weight[target] if weight is not None else torch.ones_like(loss)
    loss = loss * w
    if reduction == "mean":
        return loss.sum() / w.sum()
    if reduction == "sum":
        return loss.sum()
    return loss


class TaskLoss(nn.Module):
    """
      Holds one loss fn per task. forward(logits: dict, labels: dict) -> dict[task, scalar].
        kind: 'ce' | 'weighted_ce' | 'focal'
        class_weights: {task: Tensor(C_t)}, required for 'weighted_ce', optional alpha for 'focal'
    """
    def __init__(self, tasks, kind="ce", class_weights=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        if kind not in ("ce", "weighted_ce", "focal"):
            raise ValueError(f"Unknown task loss: {kind}")
        if kind == "weighted_ce" and not class_weights:
            raise ValueError("kind='weighted_ce' needs class_weights for every task")
        self.tasks = list(tasks)
        self.kind = kind
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        # Buffers move with .to(device) and are saved in the state_dict
        class_weights = class_weights or {}
        for task in self.tasks:
            w = class_weights.get(task)
            self.register_buffer(f"weight_{task}", None if w is None else torch.as_tensor(w, dtype=torch.float))

    def class_weight(self, task):
        return getattr(self, f"weight_{task}")

    def forward(self, logits, labels):
        losses = {}
        for task in self.tasks:
            if self.kind == "focal":
                losses[task] = focal_loss(logits[task], labels[task], self.gamma, self.class_weight(task))
            else:
                w = self.class_weight(task) if self.kind == "weighted_ce" else None
                losses[task] = F.cross_entropy(logits[task], labels[task], weight=w,
                                               label_smoothing=self.label_smoothing)
        return losses


def build_task_loss(cfg, tasks, class_weights=None) -> TaskLoss:
    """ cfg: the `loss:` section of the config. """
    return TaskLoss(tasks,
                    kind=cfg.get("task_loss", "ce"),
                    class_weights=class_weights,
                    gamma=cfg.get("focal_gamma", 2.0),
                    label_smoothing=cfg.get("label_smoothing", 0.0))


# ---------------------------------------------------------------------------
# B. Auxiliary loss: masked language modeling
# ---------------------------------------------------------------------------

def mask_tokens(input_ids, attention_mask, tokenizer, mlm_prob=0.15, generator=None):
    """
      Return (masked_input_ids, mlm_labels) with the standard BERT 80/10/10 rule:
        - candidates = real tokens (attention_mask == 1) that are not special tokens
        - each candidate is selected with prob mlm_prob (no duplicates)
        - of selected: 80% -> mask token, 10% -> random id in [0, len(tokenizer)), 10% unchanged
        - labels = original id at selected positions, -100 elsewhere
      `generator` must live on the same device as input_ids.
    """
    device = input_ids.device
    special_ids = torch.tensor(sorted(set(tokenizer.all_special_ids)), device=device)
    candidates = attention_mask.bool() & ~torch.isin(input_ids, special_ids)

    probs = torch.full(input_ids.shape, mlm_prob, device=device) * candidates
    selected = torch.bernoulli(probs, generator=generator).bool()
    labels = torch.where(selected, input_ids, torch.full_like(input_ids, -100))

    # 80% of selected -> mask; half of the remaining 20% -> random token
    to_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=device), generator=generator).bool() & selected
    to_random = (torch.bernoulli(torch.full(input_ids.shape, 0.5, device=device), generator=generator).bool()
                 & selected & ~to_mask)

    masked = input_ids.clone()
    masked[to_mask] = tokenizer.mask_token_id
    random_ids = torch.randint(len(tokenizer), input_ids.shape, generator=generator, device=device)
    masked[to_random] = random_ids[to_random]
    return masked, labels


def mlm_loss(mlm_logits, mlm_labels) -> torch.Tensor:
    """
      Cross entropy over masked positions only; already averaged over masked tokens,
      so no extra division by the number of tokens is needed.
      Returns 0 (still attached to the graph) when nothing was masked.
    """
    if not (mlm_labels != -100).any():
        return mlm_logits.sum() * 0.0
    vocab_size = mlm_logits.size(-1)
    return F.cross_entropy(mlm_logits.reshape(-1, vocab_size), mlm_labels.reshape(-1), ignore_index=-100)


# ---------------------------------------------------------------------------
# C. Regularization: SMART (Jiang et al., 2020)
# ---------------------------------------------------------------------------

def kl_loss(input, target, reduction="batchmean"):
    """ KL(softmax(target) || softmax(input)). """
    return F.kl_div(F.log_softmax(input, dim=-1), F.softmax(target, dim=-1), reduction=reduction)


def sym_kl_loss(input, target, reduction="batchmean"):
    """ Symmetric KL; each direction treats the other side as a fixed target. """
    return (kl_loss(input, target.detach(), reduction)
            + kl_loss(target, input.detach(), reduction))


def inf_norm(x):
    return x.abs().amax(dim=-1, keepdim=True)


def _input_embeddings(model):
    encoder = getattr(model, "encoder", model)
    return encoder.get_input_embeddings()


def smart_regularizer(model, input_ids, attention_mask, clean_logits,
                      eps=1e-5, step_size=1e-3, steps=1, norm_eps=1e-6) -> torch.Tensor:
    """
      Smoothness-inducing adversarial regularizer, one call for all task heads.
        1. perturb the input embeddings with small noise (padding / special positions masked)
        2. `steps` gradient-ascent steps on the noise to maximise KL(adv || clean)
        3. return sum_t symmetric_KL(adv_logits[t], clean_logits[t])   (batchmean)
      Gradients of the returned value flow to the model parameters.
    """
    embeds = _input_embeddings(model)(input_ids)
    mask = attention_mask.unsqueeze(-1).to(embeds.dtype)
    tasks = list(clean_logits)

    noise = (torch.randn_like(embeds) * eps * mask).requires_grad_()
    for _ in range(steps):
        adv_logits = model(inputs_embeds=embeds.detach() + noise, attention_mask=attention_mask)
        adv_loss = sum(kl_loss(adv_logits[t], clean_logits[t].detach()) for t in tasks)
        grad, = torch.autograd.grad(adv_loss, noise)
        if not torch.isfinite(grad).all():
            break
        step = noise + step_size * grad
        noise = (step / (inf_norm(step) + norm_eps) * mask).detach().requires_grad_()

    adv_logits = model(inputs_embeds=embeds + noise.detach(), attention_mask=attention_mask)
    return sum(sym_kl_loss(adv_logits[t], clean_logits[t]) for t in tasks)


# ---------------------------------------------------------------------------
# D. Multi-task combination strategies
# ---------------------------------------------------------------------------

class LossCombiner(nn.Module):
    """
      Common interface, so train.py never branches on the strategy name:
        total = combiner(losses, shared_params=..., step=...)
        combiner.epoch_end()   # once per epoch (used by DWA)
        combiner.weights()     # for logging
      Default behaviour is the plain sum: L = sum_t L_t.
    """
    def __init__(self, tasks, **cfg):
        super().__init__()
        self.tasks = list(tasks)

    def task_weights(self):
        return [1.0] * len(self.tasks)

    def forward(self, losses, shared_params=None, step=None):
        return sum(w * losses[t] for w, t in zip(self.task_weights(), self.tasks))

    def weights(self):
        return {t: float(w) for t, w in zip(self.tasks, self.task_weights())}

    def epoch_end(self):
        pass


class SumCombiner(LossCombiner):
    """ Paper + reference code baseline: L = sum_t L_t. """


class FixedCombiner(LossCombiner):
    """
      Reference code (`percentage`): L = alpha * L_1 + (1 - alpha) * L_2.
      For more than 2 tasks pass fixed_weights={task: w}.
    """
    def __init__(self, tasks, fixed_alpha=0.5, fixed_weights=None, **cfg):
        super().__init__(tasks)
        if fixed_weights:
            self._weights = [float(fixed_weights[t]) for t in self.tasks]
        elif len(self.tasks) == 1:
            self._weights = [1.0]
        elif len(self.tasks) == 2:
            self._weights = [fixed_alpha, 1.0 - fixed_alpha]
        else:
            raise ValueError("strategy='fixed' with more than 2 tasks needs fixed_weights")

    def task_weights(self):
        return self._weights


class UncertaintyCombiner(LossCombiner):
    """
      Kendall et al., 2018: L = sum_t exp(-s_t) * L_t + s_t, with s_t = log(sigma_t^2) learned.
      Add combiner.parameters() to the optimizer (lr = head lr).
    """
    def __init__(self, tasks, **cfg):
        super().__init__(tasks)
        self.log_vars = nn.Parameter(torch.zeros(len(self.tasks)))

    def task_weights(self):
        return torch.exp(-self.log_vars).detach().tolist()

    def forward(self, losses, shared_params=None, step=None):
        return sum(torch.exp(-s) * losses[t] + s for s, t in zip(self.log_vars, self.tasks))


class GradNormCombiner(LossCombiner):
    """
      Chen et al., 2018. Learns w_t so that ||grad_W (w_t L_t)|| ~ mean_norm * r_t^alpha,
      r_t = (L_t / L_t(0)) / mean_t(L_t / L_t(0)), W = shared_params (last shared encoder layer).
      w_t has its own optimizer, stepped inside forward, then renormalized so sum w_t = T.
      Do NOT rely on the main optimizer for w_t; the returned loss uses w_t as constants.
    """
    def __init__(self, tasks, gradnorm_alpha=1.5, gradnorm_lr=0.025, **cfg):
        super().__init__(tasks)
        self.alpha = gradnorm_alpha
        self.lr = gradnorm_lr
        self.w = nn.Parameter(torch.ones(len(self.tasks)))
        self.register_buffer("initial_losses", None)
        self._optimizer = None

    def task_weights(self):
        return self.w.detach().tolist()

    def forward(self, losses, shared_params=None, step=None):
        L = torch.stack([losses[t] for t in self.tasks])
        # clone: w is updated in place below, before the trainer calls total.backward()
        total = (self.w.detach().clone() * L).sum()

        if self.training and shared_params is not None and torch.is_grad_enabled():
            self._update_weights(L, [p for p in shared_params if p.requires_grad])
        return total

    def _update_weights(self, L, shared_params):
        if self.initial_losses is None:
            self.initial_losses = L.detach().clone()
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam([self.w], lr=self.lr)

        # ||grad_W (w_t L_t)|| = w_t * ||grad_W L_t||, so only ||grad_W L_t|| needs autograd
        norms = []
        for loss in L:
            grads = torch.autograd.grad(loss, shared_params, retain_graph=True, allow_unused=True)
            norms.append(torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None)))
        G = self.w * torch.stack(norms).detach()

        ratio = L.detach() / self.initial_losses
        r = ratio / ratio.mean()
        target = (G.mean() * r ** self.alpha).detach()
        gradnorm_loss = (G - target).abs().sum()

        self._optimizer.zero_grad()
        self.w.grad, = torch.autograd.grad(gradnorm_loss, self.w)
        self._optimizer.step()
        self.w.grad = None  # keep the main optimizer from stepping w as well
        with torch.no_grad():
            self.w.clamp_(min=1e-4)
            self.w.mul_(len(self.tasks) / self.w.sum())


class PCGradCombiner(LossCombiner):
    """
      Yu et al., 2020. Not a scalar loss: forward returns the plain sum for logging, and
      train.py calls pcgrad_backward(losses, shared_params) instead of total.backward().
    """


def pcgrad_backward(losses, shared_params, extra_loss=None):
    """
      PCGrad backward pass.
        - shared params: per-task gradients, and if g_i . g_j < 0, g_i <- g_i - (g_i.g_j / ||g_j||^2) g_j
          (random task order each step); final grad = sum of projected gradients
        - every other param (task heads): its own task gradient, as a normal backward would give
        - extra_loss (SMART / MLM terms): added with a normal backward, not projected
      Adds to existing .grad, so gradient accumulation still works.
    """
    tasks = list(losses)
    shared = [p for p in shared_params if p.requires_grad]

    flat_grads = []
    for t in tasks:
        grads = torch.autograd.grad(losses[t], shared, retain_graph=True, allow_unused=True)
        flat_grads.append(torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                                     for g, p in zip(grads, shared)]))

    projected = []
    for i, g_i in enumerate(flat_grads):
        g = g_i.clone()
        for j in torch.randperm(len(tasks)).tolist():
            if j == i:
                continue
            g_j = flat_grads[j]
            dot = torch.dot(g, g_j)
            if dot < 0:
                g = g - dot / (g_j.norm() ** 2 + 1e-12) * g_j
        projected.append(g)
    # A normal backward gives heads their task gradient and shared params the raw sum
    # (+ extra_loss); on shared params, swap the raw task sum for the projected one.
    correction = torch.stack(projected).sum(0) - torch.stack(flat_grads).sum(0)

    total = sum(losses[t] for t in tasks)
    if extra_loss is not None:
        total = total + extra_loss
    total.backward()

    offset = 0
    for p in shared:
        n = p.numel()
        fix = correction[offset:offset + n].view_as(p)
        p.grad = fix.clone() if p.grad is None else p.grad + fix
        offset += n


class DWACombiner(LossCombiner):
    """
      Liu et al., 2019: w_t(k) = T * softmax_t( (L_t(k-1) / L_t(k-2)) / temperature ),
      using epoch-average losses; w_t = 1 for the first 2 epochs.
      Call epoch_end() once at the end of every epoch.
    """
    def __init__(self, tasks, dwa_temperature=2.0, **cfg):
        super().__init__(tasks)
        self.temperature = dwa_temperature
        self.history = []  # epoch-average losses, one tensor(T) per finished epoch
        self._sum = torch.zeros(len(self.tasks))
        self._count = 0
        self._weights = [1.0] * len(self.tasks)

    def task_weights(self):
        return self._weights

    def forward(self, losses, shared_params=None, step=None):
        if self.training:
            self._sum += torch.stack([losses[t].detach().float().cpu() for t in self.tasks])
            self._count += 1
        return super().forward(losses)

    def epoch_end(self):
        if self._count:
            self.history.append(self._sum / self._count)
        self._sum = torch.zeros(len(self.tasks))
        self._count = 0
        if len(self.history) >= 2:
            ratio = self.history[-1] / self.history[-2]
            self._weights = (len(self.tasks) * F.softmax(ratio / self.temperature, dim=0)).tolist()


COMBINERS = {
    "sum": SumCombiner,
    "fixed": FixedCombiner,
    "uncertainty": UncertaintyCombiner,
    "gradnorm": GradNormCombiner,
    "pcgrad": PCGradCombiner,
    "dwa": DWACombiner,
}


def build_combiner(cfg, tasks) -> LossCombiner:
    """ cfg: the `loss:` section of the config. """
    strategy = cfg.get("strategy", "sum")
    if strategy not in COMBINERS:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(COMBINERS)}")
    kwargs = {k: v for k, v in cfg.items() if k != "strategy"}
    return COMBINERS[strategy](tasks, **kwargs)
