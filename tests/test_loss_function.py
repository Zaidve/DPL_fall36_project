"""
    Tests for utils/loss_function.py. CPU only, tiny tensors, no downloads.
    Run:  python tests/test_loss_function.py   (or `pytest tests` if pytest is installed)
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.loss_function import (COMBINERS, TaskLoss, build_combiner, class_weights_from_labels,  # noqa: E402
                                 focal_loss, legacy_token_shift, mask_tokens, mlm_loss, pcgrad_backward,
                                 smart_regularizer, sym_kl_loss)

TASKS = ["sentiment", "topic"]
NUM_CLASSES = {"sentiment": 3, "topic": 4}


class ToyEncoder(nn.Module):
    def __init__(self, vocab_size=30, hidden=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.layer = nn.Linear(hidden, hidden)

    def get_input_embeddings(self):
        return self.embed


class ToyModel(nn.Module):
    """ Shared encoder -> one linear head per task; accepts input_ids or inputs_embeds. """
    def __init__(self, tasks=TASKS):
        super().__init__()
        self.encoder = ToyEncoder()
        self.heads = nn.ModuleDict({t: nn.Linear(8, NUM_CLASSES[t]) for t in tasks})

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.encoder.embed(input_ids)
        h = torch.tanh(self.encoder.layer(inputs_embeds))
        m = attention_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1)
        return {t: head(pooled) for t, head in self.heads.items()}


class FakeTokenizer:
    """ PhoBERT-like ids: <s>=0, <pad>=1, </s>=2, <unk>=3, <mask>=4. """
    all_special_ids = [0, 1, 2, 3, 4]
    mask_token_id = 4

    def __len__(self):
        return 50


def toy_batch(batch_size=4, seq_len=6, tasks=TASKS):
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(5, 30, (batch_size, seq_len), generator=g)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, -2:] = 0
    labels = {t: torch.randint(NUM_CLASSES[t], (batch_size,), generator=g) for t in tasks}
    return input_ids, attention_mask, labels


# 1
def test_focal_gamma0_equals_ce():
    logits, y = torch.randn(16, 4), torch.randint(4, (16,))
    assert torch.allclose(focal_loss(logits, y, gamma=0), F.cross_entropy(logits, y), atol=1e-6)
    w = torch.tensor([0.5, 1.0, 2.0, 3.0])
    assert torch.allclose(focal_loss(logits, y, gamma=0, weight=w), F.cross_entropy(logits, y, weight=w), atol=1e-6)
    assert focal_loss(logits, y, gamma=2) < F.cross_entropy(logits, y)


# 2
def test_class_weights():
    labels = torch.tensor([0] * 90 + [1] * 9 + [2] * 1)
    w = class_weights_from_labels(labels, 3)
    assert w[2] > w[1] > w[0]
    assert abs(w[labels].mean().item() - 1.0) < 1e-5  # average weight per sample is 1


def test_task_loss_kinds():
    torch.manual_seed(0)
    logits = {t: torch.randn(8, NUM_CLASSES[t]) for t in TASKS}
    labels = {t: torch.randint(NUM_CLASSES[t], (8,)) for t in TASKS}
    weights = {t: torch.ones(NUM_CLASSES[t]) for t in TASKS}
    ce = TaskLoss(TASKS, "ce")(logits, labels)
    wce = TaskLoss(TASKS, "weighted_ce", class_weights=weights)(logits, labels)
    for t in TASKS:
        assert torch.allclose(ce[t], wce[t])
    focal = TaskLoss(TASKS, "focal", gamma=2.0)(logits, labels)
    assert set(focal) == set(TASKS)
    # buffers follow the module
    loss = TaskLoss(TASKS, "weighted_ce", class_weights=weights)
    assert "weight_sentiment" in loss.state_dict()


def _run_strategy(strategy, tasks=TASKS):
    torch.manual_seed(0)
    model = ToyModel(tasks)
    combiner = build_combiner({"strategy": strategy}, tasks)
    input_ids, attention_mask, labels = toy_batch(tasks=tasks)
    shared = list(model.encoder.layer.parameters())
    for step in range(2):
        model.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        losses = TaskLoss(tasks)(logits, labels)
        total = combiner(losses, shared_params=shared, step=step)
        assert total.dim() == 0 and torch.isfinite(total)
        if strategy == "pcgrad":
            pcgrad_backward(losses, shared)
        else:
            total.backward()
        combiner.epoch_end()
    for head in model.heads.values():
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert all(isinstance(v, float) for v in combiner.weights().values())
    return combiner


# 3
def test_every_strategy_backward_reaches_heads():
    for strategy in COMBINERS:
        _run_strategy(strategy)


# 4
def test_sum_single_task():
    losses = {"sentiment": torch.tensor(1.234)}
    assert torch.allclose(build_combiner({"strategy": "sum"}, ["sentiment"])(losses), losses["sentiment"])


def test_fixed_alpha():
    losses = {"sentiment": torch.tensor(2.0), "topic": torch.tensor(4.0)}
    total = build_combiner({"strategy": "fixed", "fixed_alpha": 0.7}, TASKS)(losses)
    assert torch.allclose(total, torch.tensor(0.7 * 2 + 0.3 * 4))


# 5
def test_uncertainty_learnable():
    combiner = build_combiner({"strategy": "uncertainty"}, TASKS)
    params = list(combiner.parameters())
    assert sum(p.numel() for p in params) == 2
    before = combiner.weights()
    opt = torch.optim.SGD(params, lr=0.1)
    combiner({"sentiment": torch.tensor(2.0), "topic": torch.tensor(0.5)}).backward()
    opt.step()
    assert combiner.weights() != before


def test_gradnorm_weights_sum_to_T():
    combiner = _run_strategy("gradnorm")
    assert abs(sum(combiner.weights().values()) - len(TASKS)) < 1e-5
    assert combiner.w.grad is None


def test_dwa_weights_after_two_epochs():
    combiner = build_combiner({"strategy": "dwa"}, TASKS)
    for epoch_losses in [(1.0, 1.0), (0.5, 0.9)]:
        combiner({"sentiment": torch.tensor(epoch_losses[0]), "topic": torch.tensor(epoch_losses[1])})
        combiner.epoch_end()
    w = combiner.weights()
    assert w["topic"] > w["sentiment"]  # topic loss dropped slower -> larger weight
    assert abs(sum(w.values()) - 2) < 1e-5


# 6
def test_pcgrad_projection():
    shared = nn.Parameter(torch.zeros(2))
    g1, g2 = torch.tensor([1.0, 0.2]), torch.tensor([-1.0, 0.5])
    losses = {"a": (shared * g1).sum(), "b": (shared * g2).sum()}
    pcgrad_backward(losses, [shared])
    assert torch.dot(shared.grad, g1) >= -1e-6
    assert torch.dot(shared.grad, g2) >= -1e-6
    assert not torch.allclose(shared.grad, g1 + g2)


def test_pcgrad_extra_loss_not_projected():
    shared = nn.Parameter(torch.zeros(2))
    g1, g2 = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])  # no conflict -> no projection
    losses = {"a": (shared * g1).sum(), "b": (shared * g2).sum()}
    pcgrad_backward(losses, [shared], extra_loss=(shared * 3).sum())
    assert torch.allclose(shared.grad, g1 + g2 + 3)


# 7
def test_mask_tokens():
    tok = FakeTokenizer()
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(5, 50, (100, 100), generator=g)
    input_ids[:, 0], input_ids[:, -1] = 0, 2  # <s> ... </s>
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, 80:] = 0
    input_ids[:, 80:] = 1  # padding

    masked, labels = mask_tokens(input_ids, attention_mask, tok, mlm_prob=0.15, generator=g)
    selected = labels != -100
    candidates = attention_mask.bool() & (input_ids >= 5)

    assert not (selected & ~candidates).any()  # never special tokens or padding
    rate = selected.sum().item() / candidates.sum().item()
    assert abs(rate - 0.15) < 0.03
    assert torch.equal(labels[selected], input_ids[selected])
    assert torch.equal(masked[~selected], input_ids[~selected])
    assert masked.max() < len(tok)
    frac_mask = (masked[selected] == tok.mask_token_id).float().mean().item()
    assert abs(frac_mask - 0.8) < 0.05


def test_mlm_loss():
    logits = torch.randn(2, 5, 50, requires_grad=True)
    labels = torch.full((2, 5), -100)
    labels[0, 1], labels[1, 3] = 7, 9
    expected = F.cross_entropy(torch.stack([logits[0, 1], logits[1, 3]]), torch.tensor([7, 9]))
    assert torch.allclose(mlm_loss(logits, labels), expected)
    zero = mlm_loss(logits, torch.full((2, 5), -100))
    assert zero.item() == 0 and zero.requires_grad


# 8
def test_smart_regularizer():
    torch.manual_seed(0)
    model = ToyModel()
    input_ids, attention_mask, _ = toy_batch()
    clean = model(input_ids=input_ids, attention_mask=attention_mask)
    reg = smart_regularizer(model, input_ids, attention_mask, clean, steps=1)
    assert reg.dim() == 0 and torch.isfinite(reg) and reg.item() >= 0
    reg.backward()
    assert model.encoder.layer.weight.grad is not None and model.encoder.layer.weight.grad.abs().sum() > 0
    assert model.heads["sentiment"].weight.grad.abs().sum() > 0


def test_smart_embeddings_takes_a_real_gradient_step():
    """ Default mode: the gradient w.r.t. the noise exists (the reference code's was always None). """
    grads, real_grad = [], torch.autograd.grad

    def spy(*args, **kwargs):
        out = real_grad(*args, **kwargs)
        grads.extend(out)
        return out

    torch.autograd.grad = spy
    try:
        model = ToyModel()
        input_ids, attention_mask, _ = toy_batch()
        smart_regularizer(model, input_ids, attention_mask, model(input_ids=input_ids, attention_mask=attention_mask))
    finally:
        torch.autograd.grad = real_grad
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)


def test_legacy_token_shift():
    ids = torch.randint(3, 64000, (100, 100), generator=torch.Generator().manual_seed(0))
    shifted = legacy_token_shift(ids, generator=torch.Generator().manual_seed(1))
    diff = shifted - ids
    assert set(diff.unique().tolist()) <= {-1, 0}                     # only id -> id - 1
    assert abs((diff == -1).float().mean().item() - 0.5) < 0.03        # about half of all tokens
    again = legacy_token_shift(ids, generator=torch.Generator().manual_seed(1))
    assert torch.equal(shifted, again)
    specials = legacy_token_shift(torch.tensor([[0, 1, 2] * 500]), generator=torch.Generator().manual_seed(2))
    assert set(specials[0, 0::3].tolist()) == {0}                      # <s>=0 cannot go below 0
    assert set(specials[0, 1::3].tolist()) == {0, 1}                   # <pad> -> <s> (legacy quirk)
    assert set(specials[0, 2::3].tolist()) == {1, 2}                   # </s> -> <pad> (legacy quirk)


def test_smart_token_ids_mode():
    """ Reference reproduction (tag smartref): per task, own token shift, symmetric KL with reduction sum. """
    torch.manual_seed(0)
    model = ToyModel()
    input_ids, attention_mask, _ = toy_batch()
    clean = model(input_ids=input_ids, attention_mask=attention_mask)
    torch.manual_seed(7)
    reg = smart_regularizer(model, input_ids, attention_mask, clean, mode="token_ids")
    torch.manual_seed(7)
    expected = 0.0
    for task in TASKS:
        adv = model(input_ids=legacy_token_shift(input_ids), attention_mask=attention_mask)
        expected = expected + sym_kl_loss(adv[task], clean[task], reduction="sum")
    assert torch.allclose(reg, expected) and torch.isfinite(reg) and reg.item() >= 0
    reg.backward()
    assert model.encoder.layer.weight.grad.abs().sum() > 0 and model.encoder.embed.weight.grad.abs().sum() > 0
    try:
        smart_regularizer(model, input_ids, attention_mask, clean, mode="tokens")
        raise AssertionError("unknown mode not rejected")
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
