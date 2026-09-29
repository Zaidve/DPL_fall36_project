"""
    Tests for architecture/ (models_spec.md A5). CPU only, no downloads: a tiny random BERT encoder.
    Run:  python tests/test_model.py   (or `pytest tests` if pytest is installed)
"""
import os
import sys

import torch
from transformers import AutoModel, BertConfig
from transformers import logging as hf_logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture import LegacyFlags, MTLModel, build_model  # noqa: E402
from utils.loss_function import (TaskLoss, build_combiner, mlm_loss, pcgrad_backward,  # noqa: E402
                                 smart_regularizer)

hf_logging.set_verbosity_error()
TASKS = {'sentiment': 4, 'topic': 10}


def tiny_encoder(seed=0):
    torch.manual_seed(seed)
    cfg = BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64)
    return AutoModel.from_config(cfg)


def model(tasks=TASKS, seed=0, **kw):
    return MTLModel.from_encoder(tiny_encoder(seed), tasks, **kw)


def batch(B=3, L=7, pad_last=2):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(5, 100, (B, L), generator=g)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[0, L - pad_last:] = 0
    ids[0, L - pad_last:] = 0
    return ids, mask


# 1
def test_linear_two_tasks():
    ids, mask = batch()
    out = model()(ids, mask)
    assert set(out) == {'sentiment', 'topic'}
    assert out['sentiment'].shape == (3, 4) and out['topic'].shape == (3, 10)


# 2
def test_single_task():
    ids, mask = batch()
    for head in ('linear', 'mlp'):
        out = model({'topic': 10}, head=head)(ids, mask)
        assert list(out) == ['topic'] and out['topic'].shape == (3, 10)


# 3
def test_task_aware_outputs_and_gradients():
    m = model(head='task_aware')
    ids, mask = batch()
    out = m(ids, mask, return_stage1=True)
    assert set(out) == {'sentiment', 'topic', 'sentiment__stage1', 'topic__stage1'}
    assert out['sentiment'].shape == out['sentiment__stage1'].shape == (3, 4)
    assert out['topic'].shape == out['topic__stage1'].shape == (3, 10)
    assert set(m(ids, mask)) == {'sentiment', 'topic'}                   # stage 1 only on request
    sum(v.sum() for k, v in out.items() if '__' not in k).backward()     # final logits only
    for name in ('label_emb.sentiment', 'label_emb.topic', 'gate.sentiment', 'gate.topic',
                 'stage1.sentiment', 'stage1.topic'):
        grad = m.head.get_submodule(name).weight.grad
        assert grad is not None and grad.abs().sum() > 0, name           # p is not detached
    assert m.encoder.embeddings.word_embeddings.weight.grad.abs().sum() > 0


# 4
def test_task_aware_needs_two_tasks():
    try:
        model({'sentiment': 4}, head='task_aware')
        raise AssertionError('single-task task_aware not rejected')
    except ValueError:
        pass


# 5
def test_cross_direction():
    m = model(head='task_aware', cross='sent_from_topic').eval()
    ids, mask = batch()
    out = m(ids, mask, return_stage1=True)
    assert torch.equal(out['topic'], out['topic__stage1'])
    assert not torch.allclose(out['sentiment'], out['sentiment__stage1'])
    assert set(m.head.gate) == {'sentiment'} and set(m.head.label_emb) == {'topic'}
    m2 = model(head='task_aware', cross='topic_from_sent').eval()
    out2 = m2(ids, mask, return_stage1=True)
    assert torch.equal(out2['sentiment'], out2['sentiment__stage1'])


# 6
def test_inputs_embeds_path_matches_input_ids():
    for head in ('linear', 'task_aware'):
        m = model(head=head).eval()
        ids, mask = batch()
        with torch.no_grad():
            a = m(input_ids=ids, attention_mask=mask)
            b = m(inputs_embeds=m.get_input_embeddings()(ids), attention_mask=mask)
        for t in TASKS:
            assert torch.allclose(a[t], b[t], atol=1e-5), (head, t)


# 7
def test_parameter_groups_disjoint_and_complete():
    for kw in ({}, {'head': 'task_aware'}, {'head': 'mlp', 'mlm': True}):
        m = model(**kw)
        shared, task = m.shared_parameters(), m.task_parameters()
        ids_s, ids_t = {id(p) for p in shared}, {id(p) for p in task}
        assert not ids_s & ids_t, kw
        assert ids_s | ids_t == {id(p) for p in m.parameters() if p.requires_grad}, kw
        assert len(ids_t) == len(task)                                     # no duplicates
    m = model()
    last_ids = {id(p) for p in m.last_shared_layer_params()}
    assert last_ids < {id(p) for p in m.shared_parameters()}
    assert last_ids == {id(p) for p in m.encoder.encoder.layer[-1].parameters()}


# 8
def test_same_seed_same_head_init():
    def head_weights(seed):
        enc = tiny_encoder(0)
        torch.manual_seed(seed)
        return torch.cat([p.detach().flatten() for p in MTLModel.from_encoder(enc, TASKS, head='task_aware').task_parameters()])
    assert torch.equal(head_weights(5), head_weights(5))
    assert not torch.equal(head_weights(5), head_weights(6))


def test_mean_pooling_ignores_padding():
    m = model(pooling='mean').eval()
    ids, mask = batch(B=1, L=5, pad_last=0)
    padded_ids = torch.cat([ids, torch.zeros(1, 3, dtype=torch.long)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(1, 3, dtype=torch.long)], dim=1)
    with torch.no_grad():
        a, b = m(ids, mask), m(padded_ids, padded_mask)
    assert all(torch.allclose(a[t], b[t], atol=1e-5) for t in TASKS)


def test_mlm_head_tied_untied_and_positions():
    ids, mask = batch()
    m = model(mlm=True)
    assert m.mlm_head.decoder.weight is m.get_input_embeddings().weight          # tied by default
    assert m.forward_mlm(ids, mask).shape == (3, 7, 100)                         # vocab from the encoder
    labels = torch.full_like(ids, -100)
    labels[1, 2], labels[2, 4] = ids[1, 2], ids[2, 4]
    logits = m.forward_mlm(ids, mask, positions=labels != -100)
    assert logits.shape == (2, 100)
    loss = mlm_loss(logits, labels[labels != -100])
    assert torch.isfinite(loss)
    legacy = model(mlm=True, legacy=LegacyFlags(untied_mlm_decoder=True))
    assert legacy.mlm_head.decoder.weight is not legacy.get_input_embeddings().weight
    assert isinstance(legacy.mlm_head.transform[1], torch.nn.SiLU)
    try:
        model().forward_mlm(ids, mask)
        raise AssertionError('forward_mlm without mlm head not rejected')
    except RuntimeError:
        pass


def test_losses_integration():
    """ The model plugs into utils/loss_function.py: SMART, combiners, PCGrad. """
    ids, mask = batch()
    labels = {'sentiment': torch.tensor([0, 1, 3]), 'topic': torch.tensor([2, 9, 0])}
    m = model(head='task_aware')
    out = m(ids, mask)
    losses = TaskLoss(list(TASKS))(out, labels)
    reg = smart_regularizer(m, ids, mask, {t: out[t] for t in TASKS})
    assert torch.isfinite(reg) and reg >= 0
    (build_combiner({'strategy': 'uncertainty'}, list(TASKS))(losses) + 0.02 * reg).backward()
    assert all(p.grad is not None for p in m.last_shared_layer_params())

    m.zero_grad()
    out = m(ids, mask)
    total = build_combiner({'strategy': 'gradnorm'}, list(TASKS))(TaskLoss(list(TASKS))(out, labels),
                                                                  shared_params=m.last_shared_layer_params())
    total.backward()
    m.zero_grad()
    pcgrad_backward(TaskLoss(list(TASKS))(m(ids, mask), labels), m.shared_parameters())
    assert m.head.stage2['topic'].weight.grad is not None


def test_build_model_from_config():
    cfg = {'model': {'backbone': 'phobert', 'head': 'task_aware', 'dropout': 0.2, 'pooling': 'mean',
                     'cross': 'topic_from_sent', 'aux_weight': 0.3, 'mlm': False,
                     'legacy': {'no_head_dropout': True}}}
    m = build_model(cfg, TASKS, encoder=tiny_encoder())
    assert m.head_type == 'task_aware' and m.pooling == 'mean' and m.aux_weight == 0.3
    assert m.head.cross == 'topic_from_sent' and m.legacy.no_head_dropout and m.head.dropout.p == 0.2
    counts = m.count_parameters()
    assert counts['total'] == counts['trainable'] > 0
    for bad in ({'head': 'cnn'}, {'pooling': 'max'}):
        try:
            build_model({'model': {**cfg['model'], **bad}}, TASKS, encoder=tiny_encoder())
            raise AssertionError(f'{bad} not rejected')
        except ValueError:
            pass


if __name__ == '__main__':
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith('test_')]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS  {name}')
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f'FAIL  {name}: {type(e).__name__}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
