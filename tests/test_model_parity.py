"""
    Parity with the legacy models (architecture_baseline.md §8). CPU only, no downloads.
    Run:  python tests/test_model_parity.py   (or `pytest tests` if pytest is installed)
"""
import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig
from transformers import logging as hf_logging

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from architecture import LegacyFlags, MTLModel, convert_legacy_state_dict, load_legacy_checkpoint  # noqa: E402
from architecture.heads import MLPHeads  # noqa: E402
from architecture.mlm import MLMHead  # noqa: E402
from fixtures.legacy_bert2head import BertLinear2HEAD  # noqa: E402

hf_logging.set_verbosity_error()
TASKS = {'sentiment': 4, 'topic': 10}


def n_params(module):
    return sum(p.numel() for p in {id(p): p for p in module.parameters()}.values())


def tiny_encoder(seed=0):
    torch.manual_seed(seed)
    return AutoModel.from_config(BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1,
                                            num_attention_heads=2, intermediate_size=64))


# §8.1 head parameter counts with d = 768 (architecture_baseline.md §1)
def test_head_parameter_counts():
    b1 = MLPHeads(768, TASKS)
    assert n_params(b1) == 602_894
    legacy_mlm = MLMHead(768, 64001, legacy_untied=True)                  # legacy vocab, untied
    assert n_params(b1) + n_params(legacy_mlm) == 50_411_791              # B3
    assert n_params(MLPHeads(768, {'sentiment': 4})) + n_params(legacy_mlm) == 50_404_101   # B2


# §8.3 trunk order Linear -> SiLU -> LayerNorm
def test_trunk_order():
    torch.manual_seed(0)
    head = MLPHeads(16, TASKS, no_head_dropout=True).eval()
    h = torch.randn(5, 16)
    lin, ln = head.trunk[0], head.trunk[2]
    manual = nn.functional.layer_norm(nn.functional.silu(lin(h)), (16,), ln.weight, ln.bias, ln.eps)
    out = head(h)
    assert torch.allclose(out['topic'], head.heads['topic'](manual), atol=1e-6)


# §8.5 a legacy B1 state dict loads strictly and gives the same logits
def test_legacy_b1_checkpoint_parity():
    legacy = BertLinear2HEAD(tiny_encoder(0)).eval()
    torch.manual_seed(1)
    for p in legacy.linear.parameters():                  # non-trivial head weights
        nn.init.normal_(p, std=0.2)
    ours = MTLModel.from_encoder(tiny_encoder(123), TASKS, head='mlp', legacy=LegacyFlags(no_head_dropout=True)).eval()
    dropped = load_legacy_checkpoint(ours, legacy.state_dict())
    assert dropped == []                                  # tiny BERT from_config keeps its pooler on both sides
    ids = torch.randint(5, 100, (3, 9))
    mask = torch.ones_like(ids)
    mask[0, 6:] = 0
    with torch.no_grad():
        sent, clas = legacy(ids, mask)
        out = ours(ids, mask)
    assert torch.allclose(sent, out['sentiment'], atol=1e-6) and torch.allclose(clas, out['topic'], atol=1e-6)


def test_legacy_pooler_keys_dropped():
    """ Real legacy checkpoints have encoder pooler weights; MTLModel loads encoders without a pooler. """
    legacy = BertLinear2HEAD(tiny_encoder(0))
    enc = AutoModel.from_config(tiny_encoder(0).config, add_pooling_layer=False)
    ours = MTLModel.from_encoder(enc, TASKS, head='mlp')
    dropped = load_legacy_checkpoint(ours, legacy.state_dict())
    assert dropped and all(k.startswith('encoder.pooler.') for k in dropped)


def test_key_map():
    sd = {'BertModel.embeddings.x': 1, 'linear.fc_input.weight': 2, 'linear.ln1.bias': 3,
          'linear.out_sent.weight': 4, 'linear.out_clas.bias': 5, 'linear.fc_input2.weight': 6,
          'linear.ln2.weight': 7, 'linear.MLM.bias': 8}
    assert convert_legacy_state_dict(sd) == {
        'encoder.embeddings.x': 1, 'head.trunk.0.weight': 2, 'head.trunk.2.bias': 3,
        'head.heads.sentiment.weight': 4, 'head.heads.topic.bias': 5, 'mlm_head.transform.0.weight': 6,
        'mlm_head.transform.2.weight': 7, 'mlm_head.decoder.bias': 8}
    b4 = convert_legacy_state_dict({'linear.out_sent.bias': 0, 'linear.out_clas.bias': 1, 'linear.out_topic.bias': 2},
                                   sent_task='constructive', clas_task='toxic', topic_task='topic')
    assert set(b4) == {'head.heads.constructive.bias', 'head.heads.toxic.bias', 'head.heads.topic.bias'}
    try:
        convert_legacy_state_dict({'something.else': 0})
        raise AssertionError('unknown key not rejected')
    except KeyError:
        pass


# §8.6 / §8.7 MLM vocab from the encoder; tied unless legacy
def test_mlm_vocab_and_tying():
    m = MTLModel.from_encoder(tiny_encoder(), TASKS, mlm=True)
    assert m.mlm_head.decoder.out_features == m.encoder.config.vocab_size
    assert m.mlm_head.decoder.weight is m.encoder.embeddings.word_embeddings.weight
    legacy = MTLModel.from_encoder(tiny_encoder(), TASKS, mlm=True, legacy=LegacyFlags(untied_mlm_decoder=True))
    assert legacy.mlm_head.decoder.weight is not legacy.encoder.embeddings.word_embeddings.weight


def test_legacy_flags():
    assert LegacyFlags.all_on().to_dict() == {'no_head_dropout': True, 'untied_mlm_decoder': True,
                                              'mlm_unmasked_input': True, 'smart_token_shift': True}
    assert LegacyFlags.from_dict(None) == LegacyFlags()
    try:
        LegacyFlags.from_dict({'typo_flag': True})
        raise AssertionError('unknown flag not rejected')
    except ValueError:
        pass
    assert isinstance(MLPHeads(8, TASKS, no_head_dropout=True).dropout, nn.Identity)
    assert isinstance(MLPHeads(8, TASKS).dropout, nn.Dropout)


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
