"""
    Legacy behaviour of multi-task-bert-master (architecture_baseline.md §6-7).

    LegacyFlags switches individual legacy quirks on, so the same code can run the legacy baseline
    (flags on) and the corrected baseline (flags off). Where each flag takes effect:
      no_head_dropout     §6.6  mlp head: no dropout before the output layers      -> architecture/heads/mlp.py
      untied_mlm_decoder  §6.3  MLM head: SiLU transform + random, untied decoder  -> architecture/mlm.py
      mlm_unmasked_input  §6.1  B2: MLM branch reads the clean (unmasked) pass     -> trainer
      smart_token_shift   §6.2  "SMART" = noise on token ids (tag smartref)        -> utils/loss_function.py

    convert_legacy_state_dict / load_legacy_checkpoint map legacy checkpoint keys
    (BertLinear2HEAD, BertLinear3HEAD, ...) to MTLModel keys (head='mlp').
"""
from dataclasses import asdict, dataclass, fields


@dataclass
class LegacyFlags:
    no_head_dropout: bool = False
    untied_mlm_decoder: bool = False
    mlm_unmasked_input: bool = False
    smart_token_shift: bool = False

    @classmethod
    def from_dict(cls, d=None):
        d = d or {}
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'unknown legacy flags {sorted(unknown)}')
        return cls(**d)

    @classmethod
    def all_on(cls):
        return cls(**{f.name: True for f in fields(cls)})

    def to_dict(self):
        return asdict(self)


def _key_map(sent_task, clas_task, topic_task):
    return [
        ('BertModel.', 'encoder.'),
        ('linear.fc_input2.', 'mlm_head.transform.0.'),
        ('linear.ln2.', 'mlm_head.transform.2.'),
        ('linear.MLM.', 'mlm_head.decoder.'),
        ('linear.fc_input.', 'head.trunk.0.'),
        ('linear.ln1.', 'head.trunk.2.'),
        ('linear.out_sent.', f'head.heads.{sent_task}.'),
        ('linear.out_clas.', f'head.heads.{clas_task}.'),
        ('linear.out_topic.', f'head.heads.{topic_task}.'),
    ]


def convert_legacy_state_dict(state_dict, sent_task='sentiment', clas_task='topic', topic_task='topic'):
    """
      Legacy keys -> MTLModel keys (architecture_baseline.md §7). B1/B2/B3: defaults.
      B4 (ViCTSD): sent_task='constructive', clas_task='toxic', topic_task='topic'.
    """
    mapping = _key_map(sent_task, clas_task, topic_task)
    out = {}
    for key, value in state_dict.items():
        for old, new in mapping:
            if key.startswith(old):
                out[new + key[len(old):]] = value
                break
        else:
            raise KeyError(f'unrecognised legacy key {key!r}')
    return out


def load_legacy_checkpoint(model, state_dict, **task_names):
    """
      Load a legacy state dict into an MTLModel (head='mlp') with strict key checking.
      The legacy encoders came from AutoModel with a pooler that MTLModel does not load or use;
      those pooler weights are dropped. Returns the list of dropped keys.
    """
    converted = convert_legacy_state_dict(state_dict, **task_names)
    own = set(model.state_dict())
    dropped = [k for k in converted if k not in own and k.startswith('encoder.pooler.')]
    for k in dropped:
        converted.pop(k)
    model.load_state_dict(converted, strict=True)
    return dropped
