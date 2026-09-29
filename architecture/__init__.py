"""
    Model code: MTLModel (shared encoder + task heads) and the legacy-baseline helpers.
    See architecture/model.py.
"""
from architecture.heads import HEADS, build_head
from architecture.legacy import LegacyFlags, convert_legacy_state_dict, load_legacy_checkpoint
from architecture.mlm import MLMHead
from architecture.model import MTLModel, build_model, load_encoder

__all__ = ['MTLModel', 'build_model', 'load_encoder', 'HEADS', 'build_head', 'MLMHead',
           'LegacyFlags', 'convert_legacy_state_dict', 'load_legacy_checkpoint']
