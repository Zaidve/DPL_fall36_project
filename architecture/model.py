"""
    MTLModel: shared pretrained encoder -> pooled vector -> task heads (models_spec.md Part A).

      model = build_model(cfg, {'sentiment': 4, 'topic': 10})
      out = model(input_ids, attention_mask)              # {'sentiment': (B, 4), 'topic': (B, 10)}
      out = model(inputs_embeds=..., attention_mask=...)  # same, for SMART
      out = model(..., return_stage1=True)                # task_aware: also '<task>__stage1' logits
      mlm_logits = model.forward_mlm(ids, mask, positions=labels != -100)   # (n_masked, V), mlm=True only

    One task = single-task model (same class, same code path). Parameter groups for the optimizer and
    the loss combiners: shared_parameters() (encoder), task_parameters() (heads, MLM head),
    last_shared_layer_params() (last encoder layer, for GradNorm). No device moves inside the model.
"""
import torch.nn as nn

from architecture.heads import HEADS, build_head
from architecture.legacy import LegacyFlags
from architecture.mlm import MLMHead
from architecture.pooling import POOLINGS, pool


def load_encoder(pretrained, mlm=False):
    """
      (encoder, lm_head). mlm=False: AutoModel without the unused pooler.
      mlm=True: AutoModelForMaskedLM, so the pretrained LM head (tied to the embeddings) comes with it.
    """
    if mlm:
        from transformers import AutoModelForMaskedLM
        full = AutoModelForMaskedLM.from_pretrained(pretrained)
        lm_head = getattr(full, 'lm_head', None)          # RoBERTa family (PhoBERT, XLM-R, ViSoBERT)
        if lm_head is None:
            lm_head = getattr(full, 'cls', None)          # BERT family
        return full.base_model, lm_head
    from transformers import AutoModel
    try:
        return AutoModel.from_pretrained(pretrained, add_pooling_layer=False), None
    except TypeError:
        return AutoModel.from_pretrained(pretrained), None


class MTLModel(nn.Module):

    def __init__(self, backbone, tasks, head='linear', dropout=0.1, pooling='cls', cross='both',
                 aux_weight=0.5, mlm=False, gradient_checkpointing=False, legacy=None,
                 encoder=None, lm_head=None):
        super().__init__()
        if not tasks:
            raise ValueError('tasks must name at least one task')
        if head not in HEADS:
            raise ValueError(f'unknown head {head!r}; choose from {HEADS}')
        if pooling not in POOLINGS:
            raise ValueError(f'unknown pooling {pooling!r}; choose from {POOLINGS}')
        if head == 'task_aware' and len(tasks) < 2:
            raise ValueError('task_aware heads need 2 tasks; use head linear for a single-task model')

        self.backbone = backbone
        self.tasks = dict(tasks)
        self.head_type = head
        self.pooling = pooling
        self.aux_weight = aux_weight
        self.mlm = mlm
        self.legacy = legacy if isinstance(legacy, LegacyFlags) else LegacyFlags.from_dict(legacy)

        if encoder is None:
            encoder, loaded_lm_head = load_encoder(backbone, mlm=mlm)
            if lm_head is None:
                lm_head = loaded_lm_head
        self.encoder = encoder
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        hidden = self.encoder.config.hidden_size

        self.head = build_head(head, hidden, self.tasks, dropout, cross,
                               no_head_dropout=self.legacy.no_head_dropout)
        self.mlm_head = None
        if mlm:
            if lm_head is not None and not self.legacy.untied_mlm_decoder:
                self.mlm_head = lm_head
            else:
                self.mlm_head = MLMHead(hidden, self.encoder.config.vocab_size,
                                        embeddings=self.encoder.get_input_embeddings(),
                                        legacy_untied=self.legacy.untied_mlm_decoder,
                                        layer_norm_eps=getattr(self.encoder.config, 'layer_norm_eps', 1e-5))

    @classmethod
    def from_encoder(cls, encoder, tasks, **kwargs):
        """ Build around an existing encoder (tests, or a checkpoint's encoder). """
        return cls(backbone=getattr(encoder.config, 'name_or_path', None), tasks=tasks, encoder=encoder, **kwargs)

    # ------------------------------------------------------------------
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, return_stage1=False):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        h = pool(out.last_hidden_state, attention_mask, self.pooling)
        if self.head_type == 'task_aware':
            final, stage1 = self.head(h)
            if return_stage1:
                final = {**final, **{f'{t}__stage1': z for t, z in stage1.items()}}
            return final
        return self.head(h)

    def forward_mlm(self, input_ids, attention_mask, positions=None):
        """ MLM logits: (B, L, V), or (n, V) for a boolean `positions` mask (e.g. labels != -100). """
        if self.mlm_head is None:
            raise RuntimeError('forward_mlm needs a model built with mlm=True')
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if positions is not None:
            hidden = hidden[positions]
        return self.mlm_head(hidden)

    # ------------------------------------------------------------------
    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def shared_parameters(self):
        """ Encoder parameters (shared by all tasks; PCGrad / GradNorm, encoder learning rate). """
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def task_parameters(self):
        """ Everything outside the encoder: task heads and the MLM head (tied weights stay with the encoder). """
        shared = {id(p) for p in self.encoder.parameters()}
        seen, params = set(), []
        for p in self.parameters():
            if p.requires_grad and id(p) not in shared and id(p) not in seen:
                seen.add(id(p))
                params.append(p)
        return params

    def last_shared_layer_params(self):
        """ Parameters of the last encoder layer (GradNorm's W). Falls back to all encoder parameters. """
        layers = getattr(getattr(self.encoder, 'encoder', None), 'layer', None)
        if layers is not None and len(layers):
            return [p for p in layers[-1].parameters() if p.requires_grad]
        return self.shared_parameters()

    def count_parameters(self):
        unique = {id(p): p for p in self.parameters()}.values()
        return {'total': sum(p.numel() for p in unique), 'trainable': sum(p.numel() for p in unique if p.requires_grad)}


def build_model(cfg, num_labels, encoder=None):
    """
      cfg: full config (uses the `model:` section); num_labels: {task: n_classes} in task order.
      The backbone key (phobert / xlmr / visobert) resolves to a Hugging Face id via configs/model/.
      `encoder` skips the download (tests).
    """
    from utils.config import load_model_config
    m = cfg['model']
    pretrained = None if encoder is not None else load_model_config(m['backbone'])['pretrained']
    return MTLModel(pretrained, num_labels,
                    head=m.get('head', 'linear'), dropout=m.get('dropout', 0.1), pooling=m.get('pooling', 'cls'),
                    cross=m.get('cross', 'both'), aux_weight=m.get('aux_weight', 0.5), mlm=m.get('mlm', False),
                    gradient_checkpointing=m.get('gradient_checkpointing', False),
                    legacy=LegacyFlags.from_dict(m.get('legacy')), encoder=encoder)
