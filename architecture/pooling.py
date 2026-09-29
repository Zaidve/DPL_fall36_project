"""
    Sentence vector from the encoder's token vectors.
      cls  : last_hidden_state[:, 0]  (<s> for RoBERTa-style models, [CLS] for BERT-style)
      mean : mean over real tokens (attention_mask == 1)
"""
POOLINGS = ('cls', 'mean')


def pool(hidden, attention_mask=None, mode='cls'):
    """ hidden (B, L, H) -> (B, H). """
    if mode == 'cls':
        return hidden[:, 0]
    if mode == 'mean':
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    raise ValueError(f'unknown pooling {mode!r}; choose from {POOLINGS}')
