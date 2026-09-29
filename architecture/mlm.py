"""
    MLM head, only for the optional MLM ablation (models_spec A4, architecture_baseline §6.3-6.5).

    Preferred: the backbone's own pretrained LM head (`lm_head` of AutoModelForMaskedLM, decoder tied
    to the input embeddings); MTLModel takes it directly when it loads the backbone with mlm=True.
    MLMHead below is the fallback (and the legacy variant):
      default : Linear(H, H) -> GELU -> LayerNorm -> decoder tied to the input embeddings
      legacy  : Linear(H, H) -> SiLU -> LayerNorm -> untied, randomly initialised decoder (§6.3)
    The vocabulary size always comes from the encoder (never hard-coded; legacy used 64001).
    Callers can pass only the masked positions (§6.5) to avoid a (B, L, V) logits tensor.
"""
import torch.nn as nn


class MLMHead(nn.Module):

    def __init__(self, hidden_size, vocab_size, embeddings=None, legacy_untied=False, layer_norm_eps=1e-5):
        super().__init__()
        self.transform = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                       nn.SiLU() if legacy_untied else nn.GELU(),
                                       nn.LayerNorm(hidden_size, eps=layer_norm_eps))
        self.decoder = nn.Linear(hidden_size, vocab_size)
        if embeddings is not None and not legacy_untied:
            if embeddings.weight.shape != self.decoder.weight.shape:
                raise ValueError(f'cannot tie decoder {tuple(self.decoder.weight.shape)} '
                                 f'to embeddings {tuple(embeddings.weight.shape)}')
            self.decoder.weight = embeddings.weight

    def forward(self, hidden):
        """ (..., H) -> (..., vocab) """
        return self.decoder(self.transform(hidden))
