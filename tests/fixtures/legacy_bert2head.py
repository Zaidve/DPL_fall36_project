"""
    Minimal copy of the legacy B1 model (multi-task-bert-master/architecture/bert2head/model.py:
    Linear2HEAD + BertLinear2HEAD), used only by tests/test_model_parity.py to check that a legacy
    checkpoint loads into MTLModel(head='mlp') and gives the same logits.
    Differences from the original: the encoder is passed in (no download, no .to(device)).
"""
import torch.nn as nn


class Linear2HEAD(nn.Module):
    def __init__(self, embedding_dim, dropout=0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.fc_input = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.SiLU()
        self.out_sent = nn.Linear(embedding_dim, 4)
        self.out_clas = nn.Linear(embedding_dim, 10)
        self.dropout_sent = nn.Dropout(dropout)      # defined but never used, as in the original
        self.dropout_clas = nn.Dropout(dropout)

    def forward(self, encoded):
        embedded = self.ln1(self.activation(self.fc_input(encoded)))
        return self.out_sent(embedded), self.out_clas(embedded)


class BertLinear2HEAD(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.BertModel = encoder
        self.linear = Linear2HEAD(encoder.config.hidden_size)

    def forward(self, sentences, attention):
        embedded = self.BertModel(sentences, attention_mask=attention).last_hidden_state[:, 0, :]
        return self.linear(embedded)
