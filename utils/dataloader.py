import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')

# Every loader returns a DataFrame with the same columns, so the training scripts
# can treat both datasets the same way:
#   text           : the sentence
#   sentiment      : head 1 label
#   classification : head 2 label (topic)
UIT_VSFC_LABELS = {
    'sentiment': ['negative', 'neutral', 'positive'],
    'classification': ['lecturer', 'training_program', 'facility', 'others'],
}

# NEU-ESC ships without a label description. These names were inferred by
# reading samples, so check them against the source paper before reporting.
NEU_ESC_LABELS = {
    'sentiment': ['neutral', 'positive', 'negative', 'toxic'],
    'classification': [str(i) for i in range(10)],
}


# ---------------------------------------------------------------------------
# Raw data loaders
# ---------------------------------------------------------------------------

def _read_lines(path):
    with open(path, encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]


def load_uit_vsfc(split, root=DATASET_ROOT):
    """
      split: 'train' | 'dev' | 'val' | 'test'  ('val' is an alias of 'dev')
    """
    split = 'dev' if split == 'val' else split
    folder = os.path.join(root, 'uit-vsfc', split)
    df = pd.DataFrame({
        'text': _read_lines(os.path.join(folder, 'sents.txt')),
        'sentiment': [int(x) for x in _read_lines(os.path.join(folder, 'sentiments.txt'))],
        'classification': [int(x) for x in _read_lines(os.path.join(folder, 'topics.txt'))],
    })
    return df


def load_neu_esc(split, root=DATASET_ROOT):
    """
      split: 'train' | 'val' | 'dev' | 'test'  ('dev' is an alias of 'val')
    """
    split = 'val' if split == 'dev' else split
    df = pd.read_csv(os.path.join(root, 'neu-esc', f'{split}_set.csv'))
    df = df.dropna(subset=['text', 'sentiment', 'classification']).reset_index(drop=True)
    df['text'] = df['text'].astype(str)
    df['sentiment'] = df['sentiment'].astype(int)
    df['classification'] = df['classification'].astype(int)
    return df


def load_dataset(name, split, word_segment=False, root=DATASET_ROOT):
    """
      name: 'uit-vsfc' | 'neu-esc'
      word_segment: set True for PhoBERT (vinai/phobert-*), which expects
                    word-segmented input such as `sinh_viên`.
    """
    loaders = {'uit-vsfc': load_uit_vsfc, 'neu-esc': load_neu_esc}
    df = loaders[name](split, root)
    if word_segment:
        df['text'] = segment_words(df['text'])
    return df


def segment_words(sentences):
    """
      Vietnamese word segmentation for PhoBERT (`pip install underthesea`).
      Segmenting a whole dataset is slow, so cache the result to a
      *_processed.csv the first time instead of redoing it every run.
    """
    from underthesea import word_tokenize
    return [word_tokenize(sent, format='text') for sent in sentences]


def compute_class_weights(labels, num_classes=None):
    """
      Inverse-frequency class weights for nn.CrossEntropyLoss(weight=...).
      Both datasets are heavily imbalanced (UIT neutral ~4%, NEU toxic ~2.5%).
    """
    labels = np.asarray(labels, dtype=np.int64)
    num_classes = num_classes or int(labels.max()) + 1
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float)


# ---------------------------------------------------------------------------
# Tokenize and build DataLoaders
# ---------------------------------------------------------------------------

class CreateDataset:
  """
    This class will tokenize and create a dataset for 1-2 heads models.
    Each batch is (input_ids, attention_mask, labels1, labels2).
  """
  def __init__(self, sentences, labels1, labels2, model_name, batch_size=32, max_length=128, shuffle=True):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    self.batch_size = batch_size
    self.model_name = model_name
    self.sentences = [str(s) for s in sentences]
    self.labels = [labels1, labels2]
    self.max_length = max_length
    self.shuffle = shuffle
    self.device = DEVICE

  def encoder_generator(self):
    encoded = self.tokenizer(self.sentences,
                             add_special_tokens=True,
                             max_length=self.max_length,
                             padding='max_length',
                             truncation=True,
                             return_attention_mask=True,
                             return_tensors='pt')
    self.input_ids = encoded['input_ids'].to(DEVICE)
    self.attention_masks = encoded['attention_mask'].to(DEVICE)
    self.label_tensors = [torch.tensor(np.asarray(lb), dtype=torch.long).to(DEVICE) for lb in self.labels]
    self.sent_index = torch.arange(len(self.sentences)).to(DEVICE)

  def todataloader(self):
    self.encoder_generator()

    self.dataset = TensorDataset(self.input_ids, self.attention_masks, *self.label_tensors)
    self.data_loader = DataLoader(self.dataset,
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle)
    return self.data_loader


class Create3HEADDataset(CreateDataset):
  """
    Same as `CreateDataset` but for 3 heads models.
    Each batch is (input_ids, attention_mask, labels1, labels2, labels3).
  """
  def __init__(self, sentences, labels1, labels2, labels3, model_name, batch_size=32, max_length=128, shuffle=True):
    super().__init__(sentences, labels1, labels2, model_name, batch_size, max_length, shuffle)
    self.labels = [labels1, labels2, labels3]


def build_dataloaders(name, model_name, batch_size=32, max_length=None, word_segment=None, root=DATASET_ROOT):
    """
      One call that returns (train_loader, val_loader, test_loader) for a dataset.
      Only the train loader is shuffled, so evaluation order is reproducible.

      max_length defaults: 64 for UIT-VSFC, 128 for NEU-ESC (covers ~95% of texts).
      word_segment defaults to True for PhoBERT models.
    """
    if max_length is None:
        max_length = 64 if name == 'uit-vsfc' else 128
    if word_segment is None:
        word_segment = 'phobert' in model_name.lower()

    loaders = []
    for split in ['train', 'val', 'test']:
        df = load_dataset(name, split, word_segment=word_segment, root=root)
        loaders.append(CreateDataset(df['text'], df['sentiment'], df['classification'], model_name,
                                     batch_size=batch_size, max_length=max_length,
                                     shuffle=(split == 'train')).todataloader())
    return tuple(loaders)


# ---------------------------------------------------------------------------
# Masked language modelling
# ---------------------------------------------------------------------------

# `DataCollatorForLanguageModeling` from `transformers` does not fit a customized
# model, so `DataCollatorHandMade` does the masking by hand.
#
# `random_label` picks `mlm_prob` of the real tokens (never special tokens or
# padding) in each sentence and applies the usual rule:
#   - 80% of the time, replace the token with the mask token
#   - 10% of the time, replace the token with a random token
#   - 10% of the time, keep the token unchanged
# Labels are -100 everywhere except the chosen positions.
class DataCollatorHandMade:

    def __init__(self, model_name, mlm_prob=0.3):
      self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
      self.mask_token_id = self.tokenizer.mask_token_id
      self.vocab_size = len(self.tokenizer)
      self.special_ids = torch.tensor(sorted(set(self.tokenizer.all_special_ids)))
      self.mlm_prob = mlm_prob

    def random_label(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        device = input_ids.device
        mlm_inputs = input_ids.clone()
        labels = torch.full_like(input_ids, -100)
        total_mask = 0

        candidates = attention_mask.bool() & ~torch.isin(input_ids, self.special_ids.to(device))

        for i in range(input_ids.shape[0]):
          positions = candidates[i].nonzero(as_tuple=True)[0]
          num_mask = max(1, int(len(positions) * self.mlm_prob)) if len(positions) > 0 else 0
          if num_mask == 0:
            continue
          total_mask += num_mask

          mask_pos = positions[torch.randperm(len(positions), device=device)[:num_mask]]
          labels[i, mask_pos] = input_ids[i, mask_pos]

          lucky = torch.rand(num_mask, device=device)
          to_mask = mask_pos[lucky < 0.8]
          to_random = mask_pos[(lucky >= 0.8) & (lucky < 0.9)]
          mlm_inputs[i, to_mask] = self.mask_token_id
          mlm_inputs[i, to_random] = torch.randint(0, self.vocab_size, (len(to_random),), device=device)

        return mlm_inputs.to(DEVICE), labels.to(DEVICE), total_mask


def label_for_mlm(result, mlm_labels):
    """
      Keep only the masked positions so the output fits nn.CrossEntropyLoss.
        result     : (batch, seq_len, vocab_size) MLM logits
        mlm_labels : (batch, seq_len), -100 where there is nothing to predict
      Returns y_pred (n_masked, vocab_size) and labels (n_masked,).
    """
    keep = mlm_labels != -100
    y_pred = result[keep].to(DEVICE)
    labels = mlm_labels[keep].long().to(DEVICE)
    return y_pred, labels
