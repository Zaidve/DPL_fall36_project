"""
    Training data: processed parquet -> tokenized DataLoaders (train_spec.md contract).

      build_dataloaders(df, tokenizer, tasks, text_col, max_len, batch_size, seed,
                        train_ids=None, num_workers=2) -> {'train', 'validation', 'test'}

      batch = {'input_ids':      LongTensor (B, L),     L = longest text in the batch (<= max_len)
               'attention_mask': LongTensor (B, L),
               'labels':         {task: LongTensor (B,)},
               'idx':            list[str]}             row ids from data/processed (for predictions)

    - Texts are tokenized once per split, truncated to max_len (special tokens included).
    - Train shuffles with a seeded torch.Generator; validation and test keep file order.
    - Tensors stay on the CPU; the trainer moves each batch to the device.
    - Processed data uses split 'val'; the loaders are keyed 'validation' as in the spec.

    Also: load_processed, load_subset_ids, num_labels, load_tokenizer (incl. the ViSoBERT fix),
    and build_run_dataloaders, which does all of it for one (dataset, backbone) run.
    Replaces utils/dataloader.py for training; that file stays for the EDA and preprocessing.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils.common import load_json, make_generator, processed_dir, seed_worker
from utils.config import load_data_config, load_model_config, max_len_for

LOADER_SPLITS = {'train': 'train', 'validation': 'val', 'test': 'test'}   # loader key -> split in the data
ENCODE_BATCH = 1000


# ---------------------------------------------------------------------------
# Processed data
# ---------------------------------------------------------------------------

def load_processed(dataset, folder=None):
    """ data/processed/<dataset>.parquet (columns: id, text, text_clean, text_seg, labels, split, ...). """
    path = Path(folder or processed_dir()) / load_data_config(dataset)['processed_file']
    if not path.exists():
        raise FileNotFoundError(f'{path} not found: run `python -m utils.preprocess` first')
    return pd.read_parquet(path)


def load_subset_ids(dataset, fraction, folder=None):
    """ Train ids of the nested subset for `fraction` (0.1, 0.25, 0.5); None for the full train split. """
    if fraction is None or float(fraction) >= 1.0:
        return None
    path = Path(folder or processed_dir()) / load_data_config(dataset)['subsets_file']
    subsets = load_json(path)
    for key, ids in subsets.items():
        if math.isclose(float(key), float(fraction)):
            return ids
    raise KeyError(f'no subset for fraction {fraction} in {path.name}; available: {sorted(subsets)}')


def num_labels(dataset, tasks):
    """ {task: number of classes}, from the label lists in utils.dataloader. """
    from utils.dataloader import NEU_ESC_LABELS, UIT_VSFC_LABELS
    labels = {'neu-esc': NEU_ESC_LABELS, 'uit-vsfc': UIT_VSFC_LABELS}[dataset]
    return {t: len(labels['classification' if t == 'topic' else t]) for t in tasks}


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

class SentencePieceFairseqTokenizer:
    """
      Tokenizer for models that ship only a sentencepiece model with fairseq-style ids (ViSoBERT).

      transformers 5 converts ViSoBERT's sentencepiece BPE model as if it were XLM-R's Unigram model
      and splits words wrongly, so this class encodes with the original sentencepiece model and maps
      ids the way fairseq / XLM-R do:
        <s>=0  <pad>=1  </s>=2  <unk>=3,  sentencepiece piece i (i >= 3) -> i + 1,  <mask> = n_pieces + 1
      Implements the part of the Hugging Face tokenizer API this project uses.
    """
    OFFSET = 1

    def __init__(self, model_file):
        import sentencepiece as spm
        self.model_file = str(model_file)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_file)
        self.bos_token_id, self.pad_token_id, self.eos_token_id, self.unk_token_id = 0, 1, 2, 3
        self.cls_token_id, self.sep_token_id = self.bos_token_id, self.eos_token_id
        self.mask_token_id = self.sp.get_piece_size() + self.OFFSET
        self.all_special_ids = [0, 1, 2, 3, self.mask_token_id]
        self.model_max_length = 512

    def __len__(self):
        return self.sp.get_piece_size() + self.OFFSET + 1

    def __getstate__(self):                   # picklable for DataLoader workers
        return {'model_file': self.model_file}

    def __setstate__(self, state):
        self.__init__(state['model_file'])

    def _to_ids(self, pieces):
        unk = self.sp.unk_id()
        return [self.unk_token_id if i == unk else i + self.OFFSET for i in pieces]

    def tokenize(self, text):
        return self.sp.encode(text, out_type=str)

    def encode(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = self._to_ids(self.sp.encode(text))
        if truncation and max_length:
            ids = ids[:max(max_length - (2 if add_special_tokens else 0), 0)]
        return [self.bos_token_id, *ids, self.eos_token_id] if add_special_tokens else ids

    def __call__(self, texts, add_special_tokens=True, truncation=False, max_length=None, **_):
        single = isinstance(texts, str)
        ids = [self.encode(t, add_special_tokens, truncation, max_length) for t in ([texts] if single else texts)]
        out = {'input_ids': ids, 'attention_mask': [[1] * len(x) for x in ids]}
        return {k: v[0] for k, v in out.items()} if single else out

    def convert_ids_to_tokens(self, ids):
        special = {0: '<s>', 1: '<pad>', 2: '</s>', 3: '<unk>', self.mask_token_id: '<mask>'}
        return [special.get(i) or self.sp.id_to_piece(i - self.OFFSET) for i in ids]


def load_tokenizer(model_cfg):
    """ Tokenizer for a configs/model/<key>.yaml dict (`tokenizer: auto | sentencepiece_fairseq`). """
    kind = model_cfg.get('tokenizer', 'auto')
    if kind == 'sentencepiece_fairseq':
        from huggingface_hub import hf_hub_download
        return SentencePieceFairseqTokenizer(hf_hub_download(model_cfg['pretrained'], 'sentencepiece.bpe.model'))
    if kind == 'auto':
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_cfg['pretrained'])
    raise ValueError(f'unknown tokenizer kind {kind!r}')


def encode_texts(tokenizer, texts, max_len):
    """ Token ids per text, with special tokens, truncated to max_len. """
    ids = []
    for i in range(0, len(texts), ENCODE_BATCH):
        enc = tokenizer(list(texts[i:i + ENCODE_BATCH]), add_special_tokens=True, truncation=True, max_length=max_len)
        ids += [list(x) for x in enc['input_ids']]
    return ids


# ---------------------------------------------------------------------------
# Dataset, collate, loaders
# ---------------------------------------------------------------------------

class EncodedDataset(Dataset):
    """ Pre-tokenized rows: ids, token ids and one label array per task. """

    def __init__(self, row_ids, input_ids, labels):
        self.row_ids = list(row_ids)
        self.input_ids = input_ids
        self.labels = {t: np.asarray(v, dtype=np.int64) for t, v in labels.items()}

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, i):
        return {'idx': self.row_ids[i], 'input_ids': self.input_ids[i],
                'labels': {t: int(v[i]) for t, v in self.labels.items()}}


class PadCollator:
    """ Pads a batch to its longest sequence (a class, so it pickles for DataLoader workers). """

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, items):
        width = max(len(x['input_ids']) for x in items)
        input_ids = torch.full((len(items), width), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(items), width), dtype=torch.long)
        for row, x in enumerate(items):
            n = len(x['input_ids'])
            input_ids[row, :n] = torch.tensor(x['input_ids'], dtype=torch.long)
            attention_mask[row, :n] = 1
        labels = {t: torch.tensor([x['labels'][t] for x in items], dtype=torch.long) for t in items[0]['labels']}
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels,
                'idx': [x['idx'] for x in items]}


def build_dataloaders(df, tokenizer, tasks, text_col, max_len, batch_size, seed,
                      train_ids=None, num_workers=2):
    """ {'train', 'validation', 'test'} DataLoaders for one dataset (see module docstring). """
    tasks = list(tasks)
    missing = [c for c in ['id', text_col, 'split', *tasks] if c not in df.columns]
    if missing:
        raise KeyError(f'processed data lacks columns {missing}')
    pad_id = tokenizer.pad_token_id
    loaders = {}
    for key, split in LOADER_SPLITS.items():
        part = df[df['split'] == split]
        if key == 'train' and train_ids is not None:
            wanted = set(train_ids)
            part = part[part['id'].isin(wanted)]
            if len(part) != len(wanted):
                raise ValueError(f'{len(wanted) - len(part)} train_ids are not in the train split')
        dataset = EncodedDataset(part['id'], encode_texts(tokenizer, part[text_col].tolist(), max_len),
                                 {t: part[t].to_numpy() for t in tasks})
        is_train = key == 'train'
        loaders[key] = DataLoader(
            dataset, batch_size=batch_size, shuffle=is_train,
            generator=make_generator(seed) if is_train else None,
            num_workers=num_workers, worker_init_fn=seed_worker if num_workers else None,
            persistent_workers=bool(num_workers), collate_fn=PadCollator(pad_id),
            pin_memory=torch.cuda.is_available())
    return loaders


def build_run_dataloaders(dataset, backbone_key, tasks, batch_size, seed, train_fraction=1.0,
                          num_workers=2, folder=None, tokenizer=None):
    """
      Everything for one run: processed data, the backbone's text column and tokenizer, max_len
      and the train subset. Returns (loaders, tokenizer, info).
    """
    model_cfg = load_model_config(backbone_key)
    df = load_processed(dataset, folder)
    tokenizer = tokenizer or load_tokenizer(model_cfg)
    max_len = max_len_for(dataset, backbone_key)
    train_ids = load_subset_ids(dataset, train_fraction, folder)
    loaders = build_dataloaders(df, tokenizer, tasks, model_cfg['text_column'], max_len, batch_size, seed,
                                train_ids=train_ids, num_workers=num_workers)
    info = {'dataset': dataset, 'backbone': backbone_key, 'text_column': model_cfg['text_column'],
            'max_len': max_len, 'train_fraction': float(train_fraction),
            'n_train': len(loaders['train'].dataset), 'n_validation': len(loaders['validation'].dataset),
            'n_test': len(loaders['test'].dataset), 'num_labels': num_labels(dataset, tasks),
            'vocab_size': len(tokenizer)}
    return loaders, tokenizer, info
