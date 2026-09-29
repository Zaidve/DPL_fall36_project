"""
    Tests for utils/dataset.py. CPU only, no downloads (fake tokenizer; a tiny sentencepiece model
    is trained inside the test). One optional test uses the cached ViSoBERT model if it is present.
    Run:  python tests/test_dataset.py   (or `pytest tests` if pytest is installed)
"""
import os
import pickle
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import (PadCollator, SentencePieceFairseqTokenizer, build_dataloaders,  # noqa: E402
                           load_processed, load_subset_ids, num_labels)

TASKS = ['sentiment', 'topic']


class FakeTokenizer:
    """ Whitespace tokens -> ids; <s>=0 <pad>=1 </s>=2, words from 3. """
    pad_token_id = 1

    def __init__(self):
        self.vocab = {}

    def __len__(self):
        return 3 + len(self.vocab)

    def __call__(self, texts, add_special_tokens=True, truncation=False, max_length=None, **_):
        out = []
        for t in texts:
            ids = [self.vocab.setdefault(w, 3 + len(self.vocab)) for w in t.split()]
            if truncation and max_length:
                ids = ids[:max_length - 2]
            out.append([0, *ids, 2] if add_special_tokens else ids)
        return {'input_ids': out, 'attention_mask': [[1] * len(x) for x in out]}


def synthetic(n_train=40, n_val=9, n_test=11):
    rows = []
    for split, n in (('train', n_train), ('val', n_val), ('test', n_test)):
        for i in range(n):
            words = ' '.join(f'w{j}' for j in range(1 + (i * 7) % 15))
            rows.append({'id': f'uit-vsfc-{split}-{i}', 'text_clean': words, 'text_seg': words,
                         'sentiment': i % 3, 'topic': i % 4, 'split': split})
    df = pd.DataFrame(rows)
    df['split'] = pd.Categorical(df['split'], categories=['train', 'val', 'test'], ordered=True)
    return df


def loaders_for(df, seed=42, **kw):
    args = dict(tasks=TASKS, text_col='text_clean', max_len=10, batch_size=4, seed=seed, num_workers=0)
    args.update(kw)
    return build_dataloaders(df, FakeTokenizer(), **args)


def all_ids(loader):
    return [i for b in loader for i in b['idx']]


def test_keys_and_batch_contract():
    loaders = loaders_for(synthetic())
    assert set(loaders) == {'train', 'validation', 'test'}
    b = next(iter(loaders['train']))
    assert set(b) == {'input_ids', 'attention_mask', 'labels', 'idx'}
    assert b['input_ids'].dtype == b['attention_mask'].dtype == torch.long
    assert set(b['labels']) == set(TASKS) and all(v.shape == (4,) and v.dtype == torch.long for v in b['labels'].values())
    assert isinstance(b['idx'], list) and all(isinstance(i, str) for i in b['idx'])
    assert b['input_ids'].device.type == 'cpu'


def test_per_batch_padding_and_truncation():
    for max_len in (10, 20):                         # texts have 3..17 tokens: 10 truncates, 20 does not
        widths = set()
        for b in loaders_for(synthetic(), max_len=max_len)['train']:
            lengths = b['attention_mask'].sum(1)
            assert b['input_ids'].shape[1] == int(lengths.max()) <= max_len   # padded to the longest in the batch
            pads = b['attention_mask'] == 0
            assert torch.all(b['input_ids'][pads] == 1)                        # pad id
            assert torch.all(b['input_ids'][torch.arange(len(lengths)), lengths - 1] == 2)   # </s> kept
            widths.add(b['input_ids'].shape[1])
        if max_len == 10:
            assert max(widths) == 10
        else:
            assert len(widths) > 1 and max(widths) <= 17


def test_seeded_shuffle_and_eval_order():
    df = synthetic()
    a, b, c = loaders_for(df, seed=1), loaders_for(df, seed=1), loaders_for(df, seed=2)
    assert all_ids(a['train']) == all_ids(b['train'])
    assert all_ids(c['train']) != all_ids(loaders_for(df, seed=1)['train'])
    fresh = loaders_for(df, seed=1)['train']
    first, second = all_ids(fresh), all_ids(fresh)                      # two epochs: new order, same rows
    assert first != second and sorted(first) == sorted(second)
    for key, split in (('validation', 'val'), ('test', 'test')):
        assert all_ids(a[key]) == df.loc[df['split'] == split, 'id'].tolist()   # file order, same ids


def test_train_ids_and_tasks_subset():
    df = synthetic()
    keep = [f'uit-vsfc-train-{i}' for i in range(0, 40, 4)]
    loaders = loaders_for(df, train_ids=keep, tasks=['sentiment'])
    assert sorted(all_ids(loaders['train'])) == sorted(keep)
    assert len(all_ids(loaders['validation'])) == 9 and len(all_ids(loaders['test'])) == 11
    assert set(next(iter(loaders['train']))['labels']) == {'sentiment'}
    try:
        loaders_for(df, train_ids=['uit-vsfc-train-0', 'not-an-id'])
        raise AssertionError('unknown train id not rejected')
    except ValueError:
        pass


def test_labels_match_rows():
    df = synthetic()
    by_id = df.set_index('id')
    for b in loaders_for(df)['train']:
        for task in TASKS:
            assert b['labels'][task].tolist() == by_id.loc[b['idx'], task].tolist()


def test_collator_pickles_for_workers():
    collate = pickle.loads(pickle.dumps(PadCollator(1)))
    b = collate([{'idx': 'a', 'input_ids': [0, 5, 2], 'labels': {'sentiment': 1}},
                 {'idx': 'b', 'input_ids': [0, 2], 'labels': {'sentiment': 0}}])
    assert b['input_ids'].tolist() == [[0, 5, 2], [0, 2, 1]] and b['attention_mask'].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_real_processed_data_and_subsets():
    for ds in ('neu-esc', 'uit-vsfc'):
        df = load_processed(ds)
        assert {'id', 'text_clean', 'text_seg', 'sentiment', 'topic', 'split'} <= set(df.columns)
        ids = load_subset_ids(ds, 0.1)
        train = set(df.loc[df['split'] == 'train', 'id'])
        assert ids and set(ids) <= train and abs(len(ids) - 0.1 * len(train)) < 50
        assert set(load_subset_ids(ds, 0.25)) >= set(ids)
        assert load_subset_ids(ds, 1.0) is None and load_subset_ids(ds, None) is None
    try:
        load_subset_ids('uit-vsfc', 0.3)
        raise AssertionError('unknown fraction not rejected')
    except KeyError:
        pass
    assert num_labels('neu-esc', TASKS) == {'sentiment': 4, 'topic': 10}
    assert num_labels('uit-vsfc', ['topic']) == {'topic': 4}


def _tiny_spm(folder):
    import sentencepiece as spm
    corpus = Path(folder) / 'corpus.txt'
    corpus.write_text('\n'.join(['sinh viên học rất chăm chỉ', 'thầy dạy rất hay', 'phòng học nóng quá',
                                 'giáo trình đầy đủ'] * 50), encoding='utf-8')
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(Path(folder) / 'm'), vocab_size=40,
                                   model_type='bpe', minloglevel=2)
    return Path(folder) / 'm.model'


def test_sentencepiece_fairseq_ids():
    import sentencepiece as spm
    with tempfile.TemporaryDirectory() as tmp:
        model = _tiny_spm(tmp)
        tok = SentencePieceFairseqTokenizer(model)
        sp = spm.SentencePieceProcessor(model_file=str(model))
        n = sp.get_piece_size()
        text = 'thầy dạy rất hay'
        ids = tok.encode(text)
        assert ids == [0, *[i + 1 for i in sp.encode(text)], 2]               # fairseq offset, <s> ... </s>
        assert tok.encode('xyz ⍼', add_special_tokens=False).count(3) >= 1      # unknown -> <unk>=3
        assert tok.mask_token_id == n + 1 and len(tok) == n + 2 and tok.pad_token_id == 1
        assert tok.convert_ids_to_tokens(ids)[1:-1] == sp.encode(text, out_type=str)
        cut = tok([text], truncation=True, max_length=4)['input_ids'][0]
        assert len(cut) == 4 and cut[0] == 0 and cut[-1] == 2
        clone = pickle.loads(pickle.dumps(tok))
        assert clone.encode(text) == ids


def test_visobert_tokenizer_if_cached():
    """ Uses the cached uitnlp/visobert sentencepiece model; skipped when it is not downloaded. """
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download('uitnlp/visobert', 'sentencepiece.bpe.model', local_files_only=True)
    except Exception:
        print('   (skipped: ViSoBERT not in the local Hugging Face cache)')
        return
    tok = SentencePieceFairseqTokenizer(path)
    assert tok.tokenize('thầy dạy rất hay') == ['▁thầy', '▁dạy', '▁rất', '▁hay']
    assert len(tok) == 15002 and tok.mask_token_id == 15001


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
