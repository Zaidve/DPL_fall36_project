"""
    Tests for utils/preprocess.py. CPU only, no downloads, fake or 'none' segmenter.
    Run:  python tests/test_preprocess.py   (or `pytest tests` if pytest is installed)
"""
import json
import os
import sys
import tempfile
import unicodedata

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocess import (DEFAULT_CFG, OUTPUT_COLUMNS, add_ids, add_segmented_column,  # noqa: E402
                              clean_whitespace, dedup_train, flag_leakage, load_config, load_segmenter,
                              make_train_subsets, normalize_series, normalize_teencode, normalize_text,
                              normalize_tone_marks, normalize_unicode, preprocess_dataset,
                              reduce_repeated_chars, replace_masked_names, replace_urls, segment_texts)


def fake_segmenter(texts):
    return [t.replace('sinh viên', 'sinh_viên').replace('giảng viên', 'giảng_viên') for t in texts]


fake_segmenter.used = 'fake'


def table(rows):
    """ rows: (text_clean, sentiment, topic, split); dataset 'uit-vsfc'. """
    df = pd.DataFrame(rows, columns=['text_clean', 'sentiment', 'topic', 'split'])
    df['text'] = df['text_clean']
    df['dataset'] = 'uit-vsfc'
    df['sentiment_name'] = df['sentiment'].map({0: 'negative', 1: 'neutral', 2: 'positive'})
    df['topic_name'] = df['topic'].map({0: 'lecturer', 1: 'training_program', 2: 'facility', 3: 'others'})
    return add_ids(df)


# 1
def test_normalize_unicode():
    nfd = unicodedata.normalize('NFD', 'hoà bình')
    assert nfd != 'hoà bình'
    assert normalize_unicode(nfd) == 'hoà bình'


# 2
def test_tone_marks():
    cases = {'hòa': 'hoà', 'khỏe': 'khoẻ', 'thúy': 'thuý', 'tùy': 'tuỳ', 'hòa_bình': 'hoà_bình',
             'HÒA': 'HOÀ', 'Hòa': 'Hoà', 'xõa tóc': 'xoã tóc', 'họa sĩ': 'hoạ sĩ'}
    for old, new in cases.items():
        assert normalize_tone_marks(old) == new, (old, normalize_tone_marks(old))
    for same in ('hoàng', 'thuyết', 'toán', 'hoà', 'khoẻ', 'oai', 'hoài'):
        assert normalize_tone_marks(same) == same, same


# 3
def test_replace_urls_and_phones():
    assert replace_urls('xem https://abc.vn/x?y=1 nhé') == 'xem urltoken nhé'
    assert replace_urls('xem www.abc.vn nhé') == 'xem urltoken nhé'
    assert replace_urls('gửi về abc.def@gmail.com nhé') == 'gửi về emailtoken nhé'
    for phone in ('gọi 0912 345 678 nhé', 'sđt 086.56789.18', '0971.539.830', '0912345678', '+84912345678'):
        out = replace_urls(phone)
        assert 'phonetoken' in out and not any(c.isdigit() for c in out), (phone, out)
    for same in ('ngày 13 3 2023 25 3', '03 08 2023 12', '02 01 2024 10', '00 15 8 2023',
                 '70000 1000', 'năm 1931', '30.41975 30'):
        assert replace_urls(same) == same, same


# 4
def test_replace_masked_names():
    assert replace_masked_names('thầy wzjwz36 dạy hay') == 'thầy nametoken dạy hay'
    assert replace_masked_names('cô wzjwz dạy hay') == 'cô nametoken dạy hay'


# 5
def test_reduce_repeated_chars():
    assert reduce_repeated_chars('hayyyyy!!!!') == 'hayy!!'
    assert reduce_repeated_chars('2000') == '2000'
    assert reduce_repeated_chars('😂😂😂😂') == '😂😂'
    assert reduce_repeated_chars('hayyyyy', max_repeat=0) == 'hayyyyy'


def test_clean_whitespace_and_teencode():
    assert clean_whitespace('  a\tb \n c  d  ') == 'a b c d'
    assert normalize_teencode('ko biết dc không', {'ko': 'không', 'dc': 'được'}) == 'không biết được không'
    assert normalize_teencode('kontum', {'ko': 'không'}) == 'kontum'   # whole words only


# 6
def test_normalize_text_idempotent_and_keeps_emoji():
    samples = [
        'thầy dạy rất hay 😂😂😂😂', 'hòa   bình !!!!!', 'liên hệ 0912 345 678 nhé', 'xem https://a.vn',
        'cô wzjwz12 nhiệt tình', 'khỏe không ???', 'thúy ơi ...', 'tùy bạn', 'HÒA BÌNH', 'ok 👍👍👍',
        'ngày 13 3 2023', 'email a.b@c.com', 'hayyyy quá', ' khoảng trắng  ', 'toán khó', 'hoàng hôn',
        'sinh viên năm 2', 'giáo trình đầy đủ .', 'thuyết trình', '❤️❤️❤️ yêu trường',
    ]
    for s in samples:
        once = normalize_text(s)
        assert normalize_text(once) == once, (s, once)
    assert '😂😂' in normalize_text(samples[0]) and '👍👍' in normalize_text(samples[9])
    cfg_all = {**DEFAULT_CFG, 'lowercase': True, 'teencode': True}
    for s in samples + ['KO biết WZJWZ3 DC']:
        once = normalize_text(s, cfg_all)
        assert normalize_text(once, cfg_all) == once, (s, once)
    s = pd.Series(samples)
    assert normalize_series(s).tolist() == [normalize_text(x) for x in samples]


# 7
def test_dedup_train():
    df = table([('giảng viên dạy hay', 0, 0, 'train'), ('giảng viên dạy hay', 0, 0, 'train'),
                ('giảng viên dạy hay', 1, 0, 'train'), ('giảng viên dạy hay', 2, 0, 'test'),
                ('phòng học nóng', 0, 2, 'train'), ('phòng học nóng', 1, 1, 'train')])
    out, stats = dedup_train(df)
    train = out[out['split'] == 'train']
    group = train[train['text_clean'] == 'giảng viên dạy hay']
    assert len(group) == 1 and group['sentiment'].item() == 0 and group['id'].item() == 'uit-vsfc-train-0'
    assert (out['split'] == 'test').sum() == 1                       # test copy kept
    tie = train[train['text_clean'] == 'phòng học nóng']              # tie -> first occurrence
    assert tie['sentiment'].item() == 0 and tie['topic'].item() == 2 and tie['topic_name'].item() == 'facility'
    assert stats == {'rows_removed': 3, 'duplicate_groups': 2,
                     'groups_conflicting_sentiment': 2, 'groups_conflicting_topic': 1}


# 8
def test_flag_leakage():
    df = table([('a b c', 0, 0, 'train'), ('d e f', 1, 1, 'train'), ('a b c', 0, 0, 'test'), ('g h', 2, 0, 'val')])
    out, stats = flag_leakage(df)
    assert out['in_test'].tolist() == [True, False, False, False] and stats == {'flagged': 1, 'dropped': 0}
    out, stats = flag_leakage(df, drop=True)
    assert len(out) == 3 and (out['split'] == 'test').sum() == 1 and stats['dropped'] == 1


# 9
def test_make_train_subsets():
    # 9 strata (sentiment 0-2 x topic 0-2) + topic 3 only once -> a 1-row stratum merged into 'rare'
    rows = [(f'text {i}', i % 3, (i // 3) % 3, 'train') for i in range(1200)] + [('rare text', 1, 3, 'train')]
    rows += [(f'val {i}', 0, 0, 'val') for i in range(50)]
    df = table(rows)
    n_train = int((df['split'] == 'train').sum())
    subsets = make_train_subsets(df, [0.1, 0.25, 0.5], seed=42)
    n_strata = 10
    for f, ids in subsets.items():
        assert abs(len(ids) - float(f) * n_train) <= n_strata, (f, len(ids))
        assert all(i.startswith('uit-vsfc-train-') for i in ids)
    assert set(subsets['0.1']) <= set(subsets['0.25']) <= set(subsets['0.5'])
    assert subsets == make_train_subsets(df, [0.1, 0.25, 0.5], seed=42)
    assert subsets != make_train_subsets(df, [0.1, 0.25, 0.5], seed=7)
    half = df[df['id'].isin(subsets['0.5'])]
    assert set(half['sentiment']) == {0, 1, 2} and set(half['topic']) == {0, 1, 2, 3}


# 10
def test_add_segmented_column():
    df = table([('sinh viên nametoken rất thích giảng viên', 2, 0, 'train'),
                ('liên hệ phonetoken hoặc urltoken', 1, 3, 'train'),
                ('sinh viên 😂😂 !! sinh viên', 2, 0, 'test'), ('', 1, 3, 'val')])
    out = add_segmented_column(df, fake_segmenter)
    assert out['text_seg'].tolist() == ['sinh_viên nametoken rất thích giảng_viên', 'liên hệ phonetoken hoặc urltoken',
                                        'sinh_viên 😂😂 !! sinh_viên', '']
    # a segmenter that changes punctuation or emoji cannot touch them
    noisy = lambda xs: [x.replace(' ', '_') for x in xs]
    assert segment_texts(['sinh viên 😂😂 !! tốt'], noisy) == ['sinh_viên 😂😂 !! tốt']
    assert load_segmenter('none')(['sinh viên']) == ['sinh viên']
    # a segmenter that rewrites spelling (underthesea turns hoá into hóa): keep our text, take its joins
    respell = lambda xs: [x.replace('tiêu hoá', 'tiêu_hóa') for x in xs]
    assert segment_texts(['thuốc tiêu hoá tốt'], respell) == ['thuốc tiêu_hoá tốt']
    # output that does not line up with the input syllables is ignored
    broken = lambda xs: ['xyz' for _ in xs]
    assert segment_texts(['sinh viên tốt'], broken) == ['sinh viên tốt']


# 11
def test_preprocess_dataset_end_to_end():
    raw = pd.DataFrame({
        'text': ['thầy wzjwz3 dạy hòa nhã !!!!', 'thầy wzjwz3 dạy hòa nhã !!!!', 'sinh viên   thích', 'phòng học nóng',
                 'thầy dạy hay', 'sinh viên thích', 'cơ sở vật chất tốt'],
        'sentiment': [2, 2, 2, 0, 2, 2, 2], 'topic': [0, 0, 1, 2, 0, 1, 2],
        'split': ['train', 'train', 'train', 'train', 'val', 'test', 'test'],
        'dataset': 'uit-vsfc'})
    raw['sentiment_name'] = raw['sentiment'].map({0: 'negative', 1: 'neutral', 2: 'positive'})
    raw['topic_name'] = raw['topic'].map({0: 'lecturer', 1: 'training_program', 2: 'facility', 3: 'others'})
    with tempfile.TemporaryDirectory() as tmp:
        df, report, examples = preprocess_dataset('uit-vsfc', {'segmenter': 'none'}, fake_segmenter, tmp, df=raw)
        saved = pd.read_parquet(os.path.join(tmp, 'uit-vsfc.parquet'))
        subsets = json.load(open(os.path.join(tmp, 'uit-vsfc_subsets.json'), encoding='utf-8'))
    assert list(saved.columns) == OUTPUT_COLUMNS
    assert (saved['split'] == 'val').sum() == 1 and (saved['split'] == 'test').sum() == 2   # val/test unchanged
    assert (saved['split'] == 'train').sum() == 3                                            # 1 duplicate removed
    first = saved.loc[saved['id'] == 'uit-vsfc-train-0']
    assert first['text_clean'].item() == 'thầy nametoken dạy hoà nhã !!'
    leak = saved.loc[saved['id'] == 'uit-vsfc-train-2']
    assert leak['in_test'].item() and leak['text_seg'].item() == 'sinh_viên thích'
    assert set(subsets) == {'0.1', '0.25', '0.5'}
    steps = {r['step'] for r in report}
    assert {'load', 'unicode', 'tone_marks', 'masked_names', 'dedup_train', 'flag_leakage', 'segment'} <= steps
    assert len(examples) >= 1


def test_default_config_file():
    cfg = load_config()
    assert set(DEFAULT_CFG) <= set(cfg)
    assert cfg['tone_marks'] == 'new_style' and cfg['lowercase'] is False and cfg['max_repeat_chars'] == 2


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
