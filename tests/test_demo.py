"""
    Tests for demo/ (core helpers, predictor, server API). CPU only, no downloads: a tiny random encoder,
    the hash tokenizer of test_analysis.py and a fake Reddit thread.
    Run:  python tests/test_demo.py
"""
import json
import os
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from demo import core as C  # noqa: E402
from demo import server as SV  # noqa: E402
from demo.predictor import Predictor, find_run_dirs  # noqa: E402
from test_analysis import HashTokenizer, tiny_build  # noqa: E402

REDDIT = [
    {'kind': 'Listing', 'data': {'children': [{'kind': 't3', 'data': {
        'title': 'Học phí năm nay', 'selftext': 'Mọi người thấy sao?', 'author': 'op', 'subreddit': 'VietNam',
        'score': 12, 'permalink': '/r/VietNam/comments/abc123/hoc_phi/'}}]}},
    {'kind': 'Listing', 'data': {'children': [
        {'kind': 't1', 'data': {'body': 'Đắt quá trời luôn', 'author': 'u1', 'score': 5, 'replies': {
            'kind': 'Listing', 'data': {'children': [
                {'kind': 't1', 'data': {'body': '[deleted]', 'author': None, 'score': 1, 'replies': ''}},
                {'kind': 't1', 'data': {'body': 'ko biết nữa mn ơi', 'author': 'u2', 'score': 2, 'replies': ''}}]}}}},
        {'kind': 't1', 'data': {'body': 'Totally worth it', 'author': 'u3', 'score': 1, 'replies': ''}},
        {'kind': 'more', 'data': {'children': ['x1', 'x2']}}]}},
]


YT_VIDEO = {'id': 'dQw4w9WgXcQ', 'snippet': {'title': 'Ôn thi Kinh tế vi mô', 'description': 'Tóm tắt chương 1',
                                              'channelTitle': 'NEU Channel'}}


def yt_thread(text, likes=0, replies=()):
    return {'snippet': {'topLevelComment': {'snippet': {'textOriginal': text, 'authorDisplayName': 'a', 'likeCount': likes}}},
            'replies': {'comments': [{'snippet': {'textOriginal': r, 'authorDisplayName': 'b', 'likeCount': 0}}
                                     for r in replies]} if replies else None}


YT_PAGES = [{'items': [yt_thread('Cảm ơn thầy, dễ hiểu quá', 7, ['Đúng vậy']), yt_thread('  ')], 'nextPageToken': 'p2'},
            {'items': [yt_thread('Great video')]}]


class FakeYouTube:
    """ Answers videos.list / commentThreads.list like the API; records the urls it was asked for. """

    def __init__(self, video=YT_VIDEO, pages=YT_PAGES, error=None):
        self.video, self.pages, self.error, self.urls = video, pages, error, []

    def __call__(self, url):
        from urllib.parse import parse_qs, urlparse
        self.urls.append(url)
        if self.error:
            raise self.error
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        if '/videos?' in url:
            return {'items': [self.video] if self.video else []}
        return self.pages[1] if q.get('pageToken') == 'p2' else self.pages[0]


def google_error(code, reason, details=None):
    import io
    import urllib.error
    body = json.dumps({'error': {'code': code, 'message': f'{reason} message', 'errors': [{'reason': reason}],
                                 'details': details or []}}).encode()
    return urllib.error.HTTPError('https://www.googleapis.com/youtube/v3/x?key=SECRET', code, 'err', {}, io.BytesIO(body))


def test_clean_comment_matches_processed_style():
    assert C.clean_comment('Thầy dạy HAY quá!!!  Xem https://a.vn/x nhé') == 'thầy dạy hay quá ! ! xem urltoken nhé'
    assert C.clean_comment('Gọi 0912345678, ok?') == 'gọi phonetoken , ok ?'
    assert C.clean_comment('   ') == '' and C.clean_comment(None) == ''


def test_looks_vietnamese():
    assert C.looks_vietnamese('Môn này khó vãi')
    assert C.looks_vietnamese('ko biet nua mn oi')               # unaccented social media Vietnamese
    assert not C.looks_vietnamese('a dog and a cat')
    assert not C.looks_vietnamese('This is great, thanks so much')
    assert not C.looks_vietnamese('😂😂') and not C.looks_vietnamese('')


def test_split_comments():
    text = 'một\r\n\r\nhai dòng\nvẫn hai\n\n\nba\n'
    assert C.split_comments(text, 'line') == ['một', 'hai dòng', 'vẫn hai', 'ba']
    assert C.split_comments(text, 'blank') == ['một', 'hai dòng vẫn hai', 'ba']
    try:
        C.split_comments(text, 'comma')
        raise AssertionError('unknown mode accepted')
    except ValueError:
        pass


def test_platform_and_reddit_ids():
    assert C.platform_of('https://www.reddit.com/r/VietNam/comments/1abcde/some_title/') == 'reddit'
    assert C.platform_of('https://old.reddit.com/comments/1abcde') == 'reddit'
    assert C.platform_of('https://redd.it/1abcde') == 'reddit'
    assert C.platform_of('https://www.facebook.com/groups/1/posts/2') == 'facebook'
    assert C.platform_of('https://m.facebook.com/story.php?id=1') == 'facebook'
    assert C.platform_of('https://x.com/user/status/1') == 'x' and C.platform_of('https://twitter.com/u/status/1') == 'x'
    assert C.platform_of('https://notreddit.com/comments/abc') == 'other' and C.platform_of('') == 'other'
    assert C.reddit_thread_id('https://www.reddit.com/r/VietNam/comments/1AbCdE/some_title/') == '1abcde'
    assert C.reddit_thread_id('https://redd.it/1abcde') == '1abcde'
    assert C.reddit_thread_id('https://www.reddit.com/r/VietNam/') is None
    assert C.reddit_json_url('abc', 50) == 'https://www.reddit.com/comments/abc.json?limit=50&sort=top&raw_json=1'


def test_youtube_links():
    for url in ('https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s', 'https://m.youtube.com/watch?feature=x&v=dQw4w9WgXcQ',
                'https://youtu.be/dQw4w9WgXcQ?si=abc', 'https://www.youtube.com/shorts/dQw4w9WgXcQ',
                'https://www.youtube.com/live/dQw4w9WgXcQ?feature=share', 'https://www.youtube.com/embed/dQw4w9WgXcQ'):
        assert C.platform_of(url) == 'youtube' and C.youtube_video_id(url) == 'dQw4w9WgXcQ', url
    for url in ('https://www.youtube.com/@NEUchannel', 'https://www.youtube.com/watch?v=short', 'https://youtu.be/',
                'https://notyoutube.com/watch?v=dQw4w9WgXcQ'):
        assert C.youtube_video_id(url) is None, url


def test_parse_youtube_thread():
    thread = C.parse_youtube_thread(YT_VIDEO, YT_PAGES)
    assert thread['post'] == {'title': 'Ôn thi Kinh tế vi mô', 'text': 'Tóm tắt chương 1', 'author': 'NEU Channel',
                              'source': 'YouTube · NEU Channel', 'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'}
    assert [(c['text'], c['depth'], c['score']) for c in thread['comments']] == [
        ('Cảm ơn thầy, dễ hiểu quá', 0, 7), ('Đúng vậy', 1, 0), ('Great video', 0, 0)]       # empty one skipped
    assert len(C.parse_youtube_thread(YT_VIDEO, YT_PAGES, max_comments=2)['comments']) == 2


def test_fetch_youtube_pages_and_errors():
    url = 'https://youtu.be/dQw4w9WgXcQ'
    fake = FakeYouTube()
    res = SV.fetch_comments(url, youtube=SV.YouTubeAPI('KEY', fake))
    assert res['platform'] == 'youtube' and len(res['comments']) == 3 and res['post']['author'] == 'NEU Channel'
    assert len(fake.urls) == 3 and all('key=KEY' in u for u in fake.urls)                  # video + 2 comment pages
    assert 'videoId=dQw4w9WgXcQ' in fake.urls[1] and 'maxResults=100' in fake.urls[1] and 'pageToken=p2' in fake.urls[2]

    no_key = SV.fetch_comments(url, youtube=SV.YouTubeAPI('', fake))
    assert 'YOUTUBE_API_KEY' in no_key['error'] and len(fake.urls) == 3                     # nothing requested
    assert 'needs a free API key' in SV.fetch_comments(url)['error']
    assert 'does not point to a video' in SV.fetch_comments('https://www.youtube.com/@x', youtube=SV.YouTubeAPI('K', fake))['error']
    assert 'no video' in SV.fetch_comments(url, youtube=SV.YouTubeAPI('K', FakeYouTube(video=None)))['error']
    for err, text in ((google_error(403, 'commentsDisabled'), 'turned its comments off'),
                      (google_error(403, 'quotaExceeded'), 'daily quota'),
                      (google_error(400, 'badRequest', [{'reason': 'API_KEY_INVALID'}]), 'rejected the API key'),
                      (google_error(500, 'backendError'), 'YouTube answered 500: backendError message')):
        res = SV.fetch_comments(url, youtube=SV.YouTubeAPI('SECRET', FakeYouTube(error=err)))
        assert text in res['error'] and 'SECRET' not in res['error'], res


def test_parse_reddit_thread():
    thread = C.parse_reddit_thread(REDDIT)
    assert thread['post']['title'] == 'Học phí năm nay' and thread['post']['subreddit'] == 'VietNam'
    assert thread['post']['url'] == 'https://www.reddit.com/r/VietNam/comments/abc123/hoc_phi/'
    assert [c['text'] for c in thread['comments']] == ['Đắt quá trời luôn', 'ko biết nữa mn ơi', 'Totally worth it']
    assert [c['depth'] for c in thread['comments']] == [0, 1, 0]                 # depth-first, deleted skipped
    assert len(C.parse_reddit_thread(REDDIT, max_comments=1)['comments']) == 1
    for bad in ({}, [], [{'data': {'children': []}}, {}]):
        try:
            C.parse_reddit_thread(bad)
            raise AssertionError('bad payload accepted')
        except ValueError:
            pass


def test_aggregate():
    agg = C.aggregate([[0.6, 0.4], [0.3, 0.7], [0.9, 0.1]], ['a', 'b'])
    assert agg['label'] == 'a' and abs(agg['probability'] - 0.6) < 1e-12
    assert agg['counts'] == {'a': 2, 'b': 1} and abs(agg['agreement'] - 2 / 3) < 1e-12 and agg['n'] == 3
    assert C.aggregate([], ['a', 'b']) is None
    try:
        C.aggregate([[0.5, 0.5, 0.0]], ['a', 'b'])
        raise AssertionError('wrong width accepted')
    except ValueError:
        pass


# --- predictor and server, with a tiny checkpoint ------------------------------------------------

def fake_run(root, run_id='neu-esc__phobert__mtl__seed42__smartref'):
    """ A run folder with config.yaml + best.pt of the tiny random model (4 sentiment, 10 topic labels). """
    import torch
    from trainer.runs import save_checkpoint
    from utils.common import save_yaml
    folder = Path(root) / run_id
    folder.mkdir(parents=True)
    save_yaml({'data': {'tasks': ['sentiment', 'topic']}, 'model': {'head': 'linear'}}, folder / 'config.yaml')
    save_checkpoint(tiny_build({'model': {}}, {'sentiment': 4, 'topic': 10}).half(), folder / 'best.pt')
    assert torch.load(folder / 'best.pt', weights_only=True)
    return folder


def tiny_predictor(folder, segmenter=None):
    return Predictor.from_run_dir(folder, 'cpu', build=tiny_build, tokenizer=HashTokenizer(), segmenter=segmenter)


def test_find_run_dirs_and_predictor():
    with tempfile.TemporaryDirectory() as tmp:
        folder = fake_run(tmp)
        (Path(tmp) / 'not_a_run').mkdir()
        (Path(tmp) / 'uit-vsfc__phobert__mtl__seed42__sum').mkdir()                 # no best.pt
        assert find_run_dirs([tmp]) == [folder] and find_run_dirs([folder]) == [folder]

        def segment(texts):                                                          # joins "sinh viên"
            return [t.replace('sinh viên', 'sinh_viên') for t in texts]
        segment.used = 'fake'
        pred = tiny_predictor(folder, segment)
        assert pred.dataset == 'neu-esc' and pred.tasks == ['sentiment', 'topic'] and pred.max_len == 80
        assert len(pred.labels['topic']) == 10 and pred.labels['sentiment'][3] == 'toxic'
        out = pred.predict(['Sinh viên năm nhất!', '   ', 'môn này khó'])
        assert out[0]['clean'] == 'sinh viên năm nhất !' and out[0]['input'] == 'sinh_viên năm nhất !'
        assert out[1]['sentiment'] is None and out[1]['topic'] is None               # empty text: no prediction
        assert len(out[2]['topic']) == 10 and abs(sum(out[2]['topic']) - 1) < 1e-5
        info = pred.describe()
        assert info['labels']['topic'][9] == {'name': 'club_events', 'display': 'Clubs & events'}


def test_analyze_excludes_other_languages():
    with tempfile.TemporaryDirectory() as tmp:
        pred = tiny_predictor(fake_run(tmp))
        comments = ['Đắt quá trời luôn', {'text': 'Totally worth it', 'score': 3}, 'ko biết nữa mn ơi', '']
        res = SV.analyze(pred, comments, post_text='Học phí năm nay')
        assert res['n_comments'] == 4 and res['n_used'] == 2 and res['n_other_language'] == 1
        assert [r['used'] for r in res['comments']] == [True, False, True, False]
        assert res['comments'][1]['score'] == 3 and res['post']['topic']['n'] == 2
        assert res['post_text']['topic']['label'] in pred.labels['topic']
        res = SV.analyze(pred, comments, include_other_languages=True)
        assert res['n_used'] == 3 and 'post_text' not in res
        res = SV.analyze(pred, ['hello world'])
        assert res['n_used'] == 0 and res['post'] == {'sentiment': None, 'topic': None}


def test_fetch_comments_platforms_and_errors():
    import urllib.error
    seen = []

    def fake_fetch(url):
        seen.append(url)
        return REDDIT
    res = SV.fetch_comments('https://www.reddit.com/r/VietNam/comments/abc123/hoc_phi/', fake_fetch)
    assert res['platform'] == 'reddit' and len(res['comments']) == 3 and seen[0].startswith('https://www.reddit.com/comments/abc123.json')
    for url, platform in (('https://www.facebook.com/groups/1/posts/2', 'facebook'), ('https://x.com/u/status/1', 'x'),
                          ('https://example.com/post', 'other')):
        res = SV.fetch_comments(url, fake_fetch)
        assert res['platform'] == platform and 'paste' in res['error'] and len(seen) == 1   # nothing fetched
    assert 'does not point to a post' in SV.fetch_comments('https://www.reddit.com/r/VietNam/', fake_fetch)['error']

    def refused(url):
        raise urllib.error.HTTPError(url, 429, 'Too Many Requests', {}, None)
    assert '429' in SV.fetch_comments('https://redd.it/abc123', refused)['error']


def test_server_endpoints():
    with tempfile.TemporaryDirectory() as tmp:
        folder = fake_run(tmp)
        models = SV.Models([folder], loader=tiny_predictor)
        server = SV.ThreadingHTTPServer(('127.0.0.1', 0), SV.make_handler(
            models, fetch=lambda url: REDDIT, youtube=SV.YouTubeAPI('KEY', FakeYouTube())))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f'http://127.0.0.1:{server.server_address[1]}'

        def call(path, body=None):
            req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, r.read()
            except urllib.error.HTTPError as e:
                return e.code, e.read()
        try:
            status, page = call('/')
            assert status == 200 and b'<title>Post Topic and Sentiment from Comments</title>' in page
            status, body = call('/api/models')
            assert status == 200 and json.loads(body)['models'][0]['id'] == folder.name
            status, body = call('/api/fetch', {'url': 'https://redd.it/abc123'})
            assert status == 200 and len(json.loads(body)['comments']) == 3
            status, body = call('/api/fetch', {'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'})
            assert status == 200 and json.loads(body)['post']['source'] == 'YouTube · NEU Channel'
            status, body = call('/api/analyze', {'comments': ['Đắt quá trời luôn', 'ko biết nữa mn ơi']})
            data = json.loads(body)
            assert status == 200 and data['n_used'] == 2 and data['post']['sentiment']['n'] == 2
            assert call('/api/analyze', {'comments': []})[0] == 400
            assert call('/api/analyze', {'model': 'nope', 'comments': ['a']})[0] == 400
            assert call('/missing')[0] == 404
        finally:
            server.shutdown()
            server.server_close()


def test_reddit_api_uses_oauth_host_and_caches_token():
    calls = []

    def fake_fetch_json(url, timeout=20, headers=None, data=None):
        calls.append((url, (headers or {}).get('Authorization', '')[:6], data))
        if url.endswith('/api/v1/access_token'):
            return {'access_token': 'tok', 'expires_in': 3600}
        return REDDIT
    original = SV.fetch_json
    SV.fetch_json = fake_fetch_json
    try:
        api = SV.RedditAPI('id', 'secret')
        assert api.configured and not SV.RedditAPI('', '').configured
        for _ in range(2):
            assert api(C.reddit_json_url('abc123')) == REDDIT
        assert calls[0] == ('https://www.reddit.com/api/v1/access_token', 'Basic ', b'grant_type=client_credentials')
        assert calls[1] == ('https://oauth.reddit.com/comments/abc123?limit=500&sort=top&raw_json=1', 'Bearer', None)
        assert len(calls) == 3                                                       # token fetched once
    finally:
        SV.fetch_json = original


import urllib.error  # noqa: E402  (used in test_server_endpoints)

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
