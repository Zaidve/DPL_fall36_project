"""
    Demo: guess the topic and the sentiment of a social media post from the comments under it.

      python -m demo.server [--checkpoints DIR ...] [--port 8765] [--device cuda|cpu]

    Opens a local web page (http://127.0.0.1:8765). Comments come from a YouTube or Reddit link, or are pasted /
    loaded from a .txt or .csv file (Facebook, X and other sites).
      YouTube: YouTube Data API v3 with YOUTUBE_API_KEY (a free key from Google Cloud; 1 quota unit per 100 comments).
      Reddit:  its API when REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are set (a free "script" app at
               https://www.reddit.com/prefs/apps), else its public JSON, which Reddit often refuses (403). Every comment is
    classified by a trained run (config.yaml + best.pt, default folder demo/checkpoints/), and the post-level
    answer is the mean of the comments' probabilities.

    API (JSON):
      GET  /api/models                          the loaded checkpoints and their label sets
      POST /api/fetch    {url}                  YouTube / Reddit: {post, comments}; Facebook / X: an explanation
      POST /api/analyze  {model, comments: [str | {text, ...}], post_text?, include_other_languages?}
"""
import argparse
import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode

from demo.core import (aggregate, looks_vietnamese, parse_reddit_thread, parse_youtube_thread, platform_of,
                       reddit_json_url, reddit_thread_id, youtube_video_id)

HERE = Path(__file__).resolve().parent
STATIC = HERE / 'static'
DEFAULT_CHECKPOINTS = HERE / 'checkpoints'
MAX_COMMENTS = 1000
MAX_BODY = 5_000_000
USER_AGENT = 'DPL-project-demo/0.1 (local research demo)'

NO_FETCH = {
    'facebook': ('Facebook does not let other programs read the comments of a post without a logged-in account '
                 'and an app reviewed by Meta. Open the post, expand the comments, copy them and paste them here.'),
    'x': ('X only gives comments (replies) through its paid API. Open the post, copy the replies and paste them '
          'here.'),
    'other': 'This site is not supported for reading links. Copy the comments and paste them here.',
}


class Models:
    """ Predictors keyed by run id, loaded on first use (loading PhoBERT takes a few seconds). """

    def __init__(self, run_dirs, device=None, loader=None):
        from demo.predictor import Predictor
        self.run_dirs = {Path(d).name: Path(d) for d in run_dirs}
        self.device = device
        self.loader = loader or (lambda d: Predictor.from_run_dir(d, device))
        self.loaded, self.lock = {}, threading.Lock()

    def ids(self):
        return list(self.run_dirs)

    def get(self, run_id):
        if run_id not in self.run_dirs:
            raise KeyError(f'unknown model {run_id!r}')
        with self.lock:
            if run_id not in self.loaded:
                self.loaded[run_id] = self.loader(self.run_dirs[run_id])
            return self.loaded[run_id]


def fetch_json(url, timeout=20, headers=None, data=None):
    req = urllib.request.Request(url, data=data, headers={'User-Agent': USER_AGENT, 'Accept': 'application/json',
                                                          **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


class RedditAPI:
    """ Application-only OAuth (client credentials) with the app id / secret from the environment.
        The token is kept in memory until shortly before it expires. """

    def __init__(self, client_id=None, client_secret=None):
        self.client_id = client_id or os.environ.get('REDDIT_CLIENT_ID')
        self.client_secret = client_secret or os.environ.get('REDDIT_CLIENT_SECRET')
        self.token, self.expires = None, 0.0

    @property
    def configured(self):
        return bool(self.client_id and self.client_secret)

    def _token(self):
        if self.token and time.time() < self.expires - 60:
            return self.token
        auth = base64.b64encode(f'{self.client_id}:{self.client_secret}'.encode()).decode()
        out = fetch_json('https://www.reddit.com/api/v1/access_token', headers={'Authorization': f'Basic {auth}'},
                         data=b'grant_type=client_credentials')
        if 'access_token' not in out:
            raise ValueError(f'no access token from Reddit ({out.get("error", "unknown error")})')
        self.token, self.expires = out['access_token'], time.time() + float(out.get('expires_in', 3600))
        return self.token

    def __call__(self, url):
        """ Same answer as the public JSON url, read through https://oauth.reddit.com. """
        path = url.split('reddit.com', 1)[1].replace('.json', '', 1)
        return fetch_json('https://oauth.reddit.com' + path, headers={'Authorization': f'Bearer {self._token()}'})


def reddit_fetcher():
    api = RedditAPI()
    return api if api.configured else fetch_json


class YouTubeAPI:
    """ YouTube Data API v3 with an API key (YOUTUBE_API_KEY): videos.list for the post, commentThreads.list for
        the comments, 100 per page (1 quota unit per page; the free quota is 10,000 units a day). """
    BASE = 'https://www.googleapis.com/youtube/v3/'

    def __init__(self, key=None, fetch=None):
        self.key = key if key is not None else os.environ.get('YOUTUBE_API_KEY')
        self.fetch = fetch or (lambda url: fetch_json(url))

    @property
    def configured(self):
        return bool(self.key)

    def get(self, method, **params):
        return self.fetch(self.BASE + method + '?' + urlencode({**params, 'key': self.key}))

    def thread(self, video_id, max_comments=MAX_COMMENTS):
        found = self.get('videos', part='snippet', id=video_id).get('items') or []
        if not found:
            raise LookupError('YouTube has no video with this id (it may be private or removed)')
        pages, token, n = [], None, 0
        while n < max_comments:
            params = {'part': 'snippet,replies', 'videoId': video_id, 'maxResults': 100, 'order': 'relevance',
                      'textFormat': 'plainText', **({'pageToken': token} if token else {})}
            page = self.get('commentThreads', **params)
            pages.append(page)
            n += sum(1 + len((i.get('replies') or {}).get('comments', [])) for i in page.get('items', []))
            token = page.get('nextPageToken')
            if not token:
                break
        return parse_youtube_thread(found[0], pages, max_comments)


YOUTUBE_REASONS = {
    'commentsDisabled': 'The owner of this video turned its comments off.',
    'quotaExceeded': 'The YouTube API key has used its daily quota (10,000 units); it resets at midnight Pacific time.',
    'videoNotFound': 'YouTube has no video with this id (it may be private or removed).',
    'keyInvalid': 'YouTube rejected the API key. Check YOUTUBE_API_KEY and restart the demo.',
    'accessNotConfigured': 'The API key\'s Google Cloud project does not have "YouTube Data API v3" enabled.',
    'forbidden': 'YouTube does not allow reading the comments of this video.',
}


def _google_error(e):
    """ The `reason` and message of a Google API error answer (never the request URL, which holds the key). """
    try:
        err = json.loads(e.read().decode('utf-8')).get('error', {})
    except Exception:  # noqa: BLE001
        return None, ''
    reasons = [x.get('reason') for x in err.get('errors', []) if x.get('reason')]
    if any(d.get('reason') == 'API_KEY_INVALID' for d in err.get('details', []) or []):
        reasons.insert(0, 'keyInvalid')
    return (reasons[0] if reasons else None), err.get('message', '')


def fetch_youtube(url, youtube):
    platform = 'youtube'
    video_id = youtube_video_id(url)
    if not video_id:
        return {'platform': platform, 'error': 'This YouTube link does not point to a video (watch?v=…, youtu.be/… '
                                               'or /shorts/…).'}
    if youtube is None or not youtube.configured:
        return {'platform': platform, 'error': 'Reading YouTube comments needs a free API key. Set YOUTUBE_API_KEY '
                                               '(see demo/README.md) and restart the demo, or paste the comments.'}
    try:
        return {'platform': platform, **youtube.thread(video_id, MAX_COMMENTS)}
    except urllib.error.HTTPError as e:
        reason, message = _google_error(e)
        return {'platform': platform, 'error': YOUTUBE_REASONS.get(reason) or f'YouTube answered {e.code}: '
                                                                             f'{message or e.reason}'}
    except LookupError as e:
        return {'platform': platform, 'error': f'{e}.'}
    except (urllib.error.URLError, TimeoutError) as e:
        return {'platform': platform, 'error': f'Could not reach YouTube ({getattr(e, "reason", e)}).'}
    except (ValueError, AttributeError) as e:
        return {'platform': platform, 'error': f'Unexpected answer from YouTube: {e}'}


def fetch_comments(url, fetch=fetch_json, youtube=None):
    """ {platform, post, comments} for a YouTube or Reddit link; {platform, error} for other links or failures.
        `fetch` reads Reddit's JSON; `youtube` is a YouTubeAPI (None: not configured). """
    platform = platform_of(url)
    if platform == 'youtube':
        return fetch_youtube(url, youtube)
    if platform != 'reddit':
        return {'platform': platform, 'error': NO_FETCH[platform]}
    thread_id = reddit_thread_id(url)
    if not thread_id:
        return {'platform': platform, 'error': 'This Reddit link does not point to a post (no /comments/<id>/ in it).'}
    try:
        thread = parse_reddit_thread(fetch(reddit_json_url(thread_id)), MAX_COMMENTS)
    except urllib.error.HTTPError as e:
        hint = ''
        if e.code in (401, 403, 429):
            hint = (' Reddit refuses many requests that come without an app key. Set REDDIT_CLIENT_ID and '
                    'REDDIT_CLIENT_SECRET (see demo/README.md) and restart the demo, or paste the comments.')
        return {'platform': platform, 'error': f'Reddit answered {e.code} {e.reason}.{hint}'}
    except (urllib.error.URLError, TimeoutError) as e:
        return {'platform': platform, 'error': f'Could not reach Reddit ({getattr(e, "reason", e)}).'}
    except ValueError as e:
        return {'platform': platform, 'error': f'Unexpected answer from Reddit: {e}'}
    return {'platform': platform, **thread}


def _label_result(probs, names):
    if probs is None:
        return None
    top = max(range(len(probs)), key=lambda k: probs[k])
    return {'label': names[top], 'probability': probs[top], 'distribution': dict(zip(names, probs))}


def analyze(predictor, comments, post_text=None, include_other_languages=False):
    """ Per-comment predictions and the post-level aggregate (see module docstring). """
    items = [c if isinstance(c, dict) else {'text': c} for c in comments][:MAX_COMMENTS]
    texts = [str(c.get('text') or '') for c in items]
    preds = predictor.predict(texts) if texts else []
    rows = []
    for item, text, p in zip(items, texts, preds):
        vi = looks_vietnamese(text)
        used = bool(p['clean']) and (vi or include_other_languages)
        rows.append({'text': text, 'clean': p['clean'], 'vietnamese': vi, 'used': used,
                     'author': item.get('author'), 'score': item.get('score'),
                     **{t: _label_result(p[t], predictor.labels[t]) for t in predictor.tasks}})
    post = {}
    for task in predictor.tasks:
        probs = [preds[i][task] for i, r in enumerate(rows) if r['used']]
        post[task] = aggregate(probs, predictor.labels[task]) if probs else None
    result = {'model': predictor.run_id, 'tasks': predictor.tasks, 'comments': rows, 'post': post,
              'n_comments': len(rows), 'n_used': sum(r['used'] for r in rows),
              'n_other_language': sum(1 for r in rows if r['clean'] and not r['vietnamese']),
              'n_truncated': max(0, len(comments) - MAX_COMMENTS)}
    if post_text and str(post_text).strip():
        p = predictor.predict([post_text])[0]
        result['post_text'] = {'clean': p['clean'], 'vietnamese': looks_vietnamese(post_text),
                               **{t: _label_result(p[t], predictor.labels[t]) for t in predictor.tasks}}
    return result


def make_handler(models, fetch=None, youtube=None):
    fetch = fetch or reddit_fetcher()
    youtube = youtube if youtube is not None else YouTubeAPI()

    class Handler(BaseHTTPRequestHandler):
        server_version = 'DPLDemo/0.1'

        def log_message(self, fmt, *args):            # quieter console: only errors
            if args and str(args[1] if len(args) > 1 else '').startswith(('4', '5')):
                super().log_message(fmt, *args)

        def _send(self, code, body, ctype='application/json; charset=utf-8'):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def _json_body(self):
            n = int(self.headers.get('Content-Length') or 0)
            if n > MAX_BODY:
                raise ValueError('request too large')
            return json.loads(self.rfile.read(n).decode('utf-8') or '{}')

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                return self._send(200, (STATIC / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            if self.path == '/api/models':
                out = []
                for run_id in models.ids():
                    try:
                        out.append(models.get(run_id).describe())
                    except Exception as e:  # noqa: BLE001 - shown in the page
                        out.append({'id': run_id, 'error': f'{type(e).__name__}: {e}'})
                return self._send(200, {'models': out})
            return self._send(404, {'error': 'not found'})

        def do_POST(self):
            try:
                body = self._json_body()
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, {'error': f'bad request: {e}'})
            try:
                if self.path == '/api/fetch':
                    return self._send(200, fetch_comments(body.get('url', ''), fetch, youtube))
                if self.path == '/api/analyze':
                    comments = body.get('comments') or []
                    if not isinstance(comments, list) or not comments:
                        return self._send(400, {'error': 'no comments to analyze'})
                    predictor = models.get(body.get('model') or models.ids()[0])
                    return self._send(200, analyze(predictor, comments, body.get('post_text'),
                                                   bool(body.get('include_other_languages'))))
            except KeyError as e:
                return self._send(400, {'error': str(e)})
            except Exception as e:  # noqa: BLE001 - reported to the page instead of a dropped connection
                return self._send(500, {'error': f'{type(e).__name__}: {e}'})
            return self._send(404, {'error': 'not found'})

    return Handler


def main(argv=None):
    from demo.predictor import find_run_dirs
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoints', nargs='+', default=[str(DEFAULT_CHECKPOINTS)],
                        help='run folders, or folders that contain run folders (each with config.yaml + best.pt)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', default=None, help='cuda or cpu (default: cuda when available)')
    args = parser.parse_args(argv)

    run_dirs = find_run_dirs(args.checkpoints)
    if not run_dirs:
        raise SystemExit(f'no run folder with best.pt and config.yaml in {args.checkpoints}; see demo/README.md')
    run_dirs.sort(key=lambda d: (not d.name.startswith('neu-esc'), d.name))     # NEU-ESC first: social media posts
    models = Models(run_dirs, args.device)
    print('models:', ', '.join(models.ids()))
    print('youtube:', 'API key set' if YouTubeAPI().configured else 'no key (set YOUTUBE_API_KEY to read YouTube links)')
    print('reddit:', 'API with REDDIT_CLIENT_ID' if RedditAPI().configured else 'public JSON (set REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET for the API)')
    print(f'loading {models.ids()[0]} ...')
    models.get(models.ids()[0])
    server = ThreadingHTTPServer((args.host, args.port), make_handler(models))
    print(f'demo running at http://{args.host}:{args.port}  (Ctrl+C to stop)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
