"""
    Pure helpers of the demo (no model, no network): text cleaning in the style of the training data,
    a Vietnamese-language guess, splitting pasted comments, reading a Reddit thread's JSON, and turning
    per-comment probabilities into one prediction for the post.

      clean_comment(text)                 normalized like data/processed text_clean (lowercase, spaced punctuation)
      looks_vietnamese(text)              heuristic: Vietnamese letters or common unaccented Vietnamese words
      split_comments(text, mode)          'line' (one per line) or 'blank' (separated by empty lines)
      platform_of(url)                    'reddit' | 'facebook' | 'x' | 'other'
      reddit_thread_id(url)               the base-36 post id in a reddit.com / redd.it link
      parse_reddit_thread(payload)        post + flattened comments from https://www.reddit.com/comments/<id>.json
      aggregate(probs, names)             post-level distribution, top label, agreement and counts
"""
import re
import unicodedata
from urllib.parse import urlparse

from utils.preprocess import normalize_text

# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

# The processed datasets are lowercase with a space on both sides of punctuation ("tốt quá , cảm ơn .").
_PUNCT_RE = re.compile(r'([.,!?;:()\[\]{}"“”‘’…/\\|*~^<>=+])')
_SPACE_RE = re.compile(r'\s+')


def clean_comment(text):
    """ The training text_clean normalization, then lowercase and spaced punctuation like the datasets. """
    text = normalize_text(str(text or ''))
    text = text.lower()
    text = _PUNCT_RE.sub(r' \1 ', text)
    return _SPACE_RE.sub(' ', text).strip()


# Letters that only Vietnamese uses among the languages a comment section is likely to mix (en, fr, ...).
_VI_LETTERS = set('ăâđêôơưĂÂĐÊÔƠƯ')
_VI_TONES = {'̀', '́', '̃', '̉', '̣'}         # grave, acute, tilde, hook, dot below
# Frequent Vietnamese words written without accents ("ko biet", "dc", "mn oi") in social media text.
# Only words that are not also common English words.
_VI_PLAIN = frozenset('ko khong hok dc duoc oke minh mik mn moi nguoi nha nhe nhen roi vay thi cung lam qua oi '
                      'ak em anh chi bn mk gi sao nao voi tui hoc truong lop thay sv sinh vien dang dung biet '
                      'nhieu luon cai nay'.split())
_WORD_RE = re.compile(r'[^\W\d_]+')


def _has_vi_marks(word):
    if any(c in _VI_LETTERS for c in word):
        return True
    return any(c in _VI_TONES for c in unicodedata.normalize('NFD', word))


def looks_vietnamese(text):
    """ True when at least 20% of the words carry Vietnamese letters / tone marks, or when an unaccented text
        has at least two common Vietnamese words and at least a third of its words are such words. """
    words = _WORD_RE.findall(str(text or '').lower())
    if not words:
        return False
    marked = sum(_has_vi_marks(w) for w in words)
    if marked / len(words) >= 0.2:
        return True
    plain = sum(w in _VI_PLAIN for w in words)
    return plain >= 2 and plain / len(words) >= 0.34


def split_comments(text, mode='line'):
    """ Pasted text -> list of comments. mode 'line': every non-empty line; 'blank': blocks between empty lines. """
    text = str(text or '').replace('\r\n', '\n').replace('\r', '\n')
    if mode == 'blank':
        parts = re.split(r'\n\s*\n', text)
        parts = [' '.join(p.split('\n')) for p in parts]
    elif mode == 'line':
        parts = text.split('\n')
    else:
        raise ValueError(f'unknown split mode {mode!r}; use "line" or "blank"')
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

def platform_of(url):
    host = (urlparse(str(url or '').strip()).hostname or '').lower()
    host = host[4:] if host.startswith('www.') else host
    if host == 'redd.it' or host == 'reddit.com' or host.endswith('.reddit.com'):
        return 'reddit'
    if host in ('facebook.com', 'fb.com', 'fb.watch') or host.endswith('.facebook.com'):
        return 'facebook'
    if host in ('x.com', 'twitter.com', 't.co') or host.endswith('.twitter.com') or host.endswith('.x.com'):
        return 'x'
    return 'other'


_REDDIT_ID_RE = re.compile(r'/comments/([a-z0-9]{3,12})(?:/|$)', re.IGNORECASE)


def reddit_thread_id(url):
    """ 'https://www.reddit.com/r/VietNam/comments/1abcde/title/' -> '1abcde'; redd.it/1abcde -> '1abcde'. """
    parsed = urlparse(str(url or '').strip())
    host = (parsed.hostname or '').lower()
    if host == 'redd.it':
        m = re.fullmatch(r'/([a-z0-9]{3,12})/?', parsed.path, re.IGNORECASE)
        return m.group(1).lower() if m else None
    m = _REDDIT_ID_RE.search(parsed.path)
    return m.group(1).lower() if m else None


def reddit_json_url(thread_id, limit=500):
    return f'https://www.reddit.com/comments/{thread_id}.json?limit={int(limit)}&sort=top&raw_json=1'


_GONE = {'[deleted]', '[removed]', ''}


def parse_reddit_thread(payload, max_comments=500):
    """ Reddit's thread JSON ([post listing, comment listing]) -> {'post': {...}, 'comments': [...]}.
        Comments are flattened depth-first; deleted / removed ones and 'load more' stubs are skipped. """
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError('not a Reddit thread (expected a list of two listings)')
    children = payload[0].get('data', {}).get('children', [])
    if not children:
        raise ValueError('the Reddit thread has no post')
    p = children[0].get('data', {})
    post = {'title': p.get('title', ''), 'text': p.get('selftext', '') or '', 'author': p.get('author'),
            'subreddit': p.get('subreddit'), 'score': p.get('score'),
            'url': 'https://www.reddit.com' + p['permalink'] if p.get('permalink') else None}
    comments = []

    def walk(listing, depth):
        for child in (listing or {}).get('data', {}).get('children', []):
            if len(comments) >= max_comments:
                return
            if child.get('kind') != 't1':
                continue
            d = child.get('data', {})
            body = (d.get('body') or '').strip()
            if body not in _GONE:
                comments.append({'text': body, 'author': d.get('author'), 'score': d.get('score'), 'depth': depth})
            replies = d.get('replies')
            if isinstance(replies, dict):
                walk(replies, depth + 1)

    walk(payload[1], 0)
    return {'post': post, 'comments': comments}


# ---------------------------------------------------------------------------
# Post-level prediction
# ---------------------------------------------------------------------------

def aggregate(probs, names):
    """
      probs: per-comment probability rows (list of lists or an (N, C) array); names: the C label names.
      The post's distribution is the mean of the comments' distributions. `agreement` is the share of
      comments whose own top label is the post's top label; `counts` counts the comments' top labels.
    """
    rows = [list(map(float, r)) for r in probs]
    if not rows:
        return None
    n, c = len(rows), len(names)
    if any(len(r) != c for r in rows):
        raise ValueError(f'every probability row needs {c} values')
    dist = [sum(r[k] for r in rows) / n for k in range(c)]
    top = max(range(c), key=lambda k: dist[k])
    tops = [max(range(c), key=lambda k: r[k]) for r in rows]
    counts = {names[k]: tops.count(k) for k in range(c)}
    return {'distribution': dict(zip(names, dist)), 'label': names[top], 'probability': dist[top],
            'agreement': tops.count(top) / n, 'counts': counts, 'n': n}
