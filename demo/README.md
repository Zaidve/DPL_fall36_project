# Demo: a post's topic and sentiment from its comments

A local web page. Give it the comments under a social media post (pasted, from a file, or read from a Reddit
link); it classifies every comment with a trained model of this project and reads the post's topic and
sentiment from all comments together.

```
python -m demo.server
```

Then open http://127.0.0.1:8765. Options: `--checkpoints DIR ...` (default `demo/checkpoints/`), `--port`,
`--device cpu|cuda`. It uses only packages already in `.venv`; nothing needs to be installed.

## 1. Get a model

The demo needs a run folder with `config.yaml` and `best.pt`. Checkpoints are only written on Kaggle and the
results zip leaves them out, so they have to be downloaded from the notebook's output.

**Recommended: train the demo models** (2 runs, about 1 GPU-hour on a T4). In the Kaggle runner set:

```python
EXPERIMENT = 'demo_model'
ONLY = None
MAX_STEPS = None
RESUME_FROM = None
```

This trains, with seed 42 and the checkpoint kept, the final models of Route C2-b:

| Run folder | Model | Labels |
|---|---|---|
| `neu-esc__phobert__mtl__seed42__smartref` | PhoBERT, linear heads, reference SMART | 4 sentiments, 10 topics of student forum posts |
| `uit-vsfc__phobert__mtl__seed42__sum` | PhoBERT, linear heads, equal sum | 3 sentiments, 4 topics of course feedback |

When the session ends, open the version's **Output**, and download each folder under `models/` (at least
`config.yaml` and `best.pt`, about 270 MB each) into `demo/checkpoints/<run folder>/`. Do not merge them into
`models/`: the run ids are the same as finished E1b / E1 runs.

**Or use a checkpoint that already exists** in the outputs of the earlier Kaggle sessions: every E1 `mtl/sum`
run and the E4 final models kept theirs, for example `neu-esc__phobert__mtl__seed42__sum`. Any run folder with
`config.yaml` and `best.pt` works.

## 2. Use it

- **Paste comments**: copy them from Facebook, X, TikTok, a forum or anywhere, one comment per line (or
  separated by empty lines for comments that span several lines). A `.txt` file, or a `.csv` with a
  `text` / `comment` / `message` / `body` / `content` column, can be loaded instead.
- **Post link**: Reddit links are read directly. Facebook and X do not allow other programs to read comments
  (Facebook needs a logged-in account and an app reviewed by Meta; X sells API access), so for them the page
  asks you to paste the comments.
- **Post text** (optional): the post itself is also classified and shown next to the answer from the
  comments.

### Reddit access

Reddit often refuses requests that come without an app key (403). To read Reddit links reliably, create a free
app of type "script" at https://www.reddit.com/prefs/apps and start the demo with its id and secret in the
environment:

```bash
REDDIT_CLIENT_ID=<id> REDDIT_CLIENT_SECRET=<secret> .venv/Scripts/python.exe -m demo.server
```

The demo then reads `https://oauth.reddit.com` with an application-only token. The key stays in the
environment of the demo process and is not written anywhere.

## 3. How the answer is made

1. Each comment is cleaned like the training data: Unicode and tone-mark normalization, URLs / emails / phone
   numbers replaced by placeholder words, repeated characters capped at 2, lowercase, spaces around
   punctuation. For PhoBERT the text is then word-segmented with underthesea, as in training.
2. Comments that do not look Vietnamese (no Vietnamese letters or tone marks, and few common unaccented
   Vietnamese words) are left out of the post-level answer unless the checkbox says otherwise; they are still
   listed, greyed.
3. The post's distribution over topics (and sentiments) is the mean of the comments' probabilities. The page
   shows the top label, its mean probability, and the share of comments whose own top label agrees.

## 4. Limits

- The models learned from Vietnamese student forum posts (NEU-ESC) or course feedback (UIT-VSFC). Comments on
  other subjects, in other languages, or in heavy slang are outside what they learned; treat the answers as a
  rough guide.
- They were trained on posts, not on comments; a comment often only makes sense next to its post.
- Long comments are cut at the training length (80 PhoBERT tokens for NEU-ESC, 32 for UIT-VSFC).
- The page runs on your machine only (127.0.0.1). It sends nothing anywhere except the Reddit requests.

## 5. Files

| File | What |
|---|---|
| `demo/server.py` | the local web server and the JSON API (`/api/models`, `/api/fetch`, `/api/analyze`) |
| `demo/predictor.py` | a run folder as a predictor: model, tokenizer, segmenter, both heads in one pass |
| `demo/core.py` | cleaning, Vietnamese guess, comment splitting, Reddit parsing, the post-level aggregate |
| `demo/static/index.html` | the page |
| `configs/experiment/demo_model.yaml` | the Kaggle experiment that trains the two demo models |
| `tests/test_demo.py` | 11 tests (CPU, no downloads) |
