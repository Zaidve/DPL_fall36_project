"""
    A trained run folder (config.yaml + best.pt) as a predictor for raw comments.

      Predictor.from_run_dir(run_dir)     the model, its tokenizer, max_len and (PhoBERT) the word segmenter
      predictor.predict(texts)            per text: cleaned text, model input, and both heads' probabilities
      find_run_dirs(folders)              every run folder with a best.pt under the given folders

    Inputs are prepared like the training data: clean_comment() (text_clean), then for PhoBERT word
    segmentation with underthesea, which all training runs used (data/processed text_seg).
"""
from pathlib import Path

import numpy as np
import torch

from demo.core import clean_comment
from trainer.cross_eval import load_run_model
from trainer.results import label_names
from trainer.runs import parse_run_id

DATASET_NAMES = {'neu-esc': 'NEU-ESC (student forum posts)', 'uit-vsfc': 'UIT-VSFC (course feedback)'}
BACKBONE_NAMES = {'phobert': 'PhoBERT', 'visobert': 'ViSoBERT', 'xlmr': 'XLM-R'}
MODE_NAMES = {'mtl': 'multi-task, linear heads', 'mtlaware': 'multi-task, task-aware heads',
              'st_sentiment': 'single-task sentiment', 'st_topic': 'single-task topic'}

DISPLAY = {
    'neutral': 'Neutral', 'positive': 'Positive', 'negative': 'Negative', 'toxic': 'Toxic',
    'spam': 'Spam', 'news': 'News', 'academic': 'Academic', 'other': 'Other', 'service': 'Services',
    'jobs_recruitment': 'Jobs & recruitment', 'personal_affairs': 'Personal affairs',
    'social_affairs': 'Social affairs', 'help_share': 'Help & sharing', 'club_events': 'Clubs & events',
    'lecturer': 'Lecturer', 'training_program': 'Training program', 'facility': 'Facilities', 'others': 'Others',
}


def find_run_dirs(folders):
    """ Run folders (named like a run id, with best.pt and config.yaml) in or directly under `folders`. """
    found = []
    for folder in [Path(f) for f in folders]:
        candidates = [folder] if (folder / 'best.pt').is_file() else sorted(p for p in folder.glob('*') if p.is_dir())
        for c in candidates:
            if (c / 'best.pt').is_file() and (c / 'config.yaml').is_file():
                try:
                    parse_run_id(c.name)
                except Exception:
                    continue
                found.append(c)
    return found


class Predictor:
    def __init__(self, model, tokenizer, max_len, dataset, tasks, run_id, segmenter=None, backbone=None,
                 mode=None, tag=None):
        self.model, self.tokenizer, self.max_len = model, tokenizer, int(max_len)
        self.dataset, self.tasks, self.run_id = dataset, list(tasks), run_id
        self.segmenter = segmenter
        self.backbone, self.mode, self.tag = backbone, mode, tag
        self.labels = {t: label_names(dataset, t) for t in self.tasks}

    @classmethod
    def from_run_dir(cls, run_dir, device=None, build=None, tokenizer=None, segmenter='auto'):
        """ Load a finished run with a kept checkpoint. `build` / `tokenizer` / `segmenter` are for tests. """
        from utils.config import load_model_config, max_len_for
        from utils.dataset import load_tokenizer
        run_dir = Path(run_dir)
        parts = parse_run_id(run_dir.name)
        model, cfg = load_run_model(run_dir, device, build)
        model_cfg = load_model_config(parts['backbone'])
        tokenizer = tokenizer or load_tokenizer(model_cfg)
        if segmenter == 'auto':
            segmenter = None
            if model_cfg.get('word_segmentation'):
                from utils.preprocess import load_segmenter
                segmenter = load_segmenter('underthesea')
        return cls(model, tokenizer, max_len_for(parts['dataset'], parts['backbone']), parts['dataset'],
                   cfg['data']['tasks'], run_dir.name, segmenter, parts['backbone'], parts['mode'], parts['tag'])

    def describe(self):
        return {'id': self.run_id, 'dataset': self.dataset, 'dataset_name': DATASET_NAMES.get(self.dataset, self.dataset),
                'backbone': BACKBONE_NAMES.get(self.backbone, self.backbone),
                'mode': MODE_NAMES.get(self.mode, self.mode), 'tag': self.tag, 'tasks': self.tasks,
                'labels': {t: [{'name': n, 'display': DISPLAY.get(n, n)} for n in names]
                           for t, names in self.labels.items()}}

    def model_inputs(self, texts):
        cleaned = [clean_comment(t) for t in texts]
        if self.segmenter is None:
            return cleaned, cleaned
        from utils.preprocess import segment_texts
        return cleaned, segment_texts(cleaned, self.segmenter)

    @torch.no_grad()
    def predict(self, texts, batch_size=32):
        """ [{'clean', 'input', '<task>': [probabilities]}] in the order of `texts`. Empty texts get None. """
        from utils.dataset import PadCollator, encode_texts
        texts = list(texts)
        cleaned, inputs = self.model_inputs(texts)
        keep = [i for i, t in enumerate(cleaned) if t]
        probs = {t: [None] * len(texts) for t in self.tasks}
        if keep:
            device = next(self.model.parameters()).device
            ids = encode_texts(self.tokenizer, [inputs[i] for i in keep], self.max_len)
            collate = PadCollator(self.tokenizer.pad_token_id)
            use_amp = device.type == 'cuda'
            self.model.eval()
            for start in range(0, len(ids), batch_size):
                chunk = [{'idx': str(j), 'input_ids': x, 'labels': {}} for j, x in enumerate(ids[start:start + batch_size])]
                batch = collate(chunk)
                with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                    out = self.model(input_ids=batch['input_ids'].to(device),
                                     attention_mask=batch['attention_mask'].to(device))
                for task in self.tasks:
                    p = torch.softmax(out[task].float(), dim=-1).cpu().numpy()
                    for row, i in zip(p, keep[start:start + batch_size]):
                        probs[task][i] = np.asarray(row, dtype=float).tolist()
        return [{'clean': cleaned[i], 'input': inputs[i], **{t: probs[t][i] for t in self.tasks}}
                for i in range(len(texts))]
