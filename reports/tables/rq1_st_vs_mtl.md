| dataset | backbone | task | ST mF1 | MTL mF1 | Δ mF1 | ST Acc | MTL Acc | Δ Acc | McNemar | reading |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| neu-esc | PhoBERT | sentiment | 76.26 ± 0.35 | 76.31 ± 0.27 | +0.05 | 81.80 ± 0.62 | 81.93 ± 0.39 | +0.13 |  | accuracy and macro-F1 up |
| neu-esc | PhoBERT | topic | 61.73 ± 0.43 | 61.41 ± 0.34 | -0.32 | 77.99 ± 0.64 | 78.03 ± 0.41 | +0.04 |  | accuracy up, macro-F1 not |
| neu-esc | ViSoBERT | sentiment | 76.09 ± 0.20 | 76.20 ± 0.23 | +0.11 | 82.15 ± 0.35 | 82.17 ± 0.37 | +0.02 |  | accuracy and macro-F1 up |
| neu-esc | ViSoBERT | topic | 61.02 ± 0.61 | 61.95 ± 0.31 | +0.93 | 78.12 ± 0.61 | 78.15 ± 0.26 | +0.03 |  | accuracy and macro-F1 up |
| neu-esc | XLM-R | sentiment | 76.02 ± 0.60 | 76.73 ± 0.36 | +0.72 | 81.75 ± 0.28 | 82.09 ± 0.48 | +0.34 |  | accuracy and macro-F1 up |
| neu-esc | XLM-R | topic | 62.27 ± 0.60 | 61.32 ± 0.57 | -0.94 | 77.74 ± 0.28 | 77.57 ± 0.26 | -0.17 |  | neither up |
| uit-vsfc | PhoBERT | sentiment | 82.64 ± 0.49 | 82.08 ± 2.04 | -0.56 | 93.78 ± 0.11 | 93.82 ± 0.54 | +0.04 |  | accuracy up, macro-F1 not |
| uit-vsfc | PhoBERT | topic | 80.06 ± 0.80 | 79.61 ± 0.82 | -0.45 | 89.19 ± 0.45 | 88.78 ± 0.38 | -0.41 |  | neither up |
| uit-vsfc | ViSoBERT | sentiment | 81.16 ± 0.48 | 81.27 ± 0.25 | +0.10 | 92.78 ± 0.21 | 92.75 ± 0.13 | -0.03 |  | macro-F1 up, accuracy not |
| uit-vsfc | ViSoBERT | topic | 79.17 ± 0.56 | 79.37 ± 0.94 | +0.19 | 88.51 ± 0.46 | 88.65 ± 0.59 | +0.14 |  | accuracy and macro-F1 up |
| uit-vsfc | XLM-R | sentiment | 82.17 ± 1.19 | 81.74 ± 0.69 | -0.43 | 93.22 ± 0.32 | 93.54 ± 0.24 | +0.32 |  | accuracy up, macro-F1 not |
| uit-vsfc | XLM-R | topic | 79.81 ± 0.38 | 78.98 ± 0.83 | -0.83 | 88.91 ± 0.29 | 88.51 ± 0.47 | -0.40 |  | neither up |
