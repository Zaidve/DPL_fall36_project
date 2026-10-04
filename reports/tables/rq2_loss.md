| dataset | backbone | strategy | sentiment mF1 | topic mF1 | mean mF1 | Δ mean vs sum | time/epoch (x sum) | vs smartref |
|---|---|---|---:|---:|---:|---:|---:|---|
| neu-esc | ViSoBERT | sum | 76.20 ± 0.23 | 61.95 ± 0.31 | 69.07 | +0.00 | 1.00 |  |
| neu-esc | ViSoBERT | unc | 75.84 ± 0.63 | 61.81 ± 0.26 | 68.83 | -0.25 | 1.02 |  |
| neu-esc | ViSoBERT | pcgrad | 76.17 ± 0.61 | 61.93 ± 0.60 | 69.05 | -0.03 | 7.45 |  |
| neu-esc | ViSoBERT | gradnorm | 76.25 ± 0.70 | 62.12 ± 0.54 | 69.19 | +0.11 | 3.75 |  |
| neu-esc | ViSoBERT | smartemb | 76.52 ± 0.18 | 61.95 ± 0.48 | 69.24 | +0.16 | 2.31 | n.s. |
| neu-esc | ViSoBERT | smartref | 76.53 ± 0.25† | 62.13 ± 0.25† | 69.33 | +0.26 | 2.72 |  |
| uit-vsfc | PhoBERT | sum | 82.08 ± 2.04 | 79.61 ± 0.82 | 80.84 | +0.00 | 1.00 |  |
| uit-vsfc | PhoBERT | unc | 82.29 ± 0.53 | 80.04 ± 0.53 | 81.17 | +0.32 | 1.00 |  |
| uit-vsfc | PhoBERT | pcgrad | 82.35 ± 0.65 | 80.16 ± 0.13 | 81.25 | +0.41 | 4.13 |  |
| uit-vsfc | PhoBERT | gradnorm | 82.39 ± 0.76 | 80.00 ± 0.70 | 81.19 | +0.35 | 2.07 |  |
| uit-vsfc | PhoBERT | smartemb | 81.42 ± 1.11 | 79.93 ± 0.28 | 80.68 | -0.17 | 1.65 | n.s. |
| uit-vsfc | PhoBERT | smartref | 81.48 ± 0.21 | 80.39 ± 0.91 | 80.94 | +0.09 | 1.96 |  |
