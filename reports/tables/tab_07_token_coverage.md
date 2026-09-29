| dataset | tokenizer | n_train | p50 | p90 | p95 | p99 | max | trunc%@64 | trunc%@96 | trunc%@128 | trunc%@192 | trunc%@256 | trunc%@512 | recommended_max_length | trunc%@rec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| neu-esc | vinai/phobert-base-v2 | 23,048 | 14 | 47 | 76 | 190 | 1,343 | 6.35 | 3.41 | 1.98 | 0.9806 | 0.6118 |  | 80 | 4.59 |
| neu-esc | xlm-roberta-base | 23,048 | 17 | 58 | 94 | 231 | 1,748 | 8.38 | 4.76 | 2.97 | 1.42 | 0.8808 | 0.2126 | 96 | 4.76 |
| neu-esc | bert-base-multilingual-cased | 23,048 | 17 | 60 | 97 | 241.06 | 1,869 | 8.94 | 5.05 | 3.16 | 1.52 | 0.9111 | 0.2343 | 112 | 3.96 |
| uit-vsfc | vinai/phobert-base-v2 | 11,426 | 11 | 24 | 30 | 46 | 129 | 0.2363 | 0.0175 | 0.008752 | 0 | 0 |  | 32 | 3.96 |
| uit-vsfc | xlm-roberta-base | 11,426 | 15 | 31 | 39 | 59 | 179 | 0.7264 | 0.07877 | 0.0175 | 0 | 0 | 0 | 48 | 2.34 |
| uit-vsfc | bert-base-multilingual-cased | 11,426 | 14 | 30.50 | 38 | 58 | 176 | 0.6739 | 0.08752 | 0.0175 | 0 | 0 | 0 | 48 | 2.15 |
