# EDA findings

Generated 2026-09-29T14:52:27 by `notebooks/01_eda.ipynb` (seed 42, on_kaggle = False). Every number comes from `eda_summary.json`.

## Data

- Integrity checks: **all passed**, 49,141 rows.

## Labels and imbalance (train)

| dataset | task | classes | imbalance_ratio | majority_class | majority_acc_test | majority_macro_f1_test | jsd_test | recommended_loss |
|---|---|---|---|---|---|---|---|---|
| neu-esc | sentiment | 4 | 27.24 | neutral? | 0.6892 | 0.204 | 0.003186 | focal |
| neu-esc | topic | 10 | 35.86 | t3 | 0.4358 | 0.06071 | 0.003149 | focal |
| uit-vsfc | sentiment | 3 | 12.32 | positive | 0.5022 | 0.2229 | 0.02872 | focal |
| uit-vsfc | topic | 4 | 16.43 | lecturer | 0.7233 | 0.2099 | 0.0135 | focal |

Majority-class accuracy vs macro-F1 shows why macro-F1 is the main metric.

## Sentiment × topic dependence (train)

| dataset | Cramér's V (corrected) | NMI | chi2 p | strength |
|---|---|---|---|---|
| neu-esc | 0.1768 | 0.03444 | 0 | weak |
| uit-vsfc | 0.3518 | 0.1326 | 0 | strong |

- neu-esc: V = 0.177, NMI = 0.034 (weak).
- uit-vsfc: V = 0.352, NMI = 0.133 (strong).

## Recommended max_length (train, p95 rounded up to 16)

| dataset | tokenizer | p95 | recommended_max_length | trunc% at rec |
|---|---|---|---|---|
| neu-esc | vinai/phobert-base-v2 | 76 | 80 | 4.59 |
| neu-esc | xlm-roberta-base | 94 | 96 | 4.76 |
| neu-esc | bert-base-multilingual-cased | 97 | 112 | 3.96 |
| uit-vsfc | vinai/phobert-base-v2 | 30 | 32 | 3.96 |
| uit-vsfc | xlm-roberta-base | 39 | 48 | 2.34 |
| uit-vsfc | bert-base-multilingual-cased | 38 | 48 | 2.15 |

## Leakage and label conflicts

| dataset | train∩val % of val | train∩test % of test | val∩test % of test | sentiment conflicts % train | topic conflicts % train |
|---|---|---|---|---|---|
| neu-esc | 0 | 0 | 0 | 0 | 0 |
| uit-vsfc | 0 | 0 | 0 | 0.0175 | 0 |

## Split shift

- neu-esc: train-vs-test adversarial AUC = 0.494.
- uit-vsfc: train-vs-test adversarial AUC = 0.499.

## Warnings (30)

- neu-esc / vinai/phobert-base-v2: 5.0% of sentiment=positive? train texts are longer than max_length 80
- neu-esc / vinai/phobert-base-v2: 8.0% of sentiment=negative? train texts are longer than max_length 80
- neu-esc / vinai/phobert-base-v2: 14.2% of topic=t1 train texts are longer than max_length 80
- neu-esc / vinai/phobert-base-v2: 5.7% of topic=t2 train texts are longer than max_length 80
- neu-esc / vinai/phobert-base-v2: 5.0% of topic=t4 train texts are longer than max_length 80
- neu-esc / vinai/phobert-base-v2: 17.9% of topic=t5 train texts are longer than max_length 80
- neu-esc / vinai/phobert-base-v2: 8.1% of topic=t6 train texts are longer than max_length 80
- neu-esc / vinai/phobert-base-v2: 21.1% of topic=t7 train texts are longer than max_length 80
- neu-esc / vinai/phobert-base-v2: 6.0% of topic=t8 train texts are longer than max_length 80
- neu-esc / vinai/phobert-base-v2: 22.1% of topic=t9 train texts are longer than max_length 80
- neu-esc / xlm-roberta-base: 5.2% of sentiment=positive? train texts are longer than max_length 96
- neu-esc / xlm-roberta-base: 8.4% of sentiment=negative? train texts are longer than max_length 96
- neu-esc / xlm-roberta-base: 14.0% of topic=t1 train texts are longer than max_length 96
- neu-esc / xlm-roberta-base: 5.9% of topic=t2 train texts are longer than max_length 96
- neu-esc / xlm-roberta-base: 5.6% of topic=t4 train texts are longer than max_length 96
- neu-esc / xlm-roberta-base: 18.4% of topic=t5 train texts are longer than max_length 96
- neu-esc / xlm-roberta-base: 8.4% of topic=t6 train texts are longer than max_length 96
- neu-esc / xlm-roberta-base: 23.3% of topic=t7 train texts are longer than max_length 96
- neu-esc / xlm-roberta-base: 6.0% of topic=t8 train texts are longer than max_length 96
- neu-esc / xlm-roberta-base: 21.9% of topic=t9 train texts are longer than max_length 96
- neu-esc / bert-base-multilingual-cased: 7.2% of sentiment=negative? train texts are longer than max_length 112
- neu-esc / bert-base-multilingual-cased: 12.4% of topic=t1 train texts are longer than max_length 112
- neu-esc / bert-base-multilingual-cased: 15.8% of topic=t5 train texts are longer than max_length 112
- neu-esc / bert-base-multilingual-cased: 7.2% of topic=t6 train texts are longer than max_length 112
- neu-esc / bert-base-multilingual-cased: 19.2% of topic=t7 train texts are longer than max_length 112
- neu-esc / bert-base-multilingual-cased: 5.2% of topic=t8 train texts are longer than max_length 112
- neu-esc / bert-base-multilingual-cased: 19.5% of topic=t9 train texts are longer than max_length 112
- uit-vsfc / vinai/phobert-base-v2: 7.3% of sentiment=negative train texts are longer than max_length 32
- uit-vsfc / vinai/phobert-base-v2: 6.7% of topic=training_program train texts are longer than max_length 32
- uit-vsfc / vinai/phobert-base-v2: 8.0% of topic=facility train texts are longer than max_length 32

## Note

Label names for neu-esc are **not confirmed** (sentiment names end in `?`, topics are `t0`…). Check them against the source paper before using them in the report; `tab_11_top_words` helps name the topics.
