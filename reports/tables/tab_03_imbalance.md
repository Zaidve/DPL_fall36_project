| dataset | task | classes | n_train | majority_class | imbalance_ratio | norm_entropy | class_weights | chi2_p_val | jsd_val | chi2_p_test | jsd_test | majority_acc_test | majority_macro_f1_test | recommended_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| neu-esc | sentiment | 4 | 23,048 | neutral | 27.24 | 0.6493 | 0.36, 1.99, 1.59, 9.85 | 0.9968 | 0.001815 | 0.9616 | 0.003186 | 0.6892 | 0.204 | focal |
| neu-esc | topic | 10 | 23,048 | other | 35.86 | 0.6693 | 8.20, 3.67, 0.31, 0.23, 1.40, 4.09, 2.23, 4.30, 4.96, 4.99 | 1 | 0.002034 | 1 | 0.003149 | 0.4358 | 0.06071 | focal |
| uit-vsfc | sentiment | 3 | 11,426 | positive | 12.32 | 0.7584 | 0.72, 8.32, 0.67 | 0.2096 | 0.01997 | 0.002479 | 0.02872 | 0.5022 | 0.2229 | focal |
| uit-vsfc | topic | 4 | 11,426 | lecturer | 16.43 | 0.6073 | 0.35, 1.30, 5.75, 5.08 | 0.05144 | 0.03166 | 0.4796 | 0.0135 | 0.7233 | 0.2099 | focal |
