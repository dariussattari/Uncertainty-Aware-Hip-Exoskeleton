# Leave-one-task-out — test split

AUROC, and best achievable J in brackets. `base` is the participant-split run (LG/RA/RD in training, four tasks held out at once).

Precision and F1 are **not** comparable across the two protocols: the OOD share of the test set differs by more than an order of magnitude. AUROC and best-J are prevalence-independent.

| model | base | −SA | −ST | −TR |
|---|---|---|---|---|
| vanilla/AE | 0.782 | 0.712 | 0.991 | 0.876 |
| vanilla/GAN | 0.727 | 0.534 | 0.982 | 0.777 |
| vanilla/Ensemble_GP | 0.859 | 0.698 | 0.236 | 0.607 |
| vanilla/Ensemble_synthetic | 0.813 | 0.626 | 0.722 | 0.775 |
| vanilla/Ensemble_forecast | 0.896 | 0.827 | 0.320 | 0.649 |
| prob_ensemble | 0.914 [68.0] | 0.840 [55.5] | 0.480 [41.3] | 0.509 [3.7] |

## Test-set composition

| held out | ID windows | OOD windows | OOD share |
|---|---|---|---|
| participant-split | 61,956 | 23,043 | 27.1% |
| SA | 74,726 | 10,273 | 12.1% |
| ST | 84,378 | 621 | 0.7% |
| TR | 83,005 | 1,994 | 2.3% |

_18 leave-one-task-out runs evaluated._
