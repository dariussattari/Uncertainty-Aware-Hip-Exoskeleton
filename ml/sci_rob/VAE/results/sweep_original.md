# β sweep and controls — protocol `original`

Selection metric: validation LOF AUROC. Active dims: median posterior σ² < 0.9 over ID validation windows. Controls (`sampling = False`, β=0) run the AE objective through this pipeline; `ae_control_aehp` uses the AE's learning rate 0.003, batch 1,024 and 200-epoch budget.

| variant | protocol | seed | beta | sampling | best epoch | epochs run | val KL | val NLL | active dims | KL per active dim | val AUROC (LOF) | test AUROC (LOF) | test recall % | test false alarms % | val AUROC (KL) | val AUROC (NLL) | hours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta0.25 | original | 0 | 0.25 | True | 25 | 30 | 88.08 | 3455 | 32 | 2.66 | 0.6661 | 0.6805 | 48.35 | 20.63 | 0.2657 | 0.3124 | 0.145 |
| standard | original | 0 | 1 | True | 27 | 32 | 55.88 | 3474 | 32 | 1.664 | 0.6213 | 0.6474 | 45.08 | 25.6 | 0.4147 | 0.3109 | 0.1573 |
| beta2 | original | 0 | 2 | True | 20 | 25 | 37.52 | 3516 | 24 | 1.07 | 0.6604 | 0.721 | 43.81 | 11.12 | 0.4042 | 0.3093 | 0.1221 |
| beta4 | original | 0 | 4 | True | 24 | 29 | 22.53 | 3561 | 18 | 0.6715 | 0.7055 | 0.7094 | 31.92 | 4.374 | 0.3816 | 0.317 | 0.1407 |
| beta8 | original | 0 | 8 | True | 7 | 12 | 14.35 | 3623 | 11 | 0.43 | 0.7231 | 0.7211 | 34.32 | 4.326 | 0.358 | 0.3267 | 0.0678 |
| beta16 | original | 0 | 16 | True | 20 | 25 | 10.19 | 3666 | 8 | 0.3038 | 0.7084 | 0.6998 | 29.45 | 3.648 | 0.3163 | 0.3171 | 0.1203 |
| beta32 | original | 0 | 32 | True | 24 | 29 | 6.339 | 3751 | 8 | 0.191 | 0.7983 | 0.7464 | 30.26 | 2.352 | 0.2701 | 0.3139 | 0.1382 |
| beta32 | original | 1 | 32 | True | 16 | 21 | 6.953 | 3822 | 8 | 0.1999 | 0.8108 | 0.7727 | 24.98 | 1.742 | 0.3256 | 0.3088 | 0.1022 |
| beta32 | original | 2 | 32 | True | 16 | 21 | 6.134 | 3627 | 8 | 0.1951 | 0.7859 | 0.7635 | 34.03 | 2.279 | 0.3331 | 0.3096 | 0.1064 |
| ae_control_aehp | original | 0 | 0 | False | 43 | 48 | 11.44 | 3454 | n/a |  | 0.7758 | 0.8223 | 53.07 | 7.66 | 0.2246 | 0.3128 | 1.186 |
| ae_control | original | 0 | 0 | False | 25 | 30 | 5.019 | 3455 | n/a |  | 0.7359 | 0.7987 | 56.64 | 14.83 | 0.3019 | 0.3261 | 0.1352 |
