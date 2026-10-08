# Krogh-Vedelsby decomposition — Ensemble of 7 TCNs predicting hip angle 200 ms ahead (test split)

Available only because the target is a future sensor reading. Gait phase is recovered from force plates offline, so Experiment 1 can measure branch disagreement and nothing else.

Identity `E = Ebar - A` verified: max residual **3.81e-06** over 84,999 windows.

## Which component detects

| score | AUROC | best achievable J |
|---|---|---|
| A — disagreement | **0.893** | 63.2 |
| \bar{E} — mean branch error | **0.893** | 62.9 |
| E — ensemble error | **0.867** | 59.1 |
| A/(E+\epsilon) | **0.320** | 0.9 |

Experiment 1's gait-phase ensemble reaches AUROC 0.859 for comparison. An AUROC below 0.5 means the score is *anti*-correlated with novelty — the failure the autoencoder's reconstruction error showed at 0.312 and the screening notebook predicted for prediction error at 0.31.

## Quadrant occupancy by task

| task | kind | windows | low A low E | high A low E | low A high E | high A high E |
|---|---|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 91.9% | 2.4% | 2.9% | 2.7% |
| ramp ascent (RA) | ID | 18,504 | 92.4% | 3.0% | 3.9% | 0.7% |
| ramp descent (RD) | ID | 14,784 | 93.3% | 2.6% | 3.1% | 1.0% |
| stair ascent (SA) | OOD | 10,273 | 33.4% | 8.8% | 29.7% | 28.1% |
| stair descent (SD) | OOD | 10,155 | 20.6% | 16.9% | 17.1% | 45.4% |
| transitions (TR) | OOD | 1,994 | 40.7% | 36.4% | 3.8% | 19.2% |
| standing (ST) | OOD | 621 | 0.0% | 100.0% | 0.0% | 0.0% |

The **low A, high E** column is the one a scalar score cannot reach: the branches agreed and were wrong. On in-distribution tasks it should be near zero; where it is not, the ensemble is confidently mistaken and that is a fault signature rather than unfamiliar terrain.

## Per-channel attribution

| channel | AUROC of A | AUROC of E |
|---|---|---|
| `enc_angle_r` | 0.741 | 0.674 |
| `enc_angle_l` | 0.730 | 0.653 |

**Do not select channels on this table and then report the resulting AUROC** — that is circular. Selection has to be fitted on the validation split and evaluated on test.

## Figures

- `kv_by_task` — A, Ebar and E per task, the performance-by-task view
- `kv_quadrants` — the 2-D space and per-task quadrant occupancy
- `kv_roc_compare` — every candidate score on one ROC
- `kv_identity` — the decomposition verified numerically
- `kv_per_channel` — which sensor tripped the flag