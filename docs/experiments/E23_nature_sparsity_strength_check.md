# E23: Was the sparsity too strong? (λ 0.1 and an 8 × 4 latent)

| Field | Value |
|---|---|
| ID | E23 |
| Date | 2026-10-08 to 2026-10-10 |
| Dataset | Nature 2024 |
| Status | Complete |
| Sources | Live_Update.docx, "OCT 5 – 10" §6; README "Nature awareness autoencoders" (λ and 8 × 4 rows) |
| Code / notebooks | `ml/nature/awareness/train_ae.py --sparsity-weight 0.1 --latent 8x4`; `notebooks/nature/model_exploration/onboard/01_ae_latent_spaces.ipynb` |

## Hypothesis

In the λ = 0.3 sparse AE (E22), several of the 32 units move together, which gives its plots a straight-line look. If
the redundancy comes from an overly strong penalty, a weaker penalty (λ = 0.1) or a reshaped latent (8 filters × 4 steps
instead of 4 × 8) should give more independent directions while keeping the alarm units.

## Method

Retrained on 3 s + spectral input with three seeds: sparse λ 0.1; vanilla with an 8 × 4 latent; sparse λ 0.1 with an
8 × 4 latent. Measured validation MSE, scale-free L1, novelty units, the correlation between top units, and effective
directions on held-out people.

## Result

| Model (3 s + spectral) | Validation MSE | Scale-free L1 | Novelty units |
|---|---|---|---|
| Sparse λ 0.3 (E22) | 0.306 ± 0.014 | 0.10 | 18 ± 7 |
| Sparse λ 0.1 | 0.285 ± 0.003 | 0.27 | 4 ± 5 |
| Vanilla, 8 × 4 | 0.353 ± 0.008 | 0.61 | 1 ± 1 |
| Sparse λ 0.1, 8 × 4 | 0.358 ± 0.006 | 0.14 | 17 ± 8 |

- The λ 0.3 latent's strongest units fire as one block (|r| = 1.00 between the top two).
- λ 0.1 was *more* redundant (3.1 effective directions against 4.5 on held-out people) and lost most alarm units.
- The 8 × 4 reshape tied (4.7 directions) at a higher reconstruction cost.

## Conclusion

The duplication comes from the network's structure (units firing together, which a per-unit penalty cannot see), not
from the penalty's strength. **λ = 0.3 stays.**
