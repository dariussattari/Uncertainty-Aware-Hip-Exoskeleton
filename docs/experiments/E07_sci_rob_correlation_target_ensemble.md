# E07: Ensemble with a synthetic, label-free target (summed channel correlation; Methods "Experiment 4")

| Field | Value |
|---|---|
| ID | E07 |
| Date | 2026-09-16 |
| Dataset | Science Robotics (Molinaro 2024), AB15–AB34 |
| Status | Complete |
| Sources | Methods_Experiments_Reproduction.docx §22–24, §29; Live_Update.docx, "SEP 2 – 18" §8–9 |
| Code / notebooks | `ml/sci_rob/vanilla/` (`targets.py`, `training/synthetic.py`); `notebooks/sci_rob/data_exploration/05_synthetic_targets.ipynb` (screening); `notebooks/sci_rob/model_exploration/04_correlation_ensemble.ipynb` |

## Hypothesis

Gait phase comes from force plates offline, so E03's estimator cannot run on a deployed exoskeleton. A target computed
from the input window itself should keep ensemble disagreement informative while being **available at run time**. This
completes the reference's four-architecture comparison; the reference reports F1 62.5 for this model.

## Method

- **Changed from E03:** only the target and the output head (one linear head). Every hyperparameter, the two-stage LOSO
  protocol, the filter and the threshold are inherited.
- **Target:** the sum of the 120 pairwise Pearson correlations among the 16 channels in the window. The SD denominator
  is floored at 0.01, because standing has 89% of its channels below 0.01 SD and would otherwise give random
  correlations; the floor halves the target's within-standing spread and leaves other tasks bit-identical. The target is
  standardized with training statistics.
- **Training:** LOSO best epochs ranged 4–43 (mean 24.1), so stage 2 ran 24 epochs. Threshold 3.329 × 10⁻².

## Result

- **AUROC 0.813** (0.817 on the shared window set of E10), J 35.12, best J 47.24, recall 42.4%, specificity 92.7%,
  F1 52.35 (reference 62.5), ECE 0.036.
- Flagged at its own threshold: LG 6.2%, RA 1.8%, **RD 16.4%** (the worst ID false-alarm rate of any estimator);
  SA 18.6%, SD 60.5%, TR 55.0%, ST 100%. At a matched 98.1% specificity, SA drops to 3.8%.
- **Participant specificity:** a linear probe on per-channel means and SDs explains 52% of the target's variance on
  shuffled windows but scores R² = −1.68 across participants.

## Conclusion

Label-free learning is viable. This target sits third of the four reference architectures (above the AE and the GAN,
below gait phase), which reproduces the reference's ordering, and it needs no labels. But the target is strongly
participant-specific, so an unknown share of its disagreement on held-out people is **participant shift, not task
novelty**. It lost to the hip-angle forecast target (E08).

## Caveats

- Between-participant and between-task variance were never decomposed.
- The reference's other metrics for this row were not read off the source.
- Single seed.
