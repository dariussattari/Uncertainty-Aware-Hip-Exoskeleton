# E05: Convolutional autoencoder with latent LOF (Methods "Experiment 2")

| Field | Value |
|---|---|
| ID | E05 |
| Date | 2026-09-11 |
| Dataset | Science Robotics (Molinaro 2024), AB15–AB34 |
| Status | Complete |
| Sources | Methods_Experiments_Reproduction.docx §12–16; Live_Update.docx, "SEP 2 – 18" §6 |
| Code / notebooks | `ml/common/models/autoencoder.py`, `ml/sci_rob/vanilla/` (`training/autoencoder.py`, `evaluation/autoencoder.py`); `notebooks/sci_rob/model_exploration/02_autoencoder_lof.ipynb` |

## Hypothesis

A label-free convolutional autoencoder trained on ID windows learns a compact manifold of familiar movement. **Local
density in its latent space** (Local Outlier Factor) flags held-out tasks. **Reconstruction error does not**, because,
as the reference argues, low-complexity stationary data such as standing is easy to reconstruct. The second half is a
negative result to reproduce, not assume.

## Method

- **Setup:** as E03 (16 channels, 1 s windows, stride 10, 12/4/4 participants, ID = LG/RA/RD, OOD = SA/SD/ST/TR,
  training-only scaler). No labels are needed, which recovers 19,037 extra training windows (187,993 in total).
- **Architecture:** encoder 16 → 19 → 24 → 4 channels, kernels 15/17/19, BatchNorm and ReLU, two AvgPool(5) stages
  (25× compression), giving a latent of 4 filters × 8 steps = 32 values. The decoder mirrors it with linear upsampling.
  28,558 parameters.
- **Training:** single stage, with 20% of training participants held out for early stopping (AB18, AB28), patience 5.
  It ran 42 epochs and was best at epoch 36. Adam, MSE.
- **Score:** sklearn LOF (k = 20, negated so that higher means more anomalous) on the 32-D latent, using a 50,000-point
  reference set with duplicate latents removed. The same causal median and 99.5th-percentile threshold as E03 (1.2427).
- **Deviation:** learning rate 0.003 instead of the specified 0.064. At 0.064 the reconstruction MSE plateaus at 0.959
  (the model predicts channel means), 63% of latents are identical, and the LOF threshold blows up to 3.5 × 10⁸.

## Result

| Metric | LOF on latent | Reconstruction error | Reference AE |
|---|---|---|---|
| J | 38.72 | −0.50 | 58.5 ± 16.9 |
| AUROC | **0.782** | **0.312** | – |
| Recall / specificity | 45.1% / 93.7% | 0.0% / 99.5% | 94.7% / – |
| Precision / F1 | 72.5% / 55.6% | 0 / 0 | 40.0% / 55.7% |
| ECE / Brier | 0.042 / 0.147 | – | 0.26 / 0.18 |

- Flagged per task (LOF): LG 4.6%, RA 6.6%, RD 9.3%; SA 27.0%, SD 57.6%, TR 57.3%, ST 100%. Detection rises
  monotonically with distance from the training distribution.
- Median reconstruction error ranks the tasks by **signal complexity**: ST 0.074 < TR 0.118 < SA 0.225 < RD 0.234 <
  SD 0.241 < LG 0.289 < RA 0.309. Standing is easiest to rebuild even though it is the most unfamiliar task.

## Conclusion

Latent density works and reconstruction error is anti-correlated with novelty. Both halves of the reference's claim are
reproduced on hip data. The F1 match (55.6 against 55.7) is coincidental: recall and precision are inverted relative to
the reference because this OOD set is harder. The autoencoder detects stair ascent 4× better than the gait-phase
ensemble (27.0% against 6.6%), so **the two estimators fail in different places**. At matched specificity the ensemble
is still stronger overall (E10).

## Caveats

- The lr override means the result rests on a hyperparameter chosen here, not inherited.
- LOF in 32 dimensions is at the edge of what the method suits.
- k = 20 had no sensitivity analysis.
- Single seed.
