# E28: Convolutional VAE grid (β × decoder variance): latent spaces before scoring

| Field | Value |
|---|---|
| ID | E28 |
| Date | 2026-10-13 |
| Dataset | Nature 2024 |
| Status | Complete |
| Sources | 12OCT-15OCT Verbose.docx §3.3 and §10.0; README "VAE grid" bullet |
| Code / notebooks | `ml/nature/awareness/train_vae.py`, `ml/common/models/vae.py`; `notebooks/nature/model_exploration/onboard/02_vae_latent_spaces.ipynb`; runs `notebooks/nature/model_exploration/runs/vae/` |

## Hypothesis

From E17: a VAE can add something the AE lacks only through a **learned decoder variance**. With the variance fixed at
1, posterior width showed little spread. The hypothesis was that a β grid that extends the earlier range (64, 128) and
crosses fixed with learned decoder variance would show (a) how β prunes the latent code, and (b) whether posterior width
singles out unfamiliar tasks.

## Method

- 30 cells: β in {1, 8, 32, 64, 128} × decoder variance {fixed at 1, learned per input channel} × 3 seeds. Raw 1 s
  input, the AE's settings, about 3 min each on an A100. The first cell passed jobstats on 1 core, 1 GB and 74% GPU.
- A grid rather than Bayesian optimization: β 16 and β 32 had the same active dimensions yet differed by 0.09 AUROC in
  E17, so a 1-D optimizer would chase seed noise. Optuna would be justified only at three or more dimensions.
- Each run exports posterior means and variances. A dimension counts as active when its median posterior variance on
  familiar windows is below 0.9. Selection protocol: β and the threshold are chosen on validation, and test is scored
  once (in E29).

## Result

| Decoder variance | β | Active dims (of 32) | KL per window | Validation MSE |
|---|---|---|---|---|
| Fixed | 1 | 32 | 60.2 ± 0.3 | 0.246 ± 0.003 |
| Fixed | 8 | 18.7 ± 1.5 | 17.4 ± 0.1 | 0.312 ± 0.006 |
| Fixed | 32 | 8 | 6.5 ± 0.1 | 0.404 ± 0.003 |
| Fixed | 64 | 8 | 3.9 ± 0.1 | 0.448 ± 0.007 |
| Fixed | 128 | 8 | 1.9 ± 0.1 | 0.530 ± 0.004 |
| Learned | 1 | 32 | 107.8 ± 2.4 | 0.277 ± 0.007 |
| Learned | 8 | 32 | 57.2 ± 2.2 | 0.288 ± 0.003 |
| Learned | 32 | 16 | 15.4 ± 1.0 | 0.368 ± 0.010 |
| Learned | 64 | 8 | 9.4 ± 0.0 | 0.387 ± 0.002 |
| Learned | 128 | 8 | 5.0 ± 0.1 | 0.433 ± 0.002 |
| Reference: plain / Gaussian-NLL AE | | 32 (deterministic) | | 0.247 / 0.247 |

- **β prunes by whole latent filters.** From β 32 only 8 dimensions survive: one filter across its 8 time steps. That
  is the same floor as on the Science Robotics data.
- The learned variance needs about 4× the β for the same pruning. It reconstructs hip angle to 0.09 SD and the sagittal
  gyro to 0.18 SD, against 0.7–0.8 SD for the out-of-plane axes, so the sagittal signals dominate its likelihood.
- At β 1 with fixed variance, the posterior means look like the plain AE's latent (gait rings, a loop for jumping and
  running, standing in the middle). As β grows, families mix; at the 8-dimension floor only jumping and running stay
  together.
- **Posterior width does not single out unfamiliar tasks** in any cell. It is nearly flat across the 26 tasks.

## Conclusion

The VAE's latent behaves as expected under β, and the learned variance re-weights the channels, but posterior width is
not where uncertainty shows. Training was seed-consistent in every cell, so three seeds are enough to separate cells.
Scoring against the Gaussian-NLL control (E26) follows in E29.
