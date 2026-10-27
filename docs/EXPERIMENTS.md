# Experiment log: Uncertainty-Aware Hip Exoskeleton

Every experiment run so far, in chronological order, each with a one-line result. Each `experiments/E##_*.md` file
stands alone and has a **Hypothesis, Method, Result, Conclusion** section, plus a header table giving the date, dataset,
source document and section, and code/notebook paths.

**Sources.** The experiments were extracted from three documents in
`~/Desktop/OneDrive - Harvard University/thesis/` (versions of 2026-10-18; Methods of 2026-09-21):

- `research_updates/Live_Update.docx`, the weekly log.
- `research_updates/12OCT-15OCT Verbose.docx`, the detailed Oct 12–18 write-up.
- `Methods_Experiments_Reproduction.docx`, the thesis methods draft for the Science Robotics experiments.

Gaps were filled from the repository: notebook findings (06, 09, 11, 12), `ml/sci_rob/awareness/README.md` and the main
`README.md`. Dates come from the weekly headings, git commits and result-file timestamps.

**Reading the numbers.**

- AUROC is the chance that an unfamiliar window scores above a familiar one (0.5 = chance).
- Youden J = recall + specificity − 1.
- *Science Robotics* experiments (E01–E17) use a 12/4/4 participant split, ID = level ground and ramps, held-out =
  stairs, standing and transitions. Their alarm threshold is the ankle paper's 99.5th percentile of training scores, and
  they are single-seed unless noted.
- *Nature* experiments (E18–E41) use a 14/4/4 split, ID-safe = normal, incline and weighted walking plus stairs (22
  unfamiliar tasks; dangerous = ball toss, step-ups, tug of war, sit-to-stand). Their threshold is 1% false alarms on
  the validation participants, they report mean ± SD over 3 seeds, and the test set is scored once.

## Chronological index

### Phase 1: Science Robotics dataset (Molinaro 2024): reproduce the ankle paper on hip data (Sep 2–29)

| # | Date | Experiment | Quick result |
|---|---|---|---|
| [E01](experiments/E01_sci_rob_dataset_audit.md) | Sep 3–4 | Dataset audit and sensor-axis conventions | 34 subjects, 12 h. Gyro-y is sagittal (r ≈ −0.90), IMUs not mirrored. Use phases 3–4 only (AB15–AB34) |
| [E02](experiments/E02_sci_rob_pca_geometry.md) | Sep 3–4 | Initial geometry (PCA distance and density) | PCs encode gait-cycle position. Standing sits at the ID centre, so distance scores fail |
| [E03](experiments/E03_sci_rob_gait_phase_ensemble.md) | Sep 4–5 | Gait-phase TCN ensemble (reference reproduction) | AUROC 0.859, J 37.5. Stair ascent is a blind spot (6.6% flagged). Target unavailable live |
| [E04](experiments/E04_sci_rob_kalman_nis.md) | Sep 5–9 | Kalman innovation (NIS) | Not a novelty signal (AUROC 0.413; gait-locked, model mismatch). Kept for sensor health |
| [E05](experiments/E05_sci_rob_autoencoder_lof.md) | Sep 11 | Conv autoencoder + latent LOF | LOF AUROC 0.782, reconstruction error 0.312 (inverted). Stair ascent 27% (4× the ensemble) |
| [E06](experiments/E06_sci_rob_gan_discriminator.md) | Sep 15–16 | GAN discriminator | AUROC 0.727, below raw amplitude (0.807). The score is an amplitude proxy (chance within amplitude deciles) |
| [E07](experiments/E07_sci_rob_correlation_target_ensemble.md) | Sep 16 | Ensemble with a summed-correlation target | AUROC 0.813, label-free, but participant-specific (cross-person R² −1.68) |
| [E08](experiments/E08_sci_rob_hip_angle_forecast_ensemble.md) | Sep 16 | Ensemble forecasting hip angle 200 ms ahead | **Best on this data: AUROC 0.899, J 47.9.** Live target. Stair ascent 21.9% |
| [E09](experiments/E09_sci_rob_krogh_vedelsby.md) | Sep 16–18 | Krogh–Vedelsby: disagreement against realized error | A 0.893 ≈ Ē 0.893 > E 0.867. A/(E+ε) fails (0.320). The low-A/high-E quadrant exposes confident errors |
| [E10](experiments/E10_sci_rob_five_estimator_comparison.md) | Sep 16–18 | Five estimators at matched specificity | Forecast best above about 98.5% FA tolerance; gait phase best in the strict regime. AE's J edge is an artefact |
| [E11](experiments/E11_sci_rob_probabilistic_ensemble.md) | Sep 18–19 | Probabilistic ensemble (aleatoric against epistemic) | Epistemic AUROC 0.914. Aleatoric is 2× over-confident on new people. Heavy-tailed z² |
| [E12](experiments/E12_sci_rob_leave_one_task_out.md) | Sep 19–22 | Leave-one-task-out | No score responds to familiarity. Withholding standing *lowers* scores. Winners are task-dependent |
| [E13](experiments/E13_sci_rob_umap_latent_exploration.md) | Sep 22–23 | UMAP of raw data and AE latent | SA lies next to RA (and SD next to RD) in every representation, which is why SA is hard |
| [E14](experiments/E14_sci_rob_down_holdout_awareness_layer.md) | Sep 23 | DOWN regime holdout: novelty, regime, reliability | Latent LOF 0.974–0.976. Regime probes absorb unseen regimes. Regime-aware forecasting adds nothing |
| [E15](experiments/E15_sci_rob_fault_injection.md) | Sep 23 | Injected sensor faults (four-cell grid) | LOF × reconstruction error resolves faults (0.76–1.00). Forecast error does not |
| [E16](experiments/E16_sci_rob_pelvis_imu_ablation.md) | Sep 23–24 | Pelvis IMU (UMAP + 22-channel encoder) | The pelvis IMU *hurts* novelty (0.974 → 0.948). Keep 16 channels |
| [E17](experiments/E17_sci_rob_vae_beta_sweep.md) | Sep 22–29 | VAE / β-VAE sweep | VAE never beats the AE (best 0.761). The linear-latent Gaussian-NLL control reaches 0.822 |

*Design milestone (Sep 25–29):* the pipeline was proposed: a latent gives novelty and regime, a forecaster gives
reliability, and a classifier assigns ID / OOD-safe / OOD-dangerous. The framing moved from "ID against OOD" to "which
controller, and can we still operate" (E14).

### Phase 2: Nature dataset (Molinaro, Scherpereel 2024): the three-regime awareness layer (Sep 25 – Oct 24)

| # | Date | Experiment | Quick result |
|---|---|---|---|
| [E18](experiments/E18_nature_dataset_audit_and_controller_classes.md) | Sep 25 – Oct 2 | Dataset audit and controller-class binning | Usable after a one-time fix-up. 26 tasks split into ID-safe 5 h / OOD-safe 17 h / OOD-dangerous 4 h |
| [E19](experiments/E19_nature_engineered_features_boosted_trees.md) | Oct 5–8 | Engineered features (boosted trees) | Spectral features and 3 s history help (+0.05 controller class). Jerk and orientation do not |
| [E20](experiments/E20_nature_feature_embeddings_pca_umap.md) | Oct 5–8 | Feature families in PCA/UMAP (k-NN) | 3 s + spectral best. Controller classes form no clusters in any space |
| [E21](experiments/E21_nature_spectral_no_rhythm_fix.md) | Oct 8 | Fix: rhythm features report 0 when there is no rhythm | Rankings barely move (+2.9/+3.2 points instead of +3.5/+4.4). Features now honest |
| [E22](experiments/E22_nature_plain_vs_sparse_autoencoder.md) | Oct 5–10 | Plain and sparse AE on Nature | 3 s + rhythm groups movement types. The sparse AE grows alarm units (18 ± 7) |
| [E23](experiments/E23_nature_sparsity_strength_check.md) | Oct 8–10 | Sparsity strength (λ 0.1, 8 × 4 latent) | Redundancy is structural, not penalty strength. λ 0.3 stays |
| [E24](experiments/E24_nature_ae_lof_detection.md) | Oct 8–10 | AE latent LOF detection (ankle threshold) | Plain AE 1 s: AUROC 0.89 (dangerous 0.97). Threshold too cautious (29% flagged). Quiet tasks missed |
| [E25](experiments/E25_nature_threshold_tuning_and_pca_lof.md) | Oct 11–13 | Validation-tuned threshold + PCA-k LOF | Little gain. The false-alarm rate is set by the wearer, so calibrate per wearer. The FA target is the lever |
| [E26](experiments/E26_nature_gaussian_nll_ae_control.md) | Oct 13 | Gaussian-NLL AE control (linear latent) | 3 s + rhythm: 0.911 (best at the time). Becomes the baseline the VAEs must beat |
| [E27](experiments/E27_nature_estimator_error_labels.md) | Oct 13 | Danger labels from the deployed estimator's error | Only the ball toss is clearly dangerous. 8 of 22 tasks near the boundary. Push: 12% wrong-direction torque |
| [E28](experiments/E28_nature_conv_vae_grid_latents.md) | Oct 13 | Conv VAE grid (β × decoder variance), latents | β prunes by whole filters (8-dimension floor). Posterior width is flat across tasks |
| [E29](experiments/E29_nature_native_metric_scoring.md) | Oct 13 | Every native metric scored | 1 s VAE KL at β 128, read downward: 0.901 (dangerous 0.989). Likelihood and posterior width near chance |
| [E30](experiments/E30_nature_vae_3s_score_pairs_onset_latency.md) | Oct 13–14 | 3 s VAE, cross-model score pairs, onset latency | Pair 1 s KL + 3 s LOF = 0.950. 1 s KL flags every dangerous bout within 0.9 s |
| [E31](experiments/E31_nature_sequence_vaes_input_ablation.md) | Oct 14 | LSTM/Transformer VAEs + full input ablation | **Best detector: LSTM-VAE, 3 s raw, LOF = 0.963 (dangerous 0.993).** Rhythm features hurt detection |
| [E32](experiments/E32_nature_training_class_ablation.md) | Oct 14–15 | Training-class ablation (train on safe movement) | Training on safe movement *lowers* dangerous detection (0.757 → 0.69). A supervised read-out gets 0.92 |
| [E33](experiments/E33_nature_tcn_vae.md) | Oct 15 | TCN-VAE on every input and training class | 0.950 (dangerous 0.995). Best latent for reading danger (0.924) |
| [E34](experiments/E34_nature_regime_classifier.md) | Oct 15 | Supervised regime classifier | **Macro F1 0.868. No dangerous window reaches the spline**, even for unseen tasks. 2.8% of safe windows switched off |
| [E35](experiments/E35_nature_novelty_regime_fusion_uncertainty_score.md) | Oct 15 | Novelty + classifier fusion; uncertainty score *u* | Stacking does not help (0.856). *u* = 0.5·p_safe + p_danger with cutoffs 0.25/0.825 matches F1 0.868 |
| [E36](experiments/E36_nature_gait_phase_ensemble_baseline.md) | Oct 18 | Ankle paper's gait-phase ensemble on Nature | Good alarm (dangerous 0.992) but not a regime selector (F1 0.60 against 0.868). Ψ tracks cycle distance |
| [E37](experiments/E37_nature_supplementary_latent_views.md) | Oct 20 | Latent views of the classifier's two encoders | Novelty puts OOD-safe and dangerous in the same periphery (0.63–0.78 between them) |
| [E38](experiments/E38_nature_classifier_permutation_importance.md) | Oct 20 | Classifier permutation importance | Novelty scores serve ID-safe (−0.86); latent serves dangerous (−0.78). LOF is redundant |
| [E39](experiments/E39_nature_classifier_training_participants.md) | Oct 22 | Regime classifier fitted on the 14 training participants | Features only mildly in-sample. With *u* cutoffs on val: **F1 0.875, 0% dangerous on the spline, 2.7% safe off**. Unseen dangerous switched off 41% (was 26%) |

### Phase 3: Nature dataset: the OOD-safe fallback controller (Oct 24 –)

| # | Date | Experiment | Quick result |
|---|---|---|---|
| [E40](experiments/E40_nature_moment_estimator_reproduction.md) | Oct 24 | The paper's joint moment estimator, retrained as published (Model A) | Test (Phase 3) hip R² 0.745 ± 0.006, knee 0.771; beats the deployed estimator at the hip (0.596). Unseen tasks no worse than trained. BT07 hip bias |
| [E41](experiments/E41_nature_moment_estimator_deployment_sensors.md) | Oct 24 | The estimator on hip encoder + thigh IMU (B0), trained across IMU fits (B1) | B0 matches A on trained tasks, loses 0.16 R² on unseen ones (stair ascent 0.08). Moving the IMU costs B0 0.075, B1 0.013. **B1 is the deployment candidate** (Phase 3 hip R² 0.70) |

## Where things stand (as of 2026-10-25)

- **Unfamiliar-movement alarm:** LSTM-VAE (β 1, fixed variance) on 3 s raw windows, LOF or Mahalanobis on its latent
  (E31).
- **Three-way controller choice:** a boosted-tree regime classifier on the TCN-VAE latent (trained on ID + OOD-safe, 3 s
  + rhythm), the LSTM-VAE novelty scores and the rhythm features, collapsed to *u* with two cutoffs (E34, E35). It is
  now fitted on the 14 training participants with the cutoffs set on the validation participants (E39: F1 0.875).
- **OOD-safe fallback estimator:** B1, the paper's TCN on hip encoder + thigh IMU, trained across IMU rotation and
  lever-arm perturbations (E41: Phase 3 hip R² 0.70, robust to a moved IMU).
- **Sensor health:** reconstruction error and disagreement for faults (E15), plus matched Kalman residuals for timing
  faults (E04).

**Open threads** (ordered as a deployment plan in [`deployability_status.md`](../deployment/deployability_status.md)):

- Apply the E27 labelling rule to B1, the project's own hip-and-thigh estimator (E41).
- Fit the torque ramp to the estimator's measured error.
- Test a classifier without LOF.
- Run a stair-descent LOTO holdout.
- Add a hip-encoder zero offset to B1's randomization (E40).
- Hip-angle forecaster training on Nature.
- Genuinely non-cyclic OOD data.
