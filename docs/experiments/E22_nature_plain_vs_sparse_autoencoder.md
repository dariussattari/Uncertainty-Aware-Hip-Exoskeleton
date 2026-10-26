# E22: Plain and sparse (L1) autoencoders on the Nature data: latent structure and "alarm units"

| Field | Value |
|---|---|
| ID | E22 |
| Date | 2026-10-05 to 2026-10-10 |
| Dataset | Nature 2024 |
| Status | Complete |
| Sources | Live_Update.docx, "OCT 5 – 10" §2, §4, §5, §8; README "Nature awareness autoencoders" |
| Code / notebooks | `ml/nature/awareness/ae_data.py`, `train_ae.py` (vanilla, sparse), `ml/common/models/autoencoder.py`; `notebooks/nature/model_exploration/onboard/01_ae_latent_spaces.ipynb`; runs `notebooks/nature/model_exploration/runs/ae/` |

## Hypothesis

An autoencoder trained only on familiar movement learns a compact description of normal walking. A **sparse** variant,
which pays a penalty for every latent unit that is active (like LASSO next to linear regression), forces units to
specialize. Some units might then become **alarm units**, silent on familiar movement and active on unseen movement,
which would give a simple, interpretable warning signal. The hypothesis also held that the 3 s + rhythm input (E19)
would organize the latent by movement type better than 1 s raw.

## Method

- **Data:** ID-safe tasks (normal, incline and weighted walking, stairs) of 14 training participants, with 4 validation
  and 4 test participants (never seen) across all 26 tasks. Three seeds per model.
- **Inputs:** 1 s of the 16 raw channels, and 3 s raw + the seven spectral features as constant channels (the second
  pooling stage becomes 15, so the latent stays 4 × 8 = 32).
- **Models:**
  - Plain AE: the ankle paper's conv AE.
  - Sparse AE: loss = MSE + λ × mean(z / RMS_batch(z)), with λ = 0.3. A plain L1 term fails here, because the
    decoder's BatchNorm makes reconstruction blind to latent scale, so plain L1 shrinks every unit without sparsifying
    any. λ ramps up over 5 epochs (without the ramp, one seed silenced 24 of 32 units). Validation recomputes the
    BatchNorm statistics first.
- **Training:** cluster runs used bf16 autocast. In fp32, cuDNN ran these convolutions as FFTs (551 ms against 23 ms per
  A100 step).
- A "novelty unit" responds at least 5× more strongly to some unseen task than to any training task.

## Result

| Model | Inputs | Validation MSE | Scale-free L1 (1 = dense) | Novelty units |
|---|---|---|---|---|
| Plain | 1 s raw | 0.247 ± 0.005 | 0.90 | 0–1 |
| Sparse λ 0.3 | 1 s raw | 0.313 ± 0.007 | 0.25 | 2 ± 1 |
| Plain | 3 s + spectral | 0.269 ± 0.002 | 0.84 | 0 |
| Sparse λ 0.3 | 3 s + spectral | 0.306 ± 0.014 | 0.10 | **18 ± 7** |

- **1 s input:** walking forms loops (one lap = one stride), and ramps, curbs and stairs share them.
- **3 s + rhythm:** walking collapses into one cluster. Jumping and running, squats and lifts, and standing each move to
  their own region, and stair ascent and descent begin to separate.
- **The sparse model grew alarm units** (10 of 25 active units in the seed-0 figure). They stay off during walking and
  switch on mainly for running, jumping, sit-to-stand and squats/lifts. The plain AE has 0 of 32.
- The sparse code holds on new people with 3 s + spectral input, but not with 1 s input (sparsity 0.15 on training
  people against 0.53 on new ones by epoch 15).

## Conclusion

Longer windows with rhythm features reorganize the latent by movement type. Sparsity produces interpretable alarm units
that the plain AE lacks. Whether either translates into better *detection* is tested in E24 (detection) and E23
(sparsity strength). The alarm units were later proposed as an optional second warning signal.
