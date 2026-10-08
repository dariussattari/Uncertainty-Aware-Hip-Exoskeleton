# Pelvis-IMU ablation under the `DOWN` holdout (test split)

Held out: RD, SD, ST, TR; trained on LG, RA, SA. Same encoder (β = 0, no sampling, AE settings), same LOF, same seed, same test windows; the only change is six pelvis IMU channels appended to the input (their own scaler, the 16 shared channels scaled identically). AUROC is threshold-free; OOD is the positive class; scores are causally median-filtered within trials.

| encoder | score | AUROC OOD vs ID | AUROC RD vs ID | AUROC SD vs ID | AUROC ST vs ID | AUROC TR vs ID |
|---|---|---|---|---|---|---|
| 16 channels (thigh + encoders) | LOF, 32-D latent | 0.9741 | 0.9794 | 0.9830 | 0.9976 | 0.8824 |
| 16 channels (thigh + encoders) | reconstruction NLL | 0.4863 | 0.5717 | 0.4668 | 0.0026 | 0.1037 |
| 16 channels (thigh + encoders) | LOF, PCA-12 | 0.9565 | 0.9525 | 0.9831 | 0.9976 | 0.8379 |
| 16 channels (thigh + encoders) | LOF, PCA-16 | 0.9576 | 0.9557 | 0.9813 | 0.9976 | 0.8385 |
| 22 channels (+ pelvis IMU) | LOF, 32-D latent | 0.9484 | 0.9498 | 0.9682 | 0.9976 | 0.8223 |
| 22 channels (+ pelvis IMU) | reconstruction NLL | 0.6113 | 0.6985 | 0.6234 | 0.0000 | 0.0927 |
| 22 channels (+ pelvis IMU) | LOF, PCA-12 | 0.9474 | 0.9476 | 0.9645 | 0.9985 | 0.8433 |
| 22 channels (+ pelvis IMU) | LOF, PCA-16 | 0.9537 | 0.9505 | 0.9729 | 0.9977 | 0.8659 |

## Runs

| encoder | input channels | epochs run | best epoch | best val loss | seed | hours | logistic probe ID accuracy | mlp probe ID accuracy |
|---|---|---|---|---|---|---|---|---|
| 16 channels (thigh + encoders) | 16 | 29 | 24 | 3468 | 0 | 0.6589 | 0.891 | 0.9105 |
| 22 channels (+ pelvis IMU) | 22 | 30 | 25 | 4811 | 0 | 0.7274 | 0.9266 | 0.946 |
