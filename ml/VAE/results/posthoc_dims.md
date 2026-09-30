# LOF in fewer dimensions (post hoc, no retraining)

LOF refitted on 50,000 in-distribution training latents (same seed as the production fit) and scored on validation and test with the usual per-trial median filter. Select on validation; test is reported only. "Active dims" are those with median posterior σ² < 0.9 on ID validation windows.

| model | representation | dims | val AUROC | test AUROC | variance kept |
|---|---|---|---|---|---|
| VAE original_standard_seed0 | all 32 dims | 32 | 0.6213 | 0.6474 |  |
| VAE original_standard_seed0 | active dims only | 32 | 0.6213 | 0.6474 |  |
| VAE original_standard_seed0 | PCA-8 | 8 | 0.7252 | 0.7978 | 0.4695 |
| VAE original_standard_seed0 | PCA-12 | 12 | 0.6963 | 0.7617 | 0.6027 |
| VAE original_standard_seed0 | PCA-16 | 16 | 0.65 | 0.6784 | 0.7204 |
| VAE original_standard_seed0 | PCA-18 | 18 | 0.6339 | 0.6639 | 0.7744 |
| VAE original_standard_seed0 | PCA-24 | 24 | 0.6081 | 0.645 | 0.9019 |
| VAE original_beta4_seed0 | all 32 dims | 32 | 0.7055 | 0.7094 |  |
| VAE original_beta4_seed0 | active dims only | 18 | 0.7034 | 0.7026 |  |
| VAE original_beta4_seed0 | PCA-8 | 8 | 0.6756 | 0.682 | 0.7765 |
| VAE original_beta4_seed0 | PCA-12 | 12 | 0.7064 | 0.6756 | 0.9059 |
| VAE original_beta4_seed0 | PCA-16 | 16 | 0.7017 | 0.6875 | 0.9801 |
| VAE original_beta4_seed0 | PCA-18 | 18 | 0.7035 | 0.7028 | 0.9952 |
| VAE original_beta4_seed0 | PCA-24 | 24 | 0.7056 | 0.7093 | 0.9999 |
| AE autoencoder | all 32 dims | 32 | 0.7149 | 0.7819 |  |
| AE autoencoder | PCA-8 | 8 | 0.7301 | 0.7933 | 0.8424 |
| AE autoencoder | PCA-12 | 12 | 0.7348 | 0.8102 | 0.9094 |
| AE autoencoder | PCA-16 | 16 | 0.7298 | 0.8227 | 0.9421 |
| AE autoencoder | PCA-18 | 18 | 0.7288 | 0.8208 | 0.9539 |
| AE autoencoder | PCA-24 | 24 | 0.7215 | 0.7969 | 0.9795 |
