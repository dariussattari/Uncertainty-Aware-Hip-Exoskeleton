# E31: LSTM-VAE and Transformer VAE, plus the full input ablation (window length × rhythm features)

| Field | Value |
|---|---|
| ID | E31 |
| Date | 2026-10-14 |
| Dataset | Nature 2024 |
| Status | Complete |
| Sources | 12OCT-15OCT Verbose.docx §4.0, §5.0, §13.0; README "Sequence VAEs and the full input ablation"; Live_Update.docx "OCT 10 – 16" |
| Code / notebooks | `ml/common/models/seq_vae.py` (`LSTMVAE`, `TransformerVAE`), `train_vae.py --arch lstm/transformer`, `ae_data.py --ablation-only`; notebooks `onboard/03_LSTM_vae_latent_space.ipynb`, `04_Xformer_vae_latent_space.ipynb`, `05_Novelty_Metric_comparison.ipynb` section 9 |

## Hypothesis

- **Sequence models:** an LSTM decoder gives a per-timestep NLL, which could cut onset latency (Park, Hoshi & Kemp
  2018). A Transformer is gated on the LSTM showing that sequence modelling helps at all. Neither was expected to fix
  quiet movement.
- **Input ablation:** the two inputs so far (1 s raw against 3 s raw + rhythm) differed in two ways at once. Adding the
  missing corners (1 s + rhythm, 3 s raw) separates the effect of window length from the effect of the rhythm features.

## Method

- **Architectures:** all share the 4 × 8 Gaussian latent and the ELBO, with non-autoregressive decoders (the decoder can
  reach the window only through the latent, which guards against posterior collapse).

| Model | Encoder | Decoder | Parameters |
|---|---|---|---|
| Conv VAE | AE conv blocks + pooling | mirrored conv | 30–34 k |
| LSTM-VAE | 2-layer LSTM (hidden 64), averaged over 8 segments | latent interpolated to every sample, 2-layer LSTM | 107–109 k |
| Transformer VAE | 40 patches, 3 pre-norm attention layers (width 64, 4 heads), 8 groups of 5 | latent interpolated to 40 tokens, 3 attention layers | 217–251 k |

- **The four inputs:** 1 s raw, 1 s + rhythm (the seven rhythm features from the longer history, held as constant
  channels), 3 s raw, 3 s + rhythm. All are views of the same 3 s window table, so every comparison uses the same
  windows.
- **Runs:** 138 new runs, each on 1 core and 1 A100 (first tasks: 3.4–5.6 min, 1.1–1.3 GB, 67–83% GPU).
  - LSTM and Transformer VAEs: 4 inputs × {β 1, 32, 128 fixed; β 1 learned} × 3 seeds.
  - Conv VAE and the three AEs on the two new inputs.
- **Scoring:** as in E29 and E30. Within each (model type, input), the cell and score are chosen on validation, then
  compared across model types and inputs, latent to latent with linear CKA, and per movement (onset).

## Result

Test AUROC, unfamiliar windows (dangerous in brackets):

| Model type | 1 s raw | 1 s + rhythm | 3 s raw | 3 s + rhythm |
|---|---|---|---|---|
| Plain AE | 0.887 (0.968) | 0.878 (0.933) | 0.950 (0.987) | 0.893 (0.928) |
| Sparse AE | 0.846 (0.941) | 0.762 (0.871) | 0.907 (0.937) | 0.811 (0.830) |
| Gaussian-NLL AE | 0.887 (0.966) | 0.868 (0.902) | 0.943 (0.987) | 0.914 (0.956) |
| Conv VAE | 0.901 (0.989) | 0.869 (0.967) | 0.939 (0.988) | 0.910 (0.934) |
| **LSTM-VAE** | 0.859 (0.924) | 0.868 (0.900) | **0.963 (0.993)** | 0.906 (0.949) |
| Transformer VAE | 0.896 (0.977) | 0.876 (0.967) | 0.949 (0.985) | 0.889 (0.981) |

- **3 s raw is the best input for every model type.** The best detector is the LSTM-VAE on 3 s raw (β 1, fixed
  variance, LOF): 0.963 ± 0.004, dangerous 0.993. It flags 77% of unfamiliar and 88% of dangerous windows at 0.8% false
  alarms. That beats E30's best *supervised* pair (0.950).
- **The rhythm features hurt detection.** They cost 0.03–0.10 at 3 s for every type. Going from 1 s to 3 s raw adds
  0.04–0.10 (the LSTM-VAE gains most, +0.10).
- **Metric by metric:** LOF on the latent is the best score on 20 of 24 set-ups. On the LSTM-VAE's 3 s raw latent the
  cheap scores also work: Mahalanobis reaches 0.950 (dangerous 0.997), and posterior variance 0.897 (higher =
  unfamiliar). That is the VAE uncertainty signal the conv VAE never gave.
- **The rhythm features, not the window, set the KL's direction:** downward on raw input for every architecture,
  upward with rhythm features. Correction to E30.
- The likelihood needs the rhythm features: NLL/ELBO reach 0.85–0.89 with them and 0.59–0.72 without.
- **Linear CKA:**
  - Latents with rhythm features are alike (0.67), raw latents less so (0.36), and mixed pairs least (0.15).
  - The same input makes model types alike (0.53) more than the same model makes inputs alike (0.33).
  - The rhythm features make every architecture encode whether the movement is rhythmic: a readout finds the no-rhythm
    windows at AUROC 0.97–1.00. The constant channels crowd out the movement's shape.
  - The best detector (LSTM-VAE, 3 s raw) learns the *most different* latent (CKA 0.27–0.47 with the others).
- **Per movement:** the 3 s raw VAEs catch 100% of dangerous movements at a median latency of 0.0 s, and 94–96% of
  OOD-safe movements, with only 4–6% familiar false switches. Correction to E30: the 11–20% familiar switches came from
  the rhythm features.
- **Latent spaces:**
  - β 128 collapses the sequence VAEs on raw input (KL = 0 for the LSTM on 1 s and 3 s, and the Transformer on 1 s).
  - At β 1 the Transformer uses its whole code and reconstructs best.
  - The LSTM-VAE's posterior width tracks familiarity: 0.00–0.02 on familiar tasks against 0.04–0.07 on quiet,
    posture-driven tasks. That is the first sign against the expectation that quiet movement stays invisible to any
    encoder.

## Conclusion

Run the novelty alarm on **3 s raw windows, without rhythm features**. The best detector is the **LSTM-VAE (β 1, fixed
variance)**: LOF on its latent, or Mahalanobis where a controller cannot afford LOF's neighbour search. It flags every
dangerous movement, most on the first window, with 6% familiar false switches. This supersedes E30's choice (0.901 →
0.963). Keep the rhythm features for the regime classifier, where they separate task families.
