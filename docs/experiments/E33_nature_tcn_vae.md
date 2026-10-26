# E33: TCN-VAE: a temporal-convolutional backbone on every input and training class

| Field | Value |
|---|---|
| ID | E33 |
| Date | 2026-10-15 |
| Dataset | Nature 2024 |
| Status | Complete |
| Sources | 12OCT-15OCT Verbose.docx §15.0; README "TCN-VAE" bullet |
| Code / notebooks | `ml/common/models/seq_vae.py` (`TCNVAE`), `train_vae.py --arch tcn`, `slurm/profile_tcn.py`, `slurm/tcn_tasks.txt`; `notebooks/nature/model_exploration/onboard/08_TCN_vae_latent_space.ipynb` |

## Hypothesis

Temporal convolutional networks match or beat recurrent ones on sequence modelling and forecasting (Bai, Kolter & Koltun
2018). A TCN backbone might therefore match or beat the LSTM-VAE (E31) as a novelty detector, and give a better latent
for reading dangerous movement (E32).

## Method

- **Encoder:** residual blocks of two dilated causal convolutions (kernel 3, width 64, BatchNorm, ReLU), with the
  dilation doubling per block until every output sees the whole window: 6 blocks (receptive field 253 samples) at 1 s,
  8 blocks (1,021) at 3 s. Averaged over 8 segments, giving the same 4 × 8 latent.
- **Decoder:** non-autoregressive. The latent is interpolated to every sample, followed by reversed dilations and a
  linear 1 × 1 output. 293–395 k parameters.
- **Runs:** 4 inputs × {β 1, 32, 128 fixed; β 1 learned} × 3 seeds × 3 training classes = 144 runs.
- **Speed:** profiled one A100 step (CLAUDE.md rule 6). The convolutions took only a tenth of the time; the decoder's
  upsampling backward pass, layout conversions and BatchNorm dominated. A fixed interpolation matrix cut the step from
  401 to 299 ms, and `torch.compile` cut it to 164 ms. Runs then took 2–43 min each (71% GPU, 1.5 GB).

## Result

| Model type | 1 s raw | 1 s + rhythm | 3 s raw | 3 s + rhythm |
|---|---|---|---|---|
| **TCN-VAE** | 0.863 (0.952) | 0.883 (0.940) | 0.950 (**0.995**) | 0.904 (0.961) |
| LSTM-VAE | 0.859 (0.924) | 0.868 (0.900) | **0.963** (0.993) | 0.906 (0.949) |
| Transformer VAE | 0.896 (0.977) | 0.876 (0.967) | 0.949 (0.985) | 0.889 (0.981) |
| Plain AE | 0.887 (0.968) | 0.878 (0.933) | 0.950 (0.987) | 0.893 (0.928) |

- It ties the plain AE and the Transformer and sits behind the LSTM-VAE. It ranks dangerous tasks best of any model
  (0.995), flags every dangerous movement (half on the first window), and has 7% familiar false switches.
- The input pattern is the same: 1 s → 3 s raw adds 0.087, and rhythm features cost 0.045 at 3 s. LOF is still best on
  23 of 28 set-ups.
- Its latent sits with the Transformer's (CKA 0.76). It does not collapse at β 128 (2–8 active dimensions).
- **Training-class ablation** (dangerous against OOD-safe, best per input):

| Trained on | 1 s raw | 1 s + rhythm | 3 s raw | 3 s + rhythm | Linear read-out, best |
|---|---|---|---|---|---|
| ID | 0.779 | 0.802 | 0.796 | 0.835 | 0.905 |
| ID + OOD-safe | 0.747 | 0.748 | 0.666 | 0.777 | **0.924** |
| OOD-safe only | 0.785 | 0.791 | 0.678 | 0.701 | 0.917 |

- Its ID-trained latent is the only one that tells step-ups from safe movement (0.861). Trained on safe movement, it
  drops to chance (0.53–0.56).

## Conclusion

The TCN is a good encoder but not a better novelty detector: the **LSTM-VAE on 3 s raw stays the alarm** (0.963). The
**TCN-VAE trained on ID + OOD-safe with 3 s + rhythm input is the strongest latent to read dangerous movement from**
(0.924), so it became the regime classifier's encoder (E34). E32's conclusion is confirmed most clearly here.
