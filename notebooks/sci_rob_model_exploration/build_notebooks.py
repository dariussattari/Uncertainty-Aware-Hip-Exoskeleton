"""Emit a patch for, or write, the nine exploration notebooks (no training)."""
from pathlib import Path
from textwrap import dedent
import argparse
import hashlib
import nbformat as nbf

HERE=Path(__file__).resolve().parent


def md(text):return nbf.v4.new_markdown_cell(dedent(text).strip())
def code(text):return nbf.v4.new_code_cell(dedent(text).strip())


SETUP='''
from pathlib import Path
import sys
ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / 'ml/vanilla/registry.py').exists()), None)
if ROOT is None:
    raise RuntimeError('Launch Jupyter within this repository.')
sys.path.insert(0, str(ROOT / 'notebooks/model_exploration'))
get_ipython().run_line_magic('matplotlib', 'inline')
from _exploration import *
import torch
from dataclasses import asdict
torch.set_num_threads(2)  # small architecture checks; training CLI runs separately
torch.manual_seed(0)
RUN_TRAINING = False
print('Repository:', ROOT)
print('Kernel:', sys.executable)
'''

PROTOCOL=r'''
## Data and evaluation contract

- Input: **16 channels × 200 samples**, one second at 200 Hz; stored window stride 10 samples.
- Original ID tasks: level ground (LG), ramp ascent (RA), ramp descent (RD).
  Original pseudo-OOD tasks: stair ascent (SA), stair descent (SD), standing (ST), transitions (TR).
- Participants: 12 train / 4 validation / 4 test. Task availability differs by participant.
  Input scaling is fitted on training data, never refitted on test data.
- Scoring does **not** predict a task label. High $\Psi$ means more anomalous.

$$\widetilde\Psi_t=\operatorname{median}(\Psi_{t-9},\ldots,\Psi_t),\qquad
\tau=Q_{0.995}(\widetilde\Psi_{\mathrm{train,ID}}),\qquad
\widehat o_t=\mathbf1[\widetilde\Psi_t>\tau].$$

Filtering restarts for each trial; early windows use the available history. The ten-score
filter corresponds to the repository's nominal 0.5 s smoothing setting, **not** ten raw samples.

$$\mathrm{recall}=\frac{TP}{TP+FN},\quad
\mathrm{specificity}=\frac{TN}{TN+FP},\quad J=\mathrm{recall}+\mathrm{specificity}-1.$$

AUROC measures ranking across thresholds; it is **not the fraction of OOD detected**.
OOD is the positive class. Tables below show J on −1 to 1, although source JSON stores 100J.
Scores and thresholds are not calibrated probabilities of being unsafe.
'''

ENSEMBLE_MATH=r'''
For branch $m$ and target dimension $d$, train with masked mean squared error:

$$\mathcal L=\frac{\sum_{n,m,d}a_{nd}(f_{md}(x_n)-y_{nd})^2}
{M\sum_{n,d}a_{nd}},\qquad M=7.$$

$a$ is the target-validity mask. The online score uses no target labels:

$$\bar f_d(x)=\frac1M\sum_m f_{md}(x),\qquad
\Psi(x)=\frac1D\sum_d\frac1M\sum_m(f_{md}(x)-\bar f_d(x))^2.$$

This is population variance across branches, not sample variance. Agreement is a candidate
uncertainty signal, **not a guarantee that an input is familiar or a prediction is correct**.
'''

ENS_TRAIN='''
Twelve leave-one-subject-out (LOSO) folds over the training participants select the final
epoch count from held-out-subject loss. A freshly initialized ensemble then trains on all
training participants for that duration. Defaults: Adam, learning rate 0.001, batch 1024,
maximum 60 fold epochs, patience 10. The report below is authoritative for a saved run.
Branches have independent weights/initialization but see the same training batches.
'''

COMMON_GALLERY=[
    ('psi_by_task.png','Score distributions by task. The threshold is an operating point, not an ID/OOD boundary guaranteed by the model.'),
    ('roc.png','Threshold sweep: ranking quality, not the fraction detected at the saved threshold.'),
    ('detection_by_task.png','Actual flag rates at the saved training threshold; flags on ID tasks are false alarms.'),
    ('loso.png','Training duration selection across held-out training participants; this is LOSO, not task holdout.'),
]

SPECS=[
dict(file='01_gait_phase_ensemble',key='ensemble',title='Gait-phase TCN ensemble',
 question='Can disagreement between gait-phase predictors flag unfamiliar movement?',
 math=r'''The target is bilateral sine-transformed gait phase, not a moving/stationary class:
 $$y_d(t)=\sin(2\pi\phi_d(t)),\qquad \phi_d\in[0,1).$$
 The output head is tanh. Missing leg labels are masked; windows require at least one valid label.
 Each branch has three causal Conv1d → BatchNorm → ReLU blocks, 30 filters, kernel 20,
 dilations (1,2,8). Its theoretical receptive field is
 $$R=1+(20-1)(1+2+8)=210\text{ samples},$$
 covering the available 200-sample window (with left padding).
 '''+ENSEMBLE_MATH,
 flow=[['Input\n16 × 200','7 independent TCNs\n30 filters; d=1,2,8','Last-time-step heads\n2 outputs; tanh','Branch variance\nmean over both legs']],
 model="from models.ensemble import create_ensemble\nmodel = create_ensemble(16, 2)",
 train=ENS_TRAIN, gallery=COMMON_GALLERY,
 sources=['ml/vanilla/models/ensemble.py','ml/vanilla/training/ensemble.py','ml/vanilla/evaluation/ensemble.py'],
 takeaway='Low disagreement on standing does not prove correct gait-phase prediction. In LOTO, all branches can agree on an unseen input. Inspect prediction errors separately; gait-phase labels also restrict which windows enter this model.'),
dict(file='02_autoencoder_lof',key='autoencoder',title='Convolutional autoencoder + latent LOF',
 question='Does local density in a learned sensor representation identify unfamiliar movement?',
 math=r'''The encoder compresses the input to 4 filters × 8 timesteps = 32 latent values:
 $$z=E_\theta(x),\qquad \widehat x=D_\theta(z),\qquad
 \mathcal L_{\rm recon}=\frac1{NCT}\sum_{n,c,t}(x_{nct}-\widehat x_{nct})^2.$$
 Encoder channels: 16 → 19 → 24 → 4, kernels 15/17/19; average pooling by 5 after the
 first two blocks. The mirrored decoder upsamples 8 → 40 → 200. Convolutions use same
 padding within an already-observed window; this is not a future forecasting model.

 **Detection uses latent LOF, not the training loss:**
 $$\Psi(x)=\operatorname{LOF}_{20}(E_\theta(x))
 =\frac1{|N_k(z)|}\sum_{u\in N_k(z)}\frac{\operatorname{lrd}_k(u)}{\operatorname{lrd}_k(z)}.$$
 Here lrd is local reachability density. The implementation uses
 `-sklearn.score_samples`, so larger means more anomalous. LOF is fitted on ID training
 latents (at most 50,000; exact duplicate latents removed).
 Reconstruction MSE is retained as a **separate diagnostic**, not silently substituted for LOF.
 ''',
 flow=[['Window\n16 × 200','Conv + pool encoder\n16→19→24→4','Latent\n4 × 8 = 32','Mirrored decoder\nreconstruct 16 × 200'],
       ['Training latents\nID only','Fit LOF\nk = 20','New latent\nnegative score_samples','Median filter\nand training threshold']],
 model='from models.autoencoder import create_autoencoder\nmodel = create_autoencoder()',
 train='Single-stage reconstruction training with a participant-disjoint early-stopping split inside the training pool (configured fraction 0.2). Adam, LR 0.003, batch 1024, patience 5, maximum 200 epochs. The paper-derived LR 0.064 was a declared deviation because it collapsed this AE. The saved report shows the actual internal validation participants. LOF/threshold calibration then uses ID training data.',
 gallery=[('ae_latent_space.png','PCA view of saved latents. This existing figure fits PCA to ID evaluation latents for visualization only; it is not the LOF fitting procedure.'),
          ('ae_latent_dims.png','Latent variation: look for inactive/collapsed dimensions.'),
          ('ae_reconstructions.png','Input versus reconstruction; easy reconstruction is not proof of familiarity.'),
          ('ae_score_comparison.png','LOF versus reconstruction error: different scores can rank the same tasks differently.'),
          ('ae_psi_by_task.png','Latent-density scores, per-task spread and threshold.'),
          ('ae_detection_by_task.png','Per-task flags at the saved threshold.')],
 extra='ae_umap',
 sources=['ml/vanilla/models/autoencoder.py','ml/vanilla/training/autoencoder.py','ml/vanilla/evaluation/autoencoder.py'],
 takeaway='AE + LOF has the strongest worst-holdout AUROC among the primary scores in the completed LOTO matrix, but its stair-ascent recall at the saved threshold is very low. Ranking robustness is not deployment readiness.'),
dict(file='03_gan',key='gan',title='GAN discriminator as an OOD score',
 question='Can a discriminator trained against generated sensor windows detect unfamiliar real movement?',
 math=r'''A generator maps normal noise of shape 10 × 40 to a 16 × 200 window.
 ConvTranspose1d channels are 10 → 10 → 12 → 16, with strides (5,1,1), kernel 3,
 BatchNorm and LeakyReLU on hidden layers. The discriminator has two kernel-5 spectral-normalized
 convolutions (16 → 30 → 30), ReLU/dropout 0.2, global average pooling, and a scalar logit $s(x)$.

 $$D(x)=\operatorname{sigmoid}(s(x)),\qquad \Psi(x)=1-D(x).$$
 $$\mathcal L_D=-\mathbb E_x\log D(x)-\mathbb E_z\log(1-D(G(z))),\qquad
 \mathcal L_G=-\mathbb E_z\log D(G(z)).$$
 The code uses BCEWithLogitsLoss for stability. The discriminator learns **training-real
 versus generated**, not an explicitly supervised ID/OOD classifier. Its output is not an OOD probability.
 ''',
 flow=[['Gaussian noise\n10 × 40','Generator\ntranspose convolutions','Generated window\n16 × 200','D loss: fake\nG loss: fool D'],
       ['Real window\n16 × 200','Discriminator\nspectral norm + pooling','Realness\nsigmoid(logit)','OOD score\n1 − D(x)']],
 model='from models.gan import create_gan\nmodel = create_gan()',
 train='Fixed 500 epochs, no early stopping. Adam β=(0.5,0.999); generator LR 0.0002, discriminator LR 0.00005; exponential LR decay 0.99. Five generator steps per discriminator step. Batch 256; every second stored training window (effective stride 20), but full evaluation windows. The internal participant holdout is monitored, not used to stop training.',
 gallery=[('gan_generated.png','Generated versus recorded windows; plausibility alone does not validate OOD detection.'),
          ('gan_training_curve.png','Both losses, discriminator outputs and score spread; check for saturation.'),
          ('gan_score_distribution.png','Spread and saturation: a score can retain variation without separating tasks.'),
          ('gan_psi_by_task.png','Task dependence of discriminator scores.'),
          ('gan_detection_by_task.png','Thresholded detections and ID false alarms.')],
 sources=['ml/vanilla/models/gan.py','ml/vanilla/training/gan.py','ml/vanilla/evaluation/gan.py'],
 takeaway='Inspect D(real), D(fake), and score spread together. In standing LOTO, excellent AUROC coexists with zero recall at the inherited training threshold. A healthy GAN loss is not evidence of a useful detector.'),
dict(file='04_correlation_ensemble',key='synthetic',title='Synthetic correlation-target ensemble',
 question='Can a label-free target derived from channel relationships create useful disagreement?',
 math=r'''For 16 channels, sum all 120 distinct pairs within each standardized window:
 $$s(x)=\sum_{i<j}\frac{\operatorname{cov}(x_i,x_j)}
 {\max(\operatorname{sd}(x_i),0.01)\max(\operatorname{sd}(x_j),0.01)},\qquad
 y=\frac{s-\mu_{s,\mathrm{train}}}{\sigma_{s,\mathrm{train}}}.$$
 The SD floor suppresses unstable correlations in near-constant windows. Ordinary Pearson
 correlation is scale-invariant; the fixed floor can break exact invariance near that floor.
 The target is computed from the **same observed window**, not a future measurement.
 Seven TCNs use the gait-phase trunk but have **one linear output** ($D=1$); all target masks are valid.
 '''+ENSEMBLE_MATH,
 flow=[['Observed window\n16 × 200','7 TCN regressors\n1 linear output each','7 scalar predictions','Variance across\n7 predictions'],
       ['Same observed window','Sum 120 floored\nchannel correlations','Training-fitted\ntarget standardization','MSE supervision\nfor each TCN']],
 model="from models.ensemble import create_ensemble\nmodel = create_ensemble(16, 1, activation='linear')",
 train=ENS_TRAIN+' The correlation target is standardized with training statistics; do not recompute that scaling on test participants.',
 gallery=COMMON_GALLERY,
 sources=['ml/vanilla/targets.py','ml/vanilla/dataset.py','ml/vanilla/training/synthetic.py'],
 takeaway='The target removes the need for gait-phase labels, but can encode participant-specific sensor relationships. The useful question is whether uncertainty generalizes—not simply whether the scalar target is easy to regress.'),
dict(file='05_angle_forecast_ensemble',key='forecast-angle',title='200 ms bilateral hip-angle forecast ensemble',
 question='Can disagreement predict unfamiliar motion while future encoder readings expose actual errors?',
 math=r'''Each branch predicts the two standardized hip encoder angles 40 raw samples ahead:
 $$y_t=[\theta_L(t+40),\theta_R(t+40)],\qquad 40/200=0.2\text{ s}.$$
 The trunk is unchanged, with two **linear** outputs. A target is valid only if its future
 row exists within the same trial; trial-end targets are masked, not filled with fake zeros.
 No future measurement enters the input. Prediction error becomes available only after 200 ms.
 '''+ENSEMBLE_MATH+r'''
 **Krogh–Vedelsby is post-training analysis, not an extra training loss:**
 $$E=\|\bar f-y\|^2/D,\quad
 \overline E=\frac1M\sum_m\|f_m-y\|^2/D,\quad A=\Psi,\qquad E=\overline E-A.$$
 Compute this identity on the same raw predictions/valid targets; independent median filtering
 does not generally preserve the identity. High error with low disagreement is a confident failure.
 ''',
 flow=[['History through t\n16 × 200','7 causal TCNs\n2 linear outputs each','Hip angles at\nt + 200 ms','Disagreement now\nrealized error later']],
 model="from models.ensemble import create_ensemble\nmodel = create_ensemble(16, 2, activation='linear')",
 train=ENS_TRAIN+' The regression target is future encoder angle; KV decomposition and OOD scoring are evaluated after training.',
 gallery=COMMON_GALLERY[:3]+[('kv_identity.png','Numerical check of E = mean branch error − disagreement, on compatible unfiltered quantities.'),
                            ('kv_quadrants.png','Low disagreement/high error shows why agreement alone cannot certify reliability.'),
                            ('kv_roc_compare.png','Error-based diagnostics versus disagreement. Do not confuse the failed ratio with error alone.')],
 sources=['ml/vanilla/targets.py','ml/vanilla/training/synthetic.py','ml/vanilla/evaluation/ambiguity.py'],
 takeaway='Strong original-protocol ranking and useful live targets, but standing/transition LOTO expose limits of task-novelty detection. An unseen task can be easy to predict; evaluate task novelty and actual forecast failure separately.'),
dict(file='06_all_channel_forecast',key='forecast-all',title='200 ms all-channel forecast ensemble',
 question='Does forecasting every sensor channel provide more useful uncertainty than forecasting only hip angles?',
 math=r'''The target extends bilateral angle forecasting to all 16 standardized channels:
 $$y_t=x_{\mathrm{channels}}(t+40),\qquad D=16.$$
 Seven identical TCN trunks have 16 linear outputs each. Future rows are matched within trials;
 unavailable future targets are masked. Per-channel disagreement can be inspected before averaging:
 $$\Psi_d(x)=\frac17\sum_m(f_{md}(x)-\bar f_d(x))^2,\qquad
 \Psi(x)=\frac1{16}\sum_{d=1}^{16}\Psi_d(x).$$
 Standardization changes units but does not make all channels equally predictable or equally
 relevant to control. A large channel score is not, by itself, proof that its sensor is faulty.
 '''+ENSEMBLE_MATH,
 flow=[['Observed history\n16 × 200','7 causal TCNs\nsame trunk','16 linear outputs\nat t + 200 ms','Per-channel variance\nthen average over 16']],
 model="from models.ensemble import create_ensemble\nmodel = create_ensemble(16, 16, activation='linear')",
 train=ENS_TRAIN+' This is implemented in the production registry but no completed forecast_all report/evaluation was found when this notebook was created. Do not substitute forecast_angle or smoke-run results.',
 gallery=COMMON_GALLERY,
 sources=['ml/vanilla/registry.py','ml/vanilla/targets.py','ml/vanilla/training/synthetic.py'],
 takeaway='Treat this as an executable ablation setup until a complete run exists. Compare against angle forecasting with the same horizon and protocol; inspect whether harder IMU channels dominate the aggregate score.'),
dict(file='07_probabilistic_ensemble',key='prob_ensemble',title='Probabilistic hip-angle forecast ensemble',
 question='Does learning predictive variance add information beyond disagreement between point forecasts?',
 math=r'''Each of seven TCNs emits a mean and log variance for each hip at t + 200 ms:
 $$p_m(y_d\mid x)=\mathcal N(\mu_{md}(x),\sigma_{md}^2(x)).$$
 The final linear head has four outputs (two means and two log variances); log variances
 are clamped to [−7,3]. After MSE warm-up, the masked training objective uses beta-NLL:
 $$\ell_\beta=\frac{\operatorname{stopgrad}(\sigma^2)^\beta}{2}
 \left[\log\sigma^2+\frac{(y-\mu)^2}{\sigma^2}\right],\qquad\beta=0.5.$$
 Loss is averaged across valid targets and branches; the constant Gaussian term is omitted
 from the training objective. **This beta weights likelihood gradients; it is not beta-VAE's KL weight.**

 For an equal-weight Gaussian mixture, averaged over the two output dimensions:
 $$U_{\rm alea}=\frac1D\sum_d\frac1M\sum_m\sigma_{md}^2,\quad
 U_{\rm epi}=\frac1D\sum_d\frac1M\sum_m(\mu_{md}-\bar\mu_d)^2,\quad
 U_{\rm total}=U_{\rm alea}+U_{\rm epi}.$$
 The fourth score is $U_{\rm epi}/(U_{\rm alea}+10^{-12})$. Each score has its own training
 threshold. The algebraic decomposition is exact for this mixture; interpreting its terms as
 pure physical noise versus novelty is not guaranteed. Changing the loss changes learned means.

 Calibration diagnostic: $z=(y-\mu)/\sigma$. A calibrated Gaussian requires more than a
 variance near one: inspect bias and interval coverage too. The reported `z_var` is a
 mask-weighted average of batch residual variances; the median window-averaged $z^2$ does not
 have a universal target of one and does not prove calibration.
 ''',
 flow=[['History\n16 × 200','7 causal TCNs\n4 outputs per branch','2 means +\n2 log variances','Mixture variance\nalea + epi'],
       ['3 MSE epochs\nmean warm-up','β-NLL, β=0.5\ntrain means + variances','LOSO selects duration\nthen final retraining','4 scores\n4 ID thresholds']],
 model='from prob_models import create_prob_ensemble\nmodel = create_prob_ensemble(16, 2)',
 train='Twelve separate LOSO fold jobs, collection of selected epochs, then final retraining. Defaults: Adam LR 0.001, batch 1024, patience 10, three MSE warm-up epochs followed by beta-NLL with beta 0.5. The completed original run selected ten total epochs; the saved report below supplies the actual settings. Fold selection uses unweighted Gaussian NLL after warm-up.',
 gallery=[('roc_compare.png','Compare four candidate scores with point forecasting; no score is a universal winner across protocols.'),
          ('alea_epi_plane.png','Predicted variance and branch disagreement are distinct axes, not proof of a safety classification.'),
          ('sigma_by_task.png','How predicted uncertainty and disagreement depend on task.'),
          ('scores_by_task.png','Four score distributions; each uses its own scale and threshold.'),
          ('calibration.png','Saved training/LOSO diagnostics. A reference line at one applies to residual variance, not every z statistic.'),
          ('detection_by_task.png','Fixed-threshold behavior can worsen despite improved AUROC.')],
 sources=['ml/prob_ensemble/prob_models.py','ml/prob_ensemble/losses.py','ml/prob_ensemble/train.py','ml/prob_ensemble/evaluate.py'],
 takeaway='The original run improves epistemic AUROC but reduces fixed-threshold recall versus point forecasting. LOTO shows that the ratio can help on some tasks; its poorer original result does not justify rejecting it everywhere. Predicted variance is not yet calibrated assurance on a new wearer.'),
]


LOTO_SPREAD = r'''## LOTO score spread and threshold what-if

Each figure uses one separately trained holdout run: **only the named held-out task is
orange (OOD)**; the other six are blue (ID). The black dashed line is that run's saved
training threshold. Black bars span the middle 50% (IQR), with a tick at the median.
Density shapes use at most 1,500 quantile-spaced points per task; all medians, IQRs and
flag rates use **every saved test window**. Non-GAN scores use log10 axes when positive.
The plotted scores already include the original per-trial smoothing; they are not filtered again.

$$\operatorname{recall}(\tau)=\frac{\sum_{i:y_i=1}\mathbf{1}[\widetilde\Psi_i>\tau]}{N_{\rm OOD}},
\qquad \operatorname{FPR}(\tau)=\frac{\sum_{i:y_i=0}\mathbf{1}[\widetilde\Psi_i>\tau]}{N_{\rm ID}}.$$

**Try a threshold:** replace a `None` in `LOTO_THRESHOLDS` with a numeric score threshold
(in original units, **not log10**), then rerun this cell. A purple line and exact per-task
flag rates show the what-if; `None` uses the saved threshold. The right-hand staircase
and table show recall **and ID false alarms**, not AUROC. Changing the threshold does
not change AUROC, retrain anything, or overwrite saved results.

These are **exploratory test-set what-ifs**, not validated new operating points. Choose a
final threshold using validation data, not whichever test threshold looks best. Rates
are window-weighted; overlapping windows (especially the single standing trial) are
not independent repetitions. Missing runs remain explicitly unavailable.
'''


def loto_spread_cells(key):
    score = 'epistemic' if key == 'prob_ensemble' else 'primary'
    options = "# Other saved scores: 'aleatoric', 'total', 'ratio'.\n" if key == 'prob_ensemble' else ''
    return [md(LOTO_SPREAD), code(
        "from _loto_exploration import loto_score_explorer, threshold_rates\n"
        "LOTO_THRESHOLDS = {'SA': None, 'ST': None, 'TR': None}\n"
        + options + f"LOTO_SCORE = {score!r}\n"
        "loto_bundles = loto_score_explorer(KEY, score=LOTO_SCORE, thresholds=LOTO_THRESHOLDS)\n"
        "# Exact custom query, e.g.: threshold_rates(loto_bundles['ST'], [0.59, 0.60])")]


def vae_loto_spread_cells():
    return [md(LOTO_SPREAD), code('''from _loto_exploration import loto_score_explorer, threshold_rates
LOTO_THRESHOLDS = {'SA': None, 'ST': None, 'TR': None}
LOTO_SCORE = 'latent_lof'  # Or 'reconstruction_nll', 'kl', 'negative_elbo'.
vae_loto_bundles = {}
for variant in VARIANTS:
    display(Markdown(f'## LOTO: {variant} VAE'))
    run_dirs = {task: ROOT / 'ml/VAE/runs' / f'{task}_{variant}_seed{SEED}'
                for task in ('SA', 'ST', 'TR')}
    vae_loto_bundles[variant] = loto_score_explorer(
        'vae', score=LOTO_SCORE, thresholds=LOTO_THRESHOLDS, run_dirs=run_dirs)''')]


UMAP_HOWTO = """
**Three projections of the same rows.** `umap` is nonlinear and neighbourhood-preserving:
distances, cluster areas and gaps are not measurements. `pca` is linear, fitted on the same
reference rows, and its axis labels carry the explained variance. `latent` is two or three raw
latent coordinates with no transformation at all: the only view in which a distance is a true
latent distance, at the price of showing 2-3 of 32 dimensions (`LATENT_DIMS = None` picks the
highest-variance ones on the reference rows; give a tuple such as `(0, 8, 16)` to choose).

**Layout.** In 2-D and in 3-D the projections are side-by-side panels of one figure (the VAE's
view buttons switch all panels together). Every 3-D panel is a WebGL scene, browsers allow only
about 8-16 live WebGL contexts per page, and JupyterLab is one page for every open notebook: a
three-panel 3-D figure costs three. If a 3-D panel ever shows its legend but a blank scene, close
the other notebook tabs and reload the page. 2-D maps are SVG and never hit that cap.

**How to move around (Plotly):** scroll to zoom, drag to pan (2-D) or rotate (3-D; right-drag
pans), double-click to reset. Click a task in the legend to hide it in every panel; double-click
a task to isolate it. Hover shows the participant and trial. `VIEW_DIMS` selects `(2,)`, `(3,)`
or `(2, 3)`; `PROJECTIONS` selects which projections appear. With `SAVE_HTML = True` each figure
is also written as a standalone page under `notebooks/model_exploration/interactive/` (ignored
by git) for full-screen browsing.

Every task keeps **one fixed colour and marker symbol**, whatever role a protocol assigns it.
UMAP embeddings are cached under `.cache/`; delete that folder to recompute.
"""


def ae_umap_cells():
    return [md('## Interactive views of the saved latent space: UMAP, PCA and raw latent coordinates\n\n'
               'Same 32-D test latents as `ae_latent_space.png`, now interactive and task-coloured. UMAP and PCA '
               'are **fitted on in-distribution latents only** (a random subset) and every task is then '
               'transformed into that fixed map, mirroring the discipline of the saved PCA figure; the latent '
               'panel is the untransformed coordinates. The hover text carries the saved smoothed LOF score.\n' + UMAP_HOWTO),
            code("""from _umap_exploration import ae_latent_views, task_counts
VIEW_DIMS = (2, 3)                       # (2,), (3,) or (2, 3)
PROJECTIONS = ('umap', 'pca', 'latent')  # side-by-side panels in 2-D and 3-D
LATENT_DIMS = None                       # e.g. (0, 8, 16); None = highest-variance latent coordinates
MAX_POINTS = 12_000                      # stratified sample; standing and transitions keep every window
VIEW_SEED = 0
SAVE_HTML = True                         # standalone copies under notebooks/model_exploration/interactive/
ae_views = ae_latent_views(RUN, dims=VIEW_DIMS, projections=PROJECTIONS, latent_dims=LATENT_DIMS,
                           max_points=MAX_POINTS, seed=VIEW_SEED, save_html=SAVE_HTML)""")]


def vae_umap_cells():
    return [md('### Interactive views: posterior means, sampled latent vectors, learned variance\n\n'
               'For each trained run, UMAP and PCA are fitted on the **training reference means** and three views '
               'share those coordinates (plus the raw latent coordinates). The projections are side-by-side panels of one '
               'figure per dimensionality, with `means` / `samples` / `variance` buttons switching every panel:\n\n'
               '1. **Posterior means** $\\mu_\\phi(x)$ by task.\n'
               '2. **Sampled latent vectors** $z=\\mu+\\sigma\\odot\\epsilon$: a few reparameterized draws per '
               'window, plus dense clouds of draws around two windows per task (black-outlined in 2-D) to show '
               'the local spread of $q_\\phi(z\\mid x)$.\n'
               '3. **Learned variance**: the means coloured by the average $\\sigma^2_\\phi(x)$ over the 32 '
               'latent dimensions (log10). The table underneath gives exact per-task medians from all saved windows.\n\n'
               'The variance is a property of the 32-D Gaussian; a cloud\'s footprint in a UMAP or PCA plane is '
               'a distorted picture of it, not a confidence region (in the latent panel it is an exact 2-D/3-D '
               'marginal). Runs saved before variances were stored only show view 1.\n' + UMAP_HOWTO),
            code("""from _umap_exploration import vae_latent_views
VIEW_DIMS = (2, 3)                       # (2,), (3,) or (2, 3)
PROJECTIONS = ('umap', 'pca', 'latent')  # side-by-side panels in 2-D and 3-D
LATENT_DIMS = None                       # e.g. (0, 8, 16); None = highest-variance latent coordinates
MAX_POINTS = 8_000                       # windows in the mean/variance views (stratified by task)
VIEW_SEED = 0
SAVE_HTML = True                         # standalone copies under notebooks/model_exploration/interactive/
vae_views = {}
for name, out in VAE_RUNS.items():
    display(Markdown(f'#### {name}: latent views'))
    vae_views[name] = vae_latent_views(out, dims=VIEW_DIMS, projections=PROJECTIONS, latent_dims=LATENT_DIMS,
                                       max_points=MAX_POINTS, seed=VIEW_SEED, save_html=SAVE_HTML)""")]


def all_tasks_notebook():
    return [md("""# All tasks on one interactive map: UMAP and PCA

Where do the seven ambulation tasks sit relative to one another **in the raw sensor space**, before
any model learns a representation? This notebook embeds standardized sensor windows with UMAP and
lets you pan, zoom and rotate through them in 2-D and 3-D. It is a browsing tool, not a benchmark:
nothing is trained, scored or thresholded here, and no task is labelled in-distribution or
out-of-distribution.

Learned representations have their own UMAP sections: the AE latent map in
[02_autoencoder_lof.ipynb](02_autoencoder_lof.ipynb) and the VAE posterior means, samples and
variances in [08_vae_variants.ipynb](08_vae_variants.ipynb)."""),
            code(SETUP),
            md("""## What is embedded

- Source: the mode-complete set `data/processed_all/AE_GAN` (all seven tasks in one split, the same
  windows the LOTO runs score). `SPLIT` defaults to the test participants; `train` and `val` also work
  but take longer to read.
- Each window is 16 channels × 200 samples, standardized with that set's scaler (fitted on LG/RA/RD
  training windows only, so scaling does not depend on which task is held out), then flattened to
  3,200 values.
- A stratified sample of at most `MAX_POINTS` windows is drawn; standing (621 test windows) and
  transitions keep every window. The sample is reduced to 50 principal components (the standard UMAP
  preprocessing) and UMAP is fitted on all sampled windows, since there is no reference/other split here.
  The PCA panel shows the first of those components, labelled with their share of the raw window variance.

$$x\\in\\mathbb R^{16\\times 200}\\;\\xrightarrow{\\text{standardize, flatten}}\\;\\mathbb R^{3200}
\\;\\xrightarrow{\\text{PCA}}\\;\\mathbb R^{50}\\;\\xrightarrow{\\text{UMAP}}\\;\\mathbb R^{2}\\text{ or }\\mathbb R^{3}.$$
""" + UMAP_HOWTO),
            code("""from _umap_exploration import sensor_windows, feature_views, task_counts, TASK_COLORS
SPLIT = 'test'               # 'train', 'val' or 'test' participants
VIEW_DIMS = (2, 3)           # (2,), (3,) or (2, 3)
PROJECTIONS = ('umap', 'pca')  # no latent space here: raw windows have 3,200 coordinates
MAX_POINTS = 12_000          # stratified sample of windows; small tasks keep every window
VIEW_SEED = 0
SAVE_HTML = True             # standalone copies under notebooks/model_exploration/interactive/
windows = sensor_windows(SPLIT, max_points=MAX_POINTS, seed=VIEW_SEED)
print(f"{len(windows['meta']):,} of the {SPLIT} windows sampled; 50 PCs keep "
      f"{windows['explained']:.1%} of the variance of the flattened windows.")
display(task_counts(windows['meta']))
all_task_views = feature_views(windows['features'], windows['meta'], np.arange(len(windows['meta'])),
                               dims=VIEW_DIMS, projections=PROJECTIONS, title=f'Sensor windows ({SPLIT})',
                               cache_key=f'sensor_{SPLIT}', seed=VIEW_SEED, max_points=None,
                               save_html=SAVE_HTML, fitted_on='all sampled windows',
                               pca_variance_scale=windows['explained'])"""),
            md("""## Reading the map

- Overlap between tasks in this map means their one-second sensor windows look alike to a
  neighbourhood-preserving embedding of PCA features. It does not mean a detector cannot separate
  them, and separation here does not mean a detector will.
- Windows from one trial are ten samples apart and overlap heavily, so strands of points often trace
  single trials. Hover to see which participant and trial a strand belongs to; isolate a task with a
  double-click on the legend.
- Compare with the AE latent map and the VAE posterior views to see what each model keeps or discards.
- UMAP is stochastic and seed-dependent; `VIEW_SEED` and the cache make a view reproducible, not
  canonical. Distances and cluster areas are not measurements; the PCA panel is the linear reference.
"""),
            md("""## Variant: does the pelvis IMU add task structure?

The raw trials carry a seventh sensor the processed set leaves out: a pelvis-mounted IMU
(`pelvis_accel_xyz`, `pelvis_gyro_xyz`, six channels in `exo.csv`). Adding it to the model input
is not free: few public gait datasets carry a pelvis IMU at the same placement, so every channel
added here narrows the datasets a trained model can later be checked against. Before paying that,
look at the raw data alone: does the pelvis IMU change how the seven tasks sit relative to one another?

Three channel sets are built from **the same sampled windows** (same rows, same order):

| set | channels | standardization |
|---|---|---|
| `thigh` | the processed 16: hip encoder angle + velocity and thigh IMU, both legs | the set's scaler (LG/RA/RD training windows) |
| `pelvis` | the 6 pelvis IMU channels, sliced from the raw trials at the same `end_idx` | mean/std over the same LG/RA/RD training trials |
| `thigh+pelvis` | all 22 | both |

Each set is flattened, reduced to 50 principal components and embedded exactly as above. The
`thigh` views are the figures higher up; the two new sets get their own maps below.

Looking is not enough to say "worth adding", so one number accompanies the maps: **k-NN task accuracy**
(k = 15) in the 50-PC space, and in the UMAP embeddings for reference. Windows from one trial overlap by
95 % and would label their own neighbours under a random split, so whole recordings are held out, in two ways:

- *participant held out*: one fold per test participant, the stricter question (does the structure carry
  across people?). Standing and transitions were recorded by one test participant only, so they cannot be
  scored this way (recall NaN) and they pull the plain accuracy down for every set alike.
- *trial held out*: five folds of whole trials. Transitions (five trials) are scored here; standing is a
  single trial in the test split, so it stays unscored under both schemes and its 621 windows count only
  as a distractor the other tasks must not be confused with.

The PCA-space columns are the ones to read; UMAP is stochastic and its silhouettes reward the tight strands
it draws. A set whose accuracy and per-task recall do not move has added channels without adding task
structure in the raw space; that does not rule out a learned model finding something, but it is the cheap check."""),
            code("""from _umap_exploration import sensor_window_sets, separability, FEATURE_SETS
sets = sensor_window_sets(SPLIT, tuple(FEATURE_SETS), max_points=MAX_POINTS, seed=VIEW_SEED)
assert np.array_equal(sets['thigh']['source_index'], windows['source_index']), 'the variant must reuse the rows plotted above'
for name, s in sets.items():
    print(f"{name:13s} {len(s['channels']):2d} channels; 50 PCs keep {s['explained']:.1%} of the flattened-window variance")
if sets['thigh']['dropped']:
    print(f"{sets['thigh']['dropped']} windows dropped from every set: NaN in their pelvis channels")
variant_views = {}
for name in ('pelvis', 'thigh+pelvis'):
    s = sets[name]
    variant_views[name] = feature_views(s['features'], s['meta'], np.arange(len(s['meta'])),
                                        dims=VIEW_DIMS, projections=PROJECTIONS,
                                        title=f'Sensor windows ({SPLIT}), {name}',
                                        cache_key=f"sensor_{SPLIT}_{name.replace('+', '_')}", seed=VIEW_SEED,
                                        max_points=None, save_html=SAVE_HTML, fitted_on='all sampled windows',
                                        pca_variance_scale=s['explained'])"""),
            md("""### Task separability by channel set

Rows are (channel set, held-out unit); the first two columns are the numbers to read. `per_task` gives the
PCA-space recall of each task under each scheme, so a gain (or loss) can be traced to the tasks it comes from."""),
            code("""views = {'thigh': all_task_views, **variant_views}
summary, per_task = separability({name: dict(features=sets[name]['features'],
                                             umap={d: v[d]['points']['umap'] for d in v})
                                  for name, v in views.items()}, sets['thigh']['meta'], k=15, seed=VIEW_SEED)
display(summary.round(3))
display(per_task.round(3))"""),
            code("source_links(['ml/LOTO/splits.py', 'ml/LOTO/build_data.py', 'ml/vanilla/dataset.py', 'notebooks/model_exploration/_umap_exploration.py'])")]


def existing_notebook(s):
    cells=[md(f"# {s['title']}\n\n{s['question']}\n\nRead top-to-bottom: setup → equations → architecture → training → figures → LOTO → interpretation.\n**Safe default:** saved-result exploration only; training is disabled."),
           code(SETUP),code(f"KEY = {s['key']!r}\nRUN = ROOT / RUNS[KEY]\ninventory(RUN)"),
           md(PROTOCOL),md('## Model and mathematics\n\n'+s['math']),
           code(f"diagram({s['flow']!r}, {s['title']!r})"),
           md('### Executable architecture check\n\nThis constructs **fresh random weights** and checks shapes only. The saved figures later come from trained runs; this dummy forward pass does not.'),
           code(s['model']+"\nexample = torch.zeros(2, 16, 200)\noutput = model_shapes(model, example)"),
           md('## Training\n\n'+s['train']),
           code("from registry import get\nconfig_class = get(KEY).config()\ndisplay(pd.Series(asdict(config_class()), name='current default').to_frame())") if s['key']!='prob_ensemble' else
           code("from train import ProbConfig\ndisplay(pd.Series(asdict(ProbConfig()), name='current default').to_frame())"),
           md('### Saved training configuration and learning curves\n\nCurrent defaults can differ from a past run. Below we read the report; missing reports remain explicitly missing. Loss scales from different objectives are not directly comparable.'),
           code('config_and_history(RUN)'),
           md('### Optional: run the real training pipeline\n\nThis can take hours. The commands use this kernel’s Python and write to a **new** directory. Set `RUN_TRAINING=True` in the setup cell only when ready. Existing runs are protected. After completion, set `RUN = NEW_RUN` and rerun the result cells.'),
           code("NEW_RUN = ROOT / 'notebooks/model_exploration/runs' / KEY\ncommands = training_commands(KEY, NEW_RUN)\nrun_commands(commands, NEW_RUN, enabled=RUN_TRAINING)"),
           md('## Saved test results\n\nThese values come directly from `eval_test.json`. An optimized test-ROC threshold, where reported, is diagnostic—not a validated operating point.'),
           code('results_table(RUN)'),md('## Model behavior through figures\n\nImages are embedded from the existing run, without rewriting it. Captions explain what each figure can and cannot establish.'),
           code(f"gallery(RUN, {s['gallery']!r})"),
           *(ae_umap_cells() if s.get('extra')=='ae_umap' else []),
           md('## Leave-one-task-out (LOTO) stress test\n\nSA: stair ascent; ST: standing; TR: transitions. Each run excludes the named task from **all training participants**, trains on the other six, and tests on held-out participants. This differs from LOSO epoch selection.\n\nStanding test data come from one trial of one participant: 621 overlapping windows, or 600 for the gait-phase subset. Gait-phase LOTO uses fewer labelled windows overall. Raw window counts are not independent repetitions; cross-run changes do not isolate a causal training effect.'),
           code('loto_results(KEY)'),*loto_spread_cells(s['key']),md('## Takeaway and next checks\n\n'+s['takeaway']+'\n\nKeep score/threshold selection on validation data. Report both ranking and recall at a declared false-alarm budget, repeat seeds, and quantify participant/trial-level variation before making robustness claims.'),
           code(f"source_links({s['sources']!r})")]
    return cells


def vae_sweep_cells():
    return [md(r"""## β sweep, controls and the dimension question (original protocol)

The variance result settled one thing: posterior width does not track the task, so σ² and KL
cannot be OOD scores for this model. It left two competing explanations for β=4 beating β=1,
and they lead to different next steps:

- **Dimension.** β=4 switched off 14 of 32 latent dimensions. If LOF simply works better in 18
  dimensions than in 32, the fix is a narrower bottleneck, VAE or not.
- **Regularization.** If the KL term shapes the code, pushing β higher should keep helping until
  too much collapses, giving an inverted U.

Three pieces of evidence, all selected on the **validation** split (test is shown, never used to choose):

1. **Post hoc, no retraining** (`ml/VAE/posthoc_dims.py`): LOF refitted on the same 50,000 training
   latents, on all 32 coordinates, on the *active* coordinates only, and on PCA-reduced coordinates
   with k ∈ {8, 12, 16, 18, 24}, for both VAE runs and the AE baseline. If the AE gains at 16-18, the
   story is dimension.
2. **β sweep** (`ml/VAE/slurm/sweep.sbatch`): β ∈ {0.25, 1, 2, 4, 8, 16, 32}, with the active-dimension
   count per cell (median posterior σ² < 0.9 on ID validation windows).
3. **Controls**: β=0 with sampling off is the AE objective through this pipeline. `ae_control` keeps the
   VAE optimizer settings (lr 0.001, batch 256, ≤50 epochs); `ae_control_aehp` uses the AE's (lr 0.003,
   batch 1,024, ≤200 epochs). Near the AE's 0.782 means the pipeline is fine and the Gaussian latent costs
   the difference; near 0.65-0.71 means the hyperparameters are the cause.

Seeds 1-2 are added to the winning β before any claim (`ml/VAE/slurm/seeds.sbatch`); a difference of a
few hundredths is inside seed noise. Tables and figures are written to `ml/VAE/results/` by the two
scripts and read here; missing files simply print as unavailable.
"""),
    code("""from vae_sweep import summarize, RESULTS
from IPython.display import Image
sweep = summarize('original')
if sweep.empty:
    print('No completed original-protocol runs found under ml/VAE/runs.')
else:
    cols = ['variant', 'seed', 'beta', 'sampling', 'best epoch', 'val KL', 'active dims', 'KL per active dim',
            'val AUROC (LOF)', 'test AUROC (LOF)', 'test recall %', 'test false alarms %', 'val AUROC (KL)', 'hours']
    display(sweep[cols].round(4).style.hide(axis='index'))
    seeds = sweep[sweep['seed'] > 0]
    if not seeds.empty:
        rep = sweep[sweep['variant'].isin(seeds['variant'].unique())].groupby('variant')[['val AUROC (LOF)', 'test AUROC (LOF)']]
        display(Markdown('**Seed repeats (mean ± sd, n = seeds):**'))
        display(rep.agg(['mean', 'std', 'count']).round(4))
    if (RESULTS / 'sweep_original.png').exists():
        display(Image(filename=str(RESULTS / 'sweep_original.png'), width=950))
    else:
        print('Run `python ml/VAE/vae_sweep.py --protocol original --write` to draw the sweep figure.')"""),
    md('### Posterior means of every sweep run, side by side\n\n'
       'One UMAP panel per run (every β and both controls, seed 0), the test-set posterior means coloured by task. '
       'Each panel is **its own embedding**, fitted on that run\'s training reference means, so orientation and scale '
       'are not comparable across panels; what is comparable is structure: which tasks sit apart, which overlap, and '
       'how the layout changes as β grows and dimensions switch off. Panel titles carry the validation LOF AUROC and '
       'the active-dimension count. In 3-D one scene has a run selector.'),
    code("""from _umap_exploration import sweep_mean_views
SWEEP_RUNS = sorted(p for p in (ROOT / 'ml/VAE/runs').glob('original_*_seed0') if not p.name.endswith('_smoke'))
sweep_views = sweep_mean_views(SWEEP_RUNS, dims=VIEW_DIMS, max_points=4_000, seed=VIEW_SEED, save_html=SAVE_HTML)"""),
    code("""posthoc = RESULTS / 'posthoc_dims.csv'
if posthoc.exists():
    table = pd.read_csv(posthoc)
    display(Markdown('**LOF in fewer dimensions (post hoc, LOF refitted; select on validation):**'))
    display(table.round(4).style.hide(axis='index'))
else:
    print('Run `python ml/VAE/posthoc_dims.py --write` to create ml/VAE/results/posthoc_dims.csv.')""")]


def vae_notebook():
    return [md('''# VAE exploration: standard VAE and β-VAE

**Status: notebook-local prototypes with one completed seed-0 run per cell** (standard and
β=4, each under the original protocol and the SA/ST/TR holdouts), trained on the FASRC cluster
via `ml/VAE/slurm/` and pulled into `ml/VAE/runs/`. The package is `ml/VAE/`, beside `ml/vanilla`
and `ml/prob_ensemble`; it is not registered in the vanilla CLI. Single
seed: read the numbers as a first pass, not a benchmark. Variants share architecture; only the
KL weight changes.

Question: does probabilistic latent regularization improve upon **AE + LOF**, not merely
produce a better reconstruction or a more attractive latent plot?'''),
    code(SETUP),md(PROTOCOL),md(r'''
    ## Model and equations

    The first two encoder blocks match the existing AE: 16×200 → 19×40 → 24×8.
    Two separate kernel-19 convolution heads produce means and log variances of shape 4×8.
    The existing AE decoder maps a sampled latent back to a 16×200 window.

    $$q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),\operatorname{diag}(\exp\ell_\phi(x))),\quad
    z=\mu_\phi(x)+\exp(\ell_\phi(x)/2)\odot\epsilon,\quad\epsilon\sim\mathcal N(0,I).$$
    $$p(z)=\mathcal N(0,I),\qquad p_\theta(x\mid z)=\mathcal N(D_\theta(z),I).$$

    Decoder variance is **fixed to one in standardized input units**, not learned. Latent
    log variances are clamped to [−8,8] for numerical stability. Unlike the AE's ReLU bottleneck,
    posterior means are signed and unbounded. Architecture/capacity is close, not exactly identical.

    $$R(x)=\frac12\sum_{c,t}[(x_{ct}-\widehat x_{ct})^2+\log(2\pi)],$$
    $$K(x)=\frac12\sum_j[\mu_j^2+\exp(\ell_j)-1-\ell_j],\qquad
    \mathcal L_\beta=\mathbb E_{q(z\mid x)}R(x)+\beta K(x).$$

    Reconstruction and KL are **sums per window**, then averaged over the batch.
    β=1 is the standard negative ELBO objective; β=4 is a stronger-regularization ablation.
    This β is a KL weight, **not** the probabilistic ensemble's beta-NLL reweighting.
    A near-zero KL can indicate latent collapse; latent variance is not automatically epistemic OOD uncertainty.
    '''),
    code("diagram([['Sensor window\\n16 × 200','Shared conv trunk\\n24 × 8','μ and log variance\\neach 4 × 8','Sample z, then decode\\n16 × 200'], ['Posterior mean μ\\nflatten to 32','Fit LOF on\\nID training means','Score new windows\\nLOF(μ)','Same smoothing\\nand threshold rule']], 'Standard / beta VAE: representation and detection are separate')"),
    md('### Executable model and loss\n\nThe implementation lives in `ml/VAE/` (`vae_model.py`, `vae_train.py`, `vae_plots.py`), alongside the other model packages, and is not registered in the vanilla CLI. The next pass uses random weights and dummy inputs only.'),
    code('''from vae_model import ConvVAE, loss_terms
from vae_train import VAEConfig, train_experiment
from vae_plots import plot_representation
model = ConvVAE()
x = torch.zeros(2, 16, 200)
rec, mu, logvar = model_shapes(model, x)
nll, kl = loss_terms(x, rec, mu, logvar)
display(pd.DataFrame({'reconstruction NLL': nll.detach().numpy(), 'KL': kl.detach().numpy()}))
assert rec.shape == x.shape and mu.shape == logvar.shape == (2, 4, 8)'''),
    md('### Inspect the actual implementation\n\nNo duplicate pseudocode: this is the class and loss used by the optional trainer.'),
    code('import inspect\nprint(inspect.getsource(ConvVAE))\nprint(inspect.getsource(loss_terms))'),
    md('''## What the variants test

| Variant | KL weight | Primary OOD score | Question |
|---|---:|---|---|
| Standard VAE + LOF | 1 | LOF on posterior means | Does probabilistic regularization help latent density detection? |
| β-VAE + LOF | 4 | Same LOF procedure | Does stronger regularization improve generalization or erase task distinctions? |

**Diagnostic score ablations for each model:** reconstruction NLL, KL, and negative ELBO
(reconstruction NLL + KL, regardless of training β). The latter uses eight posterior samples
per window; it is a Monte Carlo estimate of a bound, not exact negative log likelihood.
Do not call these scores calibrated OOD probabilities or choose a different winner on each test task.
'''),
    md('''## Training protocol and controls

- Default original task split; set `HELD_OUT` to `SA`, `ST`, or `TR` to use the existing LOTO builder.
- Participant-disjoint early-stopping subset inside the training pool, matching the AE approach.
  Adam LR 0.001, batch 256, ceiling 50 epochs, patience 5. These are proposed prototype settings.
- The loss samples z during both training and validation; validation uses a fixed random seed.
- LOF reference latents and the 99.5th-percentile thresholds use ID training data only.
  Evaluation uses the existing per-trial causal median filter and metric functions.
- To compare variants, keep the seed/splits fixed; repeat with seeds 0, 1, 2 before conclusions.
  For a fair new operating-point comparison, recalibrate **both** AE and VAE on the same
  validation-ID false-alarm budget. The prototype first reproduces the existing training-percentile rule.
- The optional window cap is a **smoke test**, not a benchmark. No training is run by default;
  the cluster matrix is `ml/VAE/slurm/submit.sh` and results are pulled with `ml/VAE/slurm/pull.sh`.
'''),
    code('''HELD_OUT = None  # None, 'SA', 'ST', 'TR'
SEED = 0
VARIANTS = {'standard': 1.0, 'beta4': 4.0}
protocol = HELD_OUT or 'original'
VAE_RUNS = {name: ROOT / 'ml/VAE/runs' / f'{protocol}_{name}_seed{SEED}'
            for name in VARIANTS}
configs = {name: VAEConfig(beta=beta, seed=SEED, held_out=HELD_OUT)
           for name, beta in VARIANTS.items()}
display(pd.DataFrame({name: asdict(cfg) for name, cfg in configs.items()}))
for name, out in VAE_RUNS.items():
    print(name, 'existing results' if (out/'report.json').exists() else 'NOT TRAINED', out)
if RUN_TRAINING:
    for name, cfg in configs.items():
        train_experiment(VAE_RUNS[name], cfg)
else:
    print('Training disabled. Change RUN_TRAINING explicitly to create new runs.')'''),
    md('## Learning curves, scores and learned representations\n\nThese cells will populate after training; they do not fill missing results with toy metrics. PCA is fitted on training reference means, then test means are projected into that fixed basis.'),
    code('''for name, out in VAE_RUNS.items():
    display(Markdown(f'### {name}: saved artifacts'))
    inventory(out)
    config_and_history(out)
    results_table(out)
    plot_representation(out)'''),
    *vae_umap_cells(),
    *vae_sweep_cells(),
    *vae_loto_spread_cells(),
    md('''## Read the experiment, not just the pictures

1. Do LOF scores improve over the existing AE at a comparable false-alarm rate, consistently across seeds?
2. Does a smaller KL reflect useful regularization or posterior collapse? Inspect both loss components.
3. Does standing receive low reconstruction NLL because it is simple? That would repeat the known AE failure.
4. Does the score identify large forecasting errors, or just task identity? These are different objectives.

Standing has only one test trial; overlapping windows cannot support a strong population-level
claim. The already-inspected test set is exploratory evidence, not a fresh confirmation set.
An attractive PCA plot, better likelihood, or a single favorable AUROC is not a go/no-go criterion.

**Continue** if there is repeatable improvement or useful complementary failure detection.
**Reconsider** if it only reconstructs better or preserves the same failure pattern without added information.

## Sources and further reading

- [VAE package](../../ml/VAE/README.md): [model](../../ml/VAE/vae_model.py), [training and CLI](../../ml/VAE/vae_train.py); [AE baseline](../../ml/vanilla/models/autoencoder.py).
- [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
- [Nalisnick et al.: generative likelihood can fail at OOD detection](https://arxiv.org/abs/1810.09136).
- [Higgins et al.: beta-VAE](https://openreview.net/forum?id=Sy2fzU9gl).

Literature findings motivate this experiment; they do not establish performance on hip-exoskeleton data.
''')]


def awareness_notebook():
    return [md("""# Awareness layer under a regime holdout

**Question.** A controller needs three answers per window, not one: is this movement
*unfamiliar* (novelty), *which regime* does it resemble (regime), and *can the active prediction
be trusted* (reliability)? This notebook computes those three signals from one sensor window and
lays them out side by side, per task, under a holdout that removes an **entire regime** from
training: `DOWN` holds out ramp descent, stair descent, standing and transitions, so training sees
level ground, ramp ascent and stair ascent only. The six existing models are run under the same
holdout for a model-by-model comparison.

Code and tracked results live in [`ml/awareness/`](../../ml/awareness/README.md); nothing here
trains. Scores are shown as **spreads by task with no thresholds**: this is the raw material for
deciding what to fuse, not a detector.

Read top-to-bottom: split → encoder and novelty → regime probes → forecaster → baselines → all
scores by task."""),
    code(SETUP),
    md(r"""## The holdout

`ml/LOTO/splits.py` names a holdout by the **set** of modes it removes. `DOWN` = {RD, SD, ST, TR}.
Participants, scaler, causal median filter (ten scores) and evaluators are the ones every other
run uses, so all rows below are computed on identical test windows: ID = LG, RA, SA of the four
test participants; OOD = the four held-out modes of the same participants.

$$\widetilde\Psi_t=\operatorname{median}(\Psi_{t-9},\ldots,\Psi_t)\qquad\text{for every score below.}$$

AUROC (OOD versus ID, and each held-out mode alone versus ID) is reported because it is a ranking
statistic with no threshold in it. Nothing in this notebook chooses an operating point."""),
    code("""from splits import build, held_out_modes, trained_modes
HOLDOUT = 'DOWN'
print('held out:', held_out_modes(HOLDOUT), ' trained on:', trained_modes(HOLDOUT))
_d = build('autoencoder', HOLDOUT, batch_size=256)
rows = []
for name in ('train', 'val', 'test'):
    by = _d.splits[name].meta.groupby('mode').size().to_dict()
    rows.append({'split': name, 'kind': 'ID', **by, 'total': len(_d.splits[name])})
    if name in _d.ood:
        by = _d.ood[name].meta.groupby('mode').size().to_dict()
        rows.append({'split': name, 'kind': 'OOD', **by, 'total': len(_d.ood[name])})
display(pd.DataFrame(rows).fillna(0).astype({c: int for c in ('total',)}).set_index(['split', 'kind']))
AW = ROOT / 'ml/awareness'
RUNS_AW = AW / 'runs' / HOLDOUT
RESULTS_AW = AW / 'results' / HOLDOUT"""),
    md(r"""## Stage A: encoder and novelty

The encoder is `ml/VAE`'s deterministic configuration: the AE trunk with a **linear** 4×8 latent
head, trained on the unit-variance Gaussian NLL with β=0 and no sampling, at the AE's settings
(learning rate 0.003, batch 1,024, up to 200 epochs, patience 5, participant-disjoint early
stopping). Under the original protocol it gave the best latent-LOF ranking of anything tried
(test AUROC 0.822 against the vanilla AE's 0.782), which is why it is the encoder here.

$$z=E_\theta(x)\in\mathbb R^{32},\qquad \Psi_{\rm nov}(x)=\operatorname{LOF}_{20}(z)\ \text{fitted on 50,000 ID training latents},$$

and the same LOF refitted on 12 and 16 principal components of the same reference, since fewer
dimensions helped every model in the earlier post-hoc check. Reconstruction NLL is kept as a
separate diagnostic (it ranked OOD *below* ID under the original protocol)."""),
    code("""ENC = RUNS_AW / 'encoder'
inventory(ENC)
config_and_history(ENC)
results_table(ENC)"""),
    md("""### Interactive views of the encoder's test latents

UMAP and PCA fitted on **in-distribution** test latents (LG, RA, SA), every task transformed in;
the third panel is raw latent coordinates. Task colours are fixed; the four held-out modes are
simply the four tasks the encoder never saw. In 3-D one scene has a projection selector.
""" + UMAP_HOWTO),
    code("""from _umap_exploration import ae_latent_views
VIEW_DIMS = (2, 3)
PROJECTIONS = ('umap', 'pca', 'latent')
LATENT_DIMS = None
MAX_POINTS = 12_000
VIEW_SEED = 0
SAVE_HTML = True
if (ENC / 'latents_test.npz').exists():
    enc_views = ae_latent_views(ENC, dims=VIEW_DIMS, projections=PROJECTIONS, latent_dims=LATENT_DIMS,
                                max_points=MAX_POINTS, seed=VIEW_SEED, save_html=SAVE_HTML)
else:
    print('Encoder run not pulled yet:', ENC)"""),
    md(r"""## Stage B: regime probes

Two probes read the same latent, trained on the **training split's task labels mapped to regimes**
(LG → level; RA, SA → ascend). Under `DOWN` those are the only two regimes, so a held-out window can
never be named "descend"; what is recorded is the probe's probabilities, its confidence (max
probability) and its entropy:

$$H(x)=-\sum_c p_c(x)\log p_c(x).$$

`logistic` is the linear probe (is regime linearly readable from the latent?); `mlp` (two hidden
layers of 64) is the comparison. Accuracy is meaningful only on ID windows; on the held-out modes
the interesting numbers are mean entropy and which known regime absorbs them."""),
    code("""REG = RUNS_AW / 'regime'
for probe in ('logistic', 'mlp'):
    f = REG / f'{probe}_summary.json'
    if not f.exists():
        print(probe, 'not available yet'); continue
    summary = read_json(f)
    display(Markdown(f'### {probe} probe — classes {summary["classes"]}, trained on {summary["n_train"]:,} latents'))
    acc = {k: v['accuracy'] for k, v in summary.items() if isinstance(v, dict) and 'accuracy' in v}
    display(pd.Series(acc, name='accuracy on windows with a known regime').to_frame().round(4))
    rows = []
    for key in ('test_id', 'test_ood'):
        for mode, r in summary[key]['by_mode'].items():
            rows.append({'split': key, 'mode': mode, 'windows': r['windows'], 'mean entropy': r['mean_entropy'],
                         'mean confidence': r['mean_confidence'], **{f'predicted {c}': v for c, v in r['predicted_share'].items()}})
    display(pd.DataFrame(rows).set_index(['split', 'mode']).round(3))"""),
    md(r"""## Stage C: the 200 ms hip-angle forecast ensemble

The seven-TCN forecast ensemble of Experiment 5, driven through `ml/vanilla/main.py` exactly as the
LOTO runner drives it, twice: **plain** (16 sensor channels) and **regime-conditioned** (16 channels
plus the logistic probe's two class probabilities as constant channels). Branch disagreement $A$ is
available immediately; the realized error $E$ of the ensemble mean, and the mean branch error
$\bar E$, exist 200 ms later:

$$E=\bar E-A\qquad\text{(Krogh–Vedelsby, exact per window).}$$

On a held-out window the regime input is whichever known regime the probe assigned, with its
confidence; the forecast target itself is unchanged. The first pass uses one leave-one-subject-out
fold for epoch selection (`--folds 1`) instead of twelve, to get a result back in an hour."""),
    code("""for variant in ('plain', 'regime'):
    run = RUNS_AW / f'forecast_{variant}'
    display(Markdown(f'### forecast_{variant}'))
    inventory(run)
    config_and_history(run)
    results_table(run)"""),
    md(r"""## Stage C′: regime experts, so the regime cannot be ignored

The conditioned forecaster turned out not to *use* its regime channels. The swap test below feeds
it the wrong regime on every window: on in-distribution windows the median realized error moves by
only a few percent, and on the held-out modes a wrong label even lowers it. On training data the
regime is already readable from the raw signal, so two constant input channels were never
load-bearing, and at test time a wrong value costs nothing.

Experts make the dependence structural. One ensemble is trained **per regime present in training**,
on that regime's windows only (`level` on LG, `ascend` on RA and SA); at inference the logistic
probe routes each window to an expert. A window from an unseen regime is routed to an expert that
never saw its dynamics, so its error rises by construction, and a new quantity appears: the
disagreement between the experts' mean forecasts.

$$E_{\rm routed}(x)=E_{c^*(x)}(x),\quad c^*(x)=\arg\max_c p_c(x);\qquad
E_{\min}(x)=\min_c E_c(x);\qquad
\Delta(x)=\big\|\bar f_{\rm level}(x)-\bar f_{\rm ascend}(x)\big\|^2/D.$$

Each expert sees roughly half the training windows, which is the price of this design."""),
    code("""swap = RESULTS_AW / 'regime_swap_test.csv'
if swap.exists():
    display(Markdown('**Swap test on the regime-conditioned forecaster** (median realized error per mode, true versus swapped regime input):'))
    display(pd.read_csv(swap, index_col=0).round(4))
else:
    print('Run `python ml/awareness/experts.py swap-test --holdout DOWN` to create', swap.name)
for regime in ('level', 'ascend'):
    run = RUNS_AW / f'expert_{regime}'
    display(Markdown(f'### expert_{regime}'))
    inventory(run)
    config_and_history(run)
    results_table(run)
summary = RUNS_AW / 'experts' / 'summary.json'
if summary.exists():
    s = read_json(summary)
    display(Markdown('**Expert scores, test AUROC (OOD versus ID), and the share of windows routed to each expert:**'))
    display(pd.Series({k: v for k, v in s.items() if isinstance(v, float)}, name='AUROC').to_frame().round(4))
    display(pd.Series(s['routed_share'], name='routed share').to_frame().round(3))
else:
    print('Expert scores not computed yet: pull both experts, then `python ml/awareness/experts.py score --holdout DOWN`.')"""),
    md("""## The six existing models under the same holdout

Each row is one LOTO leaf, `ml/LOTO/DOWN/<model>`, trained on LG/RA/SA and scored on the same test
windows. AUROC is ranking; recall and false alarms are at each model's own training-percentile
threshold and are shown for completeness only."""),
    code("""rows = []
for leaf in ('vanilla/AE', 'vanilla/GAN', 'vanilla/Ensemble_GP', 'vanilla/Ensemble_synthetic', 'vanilla/Ensemble_forecast', 'prob_ensemble/run'):
    r = read_json(ROOT / 'ml/LOTO' / HOLDOUT / leaf / 'eval_test.json')
    if not r:
        rows.append({'model': leaf, 'status': 'not finished'}); continue
    for score, m in r.get('scores', {'primary': r.get('metrics', {})}).items():
        rows.append({'model': leaf, 'score': score, 'AUROC': m.get('auroc'), 'recall %': m.get('recall'),
                     'ID false alarms %': 100 - m.get('specificity', 100), 'status': 'done'})
display(pd.DataFrame(rows).round(4))"""),
    md("""## Every score, by task

`ml/awareness/aw_scores.py` aligns every per-window score to the same test windows (the gait-phase
ensemble keeps its own labelled window set) and `aw_plots.py` draws one violin per task per score:
**orange = never seen in training, blue = trained on**. Black bars are the interquartile range, the
tick the median; axes span the central 99.6 % of all windows. The table gives per-mode medians and
threshold-free AUROCs. Set `RECOMPUTE_SCORES = True` to rebuild both from the pulled runs (a few
minutes: LOF refits and the forecast error decomposition from the checkpoints); otherwise the
tracked files under `ml/awareness/results/` are shown."""),
    code("""from IPython.display import Image
RECOMPUTE_SCORES = False
if RECOMPUTE_SCORES:
    import aw_scores, aw_plots
    ss = aw_scores.collect(HOLDOUT)
    summary = aw_scores.summarize(ss)
    aw_scores.write(ss, summary); aw_plots.all_groups(ss)
csv = RESULTS_AW / 'score_summary.csv'
if csv.exists():
    summary = pd.read_csv(csv)
    auc_cols = [c for c in summary.columns if c.startswith('AUROC')]
    display(summary[['score', 'windows', *auc_cols]].round(4).style.hide(axis='index'))
    display(Markdown('**Per-mode medians (each score in its own units):**'))
    display(summary[['score'] + [c for c in summary.columns if c.startswith('median ')]].round(4).style.hide(axis='index'))
    for group in ('novelty', 'regime', 'forecast', 'experts', 'baselines'):
        png = RESULTS_AW / f'spread_{group}.png'
        display(Markdown(f'### {group}'))
        display(Image(filename=str(png), width=1000)) if png.exists() else print('not available yet:', png.name)
else:
    print('No tracked results yet:', csv)"""),
    md(r"""## Making the dangerous cells fire: injected sensor faults

The held-out regime under `DOWN` turned out to be the **OOD-safe** case: unfamiliar to LOF, confidently
routed to a known regime, and still forecast acceptably. Nothing in the dataset is dangerous, so the
other cell of the novelty × health grid is populated here by construction: level-ground test windows
with a sensor fault injected into the **input only** (dead left-thigh IMU, a frozen right-thigh IMU, IMU noise at twice each channel's spread, gyro spikes, a 100 ms left-leg
delay), persistently across whole trials so the causal median filter behaves as in deployment. The
forecast target is the true future encoder reading, so realized error measures what the corrupted
input does to the prediction. A constant encoder offset is deliberately not among them: it is a
calibration error rather than a dynamic fault, and the future reading would carry the same bias.

The same five faults are applied to ramp-descent windows too, so all four cells of the grid exist
on the same participants: **ID-safe** (level ground), **ID-dangerous** (level ground + fault),
**OOD-safe** (ramp descent), **OOD-dangerous** (ramp descent + fault). Every window goes through
the same pipeline: LOF on the encoder latent, reconstruction NLL, the regime probe, the unrouted
forecast ensemble and the **routed regime experts**, whose error is the health axis in the
quadrant figure so that the regime has its say. AUROCs are against ID-safe and against OOD-safe."""),
    code("""fs = RESULTS_AW / 'faults_summary.csv'
if fs.exists():
    fsum = pd.read_csv(fs)
    keys = ['novelty_lof32', 'recon_nll', 'regime_entropy', 'forecast_disagreement', 'forecast_error', 'expert_routed_error']
    for against in ('ID-safe', 'OOD-safe'):
        cols = [f'AUROC vs {against}: {k}' for k in keys if f'AUROC vs {against}: {k}' in fsum]
        display(Markdown(f'**AUROC of each condition against {against}** (0.5 = indistinguishable; a cell is blank against itself):'))
        display(fsum[['condition', 'windows', *cols]].rename(columns=lambda c: c.replace(f'AUROC vs {against}: ', '')).round(3).style.hide(axis='index'))
    for name in ('faults_spread.png', 'faults_quadrants.png'):
        png = RESULTS_AW / name
        display(Image(filename=str(png), width=1000)) if png.exists() else print('missing', name)
else:
    print('Run `python ml/awareness/faults.py --holdout DOWN --write` to create', fs.name)"""),
    md("""### The four cells in three dimensions

Novelty (LOF on the encoder latent), the **autoencoder's reconstruction error** (the decoder's NLL
of the window, not a forecast) and the **routed expert forecast error** as three axes, all log10.
Pick a fault with the selector; each cell keeps its colour. Rotate to see that the dangerous cells
leave the safe ones along the reconstruction axis while the forecast-error axis mostly does not
separate them."""),
    code("""import plotly.graph_objects as go
from _umap_exploration import show, PLOTLY_CONFIG
fsc = RESULTS_AW / 'faults_scores.parquet'
if fsc.exists():
    fdf = pd.read_parquet(fsc)
    cells = {'ID-safe': ('#287da8', 'circle'), 'OOD-safe': ('#dd791a', 'square'), 'ID-dangerous': ('#a32d2d', 'cross'), 'OOD-dangerous': ('#4a3aa7', 'diamond')}
    axes3 = ('novelty_lof32', 'recon_nll', 'expert_routed_error')
    faults = sorted({c[c.index('+ ') + 2:c.rfind(' (')] for c in fdf.condition.unique() if '+' in c})
    rng = np.random.default_rng(0); fig = go.Figure(); groups = []; ranges = []
    CLIP = (0.5, 99.5)   # axis ranges span this percentile band of the plotted points; nothing is dropped from the tables
    for fi, fault in enumerate(faults):
        conds = {'ID-safe': 'LG (ID-safe)', 'OOD-safe': 'RD (OOD-safe)',
                 'ID-dangerous': f'LG + {fault} (ID-dangerous)', 'OOD-dangerous': f'RD + {fault} (OOD-dangerous)'}
        idx, shown = [], []
        for cell, cond in conds.items():
            g = fdf[fdf.condition == cond]; pick = rng.choice(len(g), min(1500, len(g)), replace=False); g = g.iloc[pick]
            color, symbol = cells[cell]
            xyz = np.log10(g[list(axes3)].to_numpy()); shown.append(xyz)
            fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
                                       name=cell, visible=(fi == 0), legendgroup=cell, showlegend=True,
                                       marker=dict(size=2.4, color=color, symbol=symbol, opacity=.55),
                                       text=g['subject'] + ' · ' + g['trial'], hovertemplate='%{text}<extra>' + cell + '</extra>'))
            idx.append(len(fig.data) - 1)
        groups.append(idx)
        allpts = np.concatenate(shown); lo, hi = np.percentile(allpts, CLIP[0], axis=0), np.percentile(allpts, CLIP[1], axis=0)
        pad = .05 * (hi - lo); ranges.append([[float(l - p), float(h + p)] for l, h, p in zip(lo, hi, pad)])
    def layout_for(fi, fault):
        r = ranges[fi]
        return {'title.text': f'Four cells, fault: {fault}  (axes clipped to the central {CLIP[1] - CLIP[0]:.0f} % of plotted windows)',
                'scene.xaxis.range': r[0], 'scene.yaxis.range': r[1], 'scene.zaxis.range': r[2]}
    buttons = [dict(label=fault, method='update', args=[{'visible': [i in set(groups[fi]) for i in range(len(fig.data))]}, layout_for(fi, fault)])
               for fi, fault in enumerate(faults)]
    fig.update_layout(title=layout_for(0, faults[0])['title.text'], template='plotly_white', height=680, margin=dict(l=0, r=0, t=90, b=0),
                      scene=dict(xaxis=dict(title='log10 LOF novelty', range=ranges[0][0]), yaxis=dict(title='log10 AE reconstruction NLL', range=ranges[0][1]),
                                 zaxis=dict(title='log10 routed expert forecast error', range=ranges[0][2]), aspectmode='cube'),
                      updatemenus=[dict(type='dropdown', x=0, xanchor='left', y=1.1, yanchor='top', buttons=buttons)],
                      legend=dict(title='cell (click to hide, double-click to isolate)', itemsizing='constant'))
    show(fig, 'faults_four_cells_3d', SAVE_HTML)
else:
    print('No fault scores yet:', fsc)"""),
    md("""## What other targets could make the regime matter? A label inventory

Regime conditioning failed because the 200 ms hip-angle target is locally continuous and
therefore regime-insensitive. The raw trials carry more than the 16 channels: inverse-kinematics
joint angles, inverse-dynamics **joint moments**, per-leg gait phase from heel strike and toe off,
ground reaction forces and a pelvis IMU. `ml/awareness/results/dataset_labels.md` inventories them
and their coverage; the table below is the per-mode coverage of the sampled trials. Gait phase and
moments are valid on exactly the samples where a force plate was struck, and stairs carry one or
two labelled strides per short trial. Candidate regime-dependent targets: the coming stride's
duration, stance fraction, hip flexion/extension peaks (available from the encoder itself, no lab
labels needed), and the hip moment, which is also the fallback controller's own quantity."""),
    code("""cov = AW / 'results' / 'dataset_label_coverage.csv'
if cov.exists():
    lab = pd.read_csv(cov)
    display(lab.groupby('mode').agg(trials=('trial', 'size'), seconds=('seconds', 'median'),
        gait_phase_valid_L=('gp_valid_l', 'mean'), gait_phase_valid_R=('gp_valid_r', 'mean'),
        strides_L=('strides_l', 'median'), stride_time_s=('stride_s_l', 'median'),
        moment_valid_L=('moment_valid_l', 'mean'), hip_range_rad=('hip_range_l', 'median')).round(3))
else:
    print('No coverage table yet:', cov)"""),
    md("""## How to read this, and what comes next

- **Novelty** answers "is this unlike LG/RA/SA?". The held-out modes should sit to the right of the
  trained-on ones; how far, and whether standing and transitions behave like the descents, is what
  the violins show.
- **Regime** cannot say "descend". Watch whether entropy rises on RD/SD relative to ID, and which
  known regime absorbs them; a confident wrong assignment is itself a finding.
- **Reliability** is two axes: disagreement now, realized error later. Their joint reading
  (confident and wrong) is the one a single score cannot express.
- **Baselines** show where each existing model's score puts the same windows.

No score is thresholded here. Fusion, thresholds at a declared false-alarm budget, seeds and the
other regime holdouts (ascending, level) are the next steps, in that order."""),
            md(r"""## Ablation: the pelvis IMU as six more input channels

Notebook 09's raw-space check said the pelvis IMU adds almost no task structure (k-NN task accuracy
0.600 → 0.615 with participants held out). The question that matters here is different: does a
learned encoder get anything out of it for **novelty** under the regime holdout? So Stage A and
Stage B are rerun unchanged on a 22-channel build of the same windows
(`ml/LOTO/build_data.py --channels pelvis`: the 16 channels with the *same* scaler numbers, plus
`pelvis_accel_xyz` and `pelvis_gyro_xyz` with their own LG/RA/RD-training scaler). Same β = 0 encoder,
same settings and seed, same LOF reference size, same test windows; the arrays were cut on the cluster
from the raw trials (`ml/awareness/slurm/pelvis_build.sbatch`) and the runs sit beside the 16-channel
ones as `encoder_pelvis/` and `regime_pelvis/`.

Read the table as a paired comparison: every AUROC is on the identical windows, so a difference of a
few thousandths is seed noise and a difference on one held-out mode is where to look."""),
            code("""pa = RESULTS_AW / 'pelvis_ablation.csv'
if pa.exists():
    pab = pd.read_csv(pa)
    wide = pab.pivot(index='score', columns='encoder', values=[c for c in pab.columns if c.startswith('AUROC')])
    wide.columns = [f'{a.replace("AUROC ", "")} · {b.split(" ")[0]} ch' for a, b in wide.columns]
    display(wide.loc[['LOF, 32-D latent', 'LOF, PCA-12', 'LOF, PCA-16', 'reconstruction NLL']].round(3))
    display(pd.read_csv(RESULTS_AW / 'pelvis_ablation_runs.csv', index_col=0).round(3))
    png = RESULTS_AW / 'pelvis_ablation.png'
    display(Image(filename=str(png), width=1000)) if png.exists() else print('missing', png.name)
else:
    print('Run `python ml/awareness/pelvis_ablation.py --holdout DOWN --write` after pulling encoder_pelvis/ to create', pa.name)"""),
            md("""### Interactive views of the 22-channel encoder's test latents

The same construction as the 16-channel views in Stage A: UMAP and PCA fitted on the **in-distribution**
test latents (LG, RA, SA) of `encoder_pelvis`, every task transformed in, raw latent coordinates in the
third panel; same point budget and seed, task colours fixed. Compare with the Stage A views above: the
question is whether the four held-out modes sit further from, or closer to, the trained-on gait once the
pelvis IMU is in the input. The AUROC table says closer."""),
            code("""ENC_PELVIS = RUNS_AW / 'encoder_pelvis'
if (ENC_PELVIS / 'latents_test.npz').exists():
    enc_pelvis_views = ae_latent_views(ENC_PELVIS, dims=VIEW_DIMS, projections=PROJECTIONS, latent_dims=LATENT_DIMS,
                                       max_points=MAX_POINTS, seed=VIEW_SEED, save_html=SAVE_HTML,
                                       title='22-channel encoder latent (32-D)')
else:
    print('Ablation run not pulled yet:', ENC_PELVIS)"""),
    code("source_links(['ml/awareness/README.md', 'ml/awareness/encoder.py', 'ml/awareness/regime.py', 'ml/awareness/forecast.py', 'ml/awareness/experts.py', 'ml/awareness/faults.py', 'ml/awareness/aw_scores.py', 'ml/awareness/aw_plots.py', 'ml/LOTO/splits.py'])")]


def notebooks():
    for spec in SPECS:
        yield spec['file']+'.ipynb',existing_notebook(spec)
    yield '08_vae_variants.ipynb',vae_notebook()
    yield '09_umap_all_tasks.ipynb',all_tasks_notebook()
    yield '10_awareness_regime_holdout.ipynb',awareness_notebook()


def build(filename,cells):
    for i,cell in enumerate(cells):
        cell['id']=hashlib.sha256(f'{filename}:{i}:{cell.source}'.encode()).hexdigest()[:12]
    return nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Hip Exo (.venv)','language':'python','name':'hip-exo'},
        'language_info':{'name':'python','version':'3.12'},
        'model_exploration':{'training_enabled_by_default':False}})


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--patch',action='store_true',help='print an apply_patch patch for every notebook')
    mode.add_argument('--write',nargs='*',metavar='STEM',
                      help='write notebooks directly (all, or the given stems such as 02_autoencoder_lof); '
                           'saved outputs are dropped, so execute them again afterwards')
    args=parser.parse_args()
    if args.write is not None:
        wanted={stem.removesuffix('.ipynb') for stem in args.write}
        for filename,cells in notebooks():
            if wanted and filename.removesuffix('.ipynb') not in wanted:continue
            nbf.write(build(filename,cells),HERE/filename);print('Wrote',filename)
        unknown=wanted-{f.removesuffix('.ipynb') for f,_ in notebooks()}
        if unknown:raise SystemExit(f'Unknown notebook stems: {sorted(unknown)}')
        return
    print('*** Begin Patch')
    for filename,cells in notebooks():
        nb=build(filename,cells)
        text=nbf.writes(nb)+'\n';path=HERE/filename
        if path.exists():
            print(f'*** Update File: {path}')
            print('@@')
            for line in path.read_text().splitlines():print('-'+line)
        else:
            print(f'*** Add File: {path}')
        for line in text.splitlines():print('+'+line)
    print('*** End Patch')


if __name__=='__main__':main()
