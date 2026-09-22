"""Emit an apply_patch patch for the eight exploration notebooks (no training)."""
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
           md('## Leave-one-task-out (LOTO) stress test\n\nSA: stair ascent; ST: standing; TR: transitions. Each run excludes the named task from **all training participants**, trains on the other six, and tests on held-out participants. This differs from LOSO epoch selection.\n\nStanding test data come from one trial of one participant: 621 overlapping windows, or 600 for the gait-phase subset. Gait-phase LOTO uses fewer labelled windows overall. Raw window counts are not independent repetitions; cross-run changes do not isolate a causal training effect.'),
           code('loto_results(KEY)'),md('## Takeaway and next checks\n\n'+s['takeaway']+'\n\nKeep score/threshold selection on validation data. Report both ranking and recall at a declared false-alarm budget, repeat seeds, and quantify participant/trial-level variation before making robustness claims.'),
           code(f"source_links({s['sources']!r})")]
    return cells


def vae_notebook():
    return [md('''# VAE exploration: standard VAE and β-VAE

**Status: new notebook-local prototypes, not completed benchmark results.** No production
VAE existed in this checkout at notebook creation. This notebook includes executable setup,
loss, opt-in training and evaluation. Variants share architecture; only the KL weight changes.

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
    md('### Executable model and loss\n\nThe implementation is local to this notebook folder. It is not silently added to the production registry. The next pass uses random weights and dummy inputs only.'),
    code('''from _vae_experiment import ConvVAE, VAEConfig, loss_terms, train_experiment, plot_representation
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
- The optional window cap is a **smoke test**, not a benchmark. No training is run by default.
'''),
    code('''HELD_OUT = None  # None, 'SA', 'ST', 'TR'
SEED = 0
VARIANTS = {'standard': 1.0, 'beta4': 4.0}
protocol = HELD_OUT or 'original'
VAE_RUNS = {name: ROOT / 'notebooks/model_exploration/runs/vae' / f'{protocol}_{name}_seed{SEED}'
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

- [Notebook-local implementation](_vae_experiment.py); [AE baseline](../../ml/vanilla/models/autoencoder.py).
- [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
- [Nalisnick et al.: generative likelihood can fail at OOD detection](https://arxiv.org/abs/1810.09136).
- [Higgins et al.: beta-VAE](https://openreview.net/forum?id=Sy2fzU9gl).

Literature findings motivate this experiment; they do not establish performance on hip-exoskeleton data.
''')]


def notebooks():
    for spec in SPECS:
        yield spec['file']+'.ipynb',existing_notebook(spec)
    yield '08_vae_variants.ipynb',vae_notebook()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--patch',action='store_true',required=True)
    parser.parse_args()
    print('*** Begin Patch')
    for filename,cells in notebooks():
        for i,cell in enumerate(cells):
            cell['id']=hashlib.sha256(f'{filename}:{i}:{cell.source}'.encode()).hexdigest()[:12]
        nb=nbf.v4.new_notebook(cells=cells,metadata={
            'kernelspec':{'display_name':'Hip Exo (.venv)','language':'python','name':'hip-exo'},
            'language_info':{'name':'python','version':'3.12'},
            'model_exploration':{'training_enabled_by_default':False}})
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
