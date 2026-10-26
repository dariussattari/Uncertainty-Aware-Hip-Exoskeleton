# E32: Training-class ablation: train on safe movement to detect dangerous movement?

| Field | Value |
|---|---|
| ID | E32 |
| Date | 2026-10-14 to 2026-10-15 |
| Dataset | Nature 2024 |
| Status | Complete |
| Sources | 12OCT-15OCT Verbose.docx §14.0; README "Training-class ablation"; Live_Update.docx "OCT 10 – 16" (AUROC of OOD-safe against OOD-dangerous; classifier on the latent) |
| Code / notebooks | `--train-class` in `train_ae.py` and `train_vae.py`; `ml/nature/awareness/train_class_tasks.py`, `slurm/train_class.sbatch`; `_novelty_scores.evaluate_dangerous`; `notebooks/nature/model_exploration/onboard/07_training_class_ablation.ipynb`; runs `runs_train_class/`; results `onboard/results/07_*.csv` |

## Hypothesis

Every model so far learned only the 4 ID-safe tasks, so it flags any unfamiliar movement, safe or dangerous. If a simple
heuristic recognizes ID movement (cyclic walking and stairs), the novelty model has one job left: tell OOD-dangerous
movement (exo off) from OOD-safe movement (estimator takes over). A model that has **seen safe movement** should do that
job better than one that has seen only walking.

## Method

| Training class | Trained on | Then unfamiliar |
|---|---|---|
| ID (E24–E31) | the 4 ID-safe tasks | the other 22 |
| ID + OOD-safe | the 22 safe tasks | the 4 OOD-dangerous |
| OOD-safe only | the 18 OOD-safe tasks | ID and OOD-dangerous (ID left to the heuristic) |

- OOD-dangerous movement (ball toss, step-ups, tug of war, sit-to-stand) is never trained on.
- All 70 configurations so far were retrained on each new class: 420 runs, 70 GPU-hours (jobstats: 86% of one core,
  91% GPU, 1.1 GB). Participants, arrays and cells are unchanged. Early stopping uses the same training classes' validation
  windows.
- **Protocol fixed before scoring:**
  - Target: OOD-dangerous against OOD-safe windows (ID excluded).
  - Every native score, with direction and threshold chosen on validation (1% of validation OOD-safe windows flagged).
  - Test scored once.
  - The best cell and score per (training class, model type, input) are chosen on validation.
- **Supervised reference:** a logistic regression on each run's latent, fitted on validation dangerous and OOD-safe
  windows. It has seen all four dangerous tasks.

## Result

| Trained on | Mean over 24 set-ups | Best set-up | Dangerous flagged at 1% false alarms | Higher than ID on |
|---|---|---|---|---|
| ID | **0.757** | 0.818 (LSTM-VAE, 3 s + rhythm, posterior variance) | 23% | – |
| ID + OOD-safe | 0.688 | 0.786 (Transformer, 1 s raw, posterior variance) | 14% | 2 of 24 |
| OOD-safe only | 0.693 | 0.766 (Transformer, 3 s + rhythm, posterior variance) | 1.5% | 2 of 24 |

- **Training on safe movement removes the signal.** AUROC falls on 22 of 24 set-ups (by 0.07 on average, up to 0.16).
  No set-up is a usable alarm.
- Even ID-trained models reach only 0.757 on average for dangerous against safe (against up to 0.963 for unfamiliar
  against familiar).
- The useful score shifts to reconstruction error or posterior variance; LOF is never chosen.
- Per task: step-ups stay invisible (0.52–0.66; they move like curbs and stairs). Ball toss drops from 0.85 to
  0.72–0.79, and sit-to-stand from 0.86 to about 0.80 (it moves like squats and lifts).
- **Supervised read-out of the latent:** mean 0.865 (ID-trained), 0.880 (ID + safe), 0.883 (safe only); best 0.921
  (conv VAE, 3 s + rhythm, ID + safe). That is 0.11–0.19 above the novelty scores on the same latents. Here the rhythm
  features help.
- Retraining reshapes latents: CKA with the ID-trained twin is 0.46–0.81 for most models and 0–0.33 for the sparse AE.
- With the TCN-VAE added (E33), the means over 28 set-ups are 0.763 / 0.695 / 0.699, and the best read-out is 0.924.

## Conclusion

**"Dangerous" is a label about risk, not a kind of movement the model has not seen.** Sit-to-stand moves like squats,
and step-ups like curbs and stairs. A novelty score cannot find danger, and gets worse once safe movement is trained on.
Keep the ID-trained novelty detector for unfamiliar against familiar (0.963). Move the dangerous-against-safe decision to
a **supervised regime classifier** (E34) that reads a latent trained on safe movement with 3 s + rhythm input. That
classifier must be tested on held-out dangerous tasks, since the read-out here has seen all four. The four-task danger
label itself needs revisiting (E27).
