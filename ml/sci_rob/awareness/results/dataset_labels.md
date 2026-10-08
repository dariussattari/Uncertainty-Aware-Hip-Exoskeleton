# What labels the raw dataset carries (Molinaro et al. 2024, participants AB15–AB34)

Scanned 1,505 trials over the 20 study participants (`dataset_trials.csv`); label coverage was
measured on 12 sampled trials per mode plus every standing and transition trial
(`dataset_label_coverage.csv`). Every trial folder has the same five files at 200 Hz:

| file | columns | what it is |
|---|---|---|
| `exo.csv` | 28: hip encoder angle/velocity ×2, thigh IMU accel/gyro ×2, **pelvis IMU** accel/gyro | the exoskeleton's own sensors; the 16-channel arrays are a subset (the pelvis IMU is unused so far) |
| `angle.csv` | hip, knee, ankle angle ×2 legs | motion-capture inverse kinematics |
| `moment.csv` | hip, knee, ankle moment ×2 legs | inverse dynamics; **valid only where a force plate was struck** |
| `gp.csv` | `gp_hs_{l,r}`, `gp_to_{l,r}` | gait phase as a fraction 0–1 from the last heel strike / toe off, per leg; NaN where undefined |
| `grf.csv` | 3-D force and centre of pressure ×2 | force plates |

Trial names encode the condition: `RD_Cn12p5_S0p88_UC_1` = ramp descent, −12.5°, 0.88 m/s;
`SA_C15p2_S0p0_UC_1_7` = stair ascent with a 15.2 cm riser, one short walk (about 5 s). Ramps
span 5–15°, level ground 0.6–1.9 m/s, stairs four riser heights; all trials are the
unassisted controller condition (`UC`).

## Coverage (share of samples with a valid value; per mode, sampled trials)

| mode | trial length | gait phase L / R | hip moment L / R | strides per trial (L) | median stride time |
|---|---|---|---|---|---|
| LG | 32 s | 0.81 / 0.76 | 0.81 / 0.76 | 21 | 1.24 s |
| RA | 32 s | 0.90 / 0.92 | 0.90 / 0.92 | 27 | 1.09 s |
| RD | 32 s | 0.84 / 0.85 | 0.84 / 0.85 | 27 | 1.00 s |
| SA | 5.7 s | 0.40 / 0.40 | 0.40 / 0.40 | 1 | 1.22 s |
| SD | 5.2 s | 0.35 / 0.31 | 0.35 / 0.31 | 2 | 1.03 s |
| ST | 32 s | 0.94 / 0.94 | 0.94 / 0.94 | 0 | — |
| TR | 21 s | 0.91 / 0.91 | 0.91 / 0.91 | 9 | 1.04 s |

Gait phase and moments are valid on exactly the same samples: both come from force-plate
contact, and both drop out for a few hundred samples whenever a step misses a plate. Stairs
are short and carry one or two labelled strides per trial, so per-stride labels are thin there.
Hip range of motion per trial already separates the regimes (median: RA 1.09 rad, SA 1.00,
LG 0.83, TR 0.86, RD 0.69, SD 0.63, ST 0.02), which is the kind of quantity a regime-dependent
target should be built from.

## Summary quantities that can be labelled offline (for training a predictor whose *disagreement* is used online)

| target | derived from | regime-dependent? | notes |
|---|---|---|---|
| next stride time / cadence | heel-strike resets in `gp_hs` | yes (1.00–1.24 s across modes, more across speeds) | one value per stride; needs ≥1 full stride after the window |
| stance fraction (toe-off timing) | `gp_to` versus `gp_hs` | yes | same coverage as gait phase |
| coming-stride hip flexion peak / extension peak / range | `angle.csv`, or the encoder angle in `exo.csv` itself | strongly (see ROM above) | encoder-based version needs no lab labels at all |
| hip moment (peak, or trajectory) | `moment.csv` | strongly (descent is eccentric) | the fallback controller's own quantity; force-plate coverage only; option (2), later |
| knee / ankle angles at the same instant | `angle.csv` | moderately | cross-joint consistency, not available on the exoskeleton |

The stride-level targets need heel-strike events at training time only; at run time the model
predicts them from the window and the ensemble's disagreement is the signal, as with the
angle forecast. Heel strikes could also be detected online from the encoder angle, which is
what the spline controller already does, so these targets stay realizable on the device.
