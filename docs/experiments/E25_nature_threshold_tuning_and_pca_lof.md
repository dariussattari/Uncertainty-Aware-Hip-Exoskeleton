# E25: Alarm threshold tuned on held-out people, and LOF in fewer dimensions (plan step 3.2.1)

| Field | Value |
|---|---|
| ID | E25 |
| Date | 2026-10-11 to 2026-10-13 |
| Dataset | Nature 2024 |
| Status | Complete |
| Sources | 12OCT-15OCT Verbose.docx §3.2 (1) and §7.0; README "Nature awareness autoencoders" (operating-point bullet) |
| Code / notebooks | `ml/nature/awareness/encode_split.py`; `notebooks/nature/model_exploration/onboard/05_Novelty_Metric_comparison.ipynb`, section 6 |

## Hypothesis

E24 found good ranking but a too-cautious alarm. Two hypotheses:

1. Setting the threshold on validation participants at a fixed false-alarm target (1%), instead of the ankle paper's
   99.5th percentile of training scores, raises recall at matched false alarms.
2. LOF measured in a PCA-k projection of the latent (k chosen on validation) helps, as PCA-12/16 did on the Science
   Robotics data (E17).

## Method

- The four validation participants (BT09, BT12, BT13, BT21) were encoded locally from each saved checkpoint.
  Re-encoding the cluster's test export reproduced it to within 0.5%.
- On the validation participants: a threshold flagging 1% of familiar windows, and a choice of k in {4…24} or all 32
  units. Then the four test participants were scored once.
- 0.5 s causal median; three seeds.
- Done when: recall at a matched 1% false-alarm rate per task, three seeds.

## Result

| Model | Threshold, representation | False alarms | Unfamiliar flagged | Dangerous flagged |
|---|---|---|---|---|
| Plain AE 1 s | ankle rule, 32 units | 1.6 ± 0.4% | 29.2 ± 4.0% | 44.9 ± 7.1% |
| Plain AE 1 s | validation-tuned, 32 units | 1.8 ± 0.2% | 32.0 ± 5.7% | 49.7 ± 9.8% |
| **Plain AE 1 s** | **validation-tuned, 16 PCs** | 1.7 ± 0.6% | **34.2 ± 6.0%** | **52.8 ± 10.0%** |
| Plain AE 3 s + rhythm | ankle rule, 32 units | 2.2 ± 1.0% | 36.2 ± 5.5% | 40.3 ± 4.5% |
| Plain AE 3 s + rhythm | validation-tuned, 32 units | 2.0 ± 1.0% | 34.7 ± 3.8% | 38.2 ± 2.5% |

- AUROC did not move (1 s model: 0.887 → 0.883 unfamiliar, 0.968 dangerous).
- **Thresholds do not transfer exactly between people.** A 1% target on validation gives 1.7–2.0% on test, and a 0.25%
  target gives 0.8–1.2%. The ankle rule gave 1.6–2.2% instead of 0.5%. Familiar score levels differ between people by
  about 2× in false alarms.
- Tuning alone buys little: the ankle rule already sat on the same trade-off curve.
- Fewer dimensions help only the 1 s plain model; validation keeps all 32 units for the others. The gain is inside the
  ±6–10 point seed spread.
- **The false-alarm target is the real lever.** For the 1 s, 16-PC model: a 2% target gives 44% unfamiliar and 63%
  dangerous flagged (at 2.3% false alarms); a 5% target gives 61% and 81% (at 3.7%). The curve has no knee.
- Per task (1% target): jumping 75%, side shuffle 58%, sit-to-stand 56%, step-ups 55%, squats and lifting 52–53%, tug
  of war 53%, ball toss 36% (23% under the ankle rule), standing poses 29%. Near-familiar walking stays unflagged.

## Conclusion

- Use the plain 1 s AE with LOF on 16 PCs as the novelty score.
- **Calibrate the threshold per wearer** on a short bout of normal walking, because the false-alarm rate is set by the
  wearer, not by the training set.
- Choosing the false-alarm target is a *controller* decision (what share of walking windows may be handed to the
  fallback), to be made together with the switching logic (dwell time, torque ramp).
