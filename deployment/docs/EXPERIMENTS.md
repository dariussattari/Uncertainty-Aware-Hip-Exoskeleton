# Deployment experiment log: Uncertainty-Aware Hip Exoskeleton

Experiments about getting the controller onto the exo's Jetson (Orin Nano, JetPack 6.2.1), in chronological order, each
with a one-line result. Same format as the research log (`docs/EXPERIMENTS.md`, E01–E41): each `D##_*.md` stands alone
with a header table (date, dataset, sources, code paths) and **Hypothesis, Method, Result, Conclusion**; hypotheses are
written before results are seen. The current deployment state and plan live in
[`../deployability_status.md`](../deployability_status.md).

| ID | Date | Question | Result |
|---|---|---|---|
| [D01](D01_awareness_portability_to_jetson.md) | Oct 27 | Can the validated awareness layer (E39) run on the Jetson's Python 3.10 without changing it? | Refit under scikit-learn 1.7.2 is within seed noise but a different model (0.006% dangerous to the spline). Shipped instead as plain arrays: reproduces E39 to 1e-9, no scikit-learn on the device |
| [D02](D02_b1_tensorrt_engine_on_jetson.md) | Oct 27 | Does B1 survive conversion to TensorRT on the Jetson, and how fast is it? | Engine vs PyTorch 7.6e-5 N·m/kg on real validation windows; R² 0.7530 vs the run's 0.7529; 1.98 ms per two-leg call. Beats the Nature Estimator on Phase 3 (0.725 vs 0.671) |
| [D03](D03_scirob_replay_stage_a.md) | Oct 27 | Stage A replay: B1 streamed on the Jetson over Sci-Rob exo recordings, vs Ground Truth and the Sci Robotics Estimator | Dev (4 people): Jetson = reference to 1.6e-3 N·m/kg; 2.36 ms per step (p99 2.83). B1 calibrated R² 0.707 vs the Sci Robotics Estimator 0.838; calibration +0.06 on the rolled left IMU; B1 fails on stair descent. **Final 16 (scored once): B1 0.698 vs Sci Robotics Estimator 0.843**, below it for every person and mode |
| [D04](D04_device_timing_and_stability.md) | Oct 27 | Does the device keep time? Latency budget of the whole controller, jitter sources, fp16, 12 h soak at 100 Hz | 12 h at 100 Hz: no drift (p99 2.7 ms every hour, +3% memory, ≤ 45 °C, 3.4 W); device misses 1.4e-5 (96% of misses were the test harness's own file writes). GC freezing / core pinning don't change jitter. Awareness layer 20 ms per window: needs its own 50 ms loop. fp16 7% faster, same accuracy |
| [D05](D05_encoder_tensorrt.md) | Oct 29 | Do the awareness encoders get faster as TensorRT engines? | TCN-VAE 10.97 → 2.09 ms (5.2×); LSTM-VAE 3.46 → 14.09 ms (4× slower: TensorRT loops the 600-step recurrence, PyTorch uses fused cuDNN). Mix them: awareness ≈ 11.7 ms per window |

**Names used in these experiments.** *B1*: this project's joint-moment estimator (E41). *Sci Robotics Estimator*: the
estimate recorded live on the exo during the Science Robotics study, provided in that dataset (`trq_est`). *Nature
Estimator*: the same for the Nature study (`*_torque_estimated`). *Ground Truth*: inverse dynamics provided in each
dataset. *Lab Model*: the lab's TCN in `HipExoGPUCode` (not run yet).

## Where things stand (as of 2026-10-29)

- **The device works.** B1 runs on the Jetson as a TensorRT engine that is the trained model (D02), streams at 100 Hz at
  2.2–2.4 ms per tick, and held steady for 12 h at ≤ 45 °C and 3–4 W (D04). The awareness layer runs without scikit-learn
  and reproduces E39 exactly (D01).
- **The controller must be two loops.** The awareness layer takes about 20 ms per window (11.7 ms with the TCN-VAE
  encoder as a TensorRT engine and the LSTM-VAE encoder kept in PyTorch, D05): it fits its 50 ms cadence, not a 10 ms
  tick (D04).
- **B1's accuracy depends on the device it was trained for.** On the Nature data it beats the Nature Estimator (R² 0.81
  vs 0.72, D02); on this lab's exo data it trails the Sci Robotics Estimator (0.70 vs 0.84 on 16 held-out people,
  D03), most on stairs. Per-person IMU calibration recovers 0.06 on a badly rolled IMU.
- **Real-time hygiene:** no file writes inside the control loop (they caused 96% of missed ticks in the soak); the
  residual jitter (about 1 tick in 70,000) is not Python GC or core migration (D04).

## Next steps

In rough priority; each links to the experiment that motivates it.

1. **Decide the OOD-safe estimator for this exo** (D03): fine-tune or retrain B1 on the lab's exo data, or adopt the Lab
   Model after rebuilding its engine and replaying it through the D03 pipeline for a like-for-like comparison.
2. **Cheap B1 checks on the dev participants** (D03): an output low-pass filter, and how much of the gap the 25–50 ms
   timing offset explains.
3. **Live input pipeline** (D02, D03): a streaming converter from the exo's CAN data to B1's input conventions, with
   per-wearer IMU calibration at donning; the seven rhythm features computed live for the classifier.
4. **Two-loop controller** (D04, D05): B1 every 10 ms; awareness every 50 ms in its own thread, with the trained TCN-VAE
   encoder converted to TensorRT and checked on real data; logging off the control path.
5. **Classifier without LOF** (D01, plan step 4): saves most of the remaining 5.6 ms novelty cost.
6. **Real-time scheduling** (D04): with sudo, `chrt` priority and an isolated core; hold the last command on a late tick.
7. **Before the real bus:** set the Jetson's CAN sample point to the STM32's 0.857 (it is 0.740 now).

