# Deployment experiment log: Uncertainty-Aware Hip Exoskeleton

Experiments about getting the controller onto the exo's Jetson (Orin Nano, JetPack 6.2.1), in chronological order, each
with a one-line result. Same format as the research log (`docs/EXPERIMENTS.md`, E01–E41): each `D##_*.md` stands alone
with a header table (date, dataset, sources, code paths) and **Hypothesis, Method, Result, Conclusion**; hypotheses are
written before results are seen. The current deployment state and plan live in
[`../deployability_status.md`](../deployability_status.md).

| ID | Date | Question | Result |
|---|---|---|---|
| [D01](D01_awareness_portability_to_jetson.md) | Oct 27 | Can the validated awareness layer (E39) run on the Jetson's Python 3.10 without changing it? | Refit under scikit-learn 1.7.2 is within seed noise but a different model (0.006% dangerous to the spline). Shipped instead as plain arrays: reproduces E39 to 1e-9, no scikit-learn on the device |
| [D02](D02_b1_tensorrt_engine_on_jetson.md) | Oct 27 | Does B1 survive conversion to TensorRT on the Jetson, and how fast is it? | Engine vs PyTorch 7.6e-5 N·m/kg on real validation windows; R² 0.7530 vs the run's 0.7529; 1.98 ms per two-leg call. Beats the Nature study's deployed estimator on Phase 3 (0.725 vs 0.671) |
| [D03](D03_scirob_replay_stage_a.md) | Oct 27 | Stage A replay: B1 streamed on the Jetson over Sci-Rob exo recordings, vs inverse dynamics and the exo's online estimator | Dev (4 people): Jetson = reference to 1.6e-3 N·m/kg; 2.36 ms per step (p99 2.83). B1 calibrated R² 0.707 vs the exo's online estimator 0.838; calibration +0.06 on the rolled left IMU; B1 fails on stair descent. **Final 16 (scored once): B1 0.698 vs online 0.843**, below it for every person and mode |
| [D04](D04_device_timing_and_stability.md) | Oct 27 | Does the device keep time? Latency budget of the whole controller, jitter sources, fp16, 12 h soak at 100 Hz | 12 h at 100 Hz: no drift (p99 2.7 ms every hour, +3% memory, ≤ 45 °C, 3.4 W); device misses 1.4e-5 (96% of misses were the test harness's own file writes). GC freezing / core pinning don't change jitter. Awareness layer 20 ms per window: needs its own 50 ms loop. fp16 7% faster, same accuracy |
| [D05](D05_encoder_tensorrt.md) | Oct 29 | Do the awareness encoders get faster as TensorRT engines? | TCN-VAE 10.97 → 2.09 ms (5.2×); LSTM-VAE 3.46 → 14.09 ms (4× slower: TensorRT loops the 600-step recurrence, PyTorch uses fused cuDNN). Mix them: awareness ≈ 11.7 ms per window |
