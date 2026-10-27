# D04: Device timing and stability on the Jetson: latency budget, jitter sources, fp16, and a 12 h soak

| Field | Value |
|---|---|
| ID | D04 |
| Date | 2026-10-27 (launched 16:59, unattended; auto shutdown after) |
| Dataset | Sci-Rob replay streams (D03): dev participants for the jitter and fp16 runs, all 20 for the soak |
| Status | **Running** on the Jetson (results expected 2026-10-28; review 2026-10-30) |
| Sources | D02 (B1 engine), D03 (streaming replay: 3 of 799,782 steps over 10 ms) |
| Code / notebooks | `deployment/overnight.sh` (battery), `replay_runner.py` (`--gc-freeze`, `--cpu`, `--engine`), `latency_budget.py`, `soak_test.py`; Jetson logs `~/hip-exo/logs/overnight/` (incl. `tegrastats.log`), outputs `~/hip-exo/data/replay/scirob/{predictions/jitter_*, predictions/jetson_fp16, soak/soak.npz, latency_budget.json}` |

## Hypothesis

Stated before the battery ran. A 2-trial smoke run of every step (2026-10-27 16:47) had already shown the awareness
encoders at about 21 ms per window together; (c) is written with that known.

- **(a) Jitter.** D03's rare slow steps (51 over 5 ms, 3 over 10 ms in 799,782) come from Python's garbage collector or
  from scheduling. Freezing the collector and/or pinning the process to one core removes most of them, without root.
- **(b) fp16.** The fp16 engine is no faster on real streams than fp32 (D02: not on synthetic windows either) and stays
  within its 3e-2 N·m/kg tolerance on real validation windows.
- **(c) Latency budget.** B1 fits the 10 ms tick with large margin (about 2 ms). The awareness layer (two encoders,
  novelty scores, trees, smoothing) fits its 50 ms window cadence but not inside a 10 ms tick next to B1, so on the device
  it must run on its own cadence or thread; fp16 autocast shortens the encoders.
- **(d) Soak.** Streaming at true 100 Hz pacing for 12 h shows no drift: step-time percentiles and resident memory flat
  hour to hour, temperatures plateau well below throttling, and fewer than 1 in 10⁵ ticks miss their deadline.

## Method

- **Order** (most important first, so a late failure loses least): T1 (D03's sealed final scoring), T4 latency budget,
  T2 jitter, T3 fp16, T5 soak; then summary and shutdown. Each step logs separately and a failure does not stop the next.
  `tegrastats` logs temperatures, power, frequencies and memory every second throughout.
- **T2 jitter:** the dev replay (4 participants, calibrated, about 400,000 two-leg steps each) under four conditions:
  baseline; garbage collector collected, frozen and disabled before streaming (`--gc-freeze`); process pinned to core 3
  (`--cpu 3`, away from cpu0's interrupts); both. Measure: step-time median, p99, p99.99, counts over 5 and 10 ms.
- **T3 fp16:** `moment_engine.py --fp16` (synthetic check), `check_moment_real.py --engine moment_fp16.engine` (2048
  real validation windows), dev replay with the fp16 engine; compare to the fp32 replay (D03) prediction by prediction
  and in step time.
- **T4 latency budget:** batch 1, 500 timed calls after 50 warm-up, CUDA synchronized: LSTM-VAE encoder (16 ch, 600
  samples) and TCN-VAE encoder (23 ch, 600 samples) with random weights of the deployed architectures (latency does not
  depend on weight values; no trained awareness model was sent), fp32 and fp16 autocast; novelty scores against 15,000
  reference latents; 183 stand-in trees of depth 6; smoothing and regime; the real B1 engine.
- **T5 soak:** 12 h, absolute 10 ms schedule; each tick pushes the next two 200 Hz samples (cycling through all 20
  participants' calibrated streams) and runs one two-leg engine call. Per tick: step time, wake-up lateness, missed
  deadline (step ending after the next tick's start). Resident memory every minute. Checkpointed every 10 min.
- **Not covered:** CAN I/O, building windows and rhythm features from live data, real-time scheduling priority (needs
  root), other power modes.

## Result

*Pending (battery running).*

## Conclusion

*Pending.*
