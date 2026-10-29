# D05: Do the awareness encoders get faster as TensorRT engines?

| Field | Value |
|---|---|
| ID | D05 |
| Date | 2026-10-29 |
| Dataset | None: random windows; random weights of the deployed architectures (latency does not depend on weight values) |
| Status | Complete |
| Sources | D04 (awareness layer 20.1 ms per window; TCN-VAE encoder 10.5 ms, LSTM-VAE encoder 3.4 ms in PyTorch); D02 (B1: PyTorch 5.9 → TensorRT 2.0 ms) |
| Code / notebooks | `deployment/encoder_engine_test.py`; results `data/replay/scirob/encoder_engines.json` |

## Hypothesis

From D04's conclusion, written before this was measured: converting both VAE encoders to TensorRT, as B1 was, shortens
them the way it shortened B1 (about 3×), most usefully the TCN-VAE encoder, the largest item in the awareness budget.

## Method

- LSTM-VAE encoder (16 channels) and TCN-VAE encoder (23 channels), 600-sample windows, batch 1, the deployed
  architectures (`ml/common/models/seq_vae.py`) with random weights; `encode` only (mean and log variance, 4 × 8 each).
- ONNX opset 17 (static shapes) → `trtexec` fp32 and fp16 engines on the Jetson (TensorRT 10.3).
- Checked against PyTorch fp32 on 20 random windows (max abs difference over both outputs); timed exactly as D04's
  latency budget: 500 calls after 50 warm-up, CUDA synchronized.

## Result

| Encoder | PyTorch fp32 | TensorRT fp32 | TensorRT fp16 | Engine vs PyTorch, max abs |
|---|---|---|---|---|
| TCN-VAE | 10.97 ms (p99 11.15) | **2.09 ms** (p99 2.18) | 2.08 ms (p99 2.13) | 3.0e-5 (fp16 4.2e-4) |
| LSTM-VAE | **3.46 ms** (p99 5.53) | 14.09 ms (p99 14.34) | 14.10 ms (p99 14.37) | 2.5e-5 (fp16 9.1e-5) |

- **TCN-VAE: holds, and more:** 5.2× faster as an engine; fp16 adds nothing.
- **LSTM-VAE: does not hold:** 4× slower as an engine. The likely reason: PyTorch runs the 2-layer, 600-step LSTM as one
  fused cuDNN kernel, while TensorRT executes the recurrence as an explicit loop over the 600 steps.

## Conclusion

- **Use TensorRT for the TCN-VAE encoder and keep the LSTM-VAE encoder in PyTorch (cuDNN).** The awareness layer per
  window becomes about 3.4 (LSTM) + 2.1 (TCN engine) + 5.6 (novelty) + 0.6 (trees, smoothing) ≈ **11.7 ms**, down from
  20.1; without LOF (plan step 4) roughly 6–7 ms.
- That is still more than half a 10 ms tick next to B1's 2–2.4 ms, so the two-loop design from D04 stands: B1 every
  tick, awareness on its own 50 ms cadence.
- Real engines need the trained encoders on the device; this used stand-ins and checked only conversion and speed.
