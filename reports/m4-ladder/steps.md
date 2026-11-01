# Ladder steps: `m4-ladder`

Each step paired against the configuration it builds on (ratio < 1 = faster, 95% bootstrap CI). "Within noise" = the interval reaches inside 1 ± 3%.

| Step | Compared with | batch RTF | stream TTFA | stream total |
|---|---|---|---|---|
| `L00_upstream_torch` | `L03_upstream_q4_onnx` | 4.30× [4.13–4.54] slower | – | – |
| `L01_upstream_torch_mps` | `L03_upstream_q4_onnx` | 2.17× [2.08–2.37] slower | – | – |
| `L02_upstream_q8_onnx` | `L03_upstream_q4_onnx` | 1.13× [1.10–1.15] slower | 1.04× [1.01–1.06] (within noise) | 1.07× [1.04–1.10] slower |
| `L04_upstream_q4_onnx_metal` | `L03_upstream_q4_onnx` | 0.94× [0.90–1.06] (within noise) | 0.71× [0.68–0.78] | 0.87× [0.85–0.92] |
| `L05_fast_equivalent` | `L03_upstream_q4_onnx` | 0.92× [0.90–0.93] | 0.84× [0.82–0.88] | 0.92× [0.91–0.94] |
| `L06_threads` | `L05_fast_equivalent` | 1.04× [1.02–1.05] (within noise) | 1.03× [1.01–1.07] (within noise) | 1.06× [1.06–1.07] slower |
| `L07_slice_watermark` | `L06_threads` | 1.01× [1.00–1.03] (within noise) | 1.01× [0.99–1.04] (within noise) | 0.98× [0.98–0.99] (within noise) |
| `L08_pipelined` | `L07_slice_watermark` | 1.00× [0.99–1.01] (within noise) | 1.14× [1.09–1.15] slower | 0.87× [0.86–0.88] |
| `L09_prefix_reuse` | `L08_pipelined` | 0.96× [0.94–0.98] (within noise) | 0.96× [0.91–0.99] (within noise) | 0.98× [0.96–1.00] (within noise) |
| `L10_speech_vocab` | `L09_prefix_reuse` | 0.81× [0.78–0.84] | 0.93× [0.88–0.98] (within noise) | 0.83× [0.81–0.85] |
| `L11_fixed_final_chunk` | `L10_speech_vocab` | 0.99× [0.98–1.00] (within noise) | 0.99× [0.95–1.01] (within noise) | 1.00× [0.97–1.00] (within noise) |
| `L12_flash_attn` | `L11_fixed_final_chunk` | 0.98× [0.94–1.00] (within noise) | 0.89× [0.84–0.94] | 0.97× [0.94–1.01] (within noise) |
| `L13_gpu_prefill` | `L12_flash_attn` | 0.87× [0.84–0.89] | 0.72× [0.69–0.74] | 0.86× [0.83–0.92] |
| `L14_small_first_chunk` | `L13_gpu_prefill` | 1.00× [1.00–1.01] (within noise) | 0.85× [0.83–0.87] | 1.00× [0.99–1.02] (within noise) |
| `L15_gpu_codec` | `L14_small_first_chunk` | – | 0.84× [0.81–0.85] | 0.79× [0.76–0.83] |
| `X1_int8_codec` | `L14_small_first_chunk` | 0.95× [0.95–0.96] | 0.84× [0.82–0.86] | 0.87× [0.85–0.89] |
| `X2_q8_backbone` | `L14_small_first_chunk` | 1.20× [1.17–1.21] slower | 1.08× [1.07–1.11] slower | 1.20× [1.16–1.27] slower |
