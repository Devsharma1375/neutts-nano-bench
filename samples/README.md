# Listening samples

Streamed output from the `m4-default` run (the results in `DESIGN.md` section 8.1), copied by `scripts/make_samples.py`. The rule is fixed: the first sentence of each length bucket (short, medium, long) in both voices, from every configuration that streams. Nothing was picked by ear.

Every clip is 24 kHz mono, seed 0, exactly what a listener would receive, Perth watermark included. `upstream_q4_onnx` is NeuTTS-Nano as Neuphonic documents it for low latency (the baseline); `fast_exact` generates the same speech tokens as upstream; `fast_best` and `fast_best_gpucodec` are numerically different and accepted through the quality gates. `fast_exact`'s clips are 0.02 s (one codec frame) shorter than upstream's: the same tokens, with upstream's misplaced final chunk fixed (`DESIGN.md` section 5.4).

Columns: time to first audio and total time for this request, Whisper `small.en`'s transcript and word error rate, and ECAPA speaker similarity to the reference recording.

## `s01` · voice `jo`

> Good morning, how did you sleep?

| Configuration | File | First audio | Total | Audio | Whisper heard | WER | Speaker sim |
|---|---|---|---|---|---|---|---|
| `upstream_q4_onnx` | [s01_jo_1_upstream_q4_onnx.wav](s01_jo_1_upstream_q4_onnx.wav) | 0.693 s | 0.95 s | 1.44 s | Good morning. How did you sleep? | 0.00 | 0.332 |
| `upstream_q4_onnx_metal` | [s01_jo_2_upstream_q4_onnx_metal.wav](s01_jo_2_upstream_q4_onnx_metal.wav) | 0.457 s | 0.85 s | 1.92 s | Good morning. How did you sleep? | 0.00 | 0.330 |
| `fast_exact` | [s01_jo_3_fast_exact.wav](s01_jo_3_fast_exact.wav) | 0.560 s | 0.69 s | 1.42 s | Good morning. How did you sleep? | 0.00 | 0.332 |
| `fast_best` | [s01_jo_4_fast_best.wav](s01_jo_4_fast_best.wav) | 0.282 s | 0.52 s | 1.76 s | Good morning. How did you sleep? | 0.00 | 0.400 |
| `fast_best_gpucodec` | [s01_jo_5_fast_best_gpucodec.wav](s01_jo_5_fast_best_gpucodec.wav) | 0.233 s | 0.48 s | 2.06 s | Good morning. How did you sleep? | 0.00 | 0.385 |

## `s01` · voice `dave`

> Good morning, how did you sleep?

| Configuration | File | First audio | Total | Audio | Whisper heard | WER | Speaker sim |
|---|---|---|---|---|---|---|---|
| `upstream_q4_onnx` | [s01_dave_1_upstream_q4_onnx.wav](s01_dave_1_upstream_q4_onnx.wav) | 0.434 s | 0.91 s | 2.14 s | Good morning. How did you sleep? | 0.00 | 0.390 |
| `upstream_q4_onnx_metal` | [s01_dave_2_upstream_q4_onnx_metal.wav](s01_dave_2_upstream_q4_onnx_metal.wav) | 0.336 s | 0.70 s | 1.76 s | Good morning. How did you sleep? | 0.00 | 0.492 |
| `fast_exact` | [s01_dave_3_fast_exact.wav](s01_dave_3_fast_exact.wav) | 0.341 s | 0.61 s | 2.12 s | Good morning. How did you sleep? | 0.00 | 0.389 |
| `fast_best` | [s01_dave_4_fast_best.wav](s01_dave_4_fast_best.wav) | 0.228 s | 0.62 s | 2.54 s | Good morning. How did you sleep? | 0.00 | 0.342 |
| `fast_best_gpucodec` | [s01_dave_5_fast_best_gpucodec.wav](s01_dave_5_fast_best_gpucodec.wav) | 0.178 s | 0.37 s | 1.94 s | Good morning. How did you sleep? | 0.00 | 0.410 |

## `m01` · voice `jo`

> The train to Manchester has been delayed by about twenty minutes because of a signal fault.

| Configuration | File | First audio | Total | Audio | Whisper heard | WER | Speaker sim |
|---|---|---|---|---|---|---|---|
| `upstream_q4_onnx` | [m01_jo_1_upstream_q4_onnx.wav](m01_jo_1_upstream_q4_onnx.wav) | 0.656 s | 1.76 s | 4.16 s | The train to Manchester has been delayed by about 20 minutes because of a signal fall. | 0.06 | 0.621 |
| `upstream_q4_onnx_metal` | [m01_jo_2_upstream_q4_onnx_metal.wav](m01_jo_2_upstream_q4_onnx_metal.wav) | 0.468 s | 1.61 s | 4.40 s | The train to Manchester has been delayed by about 20 minutes because of a signal fall. | 0.06 | 0.682 |
| `fast_exact` | [m01_jo_3_fast_exact.wav](m01_jo_3_fast_exact.wav) | 0.566 s | 1.22 s | 4.14 s | The train to Manchester has been delayed by about 20 minutes because of a signal fall. | 0.06 | 0.619 |
| `fast_best` | [m01_jo_4_fast_best.wav](m01_jo_4_fast_best.wav) | 0.261 s | 1.02 s | 4.38 s | The train to Manchester has been delayed by about 20 minutes because of a signal fault. | 0.00 | 0.720 |
| `fast_best_gpucodec` | [m01_jo_5_fast_best_gpucodec.wav](m01_jo_5_fast_best_gpucodec.wav) | 0.221 s | 0.84 s | 4.54 s | The train to Manchester has been delayed by about 20 minutes because of a signal fault | 0.00 | 0.632 |

## `m01` · voice `dave`

> The train to Manchester has been delayed by about twenty minutes because of a signal fault.

| Configuration | File | First audio | Total | Audio | Whisper heard | WER | Speaker sim |
|---|---|---|---|---|---|---|---|
| `upstream_q4_onnx` | [m01_dave_1_upstream_q4_onnx.wav](m01_dave_1_upstream_q4_onnx.wav) | 0.437 s | 1.58 s | 4.54 s | The train to Manchester has been delayed by about 20 minutes because of a signal fault. | 0.00 | 0.553 |
| `upstream_q4_onnx_metal` | [m01_dave_2_upstream_q4_onnx_metal.wav](m01_dave_2_upstream_q4_onnx_metal.wav) | 0.363 s | 1.79 s | 5.44 s | The train to Manchester has been delayed by about 20 minutes because of a signal fault. | 0.00 | 0.478 |
| `fast_exact` | [m01_dave_3_fast_exact.wav](m01_dave_3_fast_exact.wav) | 0.355 s | 1.04 s | 4.52 s | The train to Manchester has been delayed by about 20 minutes because of a signal fault. | 0.00 | 0.551 |
| `fast_best` | [m01_dave_4_fast_best.wav](m01_dave_4_fast_best.wav) | 0.230 s | 0.89 s | 4.18 s | The train to Manchester has been delayed by about 20 minutes because of a signal fault. | 0.00 | 0.566 |
| `fast_best_gpucodec` | [m01_dave_5_fast_best_gpucodec.wav](m01_dave_5_fast_best_gpucodec.wav) | 0.199 s | 0.86 s | 5.32 s | The train to Manchester has been delayed by about 20 minutes because of a signal fault. | 0.00 | 0.548 |

## `l01` · voice `jo`

> Before you leave the house, make sure the windows are closed, the heating is turned down, and the back door is locked, because the forecast says the storm will arrive around midnight.

| Configuration | File | First audio | Total | Audio | Whisper heard | WER | Speaker sim |
|---|---|---|---|---|---|---|---|
| `upstream_q4_onnx` | [l01_jo_1_upstream_q4_onnx.wav](l01_jo_1_upstream_q4_onnx.wav) | 0.706 s | 3.27 s | 9.00 s | Before you leave the house, make sure the windows are closed, the heating is turned down, and the back door is locked because the forecast says the storm will arrive around midnight. | 0.00 | 0.683 |
| `upstream_q4_onnx_metal` | [l01_jo_2_upstream_q4_onnx_metal.wav](l01_jo_2_upstream_q4_onnx_metal.wav) | 0.469 s | 2.73 s | 8.08 s | Before you leave the house, make sure the windows are closed, the heating is turned down, and the back door is locked because the forecast says the storm will arrive around midnight. | 0.00 | 0.679 |
| `fast_exact` | [l01_jo_3_fast_exact.wav](l01_jo_3_fast_exact.wav) | 0.626 s | 2.17 s | 8.98 s | Before you leave the house, make sure the windows are closed, the heating is turned down, and the back door is locked because the forecast says the storm will arrive around midnight. | 0.00 | 0.681 |
| `fast_best` | [l01_jo_4_fast_best.wav](l01_jo_4_fast_best.wav) | 0.310 s | 1.95 s | 8.70 s | Before you leave the house, make sure the windows are closed, the heating is turned down, and the back door is locked because the forecast says the storm will arrive around midnight. | 0.00 | 0.693 |
| `fast_best_gpucodec` | [l01_jo_5_fast_best_gpucodec.wav](l01_jo_5_fast_best_gpucodec.wav) | 0.247 s | 1.54 s | 8.60 s | Before you leave the house, make sure the windows are closed, the heating is turned down, and the back door is locked because the forecast says the storm will arrive around midnight. | 0.00 | 0.686 |

## `l01` · voice `dave`

> Before you leave the house, make sure the windows are closed, the heating is turned down, and the back door is locked, because the forecast says the storm will arrive around midnight.

| Configuration | File | First audio | Total | Audio | Whisper heard | WER | Speaker sim |
|---|---|---|---|---|---|---|---|
| `upstream_q4_onnx` | [l01_dave_1_upstream_q4_onnx.wav](l01_dave_1_upstream_q4_onnx.wav) | 0.479 s | 2.96 s | 8.82 s | before you leave the house. Make sure the windows are closed, the heating is turned down and the back door is locked because the forecast says the storm will arrive around midnight. | 0.00 | 0.521 |
| `upstream_q4_onnx_metal` | [l01_dave_2_upstream_q4_onnx_metal.wav](l01_dave_2_upstream_q4_onnx_metal.wav) | 0.363 s | 2.84 s | 8.94 s | Before you leave the house, make sure the windows are closed, the heating is turned down, and the back door is locked because the forecast says the storm will arrive around midnight. | 0.00 | 0.591 |
| `fast_exact` | [l01_dave_3_fast_exact.wav](l01_dave_3_fast_exact.wav) | 0.480 s | 1.96 s | 8.80 s | before you leave the house. Make sure the windows are closed, the heating is turned down and the back door is locked because the forecast says the storm will arrive around midnight. | 0.00 | 0.523 |
| `fast_best` | [l01_dave_4_fast_best.wav](l01_dave_4_fast_best.wav) | 0.256 s | 1.85 s | 9.26 s | Before you leave the house, make sure the windows are closed, the heating is turned down, and the back door is locked because the forecast says the storm will arrive around midnight. | 0.00 | 0.601 |
| `fast_best_gpucodec` | [l01_dave_5_fast_best_gpucodec.wav](l01_dave_5_fast_best_gpucodec.wav) | 0.188 s | 1.55 s | 9.78 s | Before you leave the house, make sure the windows are closed, the heating is turned down, and the back door is locked because the forecast says the storm will arrive around midnight. | 0.00 | 0.593 |
