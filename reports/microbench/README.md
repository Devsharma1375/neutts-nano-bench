# Microbenchmarks and diagnostics

Focused measurements behind findings in `DESIGN.md` that the suites don't cover. Same machine
as the suite runs (MacBook Air M4, 16 GB, macOS 27.0.1, AC power), 2026-10-31 to 2026-11-01.
Scripts are in `scripts/microbench/` (run with `.venv/bin/python scripts/microbench/<name>.py`);
the outputs below are copied verbatim from those runs. Unlike the suites they are single runs,
so treat small differences as noise.

"Our Q4_0" is a plain conversion of `neuphonic/neutts-nano` made with the llama.cpp version
vendored in llama-cpp-python 0.3.36 (`convert_hf_to_gguf.py` + `llama-quantize Q4_0`);
"official Q4" is `neuphonic/neutts-nano-q4-gguf` (imatrix-quantized, three Q4_1 tensors).

## 1. Thermal behaviour of the fanless M4 Air (`thermal.py`, our Q4_0)

A 256-token llama.cpp prefill on 4 threads, repeated back to back for 90 s, then probed every
10 s while idle:

```
t=  0.8s  prefill   3289 tok/s
t=  1.9s  prefill   2246 tok/s
t=  3.4s  prefill   1726 tok/s
t=  5.2s  prefill   1447 tok/s
t= 10.2s  prefill   1600 tok/s
t= 30.2s  prefill   1605 tok/s   (flat at ~1,600 until t = 91 s)
idle  10.0s  prefill   2537 tok/s
idle  20.1s  prefill   2769 tok/s
idle  40.3s  prefill   2890 tok/s
idle 121.2s  prefill   2580 tok/s
```

The same file, measured with llama-bench twice about a minute apart: pp800 2,824 → 1,215 tok/s,
tg128 348 → 215 tok/s.

## 2. Threads (`prefill_threads.py`, `decode_threads.py`, official Q4)

15 s idle before every measurement, variant order rotated each cycle, 4 cycles.

Prompt reading, a real 987-token NeuTTS prompt (voice jo), token input (full vocabulary) vs
embedding input (speech vocabulary):

```
tokens n_threads_batch= 4: median  492.4 ms ( 2004 tok/s)
embd   n_threads_batch= 4: median  491.1 ms ( 2010 tok/s)
tokens n_threads_batch= 6: median  477.4 ms ( 2067 tok/s)
embd   n_threads_batch= 6: median  476.8 ms ( 2070 tok/s)
tokens n_threads_batch=10: median  468.4 ms ( 2107 tok/s)
embd   n_threads_batch=10: median  433.6 ms ( 2276 tok/s)
```

Generation, speech-vocabulary model, 900 tokens of context, 100 tokens:

```
n_threads= 3: 2.64 ms/token (379 tok/s)
n_threads= 4: 2.50 ms/token (401 tok/s)
n_threads= 6: 2.48 ms/token (402 tok/s)
n_threads=10: 3.12 ms/token (320 tok/s)
```

## 3. Backbone: CPU vs Metal, full vs speech vocabulary (llama-bench, our Q4_0, cool)

```
| model                    | GPU layers | threads | fa | test  |              t/s |
| full vocab (176 MiB)     | 0 (CPU)  |       4 |  1 | pp800 |  2794.76 ± 4.28  |
| full vocab               | 0 (CPU)  |       4 |  1 | tg64  |   347.96 ± 1.79  |
| speech vocab (101 MiB)   | 0 (CPU)  |       4 |  1 | pp800 |  2788.25 ± 5.67  |
| speech vocab             | 0 (CPU)  |       4 |  1 | tg64  |   555.67 ± 1.39  |
| full vocab               | Metal 99 |       4 |  1 | pp800 |  8715.01 ± 41.99 |
| full vocab               | Metal 99 |       4 |  1 | tg64  |   286.88 ± 5.93  |
| speech vocab             | Metal 99 |       4 |  1 | pp800 |  8713.29 ± 50.43 |
| speech vocab             | Metal 99 |       4 |  1 | tg64  |   382.63 ± 16.69 |
```

(This speech-vocabulary file had 65,537 rows; the final 65,544-row file uses llama.cpp's repacked
kernel as well.) `op_offload` on/off with 0 GPU layers: pp800 2,761 vs 2,678 tok/s, tg 347 vs 348.

## 4. GPU prefill with KV hand-over (`hybrid.py`, `variants.py`)

Prototype, 800 random speech tokens, speech-vocabulary model, flash attention on:

```
cpu prefill 800 tok: 287.7 ms
gpu prefill 800 tok: 92.1 ms
state copy 14.8 MB: 36.4 ms
cpu decode @800 ctx: 2.80 ms/token
gpu decode @800 ctx: 3.41 ms/token
```

In the engine (`variants.py`, our Q4_0, 12 requests: 2 voices × 2 sentences × 3 seeds):

```
full_cpu         prefill  396.9 ms | first token  399.9 ms | decode 3.80 ms/tok | same tokens as full_cpu 12/12
speech_cpu       prefill  379.8 ms | first token  384.9 ms | decode 2.72 ms/tok | same tokens as full_cpu 12/12
speech_cpu_fa    prefill  298.4 ms | first token  303.0 ms | decode 2.78 ms/tok | same tokens as full_cpu 0/12
speech_gpuPF_fa  prefill   87.1 ms | first token   89.4 ms | decode 2.87 ms/tok | same tokens as full_cpu 0/12
speech_gpuPF     prefill  108.2 ms | first token  108.7 ms | decode 2.83 ms/tok | same tokens as full_cpu 0/12
```

## 5. Codec (`codec_ep.py`, `codec_torch.py`, `codec_align.py`, `mps_shapes.py`, `pad.py`)

NeuCodec ONNX decoder output length: N codes → (N − 1) × 480 samples (1 → 0, 126 → 60,000).

Real speech codes (voice jo), 82-code streaming window and 250-code clip:

```
cpu4 (ONNX, 4 threads)  82-tok window   39.6 ms | 250-tok  103.1 ms
cpu1 (ONNX, 1 thread)   82-tok window   55.6 ms | 250-tok  136.9 ms
cpu4_int8               82-tok window   31.2 ms (SNR vs FP32 26.9 dB) | 250-tok 82.9 ms (27.1 dB)
cpu1_int8               82-tok window   35.3 ms | 250-tok  89.1 ms
coreml (MLProgram, ALL / CPUAndGPU / CPUAndNeuralEngine): load failed
   "axis 3 is not in valid range [-3,2]"
coreml NeuralNetwork, dynamic shapes: window 74.2 ms
coreml MLProgram, static shapes:      window 49.6 ms (SNR 219.8 dB: effectively ran on the CPU)
torch NeuCodec, CPU:   window 49.9 ms, 250-tok 97.7 ms
torch NeuCodec, Metal: window 24.8 ms, 250-tok 56.9 ms
```

PyTorch vs ONNX decoder on the same 250 codes: lengths 120,000 vs 119,520 samples; best
alignment at a 240-sample lag with SNR 94.8 dB. PyTorch CPU vs Metal: 108.8 dB.

Metal recompiles per input length (ms; ONNX CPU 4 threads for comparison):

```
n=70:  mps   54.2 | onnx 44.0      n=70 again:  mps 28.9
n=150: mps  128.3 | onnx 69.4      n=150 again: mps 40.5
n=300: mps  389.6 | onnx 125.8     n=300 again: mps 67.5
```

Padding the codec input with extra codes changes the kept audio everywhere (ONNX decoder,
SNR of the unpadded region vs an unpadded decode): from −6.0 dB (137 codes + 60 reflected) to
27.9 dB (251 codes + 8 reflected).

## 6. Token identity diagnostics (`vocab_equiv.py`, `diverge.py`, `ties.py`)

`vocab_equiv.py`: full vs speech vocabulary, with and without prefix reuse, 2 voices × 3
sentences × 3 seeds, CPU: identical tokens in 18/18 (our Q4_0 and official Q4); on Metal also
18/18 among the Metal variants.

The one default-suite request where `fast_exact` differed from upstream (l05, voice dave, seed 0):

```
upstream      439 tokens, first divergence at step none
full          439 tokens, first divergence at step none
full_reuse    439 tokens, first divergence at step none
speech        426 tokens, first divergence at step 282
speech_reuse  426 tokens, first divergence at step 282
steps where a token outside the kept rows is in the full top-50: 0 of 440
step 282: 1 tied pairs inside the top-50, tie at the top-50 boundary: False
steps with exact logit ties in or at the edge of the top-50: 1 of 439
```

## 7. Whisper on near-identical audio (ladder, sentence m11, voice jo, streaming)

Steps L06 and L07 produce the same 167 tokens; their audio differs only in watermark placement
(41.0 dB SNR between them). Whisper `small.en` transcribed L06 (and L11) perfectly and stopped
after "The temperature will drop t" on L07–L10.
