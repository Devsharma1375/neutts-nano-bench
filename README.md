# ttsbench: on-device benchmarking and optimisation for NeuTTS-Nano

A benchmarking framework that measures how NeuTTS-Nano performs on a real device, with
numbers you can trust, and an optimised inference engine measured with it.

**Results on an M4 MacBook Air** (medians, paired 95% CIs, against NeuTTS-Nano as Neuphonic
documents it for low latency; full report in `reports/m4-default/report.md`, details in
`DESIGN.md`):

| | Upstream NeuTTS | Optimised | Change |
|---|---|---|---|
| Streaming: time to first audio | 0.602 s | **0.202 s** | **0.36×** |
| Batch: real-time factor (`fast_best`; WER gate inconclusive) | 0.325 | **0.198** | **0.60×** |
| Same speech tokens as upstream (`fast_exact`): stream TTFA / batch RTF | 0.602 s / 0.325 | 0.479 s / 0.235 | 0.83× / 0.73× |
| Worst-case memory during a request (incl. GPU) | 4.5 GiB | 2.9–3.0 GiB | −33% to −35% |

Word error rate, speaker similarity, watermark detection and streaming fidelity are gated
against upstream for every optimised configuration (section 8 of `DESIGN.md`). To hear it,
`samples/` has the same sentences from upstream and every optimised configuration.

- **Measures the whole pipeline**, not just the language model: phonemizer → prompt →
  prefill → token generation → codec → watermark, timed stage by stage. Batch and
  streaming modes are both covered.
- **Speed, quality and footprint together.**
  - Speed: time to first audio, real-time factor, tokens/s, streaming underruns.
  - Quality: word error rate via Whisper (corpus-level), speaker similarity via ECAPA,
    watermark detection, and how closely streamed audio matches a one-pass decode.
  - Footprint: memory (on macOS the physical footprint, which includes GPU memory), load
    time, cold start.
- **Trustworthy numbers.**
  - Each configuration runs in its own fresh process, after warm-up on sentences that are
    never measured, with fixed seeds.
  - Every configuration gets the same requests, so comparisons are paired.
  - Results are medians with bootstrap 95% confidence intervals, and a noise floor stops
    tiny differences being claimed as wins.
  - Requests are paced like a voice app and every configuration starts after a cool-down,
    because a fanless laptop halves its speed within seconds of continuous load. A separate
    suite measures that sustained case on purpose.
  - Hardware and software are fingerprinted automatically.
- **Quality gates.** An optimisation only counts as a win if it is faster beyond the noise
  floor *and* every quality gate passes: word error rate no more than +0.02 (corpus-level,
  paired bootstrap), speaker similarity no more than −0.02, stream fidelity no more than
  +1.5 dB worse, and the watermark detected on every output.
- **A verifiably equivalent engine.** In its output-preserving configuration (`fast_exact`)
  the optimised engine generates exactly the same speech tokens as upstream NeuTTS for the
  same seed; `make verify` checks it on the real model (90 checks). Its faster settings are
  numerically different and are accepted only through the quality gates.

**Hardware used for the submitted results:** MacBook Air (Mac16,12), Apple M4 (4 performance
+ 6 efficiency cores, 10-core GPU), 16 GB, macOS 27.0.1, fanless. Every `results/*/env.json`
records the full fingerprint.

## Quick start

```bash
# 1. NeuTTS weights are gated. On huggingface.co accept the licence of each of:
#    neuphonic/neutts-nano, neutts-nano-q4-gguf, neutts-nano-q8-gguf,
#    neucodec, neucodec-onnx-decoder, neucodec-onnx-decoder-int8
# 2. Create the environment (Python 3.10-3.13; on macOS llama.cpp is built with Metal, ~1 min)
make setup
# 3. Log in once with a Hugging Face read token
.venv/bin/hf auth login
# 4. The one command: preflight checks, benchmark, quality scoring, report
make bench
```

The report lands in `results/<timestamp>-default/report.md`, with charts beside it.

Without `make`, the equivalent is:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -c constraints.txt torch torchaudio            # Linux: add --index-url https://download.pytorch.org/whl/cpu
# macOS (Metal + native CPU kernels):
CMAKE_ARGS="-DGGML_METAL=ON -DGGML_NATIVE=ON" pip install --no-binary llama-cpp-python llama_cpp_python==0.3.36
# Linux: pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu llama_cpp_python==0.3.36
pip install -c constraints.txt -e ".[all]"
python -m ttsbench doctor
python -m ttsbench all suites/default.yaml
```

## Commands

| Command | What it does |
|---|---|
| `make bench` | Setup, preflight checks, default suite, quality scoring, report (~1.5 h on an M4 Air) |
| `make quick` | Default suite, scoring and report on a small slice (~5 minutes, no preflight): check your setup works |
| `make verify` | Proves the optimised engine produces the same tokens and audio as upstream on the real model |
| `make ladder` | The optimisation ladder: one change per step, from NeuTTS out of the box to the fastest configuration (~2 h) |
| `python -m ttsbench all suites/sustained.yaml` | Long sentences back to back: the throttled, long-form case |
| `make threads` | Generation and prompt-reading thread sweeps for your machine |
| `make native-llama` | Rebuilds llama.cpp natively for this CPU (keeps Metal on macOS); compare runs with `ttsbench compare` |
| `make test` / `make smoke` | Tests the framework itself (mock engine, no weights needed) |
| `python -m ttsbench doctor` | Checks imports, espeak, hardware, power state and Hugging Face access, with fixes |
| `python -m ttsbench env` | Prints the hardware and software fingerprint |
| `python -m ttsbench compare A B --config NAME` | Compares one configuration across two runs (e.g. two builds) |
| `python scripts/progress.py RUN:SUITE … --watch` | Live progress bar and ETA for running suites |
| `python scripts/make_samples.py RUN samples` | Copies a small listening set (fixed sentences, every configuration) out of a run |

Useful flags for `run` and `all`: `--quick`, `--only cfg1,cfg2`, `--run-id NAME`,
`--asr-model small.en`, `--skip-quality`.

## What you get

```
results/<run>/
  report.md          human-readable report (start here)
  summary.json       every statistic, machine-readable
  charts/            TTFA, RTF, stage breakdown, speed vs quality
  env.json           hardware/software fingerprint
  run.json           suite, timings, per-config status
  quality.jsonl      per-request quality scores (with what Whisper heard)
  <config>/round_<r>/records.jsonl   one line per request: stage times, chunk arrival times, tokens…
  <config>/round_<r>/audio/*.wav     the generated audio, for listening
```

The submitted runs are in `reports/` (reports, summaries, charts; audio omitted for size).
`samples/` holds 30 clips from the default run for listening: the first sentence of each length
bucket in both voices, from every configuration, with each clip's timing and transcript.

## How to read the report

- **TTFA** (time to first audio) is what a listener feels in a voice app: the silence
  before speech starts. It's the primary metric in streaming mode.
- **RTF** (real-time factor) = processing time ÷ audio duration. Below 1 means faster than
  playback. It's the primary metric in batch mode.
- `0.62× [0.60–0.64]` in the comparison table means each request took a median 62% of the
  baseline's time, and the 95% confidence interval is 60–64%.
- A **win** means faster beyond the noise floor *and* every quality gate passed. "Within
  noise" means no measurable difference; "quality needs more data" means faster, but the
  quality confidence interval still overlaps the gate.
- "Largest word-error changes" lists the sentences to listen to first.

## Extending

- **New configuration:** add an entry to a suite YAML. Any engine setting can be
  overridden per config; unknown setting names are rejected.
- **New backbone:** point `backbone:` at a Hugging Face GGUF repo or a local `.gguf` path
  (the reference engine also accepts a transformers checkpoint). `vocab: speech` builds the
  speech-only copy automatically.
- **New engine:** subclass `ttsbench.engines.base.Engine`, implement `load`, `synthesize`,
  `stream`, register it in `ttsbench/engines/__init__.py` (`create_engine`) and add its
  name and settings to `ENGINES` / `ENGINE_SETTINGS` in `ttsbench/config.py`.
- **New metric:** per-request metrics live in `worker._record`; model-based scores in
  `quality/scorers.py`; aggregation and gates in `report.py`.

See `DESIGN.md` for every decision, its reasoning and the measured results.
