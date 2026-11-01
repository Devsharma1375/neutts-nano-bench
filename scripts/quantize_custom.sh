#!/usr/bin/env bash
# Build custom GGUF quantizations of NeuTTS-Nano (an optimisation experiment).
#
# NeuTTS-Nano's hidden size is 576, which is not a multiple of 256, so llama.cpp's K-quants
# (Q4_K_M, Q5_K_M, Q6_K ...) cannot be used: llama-quantize silently falls back to Q8_0/Q5_x
# for every tensor. Only 32-element block types apply: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL.
# Neuphonic's own Q4_0 file already uses an importance matrix (see its quantize.imatrix.*
# metadata). The interesting knob left is the tied embedding/output matrix, which
# llama-quantize keeps at Q8_0 and which dominates per-token cost (see engines/vocab.py).
#
# Uses the llama.cpp sources vendored in the pinned llama-cpp-python, so converter, quantizer
# and runtime agree on the file format.
# Requires: Hugging Face access to neuphonic/neutts-nano (gated), cmake, a C++ compiler.
set -euo pipefail
OUT="${OUT:-models}"
VENV="${VENV:-.venv}"
VERSION="$(grep -i '^llama_cpp_python==' constraints.txt | cut -d= -f3)"
SRC_DIR="build/llama_cpp_python-$VERSION/vendor/llama.cpp"
mkdir -p "$OUT" build

if [ ! -d "$SRC_DIR" ]; then
  "$VENV/bin/pip" download --no-binary :all: --no-deps "llama_cpp_python==$VERSION" -d build/sdist -q
  tar xzf "build/sdist/llama_cpp_python-$VERSION.tar.gz" -C build
fi
cmake -S "$SRC_DIR" -B "$SRC_DIR/build" -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF >/dev/null
cmake --build "$SRC_DIR/build" -j --target llama-quantize llama-imatrix llama-bench >/dev/null
BIN="$SRC_DIR/build/bin"

HF="$("$VENV/bin/python" -c 'from huggingface_hub import snapshot_download; print(snapshot_download("neuphonic/neutts-nano"))')"
[ -f "$OUT/neutts-nano-F16.gguf" ] || "$VENV/bin/python" "$SRC_DIR/convert_hf_to_gguf.py" "$HF" --outtype f16 --outfile "$OUT/neutts-nano-F16.gguf"

# Calibration data for the importance matrix: real NeuTTS prompts (phonemes + speech tokens),
# written with special tokens so llama-imatrix sees what the model sees in production.
"$VENV/bin/python" - "$OUT/calibration.txt" << 'PY'
import sys
from pathlib import Path
from ttsbench.corpus import load_corpus
from ttsbench.engines.base import prepare_voices, load_voice
from ttsbench.engines.components import CACHE_DIR, Phonemizer
voices = ["jo", "dave", "emily", "paul", "sophie", "steven"]
prepare_voices(voices, CACHE_DIR / "voices")
ph = Phonemizer("en-us")
lines = []
for s in load_corpus(Path("corpus/sentences.jsonl")):
    for v in voices:
        voice = load_voice(v, CACHE_DIR / "voices")
        codes = "".join(f"<|speech_{c}|>" for c in voice.codes)
        lines.append(f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ph.to_phones(voice.text)} "
                     f"{ph.to_phones(s.text)}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes}")
Path(sys.argv[1]).write_text("\n".join(lines))
print(f"wrote {len(lines)} calibration prompts")
PY
[ -f "$OUT/imatrix.gguf" ] || "$BIN/llama-imatrix" -m "$OUT/neutts-nano-F16.gguf" -f "$OUT/calibration.txt" --parse-special -c 2048 -o "$OUT/imatrix.gguf"

# Q4_0 body with the tied embedding/output matrix also at 4 bits (llama-quantize keeps it at
# Q8_0 by default), and IQ4_NL as an alternative 4-bit codebook.
"$BIN/llama-quantize" --imatrix "$OUT/imatrix.gguf" --token-embedding-type q4_0 --output-tensor-type q4_0 \
  "$OUT/neutts-nano-F16.gguf" "$OUT/neutts-nano-Q4_0-embq4.gguf" Q4_0
"$BIN/llama-quantize" --imatrix "$OUT/imatrix.gguf" "$OUT/neutts-nano-F16.gguf" "$OUT/neutts-nano-IQ4_NL.gguf" IQ4_NL
ls -lh "$OUT"/*.gguf
echo "Use these files as 'backbone:' paths in a suite (vocab: speech works with Q4_0/Q8_0 embeddings)."
