#!/usr/bin/env bash
# Rebuild llama-cpp-python from source for THIS machine's CPU, as one benchmark experiment.
#
# Why: the prebuilt CPU wheel targets a common baseline (AVX2). CPUs with AVX-512 / VNNI
# (recent Intel, AMD Zen 4/5) or Apple silicon (Accelerate) can run faster kernels.
# Whether it helps NeuTTS-Nano is an empirical question, so the workflow is:
#
#   1. run the suite with the prebuilt wheel           -> results/A
#   2. bash scripts/build_llama_native.sh
#   3. run the same suite again                        -> results/B
#   4. python -m ttsbench compare results/A results/B --config fast_q4_onnx
#
# The llama.cpp build string is recorded in every env.json, so the two runs are labelled.
set -euo pipefail
VENV="${VENV:-.venv}"
PIP="$VENV/bin/pip"
VERSION="$(grep -i '^llama_cpp_python==' constraints.txt | cut -d= -f3)"

case "$(uname -s)" in
  Darwin) export CMAKE_ARGS="-DGGML_NATIVE=ON -DGGML_METAL=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple" ;;  # keep Metal: GPU configs need it
  Linux)
    if ldconfig -p 2>/dev/null | grep -q libopenblas; then
      export CMAKE_ARGS="-DGGML_NATIVE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
    else
      echo "OpenBLAS not found (optional: sudo apt install libopenblas-dev); building with native CPU kernels only"
      export CMAKE_ARGS="-DGGML_NATIVE=ON"
    fi ;;
  *) export CMAKE_ARGS="-DGGML_NATIVE=ON" ;;
esac

echo "Building llama-cpp-python==$VERSION with CMAKE_ARGS=$CMAKE_ARGS (takes a few minutes)"
"$PIP" install --force-reinstall --no-cache-dir --no-binary llama-cpp-python "llama-cpp-python==$VERSION"
"$VENV/bin/python" -c "import llama_cpp; print(llama_cpp.llama_print_system_info().decode())"
