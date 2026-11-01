# One command for reviewers:   make bench
# Everything else is optional.
# Python 3.10-3.13; macOS's /usr/bin/python3 is 3.9, so prefer a newer one when present.
PY      ?= $(shell for p in python3.12 python3.11 python3.13 python3.10 python3; do command -v $$p >/dev/null 2>&1 && { echo $$p; break; }; done)
VENV    ?= .venv
BIN     := $(VENV)/bin
SUITE   ?= suites/default.yaml
UNAME_S := $(shell uname -s)
TORCH_INDEX := $(if $(filter Linux,$(UNAME_S)),--index-url https://download.pytorch.org/whl/cpu,)
# llama.cpp: prebuilt CPU wheel on Linux; built from source on macOS so it has Metal and the
# host CPU's kernels (the Apple-silicon results use Metal for GPU prefill). ~1 minute.
LLAMA_VERSION := $(shell grep -i '^llama_cpp_python==' constraints.txt | cut -d= -f3)
ifeq ($(UNAME_S),Darwin)
LLAMA_INSTALL := CMAKE_ARGS="-DGGML_METAL=ON -DGGML_NATIVE=ON" $(BIN)/pip install -q --no-binary llama-cpp-python "llama_cpp_python==$(LLAMA_VERSION)"
else
LLAMA_INSTALL := $(BIN)/pip install -q --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu "llama_cpp_python==$(LLAMA_VERSION)"
endif

.PHONY: bench setup doctor quick ladder threads smoke verify test native-llama report clean

bench: setup doctor           ## setup + checks + default suite + quality + report
	$(BIN)/python -m ttsbench all $(SUITE)

setup: $(BIN)/ttsbench        ## create .venv with pinned, verified versions
$(BIN)/ttsbench: pyproject.toml constraints.txt
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -c constraints.txt torch torchaudio $(TORCH_INDEX)
	$(LLAMA_INSTALL)
	$(BIN)/pip install -q -c constraints.txt -e ".[all]"
	@touch $@

doctor: setup                 ## preflight: imports, espeak, hardware, Hugging Face access
	$(BIN)/python -m ttsbench doctor

quick: setup                  ## ~5 minute smoke run of the default suite on the real model
	$(BIN)/python -m ttsbench all $(SUITE) --quick

ladder: setup doctor          ## the optimisation ladder, one change per step (~2 h on an M4)
	$(BIN)/python -m ttsbench all suites/ladder.yaml

threads: setup                ## thread-count sweep
	$(BIN)/python -m ttsbench all suites/threads.yaml

verify: setup                 ## prove the optimised engine matches upstream on the real model
	$(BIN)/python -m ttsbench verify

smoke: setup                  ## framework self-test with the mock engine (no weights needed)
	$(BIN)/python -m ttsbench all suites/smoke.yaml --run-id smoke-$$(date +%s)

test: setup                   ## unit + end-to-end tests (no weights needed)
	$(BIN)/python -m pytest

native-llama: setup           ## rebuild llama-cpp-python for this exact CPU (an experiment)
	VENV=$(VENV) bash scripts/build_llama_native.sh

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__
