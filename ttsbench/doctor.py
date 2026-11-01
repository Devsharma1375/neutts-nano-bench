
"""``ttsbench doctor``: catch every known setup problem before a long run, with a fix."""
from __future__ import annotations

import importlib
import sys

from . import env as envmod

GATED = [
    "neuphonic/neutts-nano", "neuphonic/neutts-nano-q4-gguf", "neuphonic/neutts-nano-q8-gguf",
    "neuphonic/neucodec", "neuphonic/neucodec-onnx-decoder", "neuphonic/neucodec-onnx-decoder-int8",
]
OK, WARN, FAIL = "ok", "warn", "FAIL"


def _check_imports(results: list) -> None:
    try:
        importlib.import_module("neutts")
        results.append((OK, "neutts imports", ""))
    except ModuleNotFoundError as e:
        if "torchao" in str(e):
            results.append((FAIL, "neutts imports", "torchtune needs an older torchao: pip install 'torchao==0.15.0'"))
        else:
            results.append((FAIL, "neutts imports", f"{e}. Fix: pip install -c constraints.txt -e '.[all]'"))
    except Exception as e:  # noqa: BLE001
        results.append((FAIL, "neutts imports", f"{type(e).__name__}: {e}"))
    for mod, why in (("llama_cpp", "GGUF backbones"), ("gguf", "speech-only vocabulary"), ("onnxruntime", "ONNX codec"),
                     ("faster_whisper", "ASR quality scoring"),
                     ("speechbrain", "speaker similarity"), ("jiwer", "word error rate"), ("matplotlib", "report charts")):
        try:
            importlib.import_module(mod)
            results.append((OK, f"{mod} ({why})", ""))
        except Exception as e:  # noqa: BLE001
            results.append((FAIL, f"{mod} ({why})", f"{e}. Fix: pip install -c constraints.txt -e '.[all]'"))


def _check_espeak(results: list) -> None:
    try:
        from .engines.components import Phonemizer

        out = Phonemizer("en-us").to_phones("hello world")
        results.append((OK, "espeak phonemizer", out))
    except Exception as e:  # noqa: BLE001
        results.append((FAIL, "espeak phonemizer", f"{e}. Fix: install neutts from PyPI (bundles espeak-ng) or `brew/apt install espeak-ng`"))


def _check_hf(results: list) -> None:
    try:
        from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

        user = HfApi().whoami().get("name")
        results.append((OK, "Hugging Face login", user))
    except Exception:  # noqa: BLE001
        results.append((FAIL, "Hugging Face login",
                        "NeuTTS weights are gated. Run `hf auth login` (token from https://huggingface.co/settings/tokens)"))
        return
    for repo in GATED:
        try:
            files = HfApi().list_repo_files(repo)
            target = next((f for f in files if f.endswith((".gguf", ".onnx", ".safetensors"))), "config.json")
            get_hf_file_metadata(hf_hub_url(repo, target))
            results.append((OK, f"access {repo}", ""))
        except Exception as e:  # noqa: BLE001
            name = type(e).__name__
            fix = f"accept the license at https://huggingface.co/{repo}" if "Gated" in name or "401" in str(e) or "403" in str(e) else str(e)
            results.append((FAIL, f"access {repo}", fix))


def _check_hardware(results: list) -> None:
    e = envmod.capture()
    results.append((OK, "hardware", envmod.summary_line(e)))
    build = e.get("llama_cpp_build") or ""
    flags = set(e.get("cpu_flags") or [])
    if "avx512f" in flags and "AVX512 = 1" not in build:
        results.append((WARN, "llama.cpp build", "CPU has AVX-512 but this llama.cpp wheel doesn't use it. "
                              "Try scripts/build_llama_native.sh (benchmark it as its own experiment)"))
    power = e.get("power") or {}
    if power.get("on_ac_power") is False:
        results.append((WARN, "power", "on battery: plug in for stable numbers"))
    if power.get("low_power_mode"):
        results.append((WARN, "power", "macOS Low Power Mode is on: turn it off"))
    if power.get("cpu_governor") not in (None, "performance"):
        results.append((WARN, "cpu governor", f"{power['cpu_governor']}: `performance` gives steadier numbers"))
    if e["ram_available_gb"] < 3:
        results.append((WARN, "memory", f"only {e['ram_available_gb']} GB free: close other apps"))
    if e.get("load_avg") and e["load_avg"][0] > 0.5 * (e["logical_cores"] or 1):
        results.append((WARN, "background load", f"load average {e['load_avg'][0]:.1f}: close other apps"))


def run_doctor(check_hf: bool = True) -> bool:
    results: list = []
    results.append((OK if sys.version_info[:2] >= (3, 10) else FAIL, "python", sys.version.split()[0]))
    _check_imports(results)
    _check_espeak(results)
    _check_hardware(results)
    if check_hf:
        _check_hf(results)
    width = max(len(r[1]) for r in results)
    for status, name, detail in results:
        print(f"  [{status:>4}] {name:<{width}}  {detail}")
    failed = [r for r in results if r[0] == FAIL]
    print(f"\n{'Ready to benchmark.' if not failed else f'{len(failed)} problem(s) to fix first.'}")
    return not failed
