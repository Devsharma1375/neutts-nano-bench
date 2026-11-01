"""Everything needed to reproduce or question a number, captured automatically."""
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import subprocess
import sys
from pathlib import Path

PACKAGES = (
    "neutts", "neucodec", "llama-cpp-python", "onnxruntime", "torch", "torchao", "torchtune",
    "transformers", "numpy", "resemble-perth", "phonemizer", "faster-whisper", "speechbrain", "jiwer",
)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def cpu_model() -> str:
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def cpu_flags() -> list[str]:
    wanted = {"avx2", "avx512f", "avx512_vnni", "avx512_bf16", "amx_tile", "amx_bf16", "f16c", "fma"}
    if sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                return sorted(wanted & set(line.split(":", 1)[1].split()))
    if sys.platform == "darwin":
        return sorted(f for f in ("FEAT_DotProd", "FEAT_I8MM", "FEAT_BF16") if _run(["sysctl", "-n", f"hw.optional.arm.{f}"]) == "1")
    return []


def power_state() -> dict:
    import psutil

    out: dict = {}
    try:
        battery = psutil.sensors_battery()
        if battery is not None:
            out = {"on_ac_power": bool(battery.power_plugged), "battery_percent": battery.percent}
    except (AttributeError, NotImplementedError):
        pass
    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov.exists():
        out["cpu_governor"] = gov.read_text().strip()
    if sys.platform == "darwin":
        out["low_power_mode"] = "lowpowermode 1" in _run(["pmset", "-g"]).replace("\t", " ")
    return out


def git_state(path: Path) -> dict:
    commit = _run(["git", "-C", str(path), "rev-parse", "HEAD"])
    dirty = bool(_run(["git", "-C", str(path), "status", "--porcelain"]))
    return {"commit": commit or None, "dirty": dirty if commit else None}


def capture(repo_root: Path | None = None) -> dict:
    import psutil

    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    llama_info = None
    try:
        import llama_cpp

        llama_info = llama_cpp.llama_print_system_info().decode().strip()
    except Exception:  # noqa: BLE001 - optional dependency
        pass
    ort_info = None
    try:
        import onnxruntime as ort

        ort_info = {"version": ort.__version__, "providers": ort.get_available_providers()}
    except Exception:  # noqa: BLE001
        pass
    perf_cores = _run(["sysctl", "-n", "hw.perflevel0.physicalcpu"]) if sys.platform == "darwin" else ""
    vm = psutil.virtual_memory()
    return {
        "cpu": cpu_model(),
        "cpu_flags": cpu_flags(),
        "machine": platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "performance_cores": int(perf_cores) if perf_cores.isdigit() else None,
        "ram_gb": round(vm.total / 2**30, 1),
        "ram_available_gb": round(vm.available / 2**30, 1),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "packages": versions,
        "llama_cpp_build": llama_info,
        "onnxruntime": ort_info,
        "power": power_state(),
        "load_avg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "git": git_state(repo_root or Path.cwd()),
        "env_vars": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HOME", "TTSBENCH_CACHE") if k in os.environ},
    }


def summary_line(env: dict) -> str:
    cores = f"{env['physical_cores']}C/{env['logical_cores']}T"
    if env.get("performance_cores"):
        cores += f" ({env['performance_cores']} perf)"
    return f"{env['cpu']} · {cores} · {env['ram_gb']} GB RAM · {env['os']} · Python {env['python']}"
