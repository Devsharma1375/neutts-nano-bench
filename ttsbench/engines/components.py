"""Building blocks for the optimised engine, each independently configurable and timed."""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import os
import platform
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

CACHE_DIR = Path(os.environ.get("TTSBENCH_CACHE", Path.home() / ".cache" / "ttsbench"))


# --------------------------------------------------------------------------- threads
def physical_cores() -> int:
    import psutil

    return psutil.cpu_count(logical=False) or os.cpu_count() or 1


def performance_cores() -> int:
    """Apple silicon: performance cores only (efficiency cores slow every step down)."""
    if sys.platform == "darwin":
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"], text=True))
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            pass
    return physical_cores()


def upstream_threads() -> tuple[int, int]:
    """What NeuTTS 1.4.1 uses for llama.cpp: (n_threads, n_threads_batch)."""
    if sys.platform == "darwin":
        n = performance_cores()
    else:
        n = max((os.cpu_count() or 2) // 2, 1)
    return n, os.cpu_count() or 1


def resolve_threads(value, default: int) -> int:
    if value in (None, "auto"):
        return max(int(default), 1)
    return max(int(value), 1)


# --------------------------------------------------------------------------- phonemizer
def _load_upstream_phonemizers():
    """Load ``neutts/phonemizers.py`` without executing ``neutts/__init__``.

    ``neutts/__init__`` imports torch, transformers and the PyTorch codec. The optimised
    engine needs none of them, so skipping it cuts start-up time and memory.
    """
    spec = importlib.util.find_spec("neutts")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("neutts is not installed")
    path = Path(next(iter(spec.submodule_search_locations))) / "phonemizers.py"
    mod_spec = importlib.util.spec_from_file_location("ttsbench_neutts_phonemizers", path)
    module = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(module)
    return module


class Phonemizer:
    """Same output as upstream ``NeuTTS._to_phones``.

    Only reference transcripts are cached (``reference_phones``): they repeat on every
    request for a voice. Input text is never cached, because repeated benchmark sentences
    would otherwise get a speed-up that real traffic never sees.
    """

    def __init__(self, language: str = "en-us", cache_size: int = 64) -> None:
        mod = _load_upstream_phonemizers()
        cls = mod.CUSTOM_PHONEMIZERS.get(language)
        self._backend = cls() if cls else mod.BasePhonemizer(language_code=language)
        self.reference_phones = functools.lru_cache(maxsize=cache_size)(self.to_phones)

    def to_phones(self, text: str) -> str:
        return " ".join(self._backend.phonemize([text])[0].split())


# --------------------------------------------------------------------------- model files
def resolve_model_file(repo_or_path: str, suffix: str, filename: str | None = None) -> Path:
    """A local file path, or a Hugging Face repo whose single ``*suffix`` file is used."""
    p = Path(repo_or_path).expanduser()
    if p.is_file():
        return p
    from huggingface_hub import HfApi, hf_hub_download

    if filename is None:
        files = [f for f in HfApi().list_repo_files(repo_or_path) if f.endswith(suffix)]
        if len(files) != 1:
            raise ValueError(f"{repo_or_path}: expected exactly one *{suffix} file, found {files}")
        filename = files[0]
    return Path(hf_hub_download(repo_or_path, filename))


# --------------------------------------------------------------------------- codec
class OnnxCodec:
    """NeuCodec ONNX decoder with explicit, benchmarkable session settings.

    The upstream wrapper uses ORT defaults: as many intra-op threads as cores and
    busy-wait spinning. That is fine when the codec runs alone, but in a pipelined
    stream it fights llama.cpp for the same cores. Everything here is configurable.
    """

    def __init__(
        self,
        repo_or_path: str,
        threads: int = 1,
        opt_level: str = "all",
        allow_spinning: bool = False,
        cache_optimized: bool = True,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        self.path = resolve_model_file(repo_or_path, ".onnx", None if repo_or_path.endswith(".onnx") else "model.onnx")
        levels = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.add_session_config_entry("session.intra_op.allow_spinning", "1" if allow_spinning else "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "1" if allow_spinning else "0")

        # YAML gives providers as names or [name, {options}] pairs; ORT wants str or tuple
        providers = [p if isinstance(p, str) else (p[0], {k: str(v) for k, v in p[1].items()})
                     for p in (providers or ["CPUExecutionProvider"])]
        names = [p if isinstance(p, str) else p[0] for p in providers]
        if names != ["CPUExecutionProvider"]:
            # an optimised graph saved for the CPU contains CPU-only fused kernels that
            # other execution providers can't take, so it is never cached for them
            cache_optimized = False

        model_path = self.path
        self.loaded_from_cache = False
        if cache_optimized and opt_level != "disable":
            # optimised graphs can contain CPU-specific fused kernels, so the cache key
            # includes the ORT version, CPU and level
            key = hashlib.sha1(
                f"{self.path}|{self.path.stat().st_size}|{ort.__version__}|{platform.processor()}|{platform.machine()}|{opt_level}".encode()
            ).hexdigest()[:16]
            cached = CACHE_DIR / "ort" / f"{key}.onnx"
            if cached.exists():
                model_path, self.loaded_from_cache = cached, True
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            else:
                cached.parent.mkdir(parents=True, exist_ok=True)
                so.graph_optimization_level = levels[opt_level]
                so.optimized_model_filepath = str(cached)
        else:
            so.graph_optimization_level = levels[opt_level]

        self.session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name  # "codes" for NeuCodec
        self.settings = {
            "threads": threads,
            "opt_level": opt_level,
            "allow_spinning": allow_spinning,
            "cache_optimized": cache_optimized,
            "providers": self.session.get_providers(),
            "model": str(self.path),
        }

    def decode(self, codes) -> np.ndarray:
        arr = np.asarray(codes, dtype=np.int32).reshape(1, 1, -1)
        return self.session.run(None, {self.input_name: arr})[0].astype(np.float32)[0, 0]

    def warmup(self, lengths=(30, 82, 200)) -> None:
        for n in lengths:  # first runs allocate buffers per shape; keep that out of timings
            self.decode(np.zeros(n, dtype=np.int32))


class TorchCodec:
    """NeuCodec's PyTorch decoder (upstream's ``codec_repo="neuphonic/neucodec"``) on any
    torch device, e.g. the Apple GPU (``device="mps"``), as a drop-in for ``OnnxCodec``.

    Same weights as the ONNX export, different framing: N codes give N x 480 samples
    starting half a frame earlier, where ONNX gives (N - 1) x 480. Trimming 240 samples
    from each end makes the two match at ~95 dB SNR (measured on real speech codes), so
    the streaming schedule and every other component are unchanged. Only the decoder is
    kept: the encoder half (Wav2Vec2-BERT and friends, ~600M parameters) is dropped.
    """

    TRIM = 240

    def __init__(self, repo: str = "neuphonic/neucodec", device: str = "mps", dtype: str = "float32",
                 threads: int | None = None) -> None:
        import torch
        from neucodec import NeuCodec

        if threads:
            torch.set_num_threads(int(threads))
        model = NeuCodec.from_pretrained(repo).eval()
        for name in ("semantic_model", "feature_extractor", "SemanticEncoder_module", "CodecEnc", "fc_prior"):
            if hasattr(model, name):
                delattr(model, name)
        self._torch, self.device = torch, torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.model = model.to(self.device, self.dtype)
        self.settings = {"backend": "torch", "repo": repo, "device": device, "dtype": dtype,
                         "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}

    def decode(self, codes) -> np.ndarray:
        torch = self._torch
        x = torch.as_tensor(np.asarray(codes, dtype=np.int64)).view(1, 1, -1).to(self.device)
        with torch.inference_mode():
            audio = self.model.decode_code(x)[0, 0].float().cpu().numpy()  # .cpu() waits for the GPU
        return audio[self.TRIM : len(audio) - self.TRIM]

    def warmup(self, lengths=(30, 66, 82, 200)) -> None:
        for n in lengths:  # first runs compile kernels per shape; keep that out of timings
            self.decode(np.zeros(n, dtype=np.int32))


def make_codec(repo_or_path: str, threads: int, settings: dict):
    """``codec_backend: onnx`` (default) or ``torch`` (with ``codec_device``)."""
    if settings.get("codec_backend", "onnx") == "torch":
        return TorchCodec(settings.get("codec_torch_repo", "neuphonic/neucodec"), settings.get("codec_device", "mps"),
                          settings.get("codec_dtype", "float32"), threads=threads or None)
    onnx = settings.get("onnx") or {}
    return OnnxCodec(
        repo_or_path,
        threads=threads,
        opt_level=onnx.get("opt_level", "all"),
        allow_spinning=bool(onnx.get("allow_spinning", False)),
        cache_optimized=bool(onnx.get("cache_optimized", True)),
        providers=onnx.get("providers"),
    )


# --------------------------------------------------------------------------- watermark
def load_perth():
    """Same loader as upstream (including the setuptools ``pkg_resources`` shim)."""
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        import importlib.resources

        shim = types.ModuleType("pkg_resources")
        shim.resource_filename = lambda package, resource: str(importlib.resources.files(package) / resource)
        sys.modules["pkg_resources"] = shim
    import perth

    return perth.PerthImplicitWatermarker()


class Watermarker:
    """Perth watermark. Modes: ``slice`` (watermark only emitted audio, ours),
    ``window`` (watermark the whole decoded window, upstream), ``off`` (never in default
    suites; requires an explicit opt-in and is flagged in every report)."""

    def __init__(self, mode: str = "slice", threads: int | None = None) -> None:
        if mode not in ("slice", "window", "off"):
            raise ValueError("watermark mode must be slice, window or off")
        self.mode = mode
        self._wm = None if mode == "off" else load_perth()
        if threads and self._wm is not None:
            import torch

            torch.set_num_threads(threads)

    def apply(self, audio: np.ndarray) -> np.ndarray:
        if self._wm is None or len(audio) == 0:
            return audio
        return np.asarray(self._wm.apply_watermark(audio, sample_rate=24_000), dtype=np.float32)

    def detect(self, audio: np.ndarray) -> float:
        wm = self._wm or load_perth()
        return float(wm.get_watermark(audio, sample_rate=24_000))
