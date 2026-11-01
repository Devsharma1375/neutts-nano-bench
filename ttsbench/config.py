"""Benchmark suite definition (YAML) with strict validation.

A suite says *what* to measure (corpus, voices, seeds, repeats, modes) and lists the
configurations to compare. Each configuration inherits ``defaults`` and overrides keys.
Unknown top-level keys are errors, so a typo can't silently change an experiment.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ENGINES = {"reference", "fast", "mock"}
MODES = {"batch", "stream"}
SUITE_KEYS = {
    "name", "description", "corpus", "buckets", "voices", "seeds", "repeats", "warmup", "modes",
    "rounds", "save_audio", "stream_fidelity", "quality", "gates", "noise_floor", "baseline",
    "defaults", "configs", "allow_no_watermark", "timeout_s", "max_items", "pacing", "pacing_gap_s", "cooldown_s",
}
PACING = {"none", "playback"}
# Every setting each engine reads. A key outside these is an error, so a typo in a suite
# can't silently run a different experiment than the one its name claims.
_SHARED = {"backbone", "codec", "language", "watermark"}
ENGINE_SETTINGS = {
    "reference": _SHARED | {"backbone_device", "codec_device", "torch_dtype", "torch_threads"},
    "fast": _SHARED | {
        "gguf_filename", "vocab", "reuse_prefix", "threads", "stream", "sampling", "max_tokens_per_char",
        "n_ctx", "n_batch", "n_ubatch", "use_mlock", "use_mmap", "flash_attn", "n_gpu_layers", "offload_kqv",
        "prefill_gpu", "onnx", "codec_backend", "codec_device", "codec_dtype", "codec_torch_repo",
    },
    "mock": {"prefill_tokens_per_s", "decode_tokens_per_s", "codec_x_realtime", "tokens_per_char", "stream", "load_s",
             "backbone", "codec"},
}
NESTED_SETTINGS = {
    "threads": {"backbone", "backbone_batch", "codec", "watermark"},
    "onnx": {"opt_level", "allow_spinning", "cache_optimized", "providers"},
    "stream": {"pipelined", "frames_per_chunk", "first_chunk_frames", "lookforward", "lookback", "overlap",
               "final_alignment"},
    "sampling": {"temperature", "top_k", "top_p", "min_p", "repeat_penalty"},
}
DEFAULT_GATES = {"wer_abs": 0.02, "spk_sim_drop": 0.02, "mos_drop": 0.10, "fidelity_db": 1.5, "watermark_detect": 1.0}
# small.en: base.en hallucinated whole strings of digits on near-identical audio in real runs
DEFAULT_QUALITY = {"asr_model": "small.en", "speaker": True, "mos": False, "watermark": True}


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


@dataclass
class RunConfig:
    name: str
    engine: str
    modes: list[str]
    settings: dict

    def to_dict(self) -> dict:
        return {"name": self.name, "engine": self.engine, "modes": self.modes, "settings": self.settings}


@dataclass
class Suite:
    name: str
    corpus: Path
    voices: list[str]
    seeds: list[int]
    repeats: int
    warmup: int
    modes: list[str]
    configs: list[RunConfig]
    baseline: str
    description: str = ""
    buckets: list[str] | None = None
    rounds: int = 1
    save_audio: str = "first"
    stream_fidelity: bool = True
    quality: dict = field(default_factory=lambda: dict(DEFAULT_QUALITY))
    gates: dict = field(default_factory=lambda: dict(DEFAULT_GATES))
    noise_floor: float = 0.03
    timeout_s: int = 7200
    max_items: int | None = None
    # "playback": after each measured request, idle until its audio would have finished
    # playing (plus pacing_gap_s), as a voice app does. "none": back to back (sustained load).
    pacing: str = "none"
    pacing_gap_s: float = 0.0
    cooldown_s: float = 0.0  # idle before each configuration's process starts
    source: Path | None = None

    def config(self, name: str) -> RunConfig:
        for c in self.configs:
            if c.name == name:
                return c
        raise KeyError(name)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("configs", "source")}
        d["corpus"], d["configs"] = str(self.corpus), [c.to_dict() for c in self.configs]
        return d


def load_suite(path: str | Path, quick: bool = False, only: list[str] | None = None) -> Suite:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    unknown = set(raw) - SUITE_KEYS
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    for key in ("name", "corpus", "voices", "configs", "baseline"):
        if key not in raw:
            raise ConfigError(f"{path}: missing required key {key!r}")

    modes = list(raw.get("modes", ["batch", "stream"]))
    if not set(modes) <= MODES:
        raise ConfigError(f"modes must be a subset of {sorted(MODES)}")
    defaults = raw.get("defaults") or {}
    configs, seen = [], set()
    for entry in raw["configs"]:
        merged = deep_merge(defaults, entry)
        name, engine = merged.pop("name", None), merged.pop("engine", None)
        cfg_modes = merged.pop("modes", modes)
        if not name or name in seen:
            raise ConfigError(f"config names must be present and unique (got {name!r})")
        if engine not in ENGINES:
            raise ConfigError(f"{name}: engine must be one of {sorted(ENGINES)}")
        if not set(cfg_modes) <= MODES:
            raise ConfigError(f"{name}: bad modes {cfg_modes}")
        if merged.get("watermark") == "off" and not raw.get("allow_no_watermark"):
            raise ConfigError(f"{name}: watermark 'off' needs `allow_no_watermark: true` at suite level")
        unknown = set(merged) - ENGINE_SETTINGS[engine]
        if unknown:
            raise ConfigError(f"{name}: unknown settings for engine {engine!r}: {sorted(unknown)}")
        for key, allowed in NESTED_SETTINGS.items():
            bad = set(merged.get(key) or {}) - allowed
            if bad:
                raise ConfigError(f"{name}: unknown {key} settings: {sorted(bad)}")
        if engine != "mock":
            for req in ("backbone", "codec"):
                if req not in merged:
                    raise ConfigError(f"{name}: missing {req!r}")
        seen.add(name)
        configs.append(RunConfig(name, engine, list(cfg_modes), merged))

    if only:
        missing = set(only) - seen
        if missing:
            raise ConfigError(f"--only names not in suite: {sorted(missing)}")
        configs = [c for c in configs if c.name in only]
    if raw["baseline"] not in seen:
        raise ConfigError(f"baseline {raw['baseline']!r} is not a config name")

    corpus = Path(raw["corpus"])
    if not corpus.is_absolute():
        corpus = (path.parent / corpus).resolve() if (path.parent / corpus).exists() else corpus.resolve()
    suite = Suite(
        name=raw["name"],
        description=raw.get("description", ""),
        corpus=corpus,
        voices=list(raw["voices"]),
        seeds=[int(s) for s in raw.get("seeds", [0])],
        repeats=int(raw.get("repeats", 3)),
        warmup=int(raw.get("warmup", 2)),
        modes=modes,
        configs=configs,
        baseline=raw["baseline"],
        buckets=raw.get("buckets"),
        rounds=int(raw.get("rounds", 1)),
        save_audio=raw.get("save_audio", "first"),
        stream_fidelity=bool(raw.get("stream_fidelity", True)),
        quality=deep_merge(DEFAULT_QUALITY, raw.get("quality") or {}),
        gates=deep_merge(DEFAULT_GATES, raw.get("gates") or {}),
        noise_floor=float(raw.get("noise_floor", 0.03)),
        timeout_s=int(raw.get("timeout_s", 7200)),
        max_items=raw.get("max_items"),
        pacing=raw.get("pacing", "none"),
        pacing_gap_s=float(raw.get("pacing_gap_s", 0.0)),
        cooldown_s=float(raw.get("cooldown_s", 0.0)),
        source=path,
    )
    if suite.pacing not in PACING:
        raise ConfigError(f"pacing must be one of {sorted(PACING)}")
    if suite.save_audio not in ("first", "all", "none"):
        raise ConfigError("save_audio must be first, all or none")
    if suite.repeats < 1 or suite.rounds < 1 or suite.warmup < 0:
        raise ConfigError("repeats and rounds must be >= 1, warmup >= 0")
    if quick:  # smoke-check settings: same code paths, a fraction of the work
        suite.repeats, suite.warmup, suite.rounds, suite.seeds = 1, 1, 1, suite.seeds[:1]
        suite.voices, suite.buckets, suite.max_items = suite.voices[:1], ["short", "medium"], 4
    return suite
