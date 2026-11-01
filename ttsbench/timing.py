"""Low-overhead stage timing.

Every engine reports the same stage names, so reports compare like with like:

    phonemize   text -> phonemes (input text + reference transcript)
    prompt      building/tokenizing the prompt
    prefill     backbone reads the prompt (llama.cpp perf counter or first forward pass)
    decode      backbone generates tokens
    sampling    per-token Python/sampler overhead outside prefill+decode
    codec       tokens -> waveform
    watermark   Perth watermarking
    other       everything else inside the request (queues, copies, glue)

``other`` is computed as total minus the named stages, so stages always sum to the
total. That invariant is checked in tests and makes "unaccounted time" visible.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

STAGES = ("phonemize", "prompt", "prefill", "decode", "sampling", "codec", "watermark", "other")

now = time.perf_counter


@dataclass
class StageTimer:
    t0: float = field(default_factory=now)
    ms: dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in STAGES})
    first_audio_s: float | None = None
    first_token_s: float | None = None
    chunks: list[tuple[float, int]] = field(default_factory=list)  # (seconds since t0, samples)
    total_ms: float | None = None

    @contextmanager
    def stage(self, name: str):
        start = now()
        try:
            yield
        finally:
            self.ms[name] += (now() - start) * 1000.0

    def add(self, name: str, ms: float) -> None:
        self.ms[name] += ms

    def mark_first_token(self) -> None:
        if self.first_token_s is None:
            self.first_token_s = now() - self.t0

    def mark_chunk(self, n_samples: int) -> None:
        t = now() - self.t0
        if self.first_audio_s is None and n_samples > 0:
            self.first_audio_s = t
        self.chunks.append((t, n_samples))

    def finish(self) -> None:
        self.total_ms = (now() - self.t0) * 1000.0
        named = sum(v for k, v in self.ms.items() if k != "other")
        # never negative: overlapped stages (pipelined streaming) can exceed wall time
        self.ms["other"] = max(self.total_ms - named, 0.0)
