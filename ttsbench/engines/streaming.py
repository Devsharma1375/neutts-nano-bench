"""Chunk scheduling and incremental overlap-add for streaming synthesis.

Pure NumPy with no model dependencies, so it is unit-tested against the upstream
implementation (see tests/test_streaming.py).

All counts are in codec frames: 1 frame = 1 token = 20 ms = 480 samples at 24 kHz.

* ``frames_per_chunk``   new frames emitted per chunk (upstream: 25, i.e. 0.5 s)
* ``first_chunk_frames`` size of the first chunk only; smaller means faster first audio
* ``lookforward``        future frames the decoder must see before a chunk is emitted
* ``lookback``           past frames fed to the decoder as context (upstream: 50)
* ``overlap``            frames cross-faded between neighbouring chunks (upstream: 1)

Upstream (neutts 1.4.1, ``NeuTTS._infer_stream_ggml``) re-runs overlap-add over all
previous chunks on every chunk (quadratic in utterance length) and places the final
chunk one frame late. ``IncrementalOLA`` is sample-identical for every non-final chunk
at O(chunk) cost. ``final_alignment="fixed"`` places the final chunk correctly;
``"upstream"`` reproduces upstream byte-for-byte, for equivalence testing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

HOP = 480  # samples per codec frame (24 kHz / 50 Hz)


@dataclass(frozen=True)
class StreamParams:
    frames_per_chunk: int = 25
    first_chunk_frames: int = 25
    lookforward: int = 5
    lookback: int = 50
    overlap: int = 1
    final_alignment: Literal["fixed", "upstream"] = "fixed"

    def __post_init__(self) -> None:
        if self.frames_per_chunk < 1 or self.first_chunk_frames < 1:
            raise ValueError("chunk sizes must be at least 1 frame")
        if min(self.lookforward, self.lookback, self.overlap) < 0:
            raise ValueError("lookforward, lookback and overlap must be >= 0")
        if 2 * self.overlap >= min(self.frames_per_chunk, self.first_chunk_frames):
            raise ValueError("2 * overlap must be smaller than every chunk size")
        if self.final_alignment not in ("fixed", "upstream"):
            raise ValueError("final_alignment must be 'fixed' or 'upstream'")


@dataclass(frozen=True)
class Window:
    """One decode job.

    Decode ``tokens[token_start:token_end]``, keep ``audio[sample_start:sample_end]``
    (``None`` keeps everything to the end), add it to the output at ``out_offset``,
    then emit output up to ``emit_until`` (``None`` flushes everything).
    """

    token_start: int
    token_end: int
    sample_start: int
    sample_end: int | None
    out_offset: int
    emit_until: int | None
    is_final: bool


class ChunkPlanner:
    """Decides when a chunk is ready and which token window to decode for it.

    Tokens are counted including ``n_prefix_tokens`` (the reference codes that upstream
    places in front of generated tokens as decoder context). Output sample offsets are
    relative to the first generated token.
    """

    def __init__(self, params: StreamParams, n_prefix_tokens: int = 0) -> None:
        self.p = params
        self.prefix = n_prefix_tokens
        self.decoded = n_prefix_tokens  # tokens already emitted, including the prefix
        self.n_chunks = 0

    def _chunk_size(self) -> int:
        return self.p.first_chunk_frames if self.n_chunks == 0 else self.p.frames_per_chunk

    def ready(self, n_tokens: int) -> bool:
        return n_tokens - self.decoded >= self._chunk_size() + self.p.lookforward

    def next_window(self) -> Window:
        p, nd, c = self.p, self.decoded, self._chunk_size()
        ts = max(nd - p.lookback - p.overlap, 0)
        te = nd + c + p.lookforward + p.overlap
        ss = (nd - ts) * HOP
        window = Window(
            token_start=ts,
            token_end=te,
            sample_start=ss,
            sample_end=ss + (c + 2 * p.overlap) * HOP,
            out_offset=(nd - self.prefix) * HOP,
            emit_until=(nd - self.prefix + c) * HOP,
            is_final=False,
        )
        self.decoded += c
        self.n_chunks += 1
        return window

    def final_window(self, n_tokens: int) -> Window | None:
        p, nd = self.p, self.decoded
        remaining = n_tokens - nd
        if remaining <= 0:
            return None
        if p.final_alignment == "upstream":
            ts = max(n_tokens - (p.lookback + p.overlap + remaining), 0)
            ss = (n_tokens - ts - remaining - p.overlap) * HOP  # one frame early: upstream bug
        else:
            ts = max(nd - p.lookback - p.overlap, 0)
            ss = (nd - ts) * HOP  # starts exactly at the first un-emitted token
        self.decoded = n_tokens
        self.n_chunks += 1
        return Window(ts, n_tokens, ss, None, (nd - self.prefix) * HOP, None, True)


def _triangle(n: int) -> np.ndarray:
    t = np.linspace(0, 1, n + 2, dtype=np.float32)[1:-1]
    return 0.5 - np.abs(t - 0.5)


class IncrementalOLA:
    """Triangular-weight overlap-add (as upstream ``_linear_overlap_add``), streamed.

    Each frame's weight spans the whole frame, so a sample is final once no later frame
    can start before it. Only the pending tail is kept, so each chunk costs
    O(frame length) instead of O(total length).
    """

    def __init__(self) -> None:
        self._base = 0  # absolute sample index of the buffer's first element
        self._acc = np.zeros(0, dtype=np.float32)
        self._wsum = np.zeros(0, dtype=np.float32)
        self.emitted = 0

    def add(self, frame: np.ndarray, offset: int) -> None:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if offset < self.emitted:
            raise ValueError("frame overlaps audio that was already emitted")
        need = offset + frame.shape[0] - self._base
        if need > self._acc.shape[0]:
            pad = need - self._acc.shape[0]
            self._acc = np.concatenate([self._acc, np.zeros(pad, np.float32)])
            self._wsum = np.concatenate([self._wsum, np.zeros(pad, np.float32)])
        lo = offset - self._base
        weight = _triangle(frame.shape[0])
        self._acc[lo : lo + frame.shape[0]] += weight * frame
        self._wsum[lo : lo + frame.shape[0]] += weight

    def emit(self, until: int | None) -> np.ndarray:
        stop = self._base + self._acc.shape[0] if until is None else until
        lo, hi = self.emitted - self._base, stop - self._base
        if hi <= lo:
            return np.zeros(0, dtype=np.float32)
        if hi > self._acc.shape[0] or self._wsum[lo:hi].min() <= 0:
            raise RuntimeError("overlap-add gap: the chunk schedule left samples uncovered")
        out = self._acc[lo:hi] / self._wsum[lo:hi]
        self._acc, self._wsum = self._acc[hi:].copy(), self._wsum[hi:].copy()
        self._base, self.emitted = stop, stop
        return out


def stream_decode_offline(
    tokens: np.ndarray,
    n_prefix: int,
    decode,
    params: StreamParams,
    watermark=None,
    watermark_mode: Literal["slice", "window", "off"] = "slice",
) -> list[np.ndarray]:
    """Run the full streaming schedule over a finished token list (used by tests and by
    the stream-fidelity metric). ``decode(token_slice) -> audio``."""
    planner, ola, chunks = ChunkPlanner(params, n_prefix), IncrementalOLA(), []
    n = len(tokens)
    for available in range(n_prefix, n + 1):
        while planner.ready(available):
            chunks.append(_run_window(planner.next_window(), tokens, decode, ola, watermark, watermark_mode))
    final = planner.final_window(n)
    if final is not None:
        chunks.append(_run_window(final, tokens, decode, ola, watermark, watermark_mode))
    return chunks


def _run_window(w: Window, tokens, decode, ola, watermark, mode) -> np.ndarray:
    audio = decode(tokens[w.token_start : w.token_end])
    if watermark is not None and mode == "window":
        audio = watermark(audio)
    audio = audio[w.sample_start : w.sample_end]
    if watermark is not None and mode == "slice":
        audio = watermark(audio)
    ola.add(audio, w.out_offset)
    return ola.emit(w.emit_until)
