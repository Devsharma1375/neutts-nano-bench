"""A deterministic stand-in engine for testing the framework itself.

It needs no model weights (NeuTTS weights are gated), sleeps for configurable
"prefill/decode/codec" times, and streams through the *real* ChunkPlanner and
IncrementalOLA. Reports label mock results loudly; they say nothing about NeuTTS speed.
"""
from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np

from ..timing import StageTimer
from .base import Engine, Request, Result
from .streaming import HOP, ChunkPlanner, IncrementalOLA, StreamParams


class MockEngine(Engine):
    name = "mock"

    def load(self) -> None:
        s = self.settings
        self.prefill_tps = float(s.get("prefill_tokens_per_s", 2000))
        self.decode_tps = float(s.get("decode_tokens_per_s", 200))
        self.codec_x = float(s.get("codec_x_realtime", 30))
        self.tokens_per_char = float(s.get("tokens_per_char", 0.75))
        self.stream_params = StreamParams(**(s.get("stream") or {}))
        time.sleep(float(s.get("load_s", 0.05)))

    def _n_tokens(self, req: Request) -> int:
        return max(25, int(len(req.text) * self.tokens_per_char))

    def _tokens(self, req: Request) -> np.ndarray:
        rng = np.random.default_rng([req.seed, len(req.text)])
        return rng.integers(0, 65536, self._n_tokens(req), dtype=np.int32)

    @staticmethod
    def _audio_for(tokens: np.ndarray) -> np.ndarray:
        """Each token maps to its own fixed 480-sample frame, so any decode window yields
        the same samples: streaming must then reconstruct the full decode exactly."""
        n = np.arange(HOP, dtype=np.float32) / 24000
        freqs = 120 + (np.asarray(tokens, np.int64) % 200).astype(np.float32)
        frames = 0.2 * np.sin(2 * np.pi * freqs[:, None] * n[None, :])
        return frames.reshape(-1).astype(np.float32)

    def _sleep_codec(self, timer: StageTimer, n_frames: int) -> None:
        with timer.stage("codec"):
            time.sleep(n_frames / 50 / self.codec_x)

    def synthesize(self, req: Request, timer: StageTimer) -> Result:
        prompt = len(req.voice.codes) + len(req.text)
        with timer.stage("prefill"):
            time.sleep(prompt / self.prefill_tps)
        tokens = self._tokens(req)
        timer.mark_first_token()
        with timer.stage("decode"):
            time.sleep(len(tokens) / self.decode_tps)
        self._sleep_codec(timer, len(tokens))
        audio = self._audio_for(tokens)
        timer.mark_chunk(len(audio))
        return Result(audio, tokens, prompt, "eos", timer)

    def stream(self, req: Request, timer: StageTimer) -> Iterator[np.ndarray]:
        prompt = len(req.voice.codes) + len(req.text)
        with timer.stage("prefill"):
            time.sleep(prompt / self.prefill_tps)
        tokens = self._tokens(req)
        ref = np.zeros(len(req.voice.codes), np.int32)
        cache = list(ref)
        planner, ola = ChunkPlanner(self.stream_params, len(ref)), IncrementalOLA()

        def window(w):
            self._sleep_codec(timer, w.token_end - w.token_start)
            audio = self._audio_for(np.asarray(cache[w.token_start : w.token_end]))[w.sample_start : w.sample_end]
            ola.add(audio, w.out_offset)
            return ola.emit(w.emit_until)

        for code in tokens:
            with timer.stage("decode"):
                time.sleep(1 / self.decode_tps)
            timer.mark_first_token()
            cache.append(int(code))
            while planner.ready(len(cache)):
                yield window(planner.next_window())
        final = planner.final_window(len(cache))
        if final is not None:
            yield window(final)
        self._last_stream_tokens = tokens
        self._last_stream_info = {"prompt_tokens": prompt, "stop_reason": "eos", "extra": {}}

    def decode_full(self, tokens: np.ndarray) -> np.ndarray:
        return self._audio_for(np.asarray(tokens))
