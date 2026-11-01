"""The streaming core must match upstream exactly where upstream is right, and fix it where
it is not. Upstream comparisons import neutts and are skipped when it is not installed."""
import builtins
import re

import numpy as np
import pytest

from ttsbench.engines.streaming import (
    HOP,
    ChunkPlanner,
    IncrementalOLA,
    StreamParams,
    stream_decode_offline,
)

RNG = np.random.default_rng(0)
BASE = RNG.standard_normal((4000, HOP)).astype(np.float32) * 0.1  # audio for token id i


def identity_decode(tokens):
    return np.concatenate([BASE[int(t)] for t in tokens])


def context_decode(tokens):
    """Output depends on the whole window (like a real decoder), so overlaps matter."""
    tokens = np.asarray(tokens)
    return identity_decode(tokens) * np.float32(1.0 + 1e-3 * (int(tokens[0]) % 7))


def _upstream_stream(gen_tokens, ref_tokens, decode):
    neutts = pytest.importorskip("neutts.neutts")
    tts = object.__new__(neutts.NeuTTS)
    tts.sample_rate, tts.max_context, tts.hop_length = 24000, 4096, HOP
    tts.streaming_overlap_frames, tts.streaming_frames_per_chunk = 1, 25
    tts.streaming_lookforward, tts.streaming_lookback = 5, 50
    tts.streaming_stride_samples = 25 * HOP
    tts.watermarker, tts._seed, tts._is_quantized_model = None, 0, True

    class FakeBackbone:
        def reset(self):
            pass

        def __call__(self, prompt, **kw):
            for t in gen_tokens:
                yield {"choices": [{"text": f"<|speech_{t}|>"}]}

    tts.backbone = FakeBackbone()
    tts._ggml_prompt = lambda *a, **k: ""
    tts._decode = lambda s: decode([int(i) for i in re.findall(r"<\|speech_(\d+)\|>", s)])
    printer, builtins.print = builtins.print, lambda *a, **k: None
    try:
        return list(tts._infer_stream_ggml(list(ref_tokens), "", "", None, 1.0, 50))
    finally:
        builtins.print = printer


@pytest.mark.parametrize("n_gen", [31, 60, 137, 400])
@pytest.mark.parametrize("n_ref", [0, 37, 120])
def test_upstream_mode_is_byte_identical_to_upstream(n_gen, n_ref):
    ref = np.arange(3000, 3000 + n_ref)
    gen = np.arange(n_gen) + 100
    upstream = _upstream_stream(gen, ref, context_decode)
    ours = stream_decode_offline(
        np.concatenate([ref, gen]), n_ref, context_decode, StreamParams(final_alignment="upstream")
    )
    assert len(ours) == len(upstream)
    for a, b in zip(ours, upstream):
        np.testing.assert_allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("n_gen", [3, 31, 60, 137, 400])
@pytest.mark.parametrize(
    "params",
    [
        StreamParams(),
        StreamParams(first_chunk_frames=8),
        StreamParams(first_chunk_frames=10, frames_per_chunk=40, lookback=15),
        StreamParams(lookforward=0, overlap=2, first_chunk_frames=12),
    ],
)
def test_fixed_mode_reconstructs_exactly(n_gen, params):
    ref = np.arange(3000, 3045)
    gen = np.arange(n_gen) + 7
    out = np.concatenate(
        stream_decode_offline(np.concatenate([ref, gen]), len(ref), identity_decode, params)
    )
    np.testing.assert_allclose(out, identity_decode(gen), atol=1e-6)


def test_upstream_final_chunk_is_one_frame_late():
    gen = np.arange(137) + 7
    up = np.concatenate(_upstream_stream(gen, [], identity_decode))
    assert len(up) == len(gen) * HOP + HOP  # 20 ms too long
    fixed = np.concatenate(stream_decode_offline(gen, 0, identity_decode, StreamParams()))
    assert len(fixed) == len(gen) * HOP


def test_incremental_ola_matches_upstream_full_recompute():
    neutts = pytest.importorskip("neutts.neutts")
    stride, frame_len = 25 * HOP, 27 * HOP
    frames = [RNG.standard_normal(frame_len).astype(np.float32) for _ in range(9)]
    ola, emitted = IncrementalOLA(), []
    for k, f in enumerate(frames):
        ola.add(f, k * stride)
        emitted.append(ola.emit((k + 1) * stride))
    full = neutts._linear_overlap_add(frames, stride=stride)
    np.testing.assert_allclose(np.concatenate(emitted), full[: len(frames) * stride], atol=1e-6)


def test_first_chunk_needs_fewer_tokens():
    small = ChunkPlanner(StreamParams(first_chunk_frames=8))
    default = ChunkPlanner(StreamParams())
    assert small.ready(13) and not default.ready(13)
    assert default.ready(30)


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        StreamParams(overlap=13)
    with pytest.raises(ValueError):
        StreamParams(first_chunk_frames=0)
