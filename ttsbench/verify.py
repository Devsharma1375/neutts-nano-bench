"""``ttsbench verify``: prove on the real model that the optimised engine says exactly what
upstream says.

Speed numbers only mean something if the faster engine produces the same speech. With
identical models and seeds, upstream-compatible settings must give:

1. identical speech tokens in batch mode
2. identical streamed audio (sample-exact, upstream final-chunk alignment included)
3. identical tokens with the pipelined (two-thread) streamer
4. identical tokens with voice-prefix KV reuse (the rest of the prompt is chunked exactly where
   a full prefill splits it, so the arithmetic is the same)
5. the fixed final-chunk alignment produces exactly the one-pass decode's length, and the
   tail's measured offset against the one-pass decode (upstream's vs ours) is reported
6. identical tokens with the speech-only vocabulary (``vocab: speech``), batch and stream,
   and its logits bit-identical to the full model's for every token it keeps
"""
from __future__ import annotations

import logging

import numpy as np

from .engines import Request, create_engine
from .engines.base import load_voice
from .engines.components import CACHE_DIR
from .timing import StageTimer

log = logging.getLogger("ttsbench.verify")

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "On the third of May, Doctor Lee paid five dollars for coffee.",
    "Streaming speech should sound exactly like the full clip, from the first word to the last.",
    # long inputs: ~400 generated tokens and a prompt past the first 512-token prefill chunk,
    # where a misaligned prefix reuse once diverged (12 of 72 long requests in a full run)
    "If you need to cancel or change your booking, please let us know at least forty-eight hours "
    "in advance, otherwise a small fee may be charged to the card you used.",
    "Captain Reyes announced that the flight would land about fifteen minutes early, thanks to a "
    "strong tailwind, and that the weather in Lisbon was sunny and warm.",
]
UPSTREAM_STREAM = {"frames_per_chunk": 25, "first_chunk_frames": 25, "lookforward": 5, "lookback": 50,
                   "overlap": 1, "final_alignment": "upstream"}


def _stream(engine, req, with_chunks: bool = False):
    timer = StageTimer()
    chunks = list(engine.stream(req, timer))
    out = (np.concatenate(chunks), engine.stream_tokens())
    return (*out, chunks) if with_chunks else out


def _tail_lag(streamed: np.ndarray, one_pass: np.ndarray, final_len: int, max_lag: int = 960) -> int | None:
    """Offset (samples) of the final chunk's audio relative to the same audio in the one-pass
    decode: the lag maximising their normalised cross-correlation, measured inside the final
    chunk only (skipping its two cross-faded frames). 0 = correctly placed; +480 = one frame
    late. None if the final chunk is too short to measure."""
    start = len(streamed) - final_len + 2 * 480
    end = min(len(one_pass), len(streamed))
    if end - start < 3 * 480:
        return None
    ref = one_pass[start:end]
    padded = np.pad(streamed, (max_lag, max_lag))

    def score(lag: int) -> float:
        seg = padded[start + lag + max_lag : end + lag + max_lag]
        return float(np.dot(ref, seg) / (np.linalg.norm(seg) * np.linalg.norm(ref) + 1e-12))

    return max(range(-max_lag, max_lag + 1), key=score)


def _first_logits(engine, prompt: list[int]) -> np.ndarray:
    """Logits after reading ``prompt`` (original-vocabulary ids) into ``engine``."""
    import llama_cpp

    llm = engine.llm
    if engine.vocab_mode == "speech":
        engine._kv = engine._prefill_embd(llm, [], prompt)
    else:
        llm.reset()
        llm.eval(prompt)
    ptr = llama_cpp.llama_get_logits_ith(llm._ctx.ctx, -1)
    return np.ctypeslib.as_array(ptr, shape=(llm.n_vocab(),)).copy()


def verify(backbone: str, codec: str, voice: str = "jo", seeds=(0, 1)) -> bool:
    v = load_voice(voice, CACHE_DIR / "voices")
    ref = create_engine("reference", {"backbone": backbone, "codec": codec})
    ref.load()
    base = {"backbone": backbone, "codec": codec, "watermark": "window", "threads": {"codec": 1}}
    fast_seq = create_engine("fast", {**base, "reuse_prefix": False, "stream": {**UPSTREAM_STREAM, "pipelined": False}})
    fast_pipe = create_engine("fast", {**base, "reuse_prefix": False, "stream": {**UPSTREAM_STREAM, "pipelined": True}})
    fast_reuse = create_engine("fast", {**base, "reuse_prefix": True, "stream": {"pipelined": True}})
    fast_vocab = create_engine("fast", {**base, "reuse_prefix": True, "vocab": "speech", "stream": {"pipelined": True}})
    # unwatermarked twins for measuring where the final chunk lands
    nowm = {**base, "watermark": "off", "reuse_prefix": False}
    fast_nowm_upstream = create_engine("fast", {**nowm, "stream": {**UPSTREAM_STREAM, "pipelined": False}})
    fast_nowm_fixed = create_engine("fast", {**nowm, "stream": {**UPSTREAM_STREAM, "final_alignment": "fixed", "pipelined": False}})
    engines = (fast_seq, fast_pipe, fast_reuse, fast_vocab, fast_nowm_upstream, fast_nowm_fixed)
    for e in engines:
        e.load()

    rows, ok = [], True

    def check(name: str, passed: bool, detail: str, mandatory: bool = True) -> None:
        nonlocal ok
        rows.append((name, "PASS" if passed else ("FAIL" if mandatory else "DIFFERS"), detail))
        ok &= passed or not mandatory

    for text in SENTENCES:
        for seed in seeds:
            b = Request(text, v, seed, stream=False)
            r_ref, r_fast = ref.run(b), fast_seq.run(b)
            same = np.array_equal(r_ref.tokens, r_fast.tokens)
            check(f"batch tokens  seed={seed} '{text[:28]}…'", same, f"{len(r_ref.tokens)} vs {len(r_fast.tokens)} tokens")

            s = Request(text, v, seed, stream=True)
            a_ref, t_ref = _stream(ref, s)
            a_seq, t_seq = _stream(fast_seq, s)
            n = min(len(a_ref), len(a_seq))
            diff = float(np.max(np.abs(a_ref[:n] - a_seq[:n]))) if n else float("inf")
            check(f"stream audio  seed={seed} '{text[:28]}…'",
                  np.array_equal(t_ref, t_seq) and len(a_ref) == len(a_seq) and diff < 1e-4,
                  f"len {len(a_ref)} vs {len(a_seq)}, max |diff| {diff:.2e}")
            _, t_pipe = _stream(fast_pipe, s)
            check(f"pipelined tokens seed={seed}", np.array_equal(t_ref, t_pipe), f"{len(t_pipe)} tokens")

            fast_reuse.run(Request("A different sentence to leave something in the cache.", v, seed, False))
            r_reuse = fast_reuse.run(b)
            check(f"prefix-reuse tokens seed={seed}", np.array_equal(r_ref.tokens, r_reuse.tokens),
                  f"{len(r_reuse.tokens)} tokens with the voice prefix served from the KV cache")
            # NeuCodec decodes N tokens to (N - 1) x 480 samples. A correct stream has exactly
            # the one-pass length; upstream's final chunk starts one frame early in the decoded
            # window but is placed at the next chunk's offset, so its tail lands 480 samples late.
            a_fix, t_fix = _stream(fast_reuse, s)
            one_pass = fast_reuse.decode_full(t_fix)
            check(f"fixed final alignment seed={seed}", len(a_fix) == len(one_pass),
                  f"{len(a_fix)} samples; one-pass decode of the same {len(t_fix)} tokens: {len(one_pass)}")
            # measure the final chunk's placement on unwatermarked audio from the same tokens
            ref_one_pass = fast_seq.decode_full(t_ref)
            a_up, _, c_up = _stream(fast_nowm_upstream, s, with_chunks=True)
            a_fx, _, c_fx = _stream(fast_nowm_fixed, s, with_chunks=True)
            lag_up, lag_fx = _tail_lag(a_up, ref_one_pass, len(c_up[-1])), _tail_lag(a_fx, ref_one_pass, len(c_fx[-1]))
            # windowed decoding differs slightly from a one-pass decode (the decoder sees the whole
            # input), so a correctly placed chunk measures within a few samples of 0; a misplaced
            # one is a whole frame (480 samples) off
            check(f"final chunk lag vs one-pass seed={seed}", lag_fx is None or abs(lag_fx) < 120,
                  f"upstream {lag_up if lag_up is not None else 'n/a'} samples ({len(a_up) - len(ref_one_pass):+d} long), "
                  f"fixed {lag_fx if lag_fx is not None else 'n/a'} samples ({len(a_fx) - len(ref_one_pass):+d} long)")

            fast_vocab.run(Request("A different sentence to leave something in the cache.", v, seed, False))
            r_vocab = fast_vocab.run(b)
            prompt = fast_seq.build_prompt(b, StageTimer())
            full_logits, speech_logits = _first_logits(fast_seq, prompt), _first_logits(fast_vocab, prompt)
            off = fast_vocab._vocab_offset
            check(f"speech-vocab logits bitwise seed={seed}",
                  np.array_equal(full_logits[off : off + len(speech_logits)], speech_logits),
                  f"{len(speech_logits)} logits compared; max |diff| "
                  f"{float(np.max(np.abs(full_logits[off : off + len(speech_logits)] - speech_logits))):.1e}; "
                  f"top-50 of the full vocabulary inside the kept rows: "
                  f"{bool(np.all((np.argsort(full_logits)[-50:] >= off) & (np.argsort(full_logits)[-50:] < off + len(speech_logits))))}")
            check(f"speech-vocab tokens seed={seed}", np.array_equal(r_ref.tokens, r_vocab.tokens),
                  f"{len(r_vocab.tokens)} tokens; pruned head scores {fast_vocab.llm.n_vocab()} of "
                  f"{fast_vocab.tok.n_vocab()} tokens")
            _, t_vs = _stream(fast_vocab, s)
            check(f"speech-vocab stream tokens seed={seed}", np.array_equal(t_ref, t_vs), f"{len(t_vs)} tokens")

    width = max(len(r[0]) for r in rows)
    print("\nEquivalence: optimised engine vs upstream NeuTTS\n")
    for name, status, detail in rows:
        print(f"  {name:<{width}}  {status:<7}  {detail}")
    print(f"\n{'ALL MANDATORY CHECKS PASSED' if ok else 'SOME MANDATORY CHECKS FAILED'}\n")
    for e in (ref, *engines):
        e.close()
    return ok
