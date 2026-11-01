"""Optimised NeuTTS engine: GGUF backbone via llama.cpp, ONNX codec decoder.

It produces the same speech tokens as upstream for the same seed (``ttsbench verify``
checks this on the real model). Every optimisation is a setting, so each one is its own
benchmark experiment:

* token-ID pipeline: reference codes become token IDs by arithmetic, not by formatting
  and re-tokenizing ~650 ``<|speech_N|>`` strings; generated IDs map straight back to
  codes (upstream detokenizes to text and regex-parses it)
* ``reuse_prefix``: keep llama.cpp's KV cache between requests so the shared prompt prefix
  (instructions + reference transcript) isn't recomputed
* cached phonemes for the reference transcript
* explicit thread counts for backbone prefill/decode, codec and watermark
* tuned ONNX Runtime session (no spin-waiting, cached optimised graph)
* streaming: smaller first chunk, configurable lookback, incremental overlap-add, a
  correctly aligned final chunk, and a pipeline that decodes audio on a second thread
  while the backbone keeps generating
* ``watermark: slice``: watermark only emitted audio (2.5x less watermark work in
  streaming), detection verified per output by the quality step
* optional runaway guard: cap generated tokens relative to input length
"""
from __future__ import annotations

import math
import os
import queue
import threading
from collections.abc import Iterator

import numpy as np

from ..timing import StageTimer, now
from .base import Engine, Request, Result
from .components import (
    CACHE_DIR,
    Phonemizer,
    Watermarker,
    make_codec,
    performance_cores,
    resolve_model_file,
    resolve_threads,
    upstream_threads,
)
from .streaming import ChunkPlanner, IncrementalOLA, StreamParams

N_CODES = 65_536
_DONE = object()

# Upstream's effective sampler: NeuTTS passes temperature/top_k and llama-cpp-python's
# create_completion defaults supply the rest. Changing any of these changes the audio.
UPSTREAM_SAMPLING = dict(temperature=1.0, top_k=50, top_p=0.95, min_p=0.05, repeat_penalty=1.0)


class FastEngine(Engine):
    name = "fast"

    def load(self) -> None:
        import llama_cpp

        s = self.settings
        self._llama_cpp = llama_cpp
        stream_cfg = dict(s.get("stream") or {})
        self.pipelined = bool(stream_cfg.pop("pipelined", True))
        self.stream_params = StreamParams(**stream_cfg)
        if self.stream_params.lookforward < 1:
            # NeuCodec decodes N codes to N - 1 frames, so a window needs at least one code
            # beyond what it emits or the overlap-add leaves a gap
            raise ValueError("stream.lookforward must be >= 1 with NeuCodec")
        self.sampling = {**UPSTREAM_SAMPLING, **(s.get("sampling") or {})}
        self.reuse_prefix = bool(s.get("reuse_prefix", True))
        self._voice_prefix: dict = {}
        self._prefix_len = 0
        self.max_tokens_per_char = s.get("max_tokens_per_char")  # None = upstream behaviour

        # Thread counts: an int, "auto" (our choice) or "upstream" (NeuTTS/ORT defaults, so a
        # ladder step can isolate "better implementation" from "better thread choices").
        threads = s.get("threads") or {}
        cores = performance_cores()
        codec_cfg = threads.get("codec", 1)
        self.codec_threads = 0 if codec_cfg in ("upstream", 0) else resolve_threads(codec_cfg, 1)  # 0 = ORT default
        # a pipelined stream decodes audio on a second thread; a GPU codec needs no CPU core for it
        reserve = (self.codec_threads or 1) if self.pipelined and s.get("codec_backend") != "torch" else 0
        bb, bb_batch = threads.get("backbone"), threads.get("backbone_batch")
        n_threads = upstream_threads()[0] if bb == "upstream" else resolve_threads(bb, max(cores - reserve, 1))
        # Prompt reading (one large parallel job, split dynamically) is fastest on every core;
        # generation (one small step per token, as slow as its slowest thread) on the performance
        # cores only. Measured with a cool-down before each run on an M4: prefill 2,004 tok/s on
        # 4 threads vs 2,107-2,276 on 10; generation 401 tok/s on 4 vs 320 on 10.
        all_cores = os.cpu_count() or cores
        n_threads_batch = upstream_threads()[1] if bb_batch == "upstream" else resolve_threads(bb_batch, all_cores)
        # Generation threads per request mode. A pipelined stream leaves `reserve` cores for
        # the audio thread; a batch request decodes audio only after generation, so with
        # "auto" it takes every performance core. Switched per request (llama_set_n_threads).
        self._threads_for = {"stream": n_threads,
                             "batch": resolve_threads(None, cores) if bb in (None, "auto") else n_threads}

        if (s.get("prefill_gpu") or int(s.get("n_gpu_layers", 0)) != 0) and not llama_cpp.llama_supports_gpu_offload():
            raise RuntimeError("GPU offload requested but this llama.cpp build has no GPU backend")
        self.gguf_path = resolve_model_file(s["backbone"], ".gguf", s.get("gguf_filename"))
        # vocab: "full" runs the GGUF as shipped; "speech" runs a copy whose output head only
        # scores the tokens generation can emit (engines/vocab.py). The prompt is then tokenized
        # by the original vocabulary and fed as embeddings from the original table.
        self.vocab_mode = s.get("vocab", "full")
        if self.vocab_mode not in ("full", "speech"):
            raise ValueError("vocab must be 'full' or 'speech'")
        model_path = self.gguf_path
        if self.vocab_mode == "speech":
            from .vocab import EmbeddingTable, ensure_pruned

            model_path = ensure_pruned(self.gguf_path, CACHE_DIR / "pruned")
            self.embd = EmbeddingTable(self.gguf_path)
        llama_kw = dict(
            model_path=str(model_path),
            n_ctx=int(s.get("n_ctx", 2048)),
            n_batch=int(s.get("n_batch", 512)),  # as upstream
            n_ubatch=int(s.get("n_ubatch", 512)),
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            use_mlock=bool(s.get("use_mlock", True)),
            use_mmap=bool(s.get("use_mmap", True)),
            flash_attn=bool(s.get("flash_attn", False)),
            offload_kqv=bool(s.get("offload_kqv", True)),
            seed=0,
            verbose=False,
        )
        # Metal/GPU offload: 0 = CPU only (upstream's default), -1 = every layer
        self.llm = llama_cpp.Llama(n_gpu_layers=int(s.get("n_gpu_layers", 0)), **llama_kw)
        # prefill_gpu: a second context on the GPU reads the prompt (3x faster than the CPU on
        # Apple silicon), then hands its KV cache to the CPU context, which generates (faster
        # than the GPU per token for a model this small). Same weights, mmap-shared.
        self.llm_prefill = None
        if s.get("prefill_gpu"):
            if self.vocab_mode != "speech":
                raise ValueError("prefill_gpu requires vocab: speech (it drives llama.cpp directly)")
            self.llm_prefill = llama_cpp.Llama(n_gpu_layers=-1, **llama_kw)
            self._kv_prefill: list[int] = []
            self._state_buf = None
        self.tok = self.llm  # whatever tokenizes prompts (original vocabulary)
        if self.vocab_mode == "speech":
            self.tok = llama_cpp.Llama(model_path=str(self.gguf_path), vocab_only=True, verbose=False)
            if self.embd.n_embd != self.llm.n_embd():
                raise ValueError("embedding table and pruned model disagree on n_embd")
            self._embd_batch = llama_cpp.llama_batch_init(self.llm.n_batch, self.embd.n_embd, 1)
            self._tok_batch = llama_cpp._internals.LlamaBatch(n_tokens=1, embd=0, n_seq_max=1)
            self._kv: list[int] = []  # original-vocabulary ids of the prompt held in the KV cache
            self._vocab_offset = int(self.llm.metadata["neuphonic.vocab_offset"])
        fmt = self.tok.metadata.get("neuphonic.input_format", "phonemes")
        if fmt != "phonemes":
            raise ValueError(f"FastEngine supports phoneme-input models only; this GGUF is {fmt!r}")

        self._special = {name: self._single_token(f"<|{name}|>") for name in (
            "TEXT_PROMPT_START", "TEXT_PROMPT_END", "SPEECH_GENERATION_START", "SPEECH_GENERATION_END")}
        self._map_speech_tokens()

        self.phonemizer = Phonemizer(s.get("language", "en-us"))
        self.codec = make_codec(s["codec"], self.codec_threads, s)
        # Every window length the streaming schedule can produce (first, steady, all final
        # sizes): a GPU codec compiles kernels per input length, which must not land in a
        # measured request. Cheap for the ONNX codec, which warms a few shapes regardless.
        p = self.stream_params
        context = p.lookback + p.overlap
        lengths = {context + c + p.lookforward + p.overlap for c in (p.first_chunk_frames, p.frames_per_chunk)}
        lengths |= set(range(context + 1, context + max(p.first_chunk_frames, p.frames_per_chunk) + p.lookforward + 1))
        self.codec.warmup(sorted(lengths) if s.get("codec_backend") == "torch" else (30, 82, 200))
        self.watermark = Watermarker(s.get("watermark", "slice"), threads=resolve_threads(threads.get("watermark"), self.codec_threads or 1))
        self.resolved = {
            "gguf": str(self.gguf_path),
            "n_threads": n_threads,
            "n_threads_by_mode": self._threads_for,
            "n_threads_batch": n_threads_batch,
            "n_gpu_layers": int(s.get("n_gpu_layers", 0)),
            "flash_attn": bool(s.get("flash_attn", False)),
            "prefill_gpu": self.llm_prefill is not None,
            "vocab": self.vocab_mode,
            "n_vocab_scored": self.llm.n_vocab(),
            "codec": self.codec.settings,
            "watermark": self.watermark.mode,
            "pipelined": self.pipelined,
            "stream": self.stream_params.__dict__,
            "reuse_prefix": self.reuse_prefix,
            "sampling": self.sampling,
            "speech_ids_contiguous": self._speech_offset is not None,
        }

    # ------------------------------------------------------------------ tokens
    def _single_token(self, text: str) -> int:
        ids = self.tok.tokenize(text.encode(), add_bos=False, special=True)
        if len(ids) != 1:
            raise ValueError(f"{text} is not a single token in this GGUF (got {ids})")
        return ids[0]

    def _map_speech_tokens(self) -> None:
        first, last = self._single_token("<|speech_0|>"), self._single_token(f"<|speech_{N_CODES - 1}|>")
        probes = (1, 1234, 40_000)
        if last - first == N_CODES - 1 and all(self._single_token(f"<|speech_{p}|>") == first + p for p in probes):
            self._speech_offset, self._code_to_id, self._id_to_code = first, None, None
        else:  # fall back to explicit tables (correct for any vocabulary layout)
            self._speech_offset = None
            self._code_to_id = np.array([self._single_token(f"<|speech_{c}|>") for c in range(N_CODES)], np.int64)
            self._id_to_code = {int(t): c for c, t in enumerate(self._code_to_id)}

    def _codes_to_ids(self, codes: np.ndarray) -> list[int]:
        if self._speech_offset is not None:
            return (codes.astype(np.int64) + self._speech_offset).tolist()
        return self._code_to_id[codes].tolist()

    def _id_to_speech_code(self, token: int) -> int | None:
        if self._speech_offset is not None:
            c = token - self._speech_offset
            return c if 0 <= c < N_CODES else None
        return self._id_to_code.get(token)

    def build_prompt(self, req: Request, timer: StageTimer) -> list[int]:
        """Token-identical to llama-cpp-python's tokenization of upstream's prompt string."""
        with timer.stage("phonemize"):
            ref_ph = self.phonemizer.reference_phones(req.voice.text)
            in_ph = self.phonemizer.to_phones(req.text)
        with timer.stage("prompt"):
            head = f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_ph}"
            text = f"{head} {in_ph}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>"
            tok = self.tok
            bos = [tok.token_bos()] if tok._model.add_bos_token() and tok.token_bos() != -1 else []
            eos = [tok.token_eos()] if tok._model.add_eos_token() else []
            ids = bos + tok.tokenize(text.encode("utf-8"), add_bos=False, special=True)
            ids += self._codes_to_ids(req.voice.codes) + eos
            # The part of the prompt that is identical for every request with this voice.
            # Computed by tokenizing the head alone and keeping only the tokens that also
            # start the full prompt, so a tokenizer merge across the boundary can't leak.
            key = (req.voice.name, req.voice.text)
            if key not in self._voice_prefix:
                head_ids = bos + tok.tokenize(head.encode("utf-8"), add_bos=False, special=True)
                n = 0
                while n < min(len(head_ids), len(ids)) and head_ids[n] == ids[n]:
                    n += 1
                self._voice_prefix[key] = max(n - 1, 0)  # drop the last token: it may merge
            self._prefix_len = self._voice_prefix[key]
        return ids

    # ------------------------------------------------------------------ generation
    def _max_new(self, req: Request, n_prompt: int) -> tuple[int, str]:
        room = self.llm.n_ctx() - n_prompt
        if room <= 0:
            raise ValueError(f"prompt of {n_prompt} tokens does not fit the {self.llm.n_ctx()}-token context")
        if self.max_tokens_per_char:
            cap = max(50, math.ceil(len(req.text) * float(self.max_tokens_per_char)))
            if cap < room:
                return cap, "max_tokens"
        return room, "context"

    def _iter_codes(self, prompt: list[int], req: Request, timer: StageTimer) -> Iterator[int]:
        """Yield speech codes as the backbone produces them.

        Time is only counted while this generator is running (not while the consumer
        decodes audio between pulls), so ``sampling`` = time inside generation that isn't
        llama.cpp prefill/decode, in both sequential and pipelined modes.
        After exhaustion, ``self._last_stop`` and ``self._last_perf`` are set.
        """
        llm, lc = self.llm, self._llama_cpp
        max_new, limit_reason = self._max_new(req, len(prompt))
        busy, t_resume = 0.0, now()
        if not self.reuse_prefix and self.vocab_mode == "full":
            llm.reset()  # full re-prefill, exactly as upstream
        lc.llama_perf_context_reset(llm._ctx.ctx)
        llm.set_seed(int(req.seed))
        sp, end_id, vocab = self.sampling, self._special["SPEECH_GENERATION_END"], self.tok._model.vocab
        self._last_stop, n = limit_reason, 0
        if self.vocab_mode == "speech":
            gen = self._speech_generate(prompt, int(req.seed))
        else:
            if self.reuse_prefix:
                # Read the new part of the prompt ourselves, chunked exactly where a full prefill
                # would split it; generate() then finds the whole prompt cached and samples at once.
                # (llama-cpp-python would chunk from the reuse point instead, which changes rounding.)
                keep = self._reusable_prefix(llm.input_ids[: llm.n_tokens].tolist(), prompt)
                llm.n_tokens = keep
                for start, end in self._aligned_chunks(keep, len(prompt), llm.n_batch):
                    llm.eval(prompt[start:end])
            gen = llm.generate(
                prompt, top_k=sp["top_k"], top_p=sp["top_p"], min_p=sp["min_p"], typical_p=1.0,
                temp=sp["temperature"], repeat_penalty=sp["repeat_penalty"], frequency_penalty=0.0,
                presence_penalty=0.0, tfs_z=1.0, mirostat_mode=0,
            )
        try:
            for token in gen:
                if token == end_id or lc.llama_vocab_is_eog(vocab, token):
                    self._last_stop = "eos"
                    break
                n += 1
                code = self._id_to_speech_code(token)
                if code is not None:  # upstream's regex also skips non-speech tokens
                    timer.mark_first_token()
                    busy += now() - t_resume
                    yield code
                    t_resume = now()
                if n >= max_new:
                    break
        finally:
            gen.close()
            if self.reuse_prefix:
                # Keep only this voice's shared prefix in the KV cache. Anything longer
                # (the input text, the reference codes after it) would let a repeated
                # benchmark sentence skip prefill, which real traffic never does.
                llm.n_tokens = min(llm.n_tokens, self._prefix_len)
                if self.vocab_mode == "speech":
                    self._kv = self._kv[: self._prefix_len]
                    if self.llm_prefill is not None:
                        self._kv_prefill = self._kv_prefill[: self._prefix_len]
            busy += now() - t_resume
            perf = lc.llama_perf_context(llm._ctx.ctx)
            prefill_ms, prefill_tokens = perf.t_p_eval_ms, int(perf.n_p_eval)
            if self.llm_prefill is not None:  # GPU prompt read + KV hand-over, wall time
                # the CPU context read no prompt (llama.cpp reports n_p_eval as at least 1)
                prefill_ms, prefill_tokens = self._gpu_prefill_ms, self._last_prefill_tokens
            timer.add("prefill", prefill_ms)
            timer.add("decode", perf.t_eval_ms)
            timer.add("sampling", max(busy * 1000.0 - prefill_ms - perf.t_eval_ms, 0.0))
            self._last_perf = {"prefill_tokens": prefill_tokens, "decode_tokens": int(perf.n_eval)}

    def _speech_generate(self, prompt: list[int], seed: int) -> Iterator[int]:
        """``vocab: speech`` generation. Mirrors llama-cpp-python's ``generate`` step for step
        (prefill in ``n_batch`` chunks, the same sampler chain, sample then decode) except that
        the prompt goes in as embeddings from the original table, and yields token ids in the
        *original* vocabulary so the caller can't tell the difference."""
        lc, llm, ctx = self._llama_cpp, self.llm, self.llm._ctx
        offset = self._vocab_offset  # original id = pruned id + offset
        pf = self.llm_prefill
        self._gpu_prefill_ms, t0 = 0.0, now()
        # The mirror of what the KV cache holds is cleared while the cache is being changed,
        # so a prefill that fails part-way can never be mistaken for a reusable prefix.
        if pf is None:
            cached, self._kv = self._kv, []
            self._kv = self._prefill_embd(llm, cached, prompt)
        else:  # wall time, including waiting for the GPU (llama.cpp's counters may not)
            cached, self._kv_prefill = self._kv_prefill, []
            self._kv_prefill = self._prefill_embd(pf, cached, prompt)
            lc.llama_synchronize(pf._ctx.ctx)
            self._gpu_prefill_ms = (now() - t0) * 1000.0

        sp = self.sampling
        sampler = lc._internals.LlamaSampler()  # the chain llama-cpp-python builds for these settings
        sampler.add_penalties(llm.n_vocab(), llm.last_n_tokens_size, sp["repeat_penalty"], 0.0, 0.0)
        sampler.add_top_k(sp["top_k"])
        sampler.add_typical(1.0, 1)
        sampler.add_top_p(sp["top_p"], 1)
        sampler.add_min_p(sp["min_p"], 1)
        sampler.add_temp(sp["temperature"])
        sampler.add_dist(seed)
        n_past = len(prompt)
        try:
            token = sampler.sample(pf._ctx if pf is not None else ctx, -1)
            yield token + offset
            if pf is not None:  # hand the prompt's KV cache to the CPU context
                t0 = now()
                self._copy_kv(pf._ctx.ctx, ctx)
                self._gpu_prefill_ms += (now() - t0) * 1000.0
            while True:
                self._tok_batch.set_batch([token], n_past, False)
                ctx.decode(self._tok_batch)
                n_past += 1
                token = sampler.sample(ctx, -1)
                yield token + offset
        finally:
            sampler.close()

    def _prefill_embd(self, llm, kv: list[int], prompt: list[int]) -> list[int]:
        """Read ``prompt`` (original-vocabulary ids) into ``llm`` as embeddings, reusing the
        cached prefix shared with ``kv`` (what the cache holds) when ``reuse_prefix`` is on.
        Chunked by ``n_batch`` from the first new token, as llama-cpp-python's ``eval``."""
        import ctypes

        lc, ctx = self._llama_cpp, llm._ctx
        keep = self._reusable_prefix(kv, prompt)
        ctx.kv_cache_seq_rm(-1, keep, -1)
        b = self._embd_batch
        self._last_prefill_tokens = len(prompt) - keep
        for start, end in self._aligned_chunks(keep, len(prompt), llm.n_batch):
            chunk = prompt[start:end]
            rows = self.embd.rows(chunk)
            b.n_tokens = len(chunk)
            ctypes.memmove(b.embd, rows.ctypes.data, rows.nbytes)
            for j in range(len(chunk)):
                b.pos[j], b.n_seq_id[j], b.logits[j] = start + j, 1, 0
                b.seq_id[j][0] = 0
            b.logits[len(chunk) - 1] = 1  # as llama-cpp-python's eval: logits for each chunk's last token
            if lc.llama_decode(ctx.ctx, b) != 0:
                raise RuntimeError("llama_decode failed on the prompt")
        return list(prompt)

    PREFIX_ALIGN = 8  # llama.cpp's repacked CPU kernels process rows in groups of 4 (NEON) / 8 (AVX2)

    def _reusable_prefix(self, cached: list[int], prompt: list[int]) -> int:
        """How much of ``prompt`` can be served from the KV cache holding ``cached``: the shared
        per-voice prefix, rounded down to a multiple of PREFIX_ALIGN so every remaining prompt row
        lands in the same kernel row-group as in a full prefill (bit-identical results)."""
        if not self.reuse_prefix:
            return 0
        keep, limit = 0, min(len(cached), self._prefix_len, len(prompt) - 1)
        while keep < limit and cached[keep] == prompt[keep]:
            keep += 1
        return keep - keep % self.PREFIX_ALIGN

    @staticmethod
    def _aligned_chunks(start: int, end: int, n_batch: int):
        """Prompt chunks [a, b) whose boundaries sit at multiples of ``n_batch``, i.e. exactly where
        a full prefill from position 0 (llama-cpp-python's eval) splits the prompt."""
        while start < end:
            stop = min((start // n_batch + 1) * n_batch, end)
            yield start, stop
            start = stop

    def _copy_kv(self, src_ctx, dst) -> None:
        """Copy sequence 0's KV cache from ``src_ctx`` into the context ``dst``."""
        import ctypes

        lc = self._llama_cpp
        n = lc.llama_state_seq_get_size(src_ctx, 0)
        if self._state_buf is None or len(self._state_buf) < n:
            self._state_buf = (ctypes.c_uint8 * (n + n // 4))()  # reused: allocation is not free
        got = lc.llama_state_seq_get_data(src_ctx, self._state_buf, n, 0)
        dst.kv_cache_seq_rm(-1, 0, -1)
        if not got or lc.llama_state_seq_set_data(dst.ctx, self._state_buf, got, 0) != got:
            raise RuntimeError("KV cache transfer from the prefill context failed")

    # ------------------------------------------------------------------ batch
    def _use_threads(self, mode: str) -> None:
        llm = self.llm
        n = self._threads_for[mode]
        if llm.context_params.n_threads != n:
            self._llama_cpp.llama_set_n_threads(llm._ctx.ctx, n, llm.context_params.n_threads_batch)
            llm.context_params.n_threads = n

    def synthesize(self, req: Request, timer: StageTimer) -> Result:
        self._use_threads("batch")
        prompt = self.build_prompt(req, timer)
        codes = list(self._iter_codes(prompt, req, timer))
        if not codes:
            raise RuntimeError("backbone produced no speech tokens")
        with timer.stage("codec"):
            audio = self.codec.decode(codes)
        if self.watermark.mode != "off":
            with timer.stage("watermark"):
                audio = self.watermark.apply(audio)
        timer.mark_chunk(len(audio))
        return Result(audio, np.asarray(codes, np.int32), len(prompt), self._last_stop, timer, extra=dict(self._last_perf))

    # ------------------------------------------------------------------ streaming
    def stream(self, req: Request, timer: StageTimer) -> Iterator[np.ndarray]:
        self._use_threads("stream")
        prompt = self.build_prompt(req, timer)
        ref = req.voice.codes.astype(np.int32)
        cache: list[int] = [int(c) for c in ref]  # decoder context starts with the reference
        planner, ola = ChunkPlanner(self.stream_params, len(ref)), IncrementalOLA()
        info = {"prompt_tokens": len(prompt), "stop_reason": "eos", "extra": {}}
        self._last_stream_info = info

        def run_window(w) -> np.ndarray:
            with timer.stage("codec"):
                audio = self.codec.decode(cache[w.token_start : w.token_end])
            if self.watermark.mode == "window":
                with timer.stage("watermark"):
                    audio = self.watermark.apply(audio)
            audio = audio[w.sample_start : w.sample_end]
            if self.watermark.mode == "slice":
                with timer.stage("watermark"):
                    audio = self.watermark.apply(audio)
            ola.add(audio, w.out_offset)
            return ola.emit(w.emit_until)

        codes = self._pipelined_codes(prompt, req, timer) if self.pipelined else self._iter_codes(prompt, req, timer)
        for code in codes:
            cache.append(code)
            while planner.ready(len(cache)):
                yield run_window(planner.next_window())
        final = planner.final_window(len(cache))
        if final is not None:
            yield run_window(final)
        self._last_stream_tokens = np.asarray(cache[len(ref):], np.int32)
        info["stop_reason"] = self._last_stop
        info["extra"] = dict(self._last_perf)

    def _pipelined_codes(self, prompt: list[int], req: Request, timer: StageTimer) -> Iterator[int]:
        """Run generation on a background thread so audio decoding (codec + watermark) on
        the caller's thread overlaps with the next tokens being generated. llama.cpp and
        ONNX Runtime both release the GIL inside native calls, so this is real parallelism."""
        q: queue.SimpleQueue = queue.SimpleQueue()
        errors: list[BaseException] = []

        def produce() -> None:
            try:
                for code in self._iter_codes(prompt, req, timer):
                    q.put(code)
            except BaseException as e:  # noqa: BLE001 - re-raised on the caller's thread
                errors.append(e)
            finally:
                q.put(_DONE)

        worker = threading.Thread(target=produce, name="ttsbench-backbone", daemon=True)
        worker.start()
        try:
            while (item := q.get()) is not _DONE:
                yield item
        finally:
            worker.join()
        if errors:
            raise errors[0]

    def decode_full(self, tokens: np.ndarray) -> np.ndarray:
        return self.codec.decode(tokens)

    def describe(self) -> dict:
        return {"engine": self.name, **self.settings, "resolved": getattr(self, "resolved", {})}

    def close(self) -> None:
        if getattr(self, "_embd_batch", None) is not None:
            self._llama_cpp.llama_batch_free(self._embd_batch)
            self._embd_batch = None
        models = (getattr(self, "tok", None), getattr(self, "llm", None), getattr(self, "llm_prefill", None))
        for m in {id(x): x for x in models if x is not None}.values():
            m.close()
