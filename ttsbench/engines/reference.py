"""Upstream NeuTTS, unmodified, with stage timing attached from the outside.

This is the baseline every optimisation is measured against, so it must behave exactly
like ``pip install neutts``. Instrumentation wraps instance attributes only:

* ``_to_phones``                    -> phonemize
* ``_ggml_prompt`` / ``_apply_chat_template`` -> prompt (excluding nested phonemize time)
* the llama.cpp backbone object     -> prefill / decode (llama.cpp perf counters) and
                                       sampling (remaining time inside the backbone call)
* PyTorch backbone forward passes   -> first pass = prefill, later passes = decode
* ``_decode``                       -> codec
* ``watermarker.apply_watermark``   -> watermark

Optional, clearly named deviations (each its own experiment): ``torch_dtype: float32``
(upstream loads bfloat16, which is slow on CPUs without native BF16) and ``torch_threads``.
``backbone_device`` is not a deviation: it is upstream's own argument (``"mps"`` puts a
GGUF backbone on Metal exactly as ``NeuTTS(backbone_device="mps")`` would).
"""
from __future__ import annotations

import contextlib
import io
import re
from collections.abc import Iterator

import numpy as np

from ..timing import StageTimer, now
from .base import Engine, Request, Result

_SPEECH = re.compile(r"<\|speech_(\d+)\|>")


class _Timed:
    """Holds the timer of the request in flight; wrappers write into it."""

    timer: StageTimer | None = None


class _LlamaProxy:
    """Wraps llama_cpp.Llama so NeuTTS's calls are timed and their output captured."""

    def __init__(self, llm, owner: ReferenceEngine) -> None:
        self._llm, self._owner = llm, owner

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def __call__(self, prompt, **kwargs):
        import llama_cpp

        owner, llm = self._owner, self._llm
        llama_cpp.llama_perf_context_reset(llm._ctx.ctx)
        owner._gen_text = []

        def finish(busy_s: float) -> None:
            perf = llama_cpp.llama_perf_context(llm._ctx.ctx)
            t = owner._t.timer
            t.add("prefill", perf.t_p_eval_ms)
            t.add("decode", perf.t_eval_ms)
            t.add("sampling", max(busy_s * 1000 - perf.t_p_eval_ms - perf.t_eval_ms, 0.0))
            owner._prompt_tokens = int(perf.n_p_eval)
            owner._perf = {"prefill_tokens": int(perf.n_p_eval), "decode_tokens": int(perf.n_eval)}

        if not kwargs.get("stream"):
            t0 = now()
            out = llm(prompt, **kwargs)
            finish(now() - t0)
            owner._finish = out["choices"][0].get("finish_reason")
            owner._gen_text = [out["choices"][0]["text"]]
            return out

        def gen():
            busy, it = 0.0, llm(prompt, **kwargs)
            try:
                while True:
                    t0 = now()
                    try:
                        item = next(it)
                    except StopIteration:
                        busy += now() - t0
                        break
                    busy += now() - t0
                    owner._t.timer.mark_first_token()
                    owner._gen_text.append(item["choices"][0]["text"])
                    owner._finish = item["choices"][0].get("finish_reason") or owner._finish
                    yield item
            finally:
                finish(busy)

        owner._finish = None
        return gen()


class ReferenceEngine(Engine):
    name = "reference"

    def load(self) -> None:
        from neutts import NeuTTS

        s = self.settings
        device = s.get("backbone_device", "cpu")
        if device != "cpu" and s["backbone"].lower().endswith("gguf"):
            import llama_cpp

            if not llama_cpp.llama_supports_gpu_offload():  # upstream would silently stay on the CPU
                raise RuntimeError(f"backbone_device={device!r} but this llama.cpp build has no GPU backend")
        with contextlib.redirect_stdout(io.StringIO()):
            self.tts = NeuTTS(
                backbone_repo=s["backbone"],
                # upstream's own GPU path for GGUF: any device but "cpu" offloads every
                # layer to Metal/CUDA and turns on flash attention
                backbone_device=s.get("backbone_device", "cpu"),
                codec_repo=s["codec"],
                codec_device=s.get("codec_device", "cpu"),  # upstream's argument; ONNX codecs are CPU-only
                language=s.get("language"),
                seed=0,
            )
        self._t = _Timed()
        tts = self.tts
        self.is_gguf = tts._is_quantized_model
        self.supports_stream = self.is_gguf
        if not self.is_gguf:
            import torch

            if s.get("torch_threads"):
                torch.set_num_threads(int(s["torch_threads"]))
            if s.get("torch_dtype"):
                tts.backbone.to(getattr(torch, s["torch_dtype"]))
            self._install_torch_hooks()
        else:
            tts.backbone = _LlamaProxy(tts.backbone, self)
        self._wrap("_to_phones", "phonemize")
        self._wrap_prompt("_ggml_prompt" if self.is_gguf else "_apply_chat_template")
        self._wrap("_decode", "codec")
        if tts.watermarker is not None:
            wm, timed = tts.watermarker, self._t

            class _WM:
                def apply_watermark(self, wav, sample_rate):
                    with timed.timer.stage("watermark"):
                        return wm.apply_watermark(wav, sample_rate=sample_rate)

                def __getattr__(self, name):
                    return getattr(wm, name)

            tts.watermarker = _WM()
        self.resolved = {
            "is_gguf": self.is_gguf,
            "onnx_codec": tts._is_onnx_codec,
            "watermark": tts.watermarker is not None,
            "upstream_streaming": {
                "frames_per_chunk": tts.streaming_frames_per_chunk,
                "lookforward": tts.streaming_lookforward,
                "lookback": tts.streaming_lookback,
                "overlap": tts.streaming_overlap_frames,
            },
        }

    # ------------------------------------------------------------------ instrumentation
    def _wrap(self, attr: str, stage: str) -> None:
        fn, timed = getattr(self.tts, attr), self._t

        def wrapped(*a, **k):
            with timed.timer.stage(stage):
                return fn(*a, **k)

        setattr(self.tts, attr, wrapped)

    def _wrap_prompt(self, attr: str) -> None:
        fn, timed = getattr(self.tts, attr), self._t

        def wrapped(*a, **k):
            t = timed.timer
            before, t0 = t.ms["phonemize"], now()
            out = fn(*a, **k)
            t.add("prompt", max((now() - t0) * 1000 - (t.ms["phonemize"] - before), 0.0))
            return out

        setattr(self.tts, attr, wrapped)

    def _install_torch_hooks(self) -> None:
        import torch

        model, owner = self.tts.backbone, self
        state = {"t": 0.0, "calls": 0}
        # GPU kernels run asynchronously: wait for them before reading the clock, or the hooks
        # would time only the kernel launches
        dev = next(model.parameters()).device.type
        sync = {"mps": torch.mps.synchronize, "cuda": torch.cuda.synchronize}.get(dev, lambda: None)

        def pre(module, args, kwargs):
            sync()
            state["t"] = now()
            if state["calls"] == 0:
                ids = kwargs.get("input_ids", args[0] if args else None)
                owner._prompt_tokens = int(ids.shape[-1]) if ids is not None else 0

        def post(module, args, kwargs, output):
            sync()
            ms = (now() - state["t"]) * 1000
            owner._t.timer.add("prefill" if state["calls"] == 0 else "decode", ms)
            if state["calls"] == 1:
                owner._t.timer.mark_first_token()
            state["calls"] += 1

        model.register_forward_pre_hook(pre, with_kwargs=True)
        model.register_forward_hook(post, with_kwargs=True)
        self._hook_state = state
        infer = self.tts._infer_torch

        def timed_infer(*a, **k):
            state["calls"] = 0
            t, before = self._t.timer, dict(self._t.timer.ms)
            t0 = now()
            out = infer(*a, **k)
            spent = (now() - t0) * 1000
            fwd = (t.ms["prefill"] - before["prefill"]) + (t.ms["decode"] - before["decode"])
            t.add("sampling", max(spent - fwd, 0.0))
            self._gen_text = [out]
            self._perf = {"prefill_tokens": self._prompt_tokens, "decode_tokens": max(state["calls"] - 1, 0)}
            return out

        self.tts._infer_torch = timed_infer

    # ------------------------------------------------------------------ requests
    def _tokens(self) -> np.ndarray:
        return np.asarray([int(x) for x in _SPEECH.findall("".join(getattr(self, "_gen_text", [])))], np.int32)

    def _stop_reason(self, n_prompt: int, n_gen: int) -> str:
        finish = getattr(self, "_finish", None)
        if finish == "stop":
            return "eos"
        if finish == "length" or n_prompt + n_gen >= self.tts.max_context - 1:
            return "context"
        return "eos"

    def synthesize(self, req: Request, timer: StageTimer) -> Result:
        self._t.timer, self.tts._seed, self._finish = timer, int(req.seed), None
        with contextlib.redirect_stdout(io.StringIO()):  # upstream prints the seed per call
            audio = self.tts.infer(req.text, req.voice.codes, req.voice.text)
        timer.mark_chunk(len(audio))
        tokens = self._tokens()
        return Result(
            np.asarray(audio, np.float32), tokens, self._prompt_tokens,
            self._stop_reason(self._prompt_tokens, len(tokens)), timer, extra=dict(getattr(self, "_perf", {})),
        )

    def stream(self, req: Request, timer: StageTimer) -> Iterator[np.ndarray]:
        if not self.supports_stream:
            raise NotImplementedError("upstream streaming requires a GGUF backbone")
        self._t.timer, self.tts._seed, self._finish = timer, int(req.seed), None
        # upstream prints the seed lazily inside its generator, so keep stdout redirected
        # for the whole iteration (the worker logs to stderr, so nothing else is lost)
        with contextlib.redirect_stdout(io.StringIO()):
            for chunk in self.tts.infer_stream(req.text, req.voice.codes, req.voice.text):
                yield np.asarray(chunk, np.float32)
        tokens = self._tokens()
        self._last_stream_tokens = tokens
        self._last_stream_info = {
            "prompt_tokens": self._prompt_tokens,
            "stop_reason": self._stop_reason(self._prompt_tokens, len(tokens)),
            "extra": dict(getattr(self, "_perf", {})),
        }

    def decode_full(self, tokens: np.ndarray) -> np.ndarray:
        timer, self._t.timer = self._t.timer, StageTimer()  # untimed helper
        try:
            return np.asarray(self.tts._decode("".join(f"<|speech_{int(t)}|>" for t in tokens)), np.float32)
        finally:
            self._t.timer = timer

    def describe(self) -> dict:
        return {"engine": self.name, **self.settings, "resolved": getattr(self, "resolved", {})}
