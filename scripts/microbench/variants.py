import sys
import numpy as np
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import ttsbench.engines.fast as fast
from ttsbench.engines.base import load_voice, Request
from ttsbench.engines.components import CACHE_DIR

class DummyCodec:
    def __init__(self, *a, **k): self.settings = {}
    def warmup(self): pass
    def decode(self, codes): return np.zeros(len(codes) * 480, np.float32)
fast.OnnxCodec = DummyCodec
fast.Watermarker = lambda mode, threads=None: type("W", (), {"mode": "off", "apply": lambda s, a: a})()
gguf = sys.argv[1]
base = {"backbone": gguf, "codec": "x", "threads": {"backbone": 4, "backbone_batch": 4}, "reuse_prefix": True}
V = {
  "full_cpu":        {"vocab": "full"},
  "speech_cpu":      {"vocab": "speech"},
  "speech_cpu_fa":   {"vocab": "speech", "flash_attn": True},
  "speech_gpuPF_fa": {"vocab": "speech", "flash_attn": True, "prefill_gpu": True},
  "speech_gpuPF":    {"vocab": "speech", "prefill_gpu": True},
}
E = {}
for k, v in V.items():
    e = fast.FastEngine({**base, **v}); e.load(); E[k] = e
voices = [load_voice(v, CACHE_DIR / "voices") for v in ("jo", "dave")]
texts = ["The quick brown fox jumps over the lazy dog.", "On the third of May, Doctor Lee paid five dollars for coffee."]
for e in E.values():  # warm
    e.run(Request(texts[0], voices[0], 0, False))
stats = {k: {"pf": [], "dec": [], "first": [], "same": 0, "n": 0} for k in E}
for v in voices:
    for t in texts:
        for seed in (0, 1, 2):
            ref = None
            for k, e in E.items():
                r = e.run(Request(t, v, seed, False))
                if ref is None: ref = r.tokens
                s = stats[k]; s["n"] += 1; s["same"] += np.array_equal(r.tokens, ref)
                s["pf"].append(r.timer.ms["prefill"]); s["dec"].append(r.timer.ms["decode"] / max(r.extra["decode_tokens"], 1))
                s["first"].append(r.timer.first_token_s * 1000)
for k, s in stats.items():
    print(f"{k:16s} prefill {np.median(s['pf']):6.1f} ms | first token {np.median(s['first']):6.1f} ms | decode {np.median(s['dec']):.2f} ms/tok | same tokens as full_cpu {s['same']}/{s['n']}")
