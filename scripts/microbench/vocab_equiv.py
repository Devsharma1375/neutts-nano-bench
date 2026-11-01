"""Backbone-only equivalence: full vocab vs speech vocab, same seeds -> same tokens?"""
import sys, time
import numpy as np
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import ttsbench.engines.fast as fast
from ttsbench.engines.base import load_voice, Request
from ttsbench.engines.components import CACHE_DIR
from ttsbench.timing import StageTimer

class DummyCodec:
    def __init__(self, *a, **k): self.settings = {}
    def warmup(self): pass
    def decode(self, codes): return np.zeros(len(codes) * 480, np.float32)
fast.OnnxCodec = DummyCodec
fast.Watermarker = lambda mode, threads=None: type("W", (), {"mode": "off", "apply": lambda s, a: a})()

gguf = sys.argv[1]
ngl = int(sys.argv[2]) if len(sys.argv) > 2 else 0
base = {"backbone": gguf, "codec": "x", "threads": {"backbone": 4, "backbone_batch": 4}, "n_gpu_layers": ngl}
engines = {}
for mode in ("full", "speech"):
    for reuse in (False, True):
        e = fast.FastEngine({**base, "vocab": mode, "reuse_prefix": reuse}); e.load(); engines[(mode, reuse)] = e
voices = [load_voice(v, CACHE_DIR / "voices") for v in ("jo", "dave")]
texts = ["The quick brown fox jumps over the lazy dog.",
         "On the third of May, Doctor Lee paid five dollars for coffee.",
         "Streaming speech should sound exactly like the full clip, from the first word to the last."]
same = total = 0
for v in voices:
    for t in texts:
        for seed in (0, 1, 2):
            out = {}
            for k, e in engines.items():
                r = e.run(Request(t, v, seed, False))
                st = r.timer.ms
                out[k] = (r.tokens, r.stop_reason, st["decode"], st["sampling"], r.extra.get("decode_tokens"))
            ref = out[("full", False)][0]
            row = []
            for k, (tok, stop, dec, smp, nd) in out.items():
                eq = np.array_equal(tok, ref)
                row.append(f"{k[0][:2]}{'R' if k[1] else '-'}:{'=' if eq else 'DIFF@'+str(int(np.argmax(tok[:min(len(tok),len(ref))] != ref[:min(len(tok),len(ref))]))) } {len(tok)}t dec {dec/max(nd or 1,1):.2f}ms/t smp {smp/max(nd or 1,1):.2f}")
            total += 1; same += all(np.array_equal(o[0], ref) for o in out.values())
            print(v.name, seed, t[:20], " | ".join(row), flush=True)
print(f"identical across all 4 variants: {same}/{total}")
