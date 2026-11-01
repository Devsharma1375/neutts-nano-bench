import sys, time, numpy as np
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from ttsbench.engines.components import OnnxCodec, CACHE_DIR
from ttsbench.engines.base import load_voice
v = load_voice("jo", CACHE_DIR / "voices")
codes = v.codes  # real speech tokens (653)
win = codes[100:182]; batch = codes[:250]
cpu = OnnxCodec("neuphonic/neucodec-onnx-decoder", threads=4)
ref_w, ref_b = cpu.decode(win), cpu.decode(batch)
def bench(c, x, n=7):
    c.decode(x); ts = []
    for _ in range(n):
        t = time.perf_counter(); y = c.decode(x); ts.append(time.perf_counter() - t)
    return 1000 * np.median(ts), y
def snr(ref, y):
    return 10*np.log10(np.sum(ref**2)/max(np.sum((ref-y)**2),1e-20))
configs = [
  ("cpu4", "neuphonic/neucodec-onnx-decoder", dict(threads=4)),
  ("cpu1", "neuphonic/neucodec-onnx-decoder", dict(threads=1)),
  ("cpu4_int8", "neuphonic/neucodec-onnx-decoder-int8", dict(threads=4)),
  ("cpu1_int8", "neuphonic/neucodec-onnx-decoder-int8", dict(threads=1)),
]
for units in sys.argv[1:]:
    configs.append((f"coreml_{units}", "neuphonic/neucodec-onnx-decoder", dict(threads=4, providers=[["CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": units, "RequireStaticInputShapes": "0", "ModelCacheDirectory": str(CACHE_DIR / "coreml")}], "CPUExecutionProvider"])))
for name, repo, kw in configs:
    t = time.perf_counter()
    try:
        c = OnnxCodec(repo, **kw)
    except Exception as e:
        print(name, "LOAD FAILED", type(e).__name__, str(e)[:200]); continue
    load = time.perf_counter() - t
    mw, yw = bench(c, win); mb, yb = bench(c, batch)
    print(f"{name:22s} load {load:5.1f}s | 82-tok window {mw:6.1f} ms (SNR vs cpu {snr(ref_w, yw):5.1f} dB) | 250-tok batch {mb:6.1f} ms (SNR {snr(ref_b, yb):5.1f} dB) | providers {c.session.get_providers()}", flush=True)
