import sys, time, numpy as np, torch
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from ttsbench.engines.components import OnnxCodec, CACHE_DIR
from ttsbench.engines.base import load_voice
v = load_voice("jo", CACHE_DIR / "voices"); codes = v.codes
win, batch = codes[100:182], codes[:250]
cpu = OnnxCodec("neuphonic/neucodec-onnx-decoder", threads=4)
ref_w, ref_b = cpu.decode(win), cpu.decode(batch)
def snr(r, y): n=min(len(r),len(y)); return 10*np.log10(np.sum(r[:n]**2)/max(np.sum((r[:n]-y[:n])**2),1e-20))
# CoreML legacy NeuralNetwork format / static shapes
for fmt, static in (("NeuralNetwork", "0"), ("MLProgram", "1")):
    try:
        c = OnnxCodec("neuphonic/neucodec-onnx-decoder", threads=4, providers=[["CoreMLExecutionProvider", {"ModelFormat": fmt, "MLComputeUnits": "ALL", "RequireStaticInputShapes": static}], "CPUExecutionProvider"])
        c.decode(win); t=time.perf_counter(); y=c.decode(win); print(f"coreml {fmt} static={static}: window {1000*(time.perf_counter()-t):.1f} ms SNR {snr(ref_w,y):.1f}")
    except Exception as e: print(f"coreml {fmt} static={static}: FAILED {str(e)[:120]}")
from neucodec import NeuCodec
m = NeuCodec.from_pretrained("neuphonic/neucodec").eval()
print("torch codec params (M):", sum(p.numel() for p in m.parameters())/1e6)
for dev in ("cpu", "mps"):
    mm = m.to(dev)
    for name, x, ref in (("window", win, ref_w), ("batch", batch, ref_b)):
        t_in = torch.tensor(np.asarray(x), dtype=torch.long)[None, None, :].to(dev)
        with torch.no_grad():
            for _ in range(2): y = mm.decode_code(t_in)
            if dev == "mps": torch.mps.synchronize()
            ts = []
            for _ in range(5):
                t = time.perf_counter(); y = mm.decode_code(t_in)
                if dev == "mps": torch.mps.synchronize()
                ts.append(time.perf_counter() - t)
        y = y.float().cpu().numpy()[0, 0]
        print(f"torch {dev} {name}: {1000*np.median(ts):.1f} ms, {len(y)} samples, SNR vs onnx {snr(ref, y):.1f} dB", flush=True)
