import sys, numpy as np, torch
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from ttsbench.engines.components import OnnxCodec, CACHE_DIR
from ttsbench.engines.base import load_voice
codes = load_voice("jo", CACHE_DIR / "voices").codes[:250]
o = OnnxCodec("neuphonic/neucodec-onnx-decoder", threads=4).decode(codes)
from neucodec import NeuCodec
m = NeuCodec.from_pretrained("neuphonic/neucodec").eval()
with torch.no_grad():
    t_cpu = m.decode_code(torch.tensor(codes, dtype=torch.long)[None, None]).numpy()[0, 0]
    t_mps = m.to("mps").decode_code(torch.tensor(codes, dtype=torch.long)[None, None].to("mps")).float().cpu().numpy()[0, 0]
def best(a, b):
    res = []
    for lag in range(-960, 961, 1):
        x = a[max(lag,0): len(a)+min(lag,0)]; y = b[max(-lag,0): len(b)+min(-lag,0)]; n = min(len(x), len(y)); x, y = x[:n], y[:n]
        res.append((10*np.log10(np.sum(y**2)/np.sum((x-y)**2)), lag))
    return max(res)
print("len onnx", len(o), "torch", len(t_cpu))
print("onnx vs torch-cpu best (SNR dB, lag samples):", best(t_cpu, o))
print("torch-cpu vs torch-mps:", best(t_cpu, t_mps))
