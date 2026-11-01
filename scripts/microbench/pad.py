import sys, numpy as np
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from ttsbench.engines.components import OnnxCodec, CACHE_DIR
from ttsbench.engines.base import load_voice
o = OnnxCodec("neuphonic/neucodec-onnx-decoder", threads=4)
for v in ("jo", "dave"):
    codes = load_voice(v, CACHE_DIR / "voices").codes
    for n in (137, 251):
        x = codes[:n]; ref = o.decode(x)
        for pad in (8, 23, 60):
            for how in ("last", "edge_reflect"):
                p = np.concatenate([x, np.full(pad, x[-1])]) if how == "last" else np.concatenate([x, x[::-1][:pad]])
                y = o.decode(p)[: len(ref)]
                err = ref - y
                snr = 10*np.log10(np.sum(ref**2)/np.sum(err**2))
                tail = 10*np.log10(np.sum(ref[-4800:]**2)/max(np.sum(err[-4800:]**2),1e-20))
                body = 10*np.log10(np.sum(ref[:-4800]**2)/max(np.sum(err[:-4800]**2),1e-20))
                print(f"{v} n={n} pad={pad:2d} {how:12s} SNR all {snr:5.1f} dB | all but last 0.2s {body:6.1f} dB | last 0.2s {tail:5.1f} dB")
