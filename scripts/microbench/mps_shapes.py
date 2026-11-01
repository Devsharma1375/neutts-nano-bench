import sys, time, numpy as np
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from ttsbench.engines.components import TorchCodec, OnnxCodec
rng = np.random.default_rng(0)
c = TorchCodec(device="mps"); c.warmup()
o4 = OnnxCodec("neuphonic/neucodec-onnx-decoder", threads=4); o4.warmup()
for n in (70, 71, 72, 70, 150, 151, 150, 300, 301, 300):
    x = rng.integers(0, 65536, n)
    t = time.perf_counter(); c.decode(x); tm = time.perf_counter() - t
    t = time.perf_counter(); o4.decode(x); to = time.perf_counter() - t
    print(f"n={n}: mps {1000*tm:6.1f} ms | onnx-cpu4 {1000*to:6.1f} ms")
