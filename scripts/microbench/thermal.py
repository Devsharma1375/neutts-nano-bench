"""Probe = fixed llama.cpp prefill of 256 tokens (CPU, 4 threads). Measure throttle onset under
continuous load, then recovery while idle."""
import time, numpy as np, llama_cpp, sys
from llama_cpp import _internals
llm = llama_cpp.Llama(model_path=sys.argv[1], n_ctx=512, n_batch=256, n_ubatch=256, n_threads=4, n_threads_batch=4, flash_attn=True, verbose=False)
toks = list(np.random.default_rng(0).integers(1000, 60000, 256))
b = _internals.LlamaBatch(n_tokens=256, embd=0, n_seq_max=1)
def probe():
    llm._ctx.kv_cache_seq_rm(-1, 0, -1); b.set_batch(toks, 0, False)
    t = time.perf_counter(); llm._ctx.decode(b); return 256 / (time.perf_counter() - t)
probe()
t0 = time.time()
print("== continuous load")
while time.time() - t0 < 90:
    vals = [probe() for _ in range(10)]
    print(f"t={time.time()-t0:5.1f}s  prefill {np.median(vals):6.0f} tok/s", flush=True)
print("== idle recovery (probe every 10 s)")
t1 = time.time()
while time.time() - t1 < 120:
    time.sleep(10)
    print(f"idle {time.time()-t1:5.1f}s  prefill {probe():6.0f} tok/s", flush=True)
