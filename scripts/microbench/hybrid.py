import ctypes, time, sys
import numpy as np, llama_cpp
from llama_cpp import _internals
M = sys.argv[1]
fa = bool(int(sys.argv[2])) if len(sys.argv) > 2 else True
kw = dict(model_path=M, n_ctx=2048, n_batch=512, n_ubatch=512, flash_attn=fa, verbose=False, use_mlock=True)
cpu = llama_cpp.Llama(n_gpu_layers=0, n_threads=4, n_threads_batch=4, **kw)
gpu = llama_cpp.Llama(n_gpu_layers=-1, n_threads=4, n_threads_batch=4, **kw)
rng = np.random.default_rng(0)
prompt = rng.integers(1, 65536, 800).tolist()
def prefill(llm):
    llm._ctx.kv_cache_seq_rm(-1, 0, -1)
    b = _internals.LlamaBatch(n_tokens=512, embd=0, n_seq_max=1)
    for i in range(0, len(prompt), 512):
        b.set_batch(prompt[i:i+512], i, False); llm._ctx.decode(b)
for name, llm in (("cpu", cpu), ("gpu", gpu)):
    prefill(llm)  # warm
    ts = []
    for _ in range(5):
        t = time.perf_counter(); prefill(llm); llama_cpp.llama_synchronize(llm._ctx.ctx); ts.append(time.perf_counter() - t)
    print(f"{name} prefill 800 tok: {1000*np.median(ts):.1f} ms (fa={fa})")
# state transfer gpu -> cpu
ts = []
for _ in range(5):
    prefill(gpu)
    t = time.perf_counter()
    n = llama_cpp.llama_state_seq_get_size(gpu._ctx.ctx, 0)
    buf = (ctypes.c_uint8 * n)()
    got = llama_cpp.llama_state_seq_get_data(gpu._ctx.ctx, buf, n, 0)
    cpu._ctx.kv_cache_seq_rm(-1, 0, -1)
    put = llama_cpp.llama_state_seq_set_data(cpu._ctx.ctx, buf, n, 0)
    ts.append(time.perf_counter() - t)
print(f"state copy {n/1e6:.1f} MB: {1000*np.median(ts):.1f} ms (got {got}, put {put})")
# decode speed on cpu after transfer vs gpu
for name, llm in (("cpu", cpu), ("gpu", gpu)):
    if name == "gpu": prefill(gpu)
    b = _internals.LlamaBatch(n_tokens=1, embd=0, n_seq_max=1)
    t = time.perf_counter()
    for i in range(100):
        b.set_batch([int(rng.integers(1, 65536))], 800 + i, False); llm._ctx.decode(b)
        llama_cpp.llama_get_logits_ith(llm._ctx.ctx, -1)
    llama_cpp.llama_synchronize(llm._ctx.ctx)
    print(f"{name} decode @800 ctx: {10*(time.perf_counter()-t):.2f} ms/token")
