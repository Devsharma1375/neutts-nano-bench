"""Generation speed (pruned speech vocab, 900 tokens of context, 100 tokens) at n_threads 3/4/6/10;
15 s idle before each measurement, rotated order."""
import time, numpy as np, llama_cpp, sys, glob, os
from llama_cpp import _internals
P = sorted(glob.glob(os.path.expanduser("~/.cache/ttsbench/pruned/*.gguf")), key=os.path.getmtime)[-1]
models = {nt: llama_cpp.Llama(model_path=P, n_ctx=2048, n_batch=512, n_ubatch=512, n_threads=nt, n_threads_batch=10, verbose=False, use_mlock=True) for nt in (3, 4, 6, 10)}
rng = np.random.default_rng(0); prompt = rng.integers(8, 65544, 900).tolist()
b = _internals.LlamaBatch(n_tokens=512, embd=0, n_seq_max=1); b1 = _internals.LlamaBatch(n_tokens=1, embd=0, n_seq_max=1)
def run(llm):
    llm._ctx.kv_cache_seq_rm(-1, 0, -1)
    for i in range(0, 900, 512): b.set_batch(prompt[i:i+512], i, False); llm._ctx.decode(b)
    t = time.perf_counter()
    for j in range(100):
        b1.set_batch([int(rng.integers(8, 65544))], 900 + j, False); llm._ctx.decode(b1)
    return (time.perf_counter() - t) / 100
for m in models.values(): run(m)
res = {k: [] for k in models}; keys = list(models)
for c in range(4):
    order = keys[c:] + keys[:c]
    for k in (order if c % 2 == 0 else order[::-1]):
        time.sleep(15); res[k].append(run(models[k]))
for k, v in res.items():
    print(f"n_threads={k:2d}: {1000*np.median(v):.2f} ms/token ({1/np.median(v):.0f} tok/s)  all {[round(1000*x, 2) for x in v]}")
