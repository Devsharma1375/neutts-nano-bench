"""Prefill of a real NeuTTS prompt (voice jo, ~780 tokens): token input (full vocab) vs
embedding input (speech vocab), n_threads_batch 4/6/10. 15 s idle before every measurement,
variant order rotated each cycle so heat can't favour one variant."""
import sys, time, ctypes, numpy as np, llama_cpp
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from ttsbench.engines.base import load_voice, Request
from ttsbench.engines.components import CACHE_DIR, resolve_model_file
from ttsbench.engines.vocab import ensure_pruned, EmbeddingTable
from ttsbench.timing import StageTimer
import ttsbench.engines.fast as fast
fast.make_codec = lambda *a, **k: type("C", (), {"warmup": lambda s, l=None: None, "settings": {}})()
fast.Watermarker = lambda mode, threads=None: type("W", (), {"mode": "off"})()
src = resolve_model_file("neuphonic/neutts-nano-q4-gguf", ".gguf")
eng = fast.FastEngine({"backbone": "neuphonic/neutts-nano-q4-gguf", "codec": "x", "vocab": "full", "reuse_prefix": False}); eng.load()
prompt = eng.build_prompt(Request("The meeting has been moved to Thursday at three, so please update your calendar and let the team know.", load_voice("jo", CACHE_DIR / "voices"), 0, False), StageTimer())
print("prompt tokens", len(prompt))
pruned = ensure_pruned(src, CACHE_DIR / "pruned"); table = EmbeddingTable(src)
def make(path, nt):
    return llama_cpp.Llama(model_path=str(path), n_ctx=2048, n_batch=512, n_ubatch=512, n_threads=4, n_threads_batch=nt, verbose=False, use_mlock=True)
models = {}
for nt in (4, 6, 10):
    models[("tokens", nt)] = make(src, nt); models[("embd", nt)] = make(pruned, nt)
eb = llama_cpp.llama_batch_init(512, table.n_embd, 1)
def run_tokens(llm):
    llm.reset(); t = time.perf_counter(); llm.eval(prompt); return time.perf_counter() - t
def run_embd(llm):
    llm._ctx.kv_cache_seq_rm(-1, 0, -1); t = time.perf_counter()
    for i in range(0, len(prompt), 512):
        ch = prompt[i:i+512]; rows = table.rows(ch); eb.n_tokens = len(ch)
        ctypes.memmove(eb.embd, rows.ctypes.data, rows.nbytes)
        for j in range(len(ch)):
            eb.pos[j], eb.n_seq_id[j], eb.logits[j] = i + j, 1, 0; eb.seq_id[j][0] = 0
        eb.logits[len(ch)-1] = 1; llama_cpp.llama_decode(llm._ctx.ctx, eb)
    return time.perf_counter() - t
keys = list(models); res = {k: [] for k in keys}
for k in keys: (run_tokens if k[0] == "tokens" else run_embd)(models[k])  # warm
for cycle in range(4):
    order = keys[cycle % len(keys):] + keys[:cycle % len(keys)]
    if cycle % 2: order = order[::-1]
    for k in order:
        time.sleep(15)
        res[k].append((run_tokens if k[0] == "tokens" else run_embd)(models[k]))
for k in keys:
    v = np.array(res[k]); print(f"{k[0]:6s} n_threads_batch={k[1]:2d}: median {1000*np.median(v):6.1f} ms ({len(prompt)/np.median(v):5.0f} tok/s)  all: {[round(1000*x) for x in v]}")
