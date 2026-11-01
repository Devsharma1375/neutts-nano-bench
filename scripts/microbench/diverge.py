"""Why does l05/dave/seed 0 differ? Compare upstream vs fast variants token by token, and check
whether a non-kept token ever enters the full-vocabulary top-50 along upstream's path."""
import sys, json, numpy as np, llama_cpp
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from ttsbench.engines import create_engine, Request
from ttsbench.engines.base import load_voice
from ttsbench.engines.components import CACHE_DIR
from ttsbench.corpus import load_corpus
from pathlib import Path
from ttsbench.timing import StageTimer
text = {s.id: s.text for s in load_corpus(ROOT / "corpus" / "sentences.jsonl")}["l05"]
v = load_voice("dave", CACHE_DIR / "voices")
B, C = "neuphonic/neutts-nano-q4-gguf", "neuphonic/neucodec-onnx-decoder"
base = {"backbone": B, "codec": C, "watermark": "off"}
specs = {
  "upstream": ("reference", {"backbone": B, "codec": C}),
  "full": ("fast", {**base, "reuse_prefix": False}),
  "full_reuse": ("fast", {**base, "reuse_prefix": True}),
  "speech": ("fast", {**base, "reuse_prefix": False, "vocab": "speech"}),
  "speech_reuse": ("fast", {**base, "reuse_prefix": True, "vocab": "speech"}),
}
out = {}
for k, (kind, cfg) in specs.items():
    e = create_engine(kind, cfg); e.load()
    if "reuse" in k:  # leave another dave request in the cache first, as in the benchmark
        e.run(Request("A different sentence to leave something in the cache.", v, 0, False))
    out[k] = e.run(Request(text, v, 0, False)).tokens
    e.close(); del e
variants = {"full": create_engine("fast", specs["full"][1]), "speech": create_engine("fast", specs["speech"][1])}
for e in variants.values(): e.load()
ref = out["upstream"]
for k, t in out.items():
    n = min(len(t), len(ref)); d = np.flatnonzero(t[:n] != ref[:n])
    print(f"{k:13s} {len(t)} tokens, first divergence at step {d[0] if d.size else ('none' if len(t)==len(ref) else n)}")
# walk upstream's token path with the full model; at each step check the top-50 for non-kept ids
full = variants["full"]; llm = full.llm
prompt = full.build_prompt(Request(text, v, 0, False), StageTimer())
llm.reset(); llm.eval(prompt)
lo = variants["speech"]._vocab_offset; hi = lo + variants["speech"].llm.n_vocab()
end_id = full._special["SPEECH_GENERATION_END"]
ids = [int(c) + full._speech_offset for c in ref] + [end_id]
bad = []
for step, tok in enumerate(ids):
    logits = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(llm._ctx.ctx, -1), shape=(llm.n_vocab(),))
    top = np.argpartition(-logits, 50)[:50]
    outside = [int(t) for t in top if not (lo <= t < hi)]
    if outside:
        bad.append((step, outside[:3], [float(logits[t]) for t in outside[:3]], float(np.sort(logits)[-50])))
    if step < len(ids) - 1:
        llm.eval([tok])
print(f"steps where a token outside the kept rows is in the full top-50: {len(bad)} of {len(ids)}", bad[:5])
