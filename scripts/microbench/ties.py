import sys, numpy as np, llama_cpp
ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from pathlib import Path
from ttsbench.engines import create_engine, Request
from ttsbench.engines.base import load_voice
from ttsbench.engines.components import CACHE_DIR
from ttsbench.corpus import load_corpus
from ttsbench.timing import StageTimer
text = {s.id: s.text for s in load_corpus(ROOT / "corpus" / "sentences.jsonl")}["l05"]
v = load_voice("dave", CACHE_DIR / "voices")
base = {"backbone": "neuphonic/neutts-nano-q4-gguf", "codec": "neuphonic/neucodec-onnx-decoder", "watermark": "off", "reuse_prefix": False}
full = create_engine("fast", base); full.load()
ref = full.run(Request(text, v, 0, False)).tokens
llm = full.llm
prompt = full.build_prompt(Request(text, v, 0, False), StageTimer())
llm.reset(); llm.eval(prompt)
ids = [int(c) + full._speech_offset for c in ref]
n_tie_steps = 0
for step in range(len(ids)):
    lg = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(llm._ctx.ctx, -1), shape=(llm.n_vocab(),)).copy()
    top = np.sort(lg)[-51:][::-1]  # top 50 + the 51st
    ties = len(top[:50]) - len(np.unique(top[:50])); boundary_tie = top[49] == top[50]
    if ties or boundary_tie:
        n_tie_steps += 1
        if step in range(275, 290) or n_tie_steps <= 3:
            print(f"step {step}: {ties} tied pairs inside the top-50, tie at the top-50 boundary: {boundary_tie}")
    llm.eval([ids[step]])
print(f"steps with exact logit ties in or at the edge of the top-50: {n_tie_steps} of {len(ids)}")
