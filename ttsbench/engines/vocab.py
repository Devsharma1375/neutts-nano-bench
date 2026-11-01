"""Speech-only vocabulary for the NeuTTS backbone (``vocab: speech``).

NeuTTS-Nano's vocabulary has 194,256 tokens, but while speaking the model only ever emits
the 65,536 speech tokens or ``<|SPEECH_GENERATION_END|>``. The output head is tied to the
embedding table, which is bigger than all 24 transformer layers together (113 MiB of the
185 MiB Q4_0 file), so most of the work per generated token is spent scoring text tokens
that can never be chosen.

``prune_gguf`` writes a copy of the backbone that keeps only those 65,537 rows plus the 7
special tokens just below them (65,544 = 8 x 8,193 rows, so llama.cpp runs the same repacked
kernel on the head as on the full matrix), byte for byte; every other tensor is unchanged. The prompt's text part has no rows in the
pruned model, so prompts are fed as *embeddings* looked up in the original table
(``EmbeddingTable``), dequantized exactly as llama.cpp's ``get_rows`` does. The network
therefore sees identical inputs and computes identical logits for every token it keeps;
top-k (k=50) runs before any other sampler, so as long as the 50 most likely tokens are
among them the sampled tokens are unchanged. ``ttsbench verify`` checks the tokens against
upstream and the logits bit for bit against the full-vocabulary model.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

END_TOKEN = "<|SPEECH_GENERATION_END|>"
N_CODES = 65_536
_REPLACED_KEYS = {
    "tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges", "tokenizer.ggml.scores",
    "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token", "tokenizer.chat_template",
}


def _tokens(reader) -> list[str]:
    field = reader.fields["tokenizer.ggml.tokens"]
    return [bytes(field.parts[i]).decode("utf-8") for i in field.data]


def speech_range(tokens: list[str]) -> tuple[int, int]:
    """Original ids ``[first, last]`` to keep: END followed by speech_0..speech_65535."""
    end = tokens.index(END_TOKEN)
    first = tokens.index("<|speech_0|>")
    n = 0
    while first + n < len(tokens) and tokens[first + n] == f"<|speech_{n}|>":
        n += 1
    if first != end + 1 or n != N_CODES:
        raise ValueError(f"unexpected vocabulary layout: END at {end}, speech_0 at {first}, {n} speech tokens")
    return end, first + n - 1


ROW_MULTIPLE = 8  # llama.cpp's CPU repacked GEMV kernels need rows % 4 (NEON) or % 8 (AVX2)


def prune_gguf(src: Path, dst: Path) -> Path:
    """Write ``dst``: ``src`` with the tied embedding/output matrix cut to the END + speech
    rows, extended downwards by up to 7 real rows (the special tokens just before END) so the
    row count is a multiple of 8. Without that, llama.cpp can't repack the head (65,537 rows)
    and falls back to a generic kernel that is slower *and* accumulates in a different order,
    so logits would no longer be bit-identical. New id = original id - ``neuphonic.vocab_offset``."""
    from gguf import GGUFReader, GGUFValueType, GGUFWriter

    r = GGUFReader(src)
    tokens = _tokens(r)
    end, hi = speech_range(tokens)
    lo = end - (-(hi + 1 - end)) % ROW_MULTIPLE
    if lo < 0:
        raise ValueError("not enough rows below END to pad the head to a multiple of 8")
    keep, n_keep = slice(lo, hi + 1), hi + 1 - lo
    arch = r.fields["general.architecture"].contents()

    tmp = dst.with_suffix(".part")
    w = GGUFWriter(tmp, arch=arch)
    for name, field in r.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture" or name in _REPLACED_KEYS:
            continue
        vtype = field.types[0]
        if name == f"{arch}.vocab_size":
            w.add_uint32(name, n_keep)
        elif vtype == GGUFValueType.ARRAY:
            w.add_array(name, field.contents())
        elif vtype == GGUFValueType.STRING:
            w.add_string(name, field.contents())
        else:
            w.add_key_value(name, field.contents(), vtype)
    w.add_token_list(tokens[keep])
    w.add_token_types([int(x) for x in np.asarray(r.fields["tokenizer.ggml.token_type"].contents())[keep]])
    w.add_token_merges(["<|speech_0|> <|speech_1|>"])  # llama.cpp's BPE loader requires one; never used
    for setter in (w.add_bos_token_id, w.add_eos_token_id, w.add_pad_token_id):
        setter(end - lo)  # END's new id
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)
    if "neuphonic.input_format" not in r.fields:
        w.add_string("neuphonic.input_format", "phonemes")
    w.add_string("neuphonic.vocab", "speech")
    w.add_uint32("neuphonic.vocab_offset", lo)

    for t in r.tensors:
        data = t.data
        if t.name in ("token_embd.weight", "output.weight"):
            if int(t.shape[-1]) != len(tokens):
                raise ValueError(f"{t.name} has {int(t.shape[-1])} rows for {len(tokens)} tokens")
            data = np.ascontiguousarray(data[keep])  # whole rows: quantized blocks never span rows
        # quantized tensors arrive as uint8 [rows, bytes per row]; the writer derives the
        # element shape from the byte shape and the quantization type
        w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    tmp.rename(dst)
    return dst


def ensure_pruned(src: Path, cache_dir: Path) -> Path:
    """The pruned copy of ``src``, built once and cached (keyed by path, size and mtime)."""
    st = src.stat()
    key = hashlib.sha1(f"v2|{src.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:12]
    dst = cache_dir / f"{src.stem}.speechvocab.{key}.gguf"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        prune_gguf(src, dst)
    return dst


def _dequant_q8_0(raw: np.ndarray) -> np.ndarray:
    blocks = raw.reshape(raw.shape[0], -1, 34)
    d = blocks[..., :2].copy().view(np.float16).astype(np.float32)
    q = blocks[..., 2:].view(np.int8).astype(np.float32)
    return (q * d).reshape(raw.shape[0], -1)


def _dequant_q4_0(raw: np.ndarray) -> np.ndarray:
    blocks = raw.reshape(raw.shape[0], -1, 18)
    d = blocks[..., :2].copy().view(np.float16).astype(np.float32)
    qs = blocks[..., 2:]
    lo = (qs & 0x0F).astype(np.int8) - 8
    hi = (qs >> 4).astype(np.int8) - 8
    return (np.concatenate([lo, hi], axis=-1).astype(np.float32) * d).reshape(raw.shape[0], -1)


class EmbeddingTable:
    """Rows of the original model's ``token_embd.weight``, dequantized like llama.cpp's
    ``get_rows`` (``y = q * d`` in float32), read lazily from the memory-mapped file."""

    def __init__(self, gguf_path: Path) -> None:
        from gguf import GGMLQuantizationType, GGUFReader

        r = GGUFReader(gguf_path)
        t = next(x for x in r.tensors if x.name == "token_embd.weight")
        self.n_embd = int(t.shape[0])
        self._data = t.data
        self._dequant = {
            GGMLQuantizationType.Q8_0: _dequant_q8_0,
            GGMLQuantizationType.Q4_0: _dequant_q4_0,
            GGMLQuantizationType.F16: lambda a: a.astype(np.float32),
            GGMLQuantizationType.F32: lambda a: np.asarray(a, np.float32),
        }.get(t.tensor_type)
        if self._dequant is None:
            raise ValueError(f"vocab: speech supports Q8_0/Q4_0/F16/F32 embeddings, not {t.tensor_type.name}")
        self.type = t.tensor_type.name

    def rows(self, ids) -> np.ndarray:
        return np.ascontiguousarray(self._dequant(np.asarray(self._data[np.asarray(ids, np.int64)])), np.float32)
