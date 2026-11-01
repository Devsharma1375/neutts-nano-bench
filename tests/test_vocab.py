"""Speech-vocabulary pruning (engines/vocab.py) on a synthetic GGUF: no weights needed."""
import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter, quants

from ttsbench.engines.vocab import END_TOKEN, N_CODES, EmbeddingTable, prune_gguf, speech_range

N_EMBD = 64
PRE = ["<|begin_of_text|>", "a", "b"] + [f"<|special_{i}|>" for i in range(7)]  # END ends up at id 10
TOKENS = PRE + [END_TOKEN] + [f"<|speech_{i}|>" for i in range(N_CODES)] + ["<|extra|>"]
END_ID = len(PRE)


def _write_model(path, qtype=GGMLQuantizationType.Q8_0):
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((len(TOKENS), N_EMBD)).astype(np.float32)
    raw = quants.quantize(emb, qtype)
    w = GGUFWriter(path, arch="llama")
    w.add_uint32("llama.vocab_size", len(TOKENS))
    w.add_token_list(TOKENS)
    w.add_token_types([1] * len(TOKENS))
    w.add_token_merges(["a b"])
    w.add_bos_token_id(0)
    w.add_eos_token_id(END_ID)
    w.add_tensor("token_embd.weight", raw, raw_dtype=qtype)
    w.add_tensor("output_norm.weight", np.ones(N_EMBD, np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return raw


def test_speech_range_requires_contiguous_layout():
    assert speech_range(TOKENS) == (END_ID, END_ID + N_CODES)
    broken = TOKENS[: END_ID + 2] + TOKENS[END_ID + 3 :]  # speech_1 missing
    with pytest.raises(ValueError):
        speech_range(broken)


def test_prune_keeps_end_and_speech_rows_byte_for_byte(tmp_path):
    raw = _write_model(tmp_path / "m.gguf")
    prune_gguf(tmp_path / "m.gguf", tmp_path / "p.gguf")
    r = GGUFReader(tmp_path / "p.gguf")
    toks = r.fields["tokenizer.ggml.tokens"]
    names = [bytes(toks.parts[i]).decode() for i in toks.data]
    n_keep = N_CODES + 1 + 7  # padded down to a multiple of 8 rows with the 7 tokens below END
    lo = END_ID - 7
    assert len(names) == n_keep and n_keep % 8 == 0
    assert names == TOKENS[lo : END_ID + N_CODES + 1]
    assert r.fields["llama.vocab_size"].contents() == n_keep
    assert r.fields["tokenizer.ggml.eos_token_id"].contents() == 7  # END's new id
    assert r.fields["neuphonic.vocab_offset"].contents() == lo
    emb = next(t for t in r.tensors if t.name == "token_embd.weight")
    assert np.array_equal(np.asarray(emb.data), raw[lo : END_ID + N_CODES + 1])
    norm = next(t for t in r.tensors if t.name == "output_norm.weight")
    assert np.array_equal(np.asarray(norm.data), np.ones(N_EMBD, np.float32))


@pytest.mark.parametrize("qtype", [GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q4_0, GGMLQuantizationType.F16])
def test_embedding_rows_match_reference_dequantization(tmp_path, qtype):
    raw = _write_model(tmp_path / "m.gguf", qtype)
    table = EmbeddingTable(tmp_path / "m.gguf")
    ids = [0, 2, 3, 4, 1000, len(TOKENS) - 1]
    want = quants.dequantize(raw[ids], qtype).astype(np.float32)
    got = table.rows(ids)
    assert got.shape == (len(ids), N_EMBD) and got.dtype == np.float32
    assert np.array_equal(got, want)  # bit-exact: llama.cpp's get_rows computes the same q * d
