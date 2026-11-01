from pathlib import Path

import pytest

from ttsbench.config import ConfigError, load_suite
from ttsbench.corpus import load_corpus
from ttsbench.quality.text_norm import normalize

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("name", ["default", "ladder", "sustained", "threads", "smoke"])
def test_shipped_suites_are_valid(name):
    suite = load_suite(ROOT / "suites" / f"{name}.yaml")
    assert suite.baseline in [c.name for c in suite.configs]
    assert load_corpus(suite.corpus, suite.buckets)


def test_defaults_merge_and_override():
    s = load_suite(ROOT / "suites" / "ladder.yaml")
    assert s.config("L03_upstream_q4_onnx").settings["backbone"] == "neuphonic/neutts-nano-q4-gguf"
    assert s.config("L02_upstream_q8_onnx").settings["backbone"].endswith("q8-gguf")
    assert s.config("L00_upstream_torch").modes == ["batch"]


def test_quick_mode_shrinks_work():
    s = load_suite(ROOT / "suites" / "default.yaml", quick=True)
    assert s.repeats == 1 and len(s.voices) == 1 and s.max_items == 4


def _write(tmp_path, text):
    p = tmp_path / "s.yaml"
    p.write_text(text)
    return p


def test_typos_and_unsafe_settings_rejected(tmp_path):
    base = "name: x\ncorpus: c.jsonl\nvoices: [jo]\nbaseline: a\nconfigs:\n  - {name: a, engine: mock}\n"
    with pytest.raises(ConfigError, match="unknown keys"):
        load_suite(_write(tmp_path, base + "repeat: 3\n"))
    with pytest.raises(ConfigError, match="watermark"):
        load_suite(_write(tmp_path, base.replace("engine: mock}", "engine: mock, watermark: 'off'}")))
    with pytest.raises(ConfigError, match="baseline"):
        load_suite(_write(tmp_path, base.replace("baseline: a", "baseline: zzz")))
    # per-engine setting names: a misspelt option must not silently run the default
    with pytest.raises(ConfigError, match="unknown settings"):
        load_suite(_write(tmp_path, base.replace("engine: mock}", "engine: mock, decode_token_per_s: 900}")))
    with pytest.raises(ConfigError, match="unknown stream settings"):
        load_suite(_write(tmp_path, base.replace("engine: mock}", "engine: mock, stream: {first_chunk_frame: 10}}")))


def test_corpus_ids_unique_and_buckets_present():
    items = load_corpus(ROOT / "corpus" / "sentences.jsonl")
    assert len({i.id for i in items}) == len(items) == 36
    assert {i.bucket for i in items} == {"short", "medium", "long"}


@pytest.mark.parametrize("raw,expected", [
    ("Dr. Lee paid $5.", "doctor lee paid five dollars"),
    ("Meet at 9:30, on the 3rd!", "meet at nine thirty on the third"),
    ("Battery at 15%", "battery at fifteen percent"),
    ("In 2026 we grew", "in twenty twenty six we grew"),
    ("Don't stop", "don't stop"),
    ("Opens at 10a.m and closes at 6p.m.", "opens at ten am and closes at six pm"),
    ("Opens at 10 a.m. and closes at 6 p.m.", "opens at ten am and closes at six pm"),
    ("Doors open at 8am", "doors open at eight am"),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected
