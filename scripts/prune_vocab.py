"""Write a speech-only-vocabulary copy of a NeuTTS backbone GGUF (see ttsbench/engines/vocab.py).

The fast engine builds and caches this file itself when a config sets ``vocab: speech``;
this script is for inspecting the file or benchmarking it with llama-bench.

    python scripts/prune_vocab.py models/neutts-nano-Q4_0.gguf   # -> ..._speechvocab.gguf
"""
from __future__ import annotations

import sys
from pathlib import Path

from ttsbench.engines.vocab import prune_gguf

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) == 3 else src.with_name(src.stem + "_speechvocab.gguf")
    prune_gguf(src, dst)
    print(f"{dst}: {dst.stat().st_size / 2**20:.1f} MiB (was {src.stat().st_size / 2**20:.1f} MiB)")
