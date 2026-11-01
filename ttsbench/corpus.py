"""Test sentences and the streaming playback model."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sentence:
    id: str
    text: str
    bucket: str
    tags: tuple[str, ...] = ()


def load_corpus(path: Path, buckets: list[str] | None = None) -> list[Sentence]:
    items, seen = [], set()
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("id") or not row.get("text"):
            raise ValueError(f"{path}:{n}: every sentence needs an id and text")
        if row["id"] in seen:
            raise ValueError(f"{path}:{n}: duplicate id {row['id']}")
        seen.add(row["id"])
        s = Sentence(row["id"], row["text"].strip(), row.get("bucket", "medium"), tuple(row.get("tags", [])))
        if buckets is None or s.bucket in buckets:
            items.append(s)
    if not items:
        raise ValueError(f"{path}: no sentences selected (buckets={buckets})")
    return items


@dataclass(frozen=True)
class Playback:
    underruns: int
    stall_s: float  # total silence the listener hears after playback started
    min_headroom_s: float  # smallest margin between a chunk arriving and being needed


def simulate_playback(chunks: list[tuple[float, int]], sample_rate: int = 24_000) -> Playback:
    """Play chunks as they arrive, starting with the first one.

    A chunk is needed when the previous one finishes playing. If it arrives later, the
    listener hears a gap (an underrun) and playback resumes when it arrives.
    """
    chunks = [(t, n) for t, n in chunks if n > 0]
    if not chunks:
        return Playback(0, 0.0, 0.0)
    clock = chunks[0][0]  # playback starts when the first audio arrives
    underruns, stall, headroom = 0, 0.0, float("inf")
    for i, (arrival, n) in enumerate(chunks):
        if i > 0:  # the first chunk defines t=0 of playback, so it has no margin
            headroom = min(headroom, clock - arrival)
        if arrival > clock + 1e-9:
            underruns += 1
            stall += arrival - clock
            clock = arrival
        clock += n / sample_rate
    return Playback(underruns, stall, 0.0 if headroom == float("inf") else headroom)
