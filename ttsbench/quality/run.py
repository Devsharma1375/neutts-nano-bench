"""``ttsbench quality <run_dir>``: score every saved WAV. Resumable: already-scored
records are skipped, so an interrupted scoring pass can simply be re-run."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..corpus import load_corpus
from ..engines.base import load_voice
from ..engines.components import CACHE_DIR
from ..runner import load_records
from .signal import longest_silence_s

log = logging.getLogger("ttsbench.quality")


def score_run(run_dir: Path, asr_model: str | None = None, speaker: bool | None = None,
              mos: bool | None = None, watermark: bool | None = None) -> Path:
    run = json.loads((run_dir / "run.json").read_text())
    q = run["suite"]["quality"]
    asr_model = asr_model or q.get("asr_model", "small.en")
    speaker = q.get("speaker", True) if speaker is None else speaker
    mos = q.get("mos", False) if mos is None else mos
    watermark = q.get("watermark", True) if watermark is None else watermark

    texts = {s.id: s.text for s in load_corpus(Path(run["suite"]["corpus"]))}
    voices_dir = CACHE_DIR / "voices"
    out_path = run_dir / "quality.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.read_text().splitlines() if line.strip()}
    todo = [r for r in load_records(run_dir) if r.get("audio_path") and r["id"] not in done]
    if not todo:
        log.info("nothing to score")
        return out_path

    from . import scorers

    asr = scorers.AsrScorer(asr_model)
    spk = scorers.SpeakerScorer() if speaker else None
    mos_model = scorers.MosScorer() if mos else None
    wm = scorers.WatermarkScorer() if watermark else None
    voice_wavs: dict[str, Path] = {}

    with open(out_path, "a") as out:
        for i, rec in enumerate(todo, 1):
            wav = run_dir / rec["audio_path"]
            row: dict = {"id": rec["id"], "asr_model": asr_model}
            try:
                row.update(asr.score(wav, texts[rec["sentence_id"]]))
                if spk is not None:
                    if rec["voice"] not in voice_wavs:
                        voice_wavs[rec["voice"]] = load_voice(rec["voice"], voices_dir).wav_path
                    row["spk_sim"] = spk.similarity(wav, voice_wavs[rec["voice"]])
                if mos_model is not None:
                    row["mos"] = mos_model.score(wav)
                if wm is not None:
                    row["watermark"] = wm.score(wav)
                row["longest_silence_s"] = longest_silence_s(scorers.load_audio(wav, 24_000))
            except Exception as e:  # noqa: BLE001 - record and keep going
                row["error"] = f"{type(e).__name__}: {e}"
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 10 == 0 or i == len(todo):
                log.info("scored %d/%d", i, len(todo))
    return out_path


def load_quality(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "quality.jsonl"
    if not path.exists():
        return {}
    return {row["id"]: row for row in map(json.loads, filter(str.strip, path.read_text().splitlines()))}
