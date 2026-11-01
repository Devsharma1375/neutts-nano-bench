"""Copy a small listening set out of a run: the first sentence of each length bucket, both voices,
streamed by every configuration, with each clip's timing and what Whisper heard.

The sentences are chosen by that fixed rule, not by listening, so the set isn't cherry-picked.

    python scripts/make_samples.py results/m4-default samples
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ttsbench.corpus import load_corpus  # noqa: E402
from ttsbench.runner import load_records  # noqa: E402

MODE = "stream"


def main(run_dir: Path, out: Path) -> Path:
    run = json.loads((run_dir / "run.json").read_text())
    suite = run["suite"]
    configs = [c["name"] for c in suite["configs"] if MODE in (c.get("modes") or suite["modes"])]
    corpus = Path(suite["corpus"])  # absolute path on the machine that made the run
    if not corpus.exists():
        corpus = ROOT / "corpus" / "sentences.jsonl"
    firsts = {}
    for s in load_corpus(corpus):
        firsts.setdefault(s.bucket, s)
    sentences = [firsts[b] for b in ("short", "medium", "long") if b in firsts]
    voices = suite["voices"]

    recs = {(r["config"], r["sentence_id"], r["voice"]): r for r in load_records(run_dir)
            if r["kind"] == "measure" and r["ok"] and r["mode"] == MODE and r["seed"] == 0
            and r["repeat"] == 0 and r["round"] == 0}
    quality = {}
    qpath = run_dir / "quality.jsonl"
    if qpath.exists():
        for line in qpath.read_text().splitlines():
            if line.strip():
                q = json.loads(line)
                quality[q["id"]] = q

    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.wav"):
        old.unlink()
    lines = ["# Listening samples", "",
             f"Streamed output from the `{run_dir.name}` run (the results in `DESIGN.md` section 8.1), copied by "
             "`scripts/make_samples.py`. The rule is fixed: the first sentence of each length bucket (short, "
             "medium, long) in both voices, from every configuration that streams. Nothing was picked by ear.", "",
             "Every clip is 24 kHz mono, seed 0, exactly what a listener would receive, Perth watermark included. "
             "`upstream_q4_onnx` is NeuTTS-Nano as Neuphonic documents it for low latency (the baseline); "
             "`fast_exact` generates the same speech tokens as upstream; `fast_best` and `fast_best_gpucodec` "
             "are numerically different and accepted through the quality gates. `fast_exact`'s clips are 0.02 s "
             "(one codec frame) shorter than upstream's: the same tokens, with upstream's misplaced final chunk "
             "fixed (`DESIGN.md` section 5.4).", "",
             "Columns: time to first audio and total time for this request, Whisper `small.en`'s transcript and "
             "word error rate, and ECAPA speaker similarity to the reference recording.", ""]
    for s in sentences:
        for voice in voices:
            lines += [f"## `{s.id}` · voice `{voice}`", "", f"> {s.text}", "",
                      "| Configuration | File | First audio | Total | Audio | Whisper heard | WER | Speaker sim |",
                      "|---|---|---|---|---|---|---|---|"]
            for i, cfg in enumerate(configs, 1):
                r = recs.get((cfg, s.id, voice))
                if r is None:
                    lines.append(f"| `{cfg}` | (no record) | | | | | | |")
                    continue
                src = run_dir / cfg / f"round_{r['round']}" / "audio" / f"{MODE}__{s.id}__{voice}__s0__r0.wav"
                name = f"{s.id}_{voice}_{i}_{cfg}.wav"
                shutil.copyfile(src, out / name)
                q = quality.get(r["id"], {})
                heard = q.get("hyp", "–").replace("|", "\\|")
                wer = f"{q['wer']:.2f}" if "wer" in q else "–"
                sim = f"{q['spk_sim']:.3f}" if "spk_sim" in q else "–"
                lines.append(f"| `{cfg}` | [{name}]({name}) | {r['ttfa_s']:.3f} s | {r['total_s']:.2f} s | "
                             f"{r['audio_s']:.2f} s | {heard} | {wer} | {sim} |")
            lines.append("")
    readme = out / "README.md"
    readme.write_text("\n".join(lines))
    return readme


if __name__ == "__main__":
    print(main(Path(sys.argv[1]), Path(sys.argv[2])))
