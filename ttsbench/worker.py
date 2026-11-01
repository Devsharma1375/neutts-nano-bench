"""Runs ONE configuration in its own process: load, cold request, warm-up, measure.

Fresh process per configuration means memory, thread pools, allocator state and caches
from one experiment can never leak into the next. The parent (runner.py) only
orchestrates and never loads a model.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from itertools import product
from pathlib import Path

import numpy as np

from .config import RunConfig
from .corpus import load_corpus, simulate_playback
from .engines import Request, create_engine
from .engines.base import load_voice
from .memory import METRIC as MEM_METRIC
from .memory import MemorySampler

log = logging.getLogger("ttsbench.worker")

# Cold and warm-up requests use sentences that are never measured, so nothing a measured
# request does (shape-specific kernels, caches) has been rehearsed on exactly that input.
# Short to long, so per-length work (codec shapes, allocations) is exercised too.
WARMUP_TEXTS = (
    "Please hold while I check that for you.",
    "The library closes early on Fridays, so return your books before four o'clock.",
    "If the weather clears up tomorrow afternoon, we could take the bikes along the river "
    "and stop for a late lunch at the cafe near the old bridge.",
)


def _plan(sentences, voices, seeds, modes, repeats, round_idx, max_items):
    """The measured request order. Shuffled per repeat with a seed that depends only on
    (round, repeat), so every configuration sees the same order and comparisons are paired."""
    if max_items:
        sentences = sentences[:max_items]
    base = list(product(sentences, voices, seeds, modes))
    plan = []
    for r in range(repeats):
        rng = np.random.default_rng(10_000 * round_idx + r)
        plan += [(r, *base[i]) for i in rng.permutation(len(base))]
    return plan


def _record(cfg: RunConfig, round_idx, kind, rep, sent, voice, seed, mode, result, peak_mb, err=None):
    rec = {
        "id": f"{cfg.name}|{round_idx}|{mode}|{sent.id}|{voice.name}|{seed}|{rep}",
        "pair_key": f"{mode}|{sent.id}|{voice.name}|{seed}|{rep}|{round_idx}",
        "kind": kind, "config": cfg.name, "engine": cfg.engine, "round": round_idx, "mode": mode,
        "sentence_id": sent.id, "bucket": sent.bucket, "voice": voice.name, "seed": seed, "repeat": rep,
        "text_chars": len(sent.text), "ok": err is None, "error": err, "peak_mem_mb": round(peak_mb, 1),
    }
    if result is None:
        return rec
    t = result.timer
    audio_s = result.audio_seconds
    total_s = t.total_ms / 1000
    rec.update({
        "total_s": total_s,
        "ttfa_s": t.first_audio_s if t.first_audio_s is not None else total_s,
        "first_token_s": t.first_token_s,
        "audio_s": audio_s,
        "rtf": total_s / audio_s if audio_s > 0 else None,
        "stages_ms": {k: round(v, 3) for k, v in t.ms.items()},
        "prompt_tokens": result.prompt_tokens,
        "gen_tokens": int(len(result.tokens)),
        "prefill_tokens": result.extra.get("prefill_tokens"),
        "decode_tokens": result.extra.get("decode_tokens"),
        "stop_reason": result.stop_reason,
        "n_chunks": len(t.chunks),
    })
    if mode == "stream":
        pb = simulate_playback(t.chunks)
        rec.update({"chunks": [[round(a, 5), n] for a, n in t.chunks], "underruns": pb.underruns,
                    "stall_s": pb.stall_s, "min_headroom_s": pb.min_headroom_s})
    return rec


def run(spec: dict) -> None:
    suite, cfg = spec["suite"], RunConfig(**spec["config"])
    out_dir, round_idx = Path(spec["out_dir"]), int(spec["round"])
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    voices = [load_voice(v, Path(spec["voices_dir"])) for v in suite["voices"]]
    sentences = load_corpus(Path(suite["corpus"]), suite.get("buckets"))
    modes = [m for m in suite["modes"] if m in cfg.modes]
    mem = MemorySampler().start()
    meta: dict = {"config": cfg.to_dict(), "round": round_idx, "modes": modes}

    t0 = time.perf_counter()
    engine = create_engine(cfg.engine, cfg.settings)
    engine.load()
    meta["load_s"] = time.perf_counter() - t0
    meta["mem_metric"] = MEM_METRIC
    meta["mem_after_load_mb"] = round(mem.current_mb(), 1)
    meta["peak_mem_load_mb"] = round(mem.peak_mb(), 1)
    meta["resolved"] = engine.describe()
    if "stream" in modes and not getattr(engine, "supports_stream", True):
        log.warning("%s: engine cannot stream; measuring batch only", cfg.name)
        modes = [m for m in modes if m != "stream"]
        meta["modes"] = modes
    log.info("%s loaded in %.2fs (memory %.0f MB)", cfg.name, meta["load_s"], meta["mem_after_load_mb"])

    records = []
    # cold request: the first real request after load (what an app's first user feels), in the
    # first suite mode the config runs; recorded with its mode so reports compare like with like
    from .corpus import Sentence

    cold_mode = modes[0]
    cold = Sentence("warmup0", WARMUP_TEXTS[0], "warmup")
    mem.reset_peak()
    res = engine.run(Request(cold.text, voices[0], suite["seeds"][0], cold_mode == "stream"))
    records.append(_record(cfg, round_idx, "cold", -1, cold, voices[0], suite["seeds"][0], cold_mode, res, mem.peak_mb()))
    meta["cold_ttfa_s"], meta["cold_mode"] = records[-1]["ttfa_s"], cold_mode

    plan = _plan(sentences, voices, suite["seeds"], modes, suite["repeats"], round_idx, suite.get("max_items"))
    for i in range(suite["warmup"]):  # warm-up: run, discard (every mode and voice, unmeasured text)
        text = WARMUP_TEXTS[i % len(WARMUP_TEXTS)]
        engine.run(Request(text, voices[i % len(voices)], suite["seeds"][0], modes[i % len(modes)] == "stream"))

    pacing, gap = suite.get("pacing", "none"), float(suite.get("pacing_gap_s", 0.0))
    for n, (rep, sent, voice, seed, mode) in enumerate(plan, 1):
        if pacing == "playback" and records[-1].get("ok") and records[-1].get("audio_s"):
            # idle until the previous request's audio would have finished playing: a voice app
            # speaks, then waits, so the device is not under sustained load between utterances
            last = records[-1]
            playback_end = last["ttfa_s"] + last["audio_s"] + (last.get("stall_s") or 0.0)
            time.sleep(max(playback_end - last["total_s"], 0.0) + gap)
        elif pacing == "playback":
            time.sleep(gap)
        mem.reset_peak()
        req = Request(sent.text, voice, seed, mode == "stream")
        try:
            res = engine.run(req)
            rec = _record(cfg, round_idx, "measure", rep, sent, voice, seed, mode, res, mem.peak_mb())
        except Exception as e:  # noqa: BLE001 - one bad request must not kill the run
            log.error("request failed: %s", e)
            records.append(_record(cfg, round_idx, "measure", rep, sent, voice, seed, mode, None, mem.peak_mb(),
                                   err=f"{type(e).__name__}: {e}"))
            continue
        # ---- everything below is untimed bookkeeping
        # "first" = one copy per item: the first repeat of the first round (later rounds repeat
        # the same requests for timing; scoring them again would double-count in quality gates)
        first_copy = rep == 0 and round_idx == 0
        save = suite["save_audio"] == "all" or (suite["save_audio"] == "first" and first_copy)
        if save and res.audio.size:
            import soundfile as sf

            from .quality.signal import to_pcm16

            name = f"{mode}__{sent.id}__{voice.name}__s{seed}__r{rep}.wav"
            sf.write(audio_dir / name, to_pcm16(res.audio), res.sample_rate, subtype="PCM_16")
            rec["audio_path"] = str((audio_dir / name).relative_to(out_dir.parent.parent))
        if mode == "stream" and suite.get("stream_fidelity") and first_copy and len(res.tokens):
            try:
                from .quality.signal import stream_fidelity_db

                rec["fidelity_db"] = stream_fidelity_db(res.audio, engine.decode_full(res.tokens))
            except NotImplementedError:
                pass
        records.append(rec)
        if n % 10 == 0 or n == len(plan):
            log.info("%s round %d: %d/%d requests", cfg.name, round_idx, n, len(plan))

    mem.stop()
    meta["peak_mem_requests_mb"] = round(max([r["peak_mem_mb"] for r in records], default=0), 1)
    with open(out_dir / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (out_dir / "worker_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    engine.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ttsbench.worker")
    ap.add_argument("--spec", required=True, help="JSON spec written by the runner")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    spec = json.loads(Path(args.spec).read_text())
    try:
        run(spec)
    except Exception:  # noqa: BLE001
        Path(spec["out_dir"], "error.txt").write_text(traceback.format_exc())
        log.exception("worker failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
