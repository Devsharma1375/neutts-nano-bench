"""Orchestrates a suite run. Never loads a model itself.

Layout of a run directory::

    results/<run_id>/
        run.json                    suite, environment, per-config status
        env.json                    hardware/software fingerprint
        <config>/round_<r>/records.jsonl, worker_meta.json, worker.log, audio/*.wav
        quality.jsonl               written by `ttsbench quality`
        report.md, summary.json, charts/   written by `ttsbench report`
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from . import env as envmod
from .config import Suite
from .engines.base import prepare_voices
from .engines.components import CACHE_DIR

log = logging.getLogger("ttsbench.runner")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_voices(suite: Suite) -> Path:
    voices_dir = CACHE_DIR / "voices"
    missing = [v for v in suite.voices if not (voices_dir / f"{v}.npy").exists()]
    if missing:
        if all(c.engine == "mock" for c in suite.configs):
            _synthetic_voices(missing, voices_dir)
        else:
            log.info("preparing reference voices: %s", missing)
            prepare_voices(missing, voices_dir)
    return voices_dir


def _prepare_models(suite: Suite) -> None:
    """Build one-off derived files (the speech-vocabulary GGUF) before any timed worker
    starts, so their construction never lands in a configuration's load time or memory."""
    from .engines.components import resolve_model_file
    from .engines.vocab import ensure_pruned

    for cfg in suite.configs:
        if cfg.engine == "fast" and cfg.settings.get("vocab") == "speech":
            src = resolve_model_file(cfg.settings["backbone"], ".gguf", cfg.settings.get("gguf_filename"))
            log.info("speech-vocabulary backbone for %s: %s", cfg.name, ensure_pruned(src, CACHE_DIR / "pruned").name)


def _synthetic_voices(names: list[str], voices_dir: Path) -> None:
    """Mock-only suites can run without neutts installed (CI)."""
    import numpy as np
    import soundfile as sf

    voices_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        rng = np.random.default_rng(i)
        np.save(voices_dir / f"{name}.npy", rng.integers(0, 65536, 300 + 50 * i, dtype=np.int32))
        (voices_dir / f"{name}.txt").write_text("This is a synthetic reference voice for testing.\n")
        wav = voices_dir / f"{name}.wav"
        sf.write(wav, (0.1 * np.sin(np.arange(24000 * 6) * 2 * np.pi * 180 / 24000)).astype("float32"), 24000)
        (voices_dir / f"{name}.json").write_text(json.dumps({"wav": str(wav), "synthetic": True}))


def run_suite(suite: Suite, out_root: Path, run_id: str | None = None) -> Path:
    run_id = run_id or f"{dt.datetime.now():%Y%m%d-%H%M%S}-{suite.name}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    voices_dir = _ensure_voices(suite)
    _prepare_models(suite)

    environment = envmod.capture(REPO_ROOT)
    (run_dir / "env.json").write_text(json.dumps(environment, indent=2))
    log.info("hardware: %s", envmod.summary_line(environment))
    if environment["power"].get("on_ac_power") is False:
        log.warning("running on battery: numbers will be slower and noisier than on AC power")

    status: dict[str, list] = {c.name: [] for c in suite.configs}
    started = time.time()
    for r in range(suite.rounds):
        # ABBA ordering: alternate the config order each round so slow drift (thermal,
        # background load) affects every configuration equally
        order = suite.configs if r % 2 == 0 else list(reversed(suite.configs))
        for cfg in order:
            out_dir = run_dir / cfg.name / f"round_{r}"
            out_dir.mkdir(parents=True)
            spec = {"suite": suite.to_dict(), "config": cfg.to_dict(), "round": r,
                    "out_dir": str(out_dir), "voices_dir": str(voices_dir)}
            spec_path = out_dir / "spec.json"
            spec_path.write_text(json.dumps(spec, indent=2, default=str))
            if suite.cooldown_s:
                # every configuration starts from the same, recovered thermal state (fanless
                # laptops throttle within seconds of sustained load and recover within ~10 s idle)
                log.info("cooling down %.0f s", suite.cooldown_s)
                time.sleep(suite.cooldown_s)
            log.info("[round %d] running %s (%s)", r, cfg.name, cfg.engine)
            t0 = time.time()
            with open(out_dir / "worker.log", "w") as logf:
                try:
                    proc = subprocess.run(
                        [sys.executable, "-m", "ttsbench.worker", "--spec", str(spec_path)],
                        stdout=logf, stderr=subprocess.STDOUT, timeout=suite.timeout_s, check=False,
                        env={**os.environ, "PYTHONUNBUFFERED": "1"}, cwd=REPO_ROOT,
                    )
                    code = proc.returncode
                except subprocess.TimeoutExpired:
                    code = "timeout"
            ok = code == 0 and (out_dir / "records.jsonl").exists()
            status[cfg.name].append({"round": r, "ok": ok, "exit": code, "seconds": round(time.time() - t0, 1)})
            if not ok:
                tail = (out_dir / "worker.log").read_text()[-1500:]
                log.error("%s failed (exit %s). Last log lines:\n%s", cfg.name, code, tail)

    run_info = {
        "run_id": run_id,
        "suite": suite.to_dict(),
        "started": dt.datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "mock": any(c.engine == "mock" for c in suite.configs),
        "hardware": envmod.summary_line(environment),
    }
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2, default=str))
    return run_dir


def load_records(run_dir: Path) -> list[dict]:
    records = []
    for path in sorted(run_dir.glob("*/round_*/records.jsonl")):
        records += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return records


def load_worker_meta(run_dir: Path) -> dict[str, list[dict]]:
    meta: dict[str, list[dict]] = {}
    for path in sorted(run_dir.glob("*/round_*/worker_meta.json")):
        m = json.loads(path.read_text())
        meta.setdefault(m["config"]["name"], []).append(m)
    return meta
