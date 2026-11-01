"""``ttsbench`` command line.

    ttsbench all suites/default.yaml        run + quality + report (the one command)
    ttsbench run suites/ladder.yaml         timing only
    ttsbench quality results/<run>          score saved audio
    ttsbench report results/<run>           summary.json, report.md, charts
    ttsbench compare A B --config fast_best cross-run comparison (e.g. native build)
    ttsbench verify                         fast engine == upstream, on the real model
    ttsbench doctor                         preflight checks with fixes
    ttsbench env                            print the hardware/software fingerprint
    ttsbench prepare-voices jo dave         convert reference voices once
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

RESULTS = Path("results")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S")


def _run(args) -> Path:
    from .config import load_suite
    from .runner import run_suite

    suite = load_suite(args.suite, quick=args.quick, only=args.only.split(",") if args.only else None)
    run_dir = run_suite(suite, Path(args.out), args.run_id)
    print(f"\nTiming results: {run_dir}")
    return run_dir


def _quality(run_dir: Path, args) -> None:
    from .quality.run import score_run

    score_run(run_dir, asr_model=getattr(args, "asr_model", None))


def _report(run_dir: Path) -> Path:
    from .report import write_report

    path = write_report(run_dir)
    print(f"Report: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ttsbench", description="Benchmark and optimise NeuTTS on-device.")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("run", "all"):
        p = sub.add_parser(name, help="run a suite" + (" then score and report" if name == "all" else ""))
        p.add_argument("suite", type=Path)
        p.add_argument("--out", default=str(RESULTS))
        p.add_argument("--run-id")
        p.add_argument("--only", help="comma-separated config names")
        p.add_argument("--quick", action="store_true", help="smoke-test sizes (1 repeat, 4 sentences)")
        p.add_argument("--asr-model")
        p.add_argument("--skip-quality", action="store_true")

    p = sub.add_parser("quality", help="score saved audio of a run")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--asr-model")
    p = sub.add_parser("report", help="build report.md / summary.json / charts")
    p.add_argument("run_dir", type=Path)
    p = sub.add_parser("compare", help="compare one config across two runs")
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    p.add_argument("--config", required=True, help="config name in A (and in B unless --config-b)")
    p.add_argument("--config-b")
    p.add_argument("--out", type=Path)
    p = sub.add_parser("verify", help="prove the fast engine matches upstream")
    p.add_argument("--backbone", default="neuphonic/neutts-nano-q4-gguf")
    p.add_argument("--codec", default="neuphonic/neucodec-onnx-decoder")
    p.add_argument("--voice", default="jo")
    p = sub.add_parser("doctor", help="preflight checks")
    p.add_argument("--offline", action="store_true", help="skip Hugging Face access checks")
    sub.add_parser("env", help="print the environment fingerprint")
    p = sub.add_parser("prepare-voices", help="convert NeuTTS reference voices to .npy")
    p.add_argument("voices", nargs="+")

    args = ap.parse_args(argv)
    _setup_logging(args.verbose)

    if args.cmd in ("run", "all"):
        run_dir = _run(args)
        if args.cmd == "all":
            run = json.loads((run_dir / "run.json").read_text())
            if not run["mock"] and not args.skip_quality:
                _quality(run_dir, args)
            _report(run_dir)
        return 0
    if args.cmd == "quality":
        _quality(args.run_dir, args)
        return 0
    if args.cmd == "report":
        _report(args.run_dir)
        return 0
    if args.cmd == "compare":
        from .report import compare_runs

        out = args.out or args.run_b / f"compare_vs_{args.run_a.name}.md"
        print(compare_runs(args.run_a, args.run_b, args.config, args.config_b or args.config, out).read_text())
        return 0
    if args.cmd == "verify":
        from .engines.base import prepare_voices
        from .engines.components import CACHE_DIR
        from .verify import verify

        if not (CACHE_DIR / "voices" / f"{args.voice}.npy").exists():
            prepare_voices([args.voice], CACHE_DIR / "voices")
        return 0 if verify(args.backbone, args.codec, args.voice) else 1
    if args.cmd == "doctor":
        from .doctor import run_doctor

        return 0 if run_doctor(check_hf=not args.offline) else 1
    if args.cmd == "env":
        from .env import capture

        print(json.dumps(capture(), indent=2))
        return 0
    if args.cmd == "prepare-voices":
        from .engines.base import prepare_voices
        from .engines.components import CACHE_DIR

        for p in prepare_voices(args.voices, CACHE_DIR / "voices"):
            print(p)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
