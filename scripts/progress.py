"""Live progress and ETA for benchmark runs in results/: a few file reads per refresh, no
model imports, so it doesn't disturb the measurements.

    python scripts/progress.py m4-default:suites/default.yaml m4-sustained:suites/sustained.yaml --watch
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ttsbench.config import load_suite  # noqa: E402
from ttsbench.corpus import load_corpus  # noqa: E402

# seconds per unit of work when a suite has no completed configuration to learn from
PRIOR = {"request_paced": 6.5, "request_unpaced": 3.0, "config_overhead": 60.0, "quality": 0.7}
DONE_RE = re.compile(r"(\d+)/(\d+) requests")


def _expected(suite) -> dict[str, int]:
    n_sent = len(load_corpus(suite.corpus, suite.buckets))
    if suite.max_items:
        n_sent = min(n_sent, suite.max_items)
    base = n_sent * len(suite.voices) * len(suite.seeds) * suite.repeats
    return {c.name: base * len([m for m in suite.modes if m in c.modes]) for c in suite.configs}


def _suite_state(run_dir: Path, suite) -> dict:
    exp = _expected(suite)
    rounds, now = suite.rounds, time.time()
    done_req = total_req = 0
    spent, finished_req = 0.0, 0
    current = None
    for r in range(rounds):
        for c in suite.configs:
            total_req += exp[c.name]
            d = run_dir / c.name / f"round_{r}"
            log = d / "worker.log"
            if (d / "records.jsonl").exists():
                done_req += exp[c.name]
                finished_req += exp[c.name]
                spent += log.stat().st_mtime - (d / "spec.json").stat().st_mtime
            elif log.exists():
                m = DONE_RE.findall(log.read_text()[-4000:])
                n = int(m[-1][0]) if m else 0
                done_req += n
                current = (c.name, r, n, exp[c.name], now - (d / "spec.json").stat().st_mtime)
    quality_total = sum(exp.values())  # one scored copy per item (first round, first repeat)
    q = run_dir / "quality.jsonl"
    quality_done = sum(1 for _ in q.open()) if q.exists() else 0
    report = (run_dir / "report.md").exists()
    per_req = spent / finished_req if finished_req else (PRIOR["request_paced"] if suite.pacing == "playback"
                                                        else PRIOR["request_unpaced"])
    remaining = 0.0 if report else (total_req - done_req) * per_req + (quality_total - quality_done) * PRIOR["quality"]
    if not finished_req and not report:  # learn nothing yet: add per-config load + cool-down
        remaining += len(suite.configs) * rounds * (PRIOR["config_overhead"] + suite.cooldown_s)
    return {"done": done_req, "total": total_req, "q_done": quality_done, "q_total": quality_total,
            "current": current, "remaining_s": remaining, "report": report, "started": run_dir.exists()}


def _bar(frac: float, width: int = 30) -> str:
    frac = max(0.0, min(1.0, frac))
    full = int(frac * width)
    return "█" * full + ("▌" if frac * width - full >= 0.5 else "") + "·" * (width - full - (1 if frac * width - full >= 0.5 else 0))


def _fmt(s: float) -> str:
    s = int(max(s, 0))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def render(specs: list[tuple[str, Path]]) -> str:
    lines = [f"ttsbench progress · {dt.datetime.now():%H:%M:%S}", ""]
    total_left, all_done, all_total = 0.0, 0, 0
    for run_id, suite_path in specs:
        suite = load_suite(suite_path)
        st = _suite_state(ROOT / "results" / run_id, suite)
        work = st["total"] + st["q_total"]
        done = st["done"] + st["q_done"]
        all_done, all_total = all_done + done, all_total + work
        total_left += st["remaining_s"]
        status = "done" if st["report"] else ("waiting" if not st["started"] else "running")
        lines.append(f"{suite.name:<10} {_bar(done / work if work else 1)} {100 * done / max(work, 1):5.1f}%  "
                     f"{status:<7} requests {st['done']}/{st['total']}, quality {st['q_done']}/{st['q_total']}")
        if st["current"] and not st["report"]:
            name, r, n, e, el = st["current"]
            lines.append(f"{'':<10} └ {name} (round {r}): {n}/{e} requests, {_fmt(el)} in")
    eta = dt.datetime.now() + dt.timedelta(seconds=total_left)
    lines += ["", f"{'overall':<10} {_bar(all_done / max(all_total, 1))} {100 * all_done / max(all_total, 1):5.1f}%",
              f"time left ≈ {_fmt(total_left)}  →  finishes ≈ {eta:%H:%M}" if total_left else "all suites finished"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run_id:suite.yaml pairs, in execution order")
    ap.add_argument("--watch", action="store_true", help="refresh every 15 s until everything is finished")
    a = ap.parse_args(argv)
    specs = [(r.split(":", 1)[0], ROOT / r.split(":", 1)[1]) for r in a.runs]
    while True:
        out = render(specs)
        print("\033[2J\033[H" + out if a.watch else out, flush=True)
        if not a.watch or "all suites finished" in out:
            return 0
        time.sleep(15)


if __name__ == "__main__":
    sys.exit(main())
