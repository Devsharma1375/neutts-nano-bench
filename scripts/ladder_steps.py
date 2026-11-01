"""Step-by-step attribution for a ladder run: each step paired against the step it builds on.

The report compares every configuration with the suite baseline; this answers "what did this
one change buy?" instead. Writes <run>/steps.md.

    python scripts/ladder_steps.py results/m4-ladder
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ttsbench import stats  # noqa: E402
from ttsbench.runner import load_records  # noqa: E402

# step -> the configuration it is compared with (default: the previous one in the suite)
PARENT = {"L02_upstream_q8_onnx": "L03_upstream_q4_onnx", "L04_upstream_q4_onnx_metal": "L03_upstream_q4_onnx",
          "L05_fast_equivalent": "L03_upstream_q4_onnx", "X1_int8_codec": "L14_small_first_chunk",
          "X2_q8_backbone": "L14_small_first_chunk", "L00_upstream_torch": "L03_upstream_q4_onnx",
          "L01_upstream_torch_mps": "L03_upstream_q4_onnx"}
METRICS = (("batch", "rtf", "batch RTF"), ("stream", "ttfa_s", "stream TTFA"), ("stream", "total_s", "stream total"))


def main(run_dir: Path) -> Path:
    run = json.loads((run_dir / "run.json").read_text())
    order = [c["name"] for c in run["suite"]["configs"]]
    noise = run["suite"]["noise_floor"]
    by: dict = {}
    for r in load_records(run_dir):
        if r["kind"] == "measure" and r["ok"]:
            by.setdefault((r["config"], r["mode"]), {})[r["pair_key"]] = r

    def cell(a: str, b: str, mode: str, key: str) -> str:
        base, cand = by.get((a, mode)), by.get((b, mode))
        if not base or not cand:
            return "–"
        c = stats.paired_ratio(stats.pair_up({k: v[key] for k, v in base.items()}, {k: v[key] for k, v in cand.items()}))
        if not c.get("n"):
            return "–"
        verdict = stats.speed_verdict(c, noise)
        tag = {"faster": "", "slower": " slower", "within noise": " (within noise)"}.get(verdict, f" ({verdict})")
        return f"{c['ratio']:.2f}× [{c['ci_low']:.2f}–{c['ci_high']:.2f}]{tag}"

    lines = [f"# Ladder steps: `{run_dir.name}`", "",
             "Each step paired against the configuration it builds on (ratio < 1 = faster, 95% bootstrap CI). "
             f"\"Within noise\" = the interval reaches inside 1 ± {noise:.0%}.", "",
             "| Step | Compared with | " + " | ".join(m[2] for m in METRICS) + " |",
             "|---|---|" + "---|" * len(METRICS)]
    for i, name in enumerate(order):
        parent = PARENT.get(name) or (order[i - 1] if i else None)
        if parent is None or name == parent or name == run["suite"]["baseline"]:
            continue
        lines.append(f"| `{name}` | `{parent}` | " + " | ".join(cell(parent, name, m, k) for m, k, _ in METRICS) + " |")
    out = run_dir / "steps.md"
    out.write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    print(main(Path(sys.argv[1])).read_text())
