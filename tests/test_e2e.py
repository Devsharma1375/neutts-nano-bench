"""Runs the whole framework (runner -> isolated workers -> stats -> report) on the mock
engine. No model weights or network needed."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_smoke_suite_end_to_end(tmp_path):
    cmd = [sys.executable, "-m", "ttsbench", "all", str(ROOT / "suites" / "smoke.yaml"),
           "--out", str(tmp_path), "--run-id", "e2e"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    run_dir = tmp_path / "e2e"
    summary = json.loads((run_dir / "summary.json").read_text())
    for name in ("mock_slow", "mock_fast"):
        for mode in ("batch", "stream"):
            m = summary["configs"][name]["modes"][mode]
            assert m["n_ok"] == m["n"] > 0
    # the faster mock must be detected as faster, with a real paired comparison
    cmp = summary["comparisons"]["mock_fast"]["stream"]
    assert cmp["speed_verdict"] == "faster"
    assert cmp["overall"] == "faster, quality not scored"  # mock runs skip quality scoring
    # the streaming path reconstructs the one-pass decode exactly
    assert summary["configs"]["mock_fast"]["modes"]["stream"]["metrics"]["fidelity_db"]["median"] < 1e-3
    report = (run_dir / "report.md").read_text()
    assert "MOCK RUN" in report and (run_dir / "charts").is_dir()
