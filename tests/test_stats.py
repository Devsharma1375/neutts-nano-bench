import math

import numpy as np

from ttsbench import stats


def test_summary_and_ci_contain_median():
    x = np.random.default_rng(1).lognormal(0, 0.2, 200)
    s = stats.summarize(x)
    assert s["n"] == 200 and s["ci_low"] <= s["median"] <= s["ci_high"]
    assert s["p95"] >= s["median"]


def test_ignores_none_and_nan():
    assert stats.summarize([1.0, None, float("nan"), 3.0])["n"] == 2
    assert stats.summarize([])["n"] == 0


def test_paired_ratio_recovers_known_speedup():
    rng = np.random.default_rng(2)
    base = rng.uniform(0.5, 3.0, 100)  # very different sentence lengths
    cand = base * 0.8 * rng.normal(1, 0.02, 100)
    r = stats.paired_ratio(list(zip(base, cand)))
    assert abs(r["ratio"] - 0.8) < 0.01 and r["ci_high"] < 0.85
    assert stats.speed_verdict(r, 0.03) == "faster"


def test_noise_floor_blocks_tiny_wins():
    base = np.linspace(1, 2, 200)
    r = stats.paired_ratio(list(zip(base, base * 0.99)))
    assert r["ci_high"] < 1  # statistically "faster"...
    assert stats.speed_verdict(r, 0.03) == "within noise"  # ...but not practically


def test_paired_delta_and_insufficient_data():
    d = stats.paired_delta([(0.1, 0.12)] * 10)
    assert math.isclose(d["delta"], 0.02, abs_tol=1e-9)
    assert stats.speed_verdict({"n": 2}, 0.03) == "insufficient data"
