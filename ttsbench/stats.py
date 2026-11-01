"""Statistics that make the numbers trustworthy.

* Medians and p95, never means of latencies (latency distributions have long tails).
* Bootstrap confidence intervals: no normality assumption, works for medians.
* Paired comparisons: every configuration runs the same (sentence, voice, seed, repeat,
  round) requests, so we compare request-by-request. That removes sentence-length
  variance and needs far fewer runs than comparing two independent distributions.
* A noise floor: a change smaller than the run-to-run noise is reported as
  "within noise", even when its confidence interval happens to exclude zero.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

SEED = 20_240_531
N_BOOT = 2000


def _clean(values: Sequence[float | None]) -> np.ndarray:
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], float)
    return arr


def bootstrap_ci(values, stat: Callable = np.median, n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float]:
    x = _clean(values)
    if x.size < 2:
        v = float(stat(x)) if x.size else float("nan")
        return v, v
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.apply_along_axis(stat, 1, x[idx]) if stat is not np.median else np.median(x[idx], axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def summarize(values) -> dict:
    x = _clean(values)
    if x.size == 0:
        return {"n": 0}
    lo, hi = bootstrap_ci(x)
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "ci_low": lo,
        "ci_high": hi,
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "min": float(x.min()),
        "max": float(x.max()),
    }


def paired_ratio(pairs: list[tuple[float, float]]) -> dict:
    """Median of per-request ratios candidate/baseline, with a bootstrap CI.
    < 1 means the candidate is faster (for time metrics)."""
    logs = _clean([math.log(b / a) for a, b in pairs if a and b and a > 0 and b > 0])
    if logs.size == 0:
        return {"n": 0}
    lo, hi = bootstrap_ci(logs)
    med = float(np.median(logs))
    return {"n": int(logs.size), "ratio": math.exp(med), "ci_low": math.exp(lo), "ci_high": math.exp(hi)}


def paired_delta(pairs: list[tuple[float, float]]) -> dict:
    """Mean of per-item differences candidate - baseline (for quality scores)."""
    d = _clean([b - a for a, b in pairs if a is not None and b is not None])
    if d.size == 0:
        return {"n": 0}
    lo, hi = bootstrap_ci(d, stat=np.mean)
    return {"n": int(d.size), "delta": float(d.mean()), "ci_low": lo, "ci_high": hi}


def speed_verdict(cmp: dict, noise_floor: float) -> str:
    """'faster' / 'slower' only if the whole CI is beyond the noise floor."""
    if not cmp or cmp.get("n", 0) < 3:
        return "insufficient data"
    if cmp["ci_high"] < 1 - noise_floor:
        return "faster"
    if cmp["ci_low"] > 1 + noise_floor:
        return "slower"
    return "within noise"


def pair_up(base: dict[str, float], cand: dict[str, float]) -> list[tuple[float, float]]:
    return [(base[k], cand[k]) for k in base.keys() & cand.keys()]


def paired_corpus_delta(items: list[tuple[float, float, float]]) -> dict:
    """Corpus-level error-rate difference, candidate - baseline, from per-item
    ``(baseline_errors, candidate_errors, reference_words)``: sum of errors over sum of words,
    as ASR word error rate is normally reported, with a paired bootstrap over items for the CI
    (Bisani & Ney, 2004). Unlike a mean of per-sentence rates, one short sentence with a
    hallucinated tail can't outweigh a whole long one."""
    arr = np.asarray([x for x in items if x[2] and x[2] > 0], float)
    if arr.size == 0:
        return {"n": 0}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(arr), size=(N_BOOT, len(arr)))
    boots = (arr[idx, 1].sum(axis=1) - arr[idx, 0].sum(axis=1)) / arr[idx, 2].sum(axis=1)
    return {"n": int(len(arr)), "delta": float((arr[:, 1].sum() - arr[:, 0].sum()) / arr[:, 2].sum()),
            "ci_low": float(np.quantile(boots, 0.025)), "ci_high": float(np.quantile(boots, 0.975))}
