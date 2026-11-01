"""Turn raw records into ``summary.json``, ``report.md`` and ``charts/*.png``."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import stats
from .quality.run import load_quality
from .runner import load_records, load_worker_meta
from .timing import STAGES

PRIMARY = {"stream": ("ttfa_s", "time to first audio"), "batch": ("rtf", "real-time factor")}


# ----------------------------------------------------------------------------- derive
def _per_record_metrics(r: dict) -> dict:
    st, audio = r.get("stages_ms", {}), r.get("audio_s") or 0
    m = {k: r.get(k) for k in ("ttfa_s", "total_s", "rtf", "first_token_s", "peak_mem_mb", "underruns", "stall_s",
                               "min_headroom_s", "fidelity_db", "gen_tokens", "prompt_tokens")}
    if st.get("prefill") and r.get("prefill_tokens"):
        m["prefill_tok_s"] = r["prefill_tokens"] / (st["prefill"] / 1000)
    if st.get("decode") and r.get("decode_tokens"):
        m["decode_tok_s"] = r["decode_tokens"] / (st["decode"] / 1000)
    if audio > 0:
        for k in STAGES:
            m[f"{k}_ms_per_s"] = st.get(k, 0.0) / audio
    return m


def build_summary(run_dir: Path) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    suite = run["suite"]
    records = load_records(run_dir)
    quality = load_quality(run_dir)
    meta = load_worker_meta(run_dir)
    order = [c["name"] for c in suite["configs"]]
    baseline, noise, gates = suite["baseline"], suite["noise_floor"], suite["gates"]

    # reference word counts (after the same normalisation the scorer applies), for corpus-level WER
    from .corpus import load_corpus
    from .quality.text_norm import normalize

    try:
        n_words = {s.id: len(normalize(s.text).split()) for s in load_corpus(Path(suite["corpus"]))}
    except (OSError, ValueError):
        n_words = {}

    def wer_errors(rec: dict) -> tuple[float, float] | None:
        row, n = quality.get(rec["id"], {}), n_words.get(rec["sentence_id"])
        return None if row.get("wer") is None or not n else (row["wer"] * n, n)

    measured = [r for r in records if r["kind"] == "measure"]
    groups: dict[tuple, list] = defaultdict(list)
    for r in measured:
        groups[(r["config"], r["mode"])].append(r)

    configs: dict[str, dict] = {}
    for name in order:
        wm = meta.get(name, [])
        cfg: dict = {
            "load_s": stats.summarize([m.get("load_s") for m in wm]),
            "cold_ttfa_s": stats.summarize([m.get("cold_ttfa_s") for m in wm]),
            "mem_metric": wm[0].get("mem_metric", "rss") if wm else None,
            "mem_after_load_mb": max([m.get("mem_after_load_mb", 0) for m in wm], default=None),
            "peak_mem_load_mb": max([m.get("peak_mem_load_mb", 0) for m in wm], default=None),
            "cold_mode": wm[0].get("cold_mode") if wm else None,
            "resolved": wm[0].get("resolved") if wm else None,
            "status": run["status"].get(name),
            "modes": {},
        }
        for mode in ("batch", "stream"):
            rows = groups.get((name, mode), [])
            if not rows:
                continue
            ok = [r for r in rows if r["ok"]]
            per = [_per_record_metrics(r) for r in ok]
            keys = sorted({k for p in per for k in p})
            ms = {k: stats.summarize([p.get(k) for p in per]) for k in keys}
            qrows = [quality[r["id"]] for r in ok if r["id"] in quality and "error" not in quality[r["id"]]]
            q = {k: stats.summarize([x.get(k) for x in qrows]) for k in ("wer", "cer", "spk_sim", "mos", "longest_silence_s")}
            counted = [e for e in (wer_errors(r) for r in ok) if e]
            q["wer_corpus"] = sum(e for e, _ in counted) / sum(n for _, n in counted) if counted else None
            wm_scores = [x["watermark"] for x in qrows if x.get("watermark") is not None]
            failures = {
                "errors": sum(not r["ok"] for r in rows),
                "non_eos_stops": sum(r.get("stop_reason") not in (None, "eos") for r in ok),
                "long_silences": sum((x.get("longest_silence_s") or 0) > 1.0 for x in qrows),
                "watermark_missing": sum(s < 0.5 for s in wm_scores),
                "underrun_requests": sum((r.get("underruns") or 0) > 0 for r in ok),
            }
            cfg["modes"][mode] = {
                "n": len(rows), "n_ok": len(ok), "metrics": ms, "quality": q,
                "watermark_detect_rate": (sum(s >= 0.5 for s in wm_scores) / len(wm_scores)) if wm_scores else None,
                "failures": failures,
            }
        configs[name] = cfg

    # paired comparisons against the baseline
    comparisons: dict[str, dict] = {}
    by_key = {key: {r["pair_key"]: r for r in rows if r["ok"]} for key, rows in groups.items()}
    for name in order:
        if name == baseline:
            continue
        comparisons[name] = {}
        for mode in ("batch", "stream"):
            base, cand = by_key.get((baseline, mode)), by_key.get((name, mode))
            if not base or not cand:
                continue
            metric, _ = PRIMARY[mode]
            speed = {k: stats.paired_ratio(_pairs(base, cand, lambda r, k=k: r.get(k)))
                     for k in ("ttfa_s", "total_s", "rtf")}
            qual = {k: stats.paired_delta(_pairs(base, cand, lambda r, k=k: quality.get(r["id"], {}).get(k)))
                    for k in ("spk_sim", "mos")}
            # WER gate: corpus-level (errors / words) with a paired bootstrap, the ASR standard
            items, diffs = [], []
            for key in base.keys() & cand.keys():
                eb, ec = wer_errors(base[key]), wer_errors(cand[key])
                if eb and ec:
                    items.append((eb[0], ec[0], eb[1]))
                    diffs.append(((ec[0] - eb[0]) / eb[1], cand[key]["sentence_id"], cand[key]["voice"],
                                  quality[base[key]["id"]].get("hyp", ""), quality[cand[key]["id"]].get("hyp", "")))
            qual["wer"] = stats.paired_corpus_delta(items)
            qual["wer_mean_item"] = stats.paired_delta(_pairs(base, cand, lambda r: quality.get(r["id"], {}).get("wer")))
            qual["largest_wer_changes"] = sorted(diffs, key=lambda d: -abs(d[0]))[:3]
            fid_b = configs[baseline]["modes"].get(mode, {}).get("metrics", {}).get("fidelity_db", {}).get("median")
            fid_c = configs[name]["modes"].get(mode, {}).get("metrics", {}).get("fidelity_db", {}).get("median")
            gate = _gates(qual, configs[name]["modes"][mode], fid_b, fid_c, gates)
            verdict = stats.speed_verdict(speed[metric], noise)
            comparisons[name][mode] = {"primary": metric, "speed": speed, "quality": qual, "gates": gate,
                                       "speed_verdict": verdict, "overall": _overall(verdict, gate, suite["quality"])}

    best = _pick_best(comparisons)
    env = json.loads((run_dir / "env.json").read_text())
    asr_models = sorted({row.get("asr_model") for row in quality.values() if row.get("asr_model")})
    return {"run": run, "env": env, "configs": configs, "comparisons": comparisons, "best": best, "asr_models": asr_models}


def _pairs(base: dict, cand: dict, value) -> list[tuple[float, float]]:
    """Match baseline and candidate records on pair_key and extract ``value(record)``."""
    return stats.pair_up({k: value(r) for k, r in base.items()}, {k: value(r) for k, r in cand.items()})


def _gates(qual: dict, mode_summary: dict, fid_base, fid_cand, g: dict) -> dict:
    def judge(cmp: dict, limit: float, sign: int) -> str:
        if not cmp or cmp.get("n", 0) < 3:
            return "not scored"
        lo, hi = sorted((sign * cmp["ci_low"], sign * cmp["ci_high"]))
        if lo > limit:
            return "fail"
        return "pass" if hi <= limit else "inconclusive"

    out = {
        "wer": judge(qual.get("wer"), g["wer_abs"], +1),  # increase in WER
        "spk_sim": judge(qual.get("spk_sim"), g["spk_sim_drop"], -1),  # decrease in similarity
        "mos": judge(qual.get("mos"), g["mos_drop"], -1),
    }
    rate = mode_summary.get("watermark_detect_rate")
    out["watermark"] = "not scored" if rate is None else ("pass" if rate >= g["watermark_detect"] else "fail")
    if fid_base is not None and fid_cand is not None and not (math.isnan(fid_base) or math.isnan(fid_cand)):
        out["fidelity"] = "pass" if fid_cand - fid_base <= g["fidelity_db"] else "fail"
    errs = mode_summary["failures"]
    out["failures"] = "pass" if errs["errors"] == 0 and errs["watermark_missing"] == 0 else "fail"
    return out


def _overall(verdict: str, gates: dict, quality: dict) -> str:
    states = set(gates.values())
    if "fail" in states:
        return "rejected: quality gate failed"
    # a speed-up only counts as a win once the quality the suite asks for was actually scored
    required = {"wer"} | ({"spk_sim"} if quality.get("speaker") else set()) | ({"watermark"} if quality.get("watermark") else set())
    if verdict == "faster" and any(gates.get(k) == "not scored" for k in required):
        return "faster, quality not scored"
    if verdict == "faster":
        return "win" if "inconclusive" not in states else "faster, quality needs more data"
    if verdict == "slower":
        return "slower"
    return "no measurable change"


def _pick_best(comparisons: dict) -> dict:
    best = {}
    for mode in ("stream", "batch"):
        cands = [(c[mode]["speed"][c[mode]["primary"]]["ratio"], name) for name, c in comparisons.items()
                 if mode in c and c[mode]["overall"] == "win" and c[mode]["speed"][c[mode]["primary"]].get("n")]
        if cands:
            ratio, name = min(cands)
            best[mode] = {"config": name, "ratio": ratio}
    return best


# ----------------------------------------------------------------------------- charts
def make_charts(summary: dict, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(exist_ok=True)
    files, order = [], [c["name"] for c in summary["run"]["suite"]["configs"]]
    baseline = summary["run"]["suite"]["baseline"]
    accent, base_c, grid = "#11887D", "#A8701A", "#D3DCD7"

    for mode, (metric, label) in PRIMARY.items():
        names = [n for n in order if mode in summary["configs"][n]["modes"] and summary["configs"][n]["modes"][mode]["metrics"].get(metric, {}).get("n")]
        if not names:
            continue
        ms = [summary["configs"][n]["modes"][mode]["metrics"][metric] for n in names]
        med = np.array([m["median"] for m in ms])
        err = np.array([[m["median"] - m["ci_low"] for m in ms], [m["ci_high"] - m["median"] for m in ms]])
        fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 1.6))
        colors = [base_c if n == baseline else accent for n in names]
        ax.barh(names, med, xerr=err, color=colors, capsize=3)
        right = med + err[1]
        for y, (v, x) in enumerate(zip(med, right)):
            ax.text(x, y, f"  {v:.3f}", va="center", fontsize=9)
        ax.set_xlim(0, float(right.max()) * 1.15)
        ax.invert_yaxis()
        ax.set_xlabel(f"median {label} ({'s' if metric == 'ttfa_s' else 'x'}), 95% CI; lower is better")
        ax.set_title(f"{mode} mode: {label} by configuration (baseline in brown)", loc="left", fontsize=11)
        if metric == "rtf":
            ax.axvline(1.0, color="#B0412A", lw=1, ls="--")
        ax.grid(axis="x", color=grid)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = out / f"{mode}_{metric}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        files.append(path.name)

    # stage breakdown per audio-second (batch preferred, stream otherwise)
    for mode in ("batch", "stream"):
        names = [n for n in order if mode in summary["configs"][n]["modes"]]
        if not names:
            continue
        fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 2.4))
        left = np.zeros(len(names))
        palette = ["#7A8C87", "#B9C4C0", "#11887D", "#A8701A", "#8A6FB0", "#5E7FB8", "#B0412A", "#D3DCD7"]
        for stage, color in zip(STAGES, palette):
            vals = np.array([summary["configs"][n]["modes"][mode]["metrics"].get(f"{stage}_ms_per_s", {}).get("median", 0) or 0 for n in names])
            ax.barh(names, vals, left=left, color=color, label=stage)
            left += vals
        ax.invert_yaxis()
        ax.set_xlabel("milliseconds of work per second of audio (median per stage)")
        ax.set_title(f"Where the time goes ({mode} mode)", loc="left", fontsize=11)
        ax.legend(ncol=8, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.22))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = out / f"stages_{mode}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        files.append(path.name)
        break

    # speed vs quality
    pts = []
    for n in order:
        for mode, marker in (("batch", "o"), ("stream", "^")):
            b = summary["configs"][n]["modes"].get(mode)
            if b and b["metrics"].get("rtf", {}).get("n") and b["quality"].get("wer", {}).get("n"):
                pts.append((n, mode, marker, b["metrics"]["rtf"]["median"], b["quality"]["wer"]["mean"]))
    if pts:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for n, mode, marker, x, y in pts:
            ax.scatter(x, y, color=base_c if n == baseline else accent, s=50, marker=marker)
            ax.annotate(f"{n} ({mode})", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("median real-time factor (log scale; lower is faster)")
        ax.set_ylabel("mean word error rate (lower is better)")
        ax.set_title("Speed vs intelligibility", loc="left", fontsize=11)
        ax.grid(color=grid)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "speed_vs_quality.png", dpi=140)
        plt.close(fig)
        files.append("speed_vs_quality.png")
    return files


# ----------------------------------------------------------------------------- markdown
def _scorers(summary: dict) -> str:
    models = sorted(set(summary.get("asr_models") or []))
    if not models:
        return "not run"
    return (f"Whisper `{', '.join(models)}` (word error rate, corpus-level), ECAPA-TDNN speaker similarity, "
            "Perth watermark detection")


def _f(x, nd=3, unit=""):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}{unit}"


def _ci(m: dict, nd=3) -> str:
    if not m or not m.get("n"):
        return "–"
    return f"{_f(m['median'], nd)} [{_f(m['ci_low'], nd)}–{_f(m['ci_high'], nd)}]"


def _ratio(c: dict) -> str:
    if not c or not c.get("n"):
        return "–"
    return f"{c['ratio']:.3f}× [{c['ci_low']:.3f}–{c['ci_high']:.3f}]"


def render_markdown(summary: dict, charts: list[str]) -> str:
    run, env, suite = summary["run"], summary["env"], summary["run"]["suite"]
    baseline = suite["baseline"]
    L = [f"# NeuTTS benchmark report: `{run['run_id']}`", ""]
    if run.get("mock"):
        L += ["> **MOCK RUN.** These numbers come from the mock engine that tests the framework itself.",
              "> They say nothing about NeuTTS performance.", ""]
    L += [
        f"**Hardware:** {run['hardware']}  ",
        f"**CPU features:** {', '.join(env.get('cpu_flags') or []) or '–'}  ",
        f"**llama.cpp build:** `{env.get('llama_cpp_build') or 'n/a'}`  ",
        f"**Power:** {json.dumps(env.get('power') or {})}  ",
        f"**Suite:** `{suite['name']}`: {len(suite['configs'])} configs, {suite['repeats']} repeats × "
        f"{len(suite['seeds'])} seeds × {len(suite['voices'])} voices, {suite['rounds']} round(s), "
        f"warm-up {suite['warmup']}, noise floor ±{suite['noise_floor']:.0%}. Baseline: `{baseline}`.  ",
        "**Load pattern:** "
        + ("requests paced at playback speed (each waits until the previous audio would have finished playing"
           + (f", plus {suite.get('pacing_gap_s', 0):g} s" if suite.get("pacing_gap_s") else "") + "), as a voice app runs"
           if suite.get("pacing") == "playback" else "back to back (sustained load)")
        + (f"; {suite.get('cooldown_s', 0):g} s idle before each configuration" if suite.get("cooldown_s") else "") + ".  ",
        f"**Quality scorers:** {_scorers(summary)}  ",
        f"**Run:** {run['started']} → {run['finished']}",
        "",
        "Every number is a median with a 95% bootstrap confidence interval in brackets. "
        "Comparisons are paired: each configuration ran the identical requests (sentence, voice, seed, repeat).",
        "",
        "## Headline",
        "",
    ]
    if not summary["best"]:
        L.append("No configuration beat the baseline beyond the noise floor while passing every quality gate.")
    for mode, b in summary["best"].items():
        metric, label = PRIMARY[mode]
        base_m = summary["configs"][baseline]["modes"][mode]["metrics"][metric]
        best_m = summary["configs"][b["config"]]["modes"][mode]["metrics"][metric]
        L.append(f"- **{mode}:** {label} {_f(base_m['median'])} → {_f(best_m['median'])} with `{b['config']}` "
                 f"(paired ratio {_ratio(summary['comparisons'][b['config']][mode]['speed'][metric])}, quality gates passed).")
    L.append("")

    for mode in ("stream", "batch"):
        names = [n for n, c in summary["configs"].items() if mode in c["modes"]]
        if not names:
            continue
        L += [f"## {mode.capitalize()} mode", "",
              "| Config | TTFA s | Total s | RTF | Prefill tok/s | Decode tok/s | Peak memory MB (requests) | OK |",
              "|---|---|---|---|---|---|---|---|"]
        for n in names:
            m = summary["configs"][n]["modes"][mode]
            mt = m["metrics"]
            L.append(f"| `{n}` | {_ci(mt.get('ttfa_s'))} | {_ci(mt.get('total_s'))} | {_ci(mt.get('rtf'))} | "
                     f"{_f(mt.get('prefill_tok_s', {}).get('median'), 0)} | {_f(mt.get('decode_tok_s', {}).get('median'), 0)} | "
                     f"{_f(m['metrics'].get('peak_mem_mb', {}).get('max'), 0)} | {m['n_ok']}/{m['n']} |")
        L.append("")
        L += ["**Where the time goes** (median ms of work per second of audio):", "",
              "| Config | " + " | ".join(STAGES) + " |", "|---|" + "---|" * len(STAGES)]
        for n in names:
            mt = summary["configs"][n]["modes"][mode]["metrics"]
            L.append(f"| `{n}` | " + " | ".join(_f(mt.get(f"{s}_ms_per_s", {}).get("median"), 1) for s in STAGES) + " |")
        L.append("")
        if mode == "stream":
            L += ["**Streaming health:**", "", "| Config | Requests with underruns | Median stall s | Min headroom s (p50) | Fidelity dB (vs one-pass decode) |", "|---|---|---|---|---|"]
            for n in names:
                m = summary["configs"][n]["modes"][mode]
                mt = m["metrics"]
                L.append(f"| `{n}` | {m['failures']['underrun_requests']}/{m['n_ok']} | {_f(mt.get('stall_s', {}).get('median'))} | "
                         f"{_f(mt.get('min_headroom_s', {}).get('median'))} | {_f(mt.get('fidelity_db', {}).get('median'), 2)} |")
            L.append("")

    L += ["## Compared with the baseline", "",
          "Ratio = candidate ÷ baseline per request (below 1 is faster). Quality deltas are candidate − baseline; "
          "WER Δ is corpus-level (errors ÷ words over all sentences) with a paired bootstrap.", "",
          "| Config | Mode | Primary metric ratio | Speed | WER Δ | Speaker sim Δ | Gates | Verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for n, modes in summary["comparisons"].items():
        for mode, c in modes.items():
            q = c["quality"]
            gates = ", ".join(f"{k}:{v}" for k, v in c["gates"].items() if v != "not scored") or "not scored"
            wer = f"{q['wer']['delta']:+.3f}" if q.get("wer", {}).get("n") else "–"
            spk = f"{q['spk_sim']['delta']:+.3f}" if q.get("spk_sim", {}).get("n") else "–"
            L.append(f"| `{n}` | {mode} | {_ratio(c['speed'][c['primary']])} | {c['speed_verdict']} | {wer} | {spk} | {gates} | **{c['overall']}** |")
    L.append("")
    changes = [(n, mode, d) for n, modes in summary["comparisons"].items() for mode, c in modes.items()
               for d in c["quality"].get("largest_wer_changes", []) if abs(d[0]) > 0]
    if changes:
        L += ["**Largest word-error changes against the baseline** (per sentence; listen to these first):", "",
              "| Config | Mode | Sentence | Voice | WER Δ | Baseline heard | Candidate heard |", "|---|---|---|---|---|---|---|"]
        for n, mode, (d, sid, voice, hb, hc) in changes:
            L.append(f"| `{n}` | {mode} | {sid} | {voice} | {d:+.2f} | {hb[:90]} | {hc[:90]} |")
        L.append("")

    L += ["## Quality", "", "| Config | Mode | WER (corpus) | WER (mean per sentence) | CER (mean) | Speaker sim (mean) | MOS | Watermark detected | Errors | Non-EOS stops | Long silences |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, c in summary["configs"].items():
        for mode, m in c["modes"].items():
            q = m["quality"]
            wd = m["watermark_detect_rate"]
            L.append(f"| `{n}` | {mode} | {_f(q.get('wer_corpus'))} | {_f(q['wer'].get('mean'))} | {_f(q['cer'].get('mean'))} | {_f(q['spk_sim'].get('mean'))} | "
                     f"{_f(q['mos'].get('mean'), 2)} | {'–' if wd is None else f'{wd:.0%}'} | {m['failures']['errors']} | "
                     f"{m['failures']['non_eos_stops']} | {m['failures']['long_silences']} |")
    metric = next((c["mem_metric"] for c in summary["configs"].values() if c.get("mem_metric")), "rss")
    L += ["", "## Start-up and memory", "",
          f"Memory is the process's {'physical footprint (includes Metal/GPU allocations)' if metric == 'phys_footprint' else 'resident set size'}.",
          "The cold request is the first real request after load, in the mode shown.", "",
          "| Config | Load s | Cold first request: TTFA s (mode) | Memory after load MB | Peak during load MB |",
          "|---|---|---|---|---|"]
    for n, c in summary["configs"].items():
        L.append(f"| `{n}` | {_f(c['load_s'].get('median'), 2)} | {_f(c['cold_ttfa_s'].get('median'), 3)} ({c.get('cold_mode') or '–'}) | "
                 f"{_f(c['mem_after_load_mb'], 0)} | {_f(c['peak_mem_load_mb'], 0)} |")
    L.append("")
    if charts:
        L += ["## Charts", ""] + [f"![{c}](charts/{c})" for c in charts] + [""]
    failed = {n: s for n, s in run["status"].items() if not all(x["ok"] for x in s)}
    if failed:
        L += ["## Failed configurations", "", *[f"- `{n}`: {s}" for n, s in failed.items()], ""]
    src = suite.get("source") or "suites/<suite>.yaml"
    L += ["## Reproduce", "", "```bash", f"python -m ttsbench all {src}", "```", ""]
    return "\n".join(L)


def write_report(run_dir: Path) -> Path:
    summary = build_summary(run_dir)
    charts = make_charts(summary, run_dir / "charts")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    path = run_dir / "report.md"
    path.write_text(render_markdown(summary, charts))
    return path


def compare_runs(run_a: Path, run_b: Path, config_a: str, config_b: str, out: Path) -> Path:
    """Cross-run comparison, e.g. the same config before and after a native llama.cpp build."""
    rec_a = {r["pair_key"]: r for r in load_records(run_a) if r["kind"] == "measure" and r["ok"] and r["config"] == config_a}
    rec_b = {r["pair_key"]: r for r in load_records(run_b) if r["kind"] == "measure" and r["ok"] and r["config"] == config_b}
    noise = json.loads((run_a / "run.json").read_text())["suite"]["noise_floor"]
    env_a, env_b = (json.loads((p / "env.json").read_text()) for p in (run_a, run_b))
    L = [f"# Comparison: `{run_a.name}:{config_a}` → `{run_b.name}:{config_b}`", "",
         f"- A: {env_a.get('llama_cpp_build')}", f"- B: {env_b.get('llama_cpp_build')}", "",
         "| Mode | Metric | Ratio B/A [95% CI] | Verdict | Pairs |", "|---|---|---|---|---|"]
    for mode in ("stream", "batch"):
        a = {k: v for k, v in rec_a.items() if v["mode"] == mode}
        b = {k: v for k, v in rec_b.items() if v["mode"] == mode}
        for metric in ("ttfa_s", "total_s", "rtf"):
            c = stats.paired_ratio(stats.pair_up({k: v.get(metric) for k, v in a.items()}, {k: v.get(metric) for k, v in b.items()}))
            if c.get("n"):
                L.append(f"| {mode} | {metric} | {_ratio(c)} | {stats.speed_verdict(c, noise)} | {c['n']} |")
    out.write_text("\n".join(L) + "\n")
    return out
