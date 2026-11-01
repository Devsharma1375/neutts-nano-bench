import time

from ttsbench.corpus import simulate_playback
from ttsbench.timing import STAGES, StageTimer


def test_no_underrun_when_chunks_arrive_early():
    # 0.5 s chunks arriving every 0.3 s
    chunks = [(0.5 + 0.3 * i, 12000) for i in range(5)]
    p = simulate_playback(chunks)
    assert p.underruns == 0 and p.stall_s == 0 and p.min_headroom_s > 0


def test_underruns_counted_and_stall_measured():
    # 0.5 s chunks arriving every 0.6 s -> each later chunk is 0.1 s late
    chunks = [(1.0 + 0.6 * i, 12000) for i in range(4)]
    p = simulate_playback(chunks)
    assert p.underruns == 3 and abs(p.stall_s - 0.3) < 1e-9 and p.min_headroom_s < 0


def test_stages_sum_to_total():
    t = StageTimer()
    with t.stage("decode"):
        time.sleep(0.02)
    t.add("codec", 5.0)
    time.sleep(0.01)
    t.finish()
    assert set(t.ms) == set(STAGES)
    assert abs(sum(t.ms.values()) - t.total_ms) < 1e-6 or t.ms["other"] == 0
    assert t.ms["other"] > 0
