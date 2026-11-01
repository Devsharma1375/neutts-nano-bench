"""Peak memory sampling.

``resource.getrusage().ru_maxrss`` only gives the lifetime peak (and is kilobytes on
Linux but bytes on macOS), so a background thread samples instead. That gives per-phase
peaks: load, cold request, and each measured request.

What is sampled:

* macOS: the process's *physical footprint* (``proc_pid_rusage`` ``ri_phys_footprint``,
  the "Memory" column of Activity Monitor). Unlike RSS it includes Metal/GPU allocations
  (llama.cpp's Metal buffers, PyTorch MPS tensors), which on Apple's unified memory are the
  same RAM the rest of the system needs. RSS alone would make GPU configurations look
  smaller than they are.
* elsewhere: resident set size (RSS).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import threading

import psutil

METRIC = "phys_footprint" if sys.platform == "darwin" else "rss"


def _footprint_reader():
    """Return a zero-argument function giving this process's physical footprint in bytes."""
    lib = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
    buf = (ctypes.c_uint64 * 64)()  # rusage_info_v4 is 296 bytes
    pid = os.getpid()
    rusage_info_v4 = 4
    # struct rusage_info_v4: uint8 ri_uuid[16], then uint64 user_time, system_time,
    # pkg_idle_wkups, interrupt_wkups, pageins, wired_size, resident_size, phys_footprint
    footprint_index = 2 + 7

    def read() -> int:
        if lib.proc_pid_rusage(pid, rusage_info_v4, buf) != 0:
            raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
        return int(buf[footprint_index])

    read()
    return read


class MemorySampler:
    def __init__(self, interval_s: float = 0.01) -> None:
        self._proc = psutil.Process()
        self._read = _footprint_reader() if METRIC == "phys_footprint" else (lambda: self._proc.memory_info().rss)
        self._interval = interval_s
        self._peak = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mem-sampler", daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._observe()

    def _observe(self) -> int:
        used = self._read()
        with self._lock:
            self._peak = max(self._peak, used)
        return used

    def start(self) -> MemorySampler:
        self._observe()
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)

    def reset_peak(self) -> None:
        with self._lock:
            self._peak = 0
        self._observe()

    def peak_mb(self) -> float:
        self._observe()
        with self._lock:
            return self._peak / 2**20

    def current_mb(self) -> float:
        return self._observe() / 2**20
