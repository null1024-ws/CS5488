"""Shared timing protocol: one warmup, repeated runs, median is computed later."""

from __future__ import annotations

import statistics
import threading
import time

import psutil


def bind(workers: int) -> None:
    proc = psutil.Process()
    n = psutil.cpu_count(logical=True) or 1
    proc.cpu_affinity(list(range(min(max(workers, 1), n))))


def unbind() -> None:
    proc = psutil.Process()
    proc.cpu_affinity(list(range(psutil.cpu_count(logical=True) or 1)))


def _rss_mb() -> float:
    proc = psutil.Process()
    rss = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.Error:
            pass
    return rss / 2**20


def run_timed(fn, workers: int, repeats: int = 5, warmup: int = 1) -> list[dict]:
    bind(workers)
    rows = []
    try:
        for _ in range(warmup):
            fn()
        for repeat in range(repeats):
            stop = threading.Event()
            samples = []

            def sample():
                psutil.cpu_percent(interval=None)
                while not stop.wait(0.2):
                    samples.append(psutil.cpu_percent(interval=None))

            thread = threading.Thread(target=sample, daemon=True)
            thread.start()
            t0 = time.perf_counter()
            extra = fn() or {}
            wall = time.perf_counter() - t0
            stop.set()
            thread.join()
            cpu = statistics.median(samples) if samples else float("nan")
            row = {
                "repeat": repeat,
                "wall_s": wall,
                "cpu_pct": cpu,
                "rss_mb": _rss_mb(),
            }
            if isinstance(extra, dict):
                for key in ("read_s", "clean_s", "agg_s", "merge_s"):
                    if key in extra:
                        row[key] = extra[key]
            rows.append(row)
    finally:
        unbind()
    return rows
