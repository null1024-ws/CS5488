"""Scaling matrix. Every repeat is appended to results/runs.csv.

T3 runs before Numba is imported. Forking a process pool after Numba has
started its thread pool is a known deadlock on Linux.

Numba freezes NUMBA_NUM_THREADS from the process CPU affinity at import.
`run_timed` pins affinity to p cores before the timed call, so the pool is
opened here, on the full set of CPUs, before any pin. The pool is sized to
8 so the p=8 oversubscription point can call `set_num_threads(8)`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
from pathlib import Path

import psutil

from bench.bench import run_timed, unbind
from pipeline.etl_parallel import list_trip_files, process_month, load_valid_zones

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs.csv"
FIELDS = [
    "workload", "mode", "p", "chunksize", "block", "n_input", "repeat",
    "wall_s", "read_s", "clean_s", "agg_s", "merge_s", "cpu_pct", "rss_mb", "note",
]


def _write_machine():
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "memory_bytes": psutil.virtual_memory().total,
    }
    freq = psutil.cpu_freq()
    if freq:
        info["cpu_freq_mhz"] = freq.current
    (RESULTS / "machine.json").write_text(json.dumps(info, indent=2))
    return info


def _append(rows: list[dict]):
    RESULTS.mkdir(parents=True, exist_ok=True)
    new = not RUNS.exists()
    with RUNS.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})


def _record(workload, mode, p, n_input, timed, chunksize="", block="", note=""):
    rows = []
    for item in timed:
        rows.append({
            "workload": workload,
            "mode": mode,
            "p": p,
            "chunksize": chunksize,
            "block": block,
            "n_input": n_input,
            "repeat": item["repeat"],
            "wall_s": f"{item['wall_s']:.6f}",
            "read_s": f"{item.get('read_s', '')}" if "read_s" in item else "",
            "clean_s": f"{item.get('clean_s', '')}" if "clean_s" in item else "",
            "agg_s": f"{item.get('agg_s', '')}" if "agg_s" in item else "",
            "merge_s": f"{item.get('merge_s', '')}" if "merge_s" in item else "",
            "cpu_pct": f"{item['cpu_pct']:.4f}" if item["cpu_pct"] == item["cpu_pct"] else "",
            "rss_mb": f"{item['rss_mb']:.2f}",
            "note": note,
        })
    _append(rows)
    walls = [item["wall_s"] for item in timed]
    print(f"  {workload} {mode} p={p} n={n_input} median={statistics.median(walls):.3f}s", flush=True)


def bench_etl(files: list[Path], workers: int, mode: str, repeats: int, chunksize: int = 1, note: str = ""):
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    import pipeline.etl_parallel as etl

    zones = load_valid_zones()
    etl._VALID = zones
    paths = [str(p) for p in files]

    def once():
        pool = ProcessPoolExecutor if mode == "process" else ThreadPoolExecutor
        kwargs = {"max_workers": workers}
        if mode == "process":
            kwargs["initializer"] = etl._init_worker
            kwargs["initargs"] = (zones,)
        with pool(**kwargs) as ex:
            parts = list(ex.map(etl._task_timeonly, paths, chunksize=chunksize))
        return {
            "read_s": sum(part["times"]["read_s"] for part in parts),
            "clean_s": sum(part["times"]["clean_s"] for part in parts),
            "agg_s": sum(part["times"]["agg_s"] for part in parts),
            "merge_s": 0.0,
        }

    timed = run_timed(once, workers, repeats=repeats, warmup=1)
    _record("T1", mode, workers, len(files), timed, chunksize=chunksize, note=note or f"files={len(files)}")


def _python_records(path: Path, n: int = 400_000):
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["trip_distance", "fare_amount", "tpep_pickup_datetime", "tpep_dropoff_datetime"])
    df = table.to_pandas()
    dur = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds().fillna(-1).to_numpy()
    dist = df["trip_distance"].to_numpy()
    fare = df["fare_amount"].to_numpy()
    n = min(n, len(df))
    return list(zip(dist[:n].tolist(), fare[:n].tolist(), dur[:n].tolist()))


def bench_gil(records, workers: int, mode: str, repeats: int):
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    from bench.gil import python_sum

    chunks = [records[i::workers] for i in range(workers)]

    def once():
        pool = ProcessPoolExecutor if mode == "process" else ThreadPoolExecutor
        with pool(max_workers=workers) as ex:
            list(ex.map(python_sum, chunks))
        return {}

    timed = run_timed(once, workers, repeats=repeats, warmup=1)
    _record("GIL", mode, workers, len(records), timed, note="pure Python row loop")


def bench_t3(transactions, workers: int, repeats: int):
    from algorithms.fpgrowth import mine_parallel

    def once():
        mine_parallel(transactions, 0.02, workers)
        return {}

    timed = run_timed(once, workers, repeats=repeats, warmup=1)
    _record("T3", "fpgrowth", workers, len(transactions), timed, note="min_support=0.02")


def prime_numba(max_threads: int = 8) -> None:
    """Open the Numba pool at `max_threads` before any affinity pin."""
    unbind()
    os.environ["NUMBA_NUM_THREADS"] = str(max_threads)
    import numba as nb

    nb.set_num_threads(1)
    nb.set_num_threads(max_threads)


def bench_t2(X, init, workers: int, repeats: int, block: int, parallel_update: bool, note: str = ""):
    from algorithms.kmeans_par import kmeans

    def once():
        kmeans(X, init, n_iter=30, n_threads=workers, block=block, parallel_update=parallel_update)
        return {}

    timed = run_timed(once, workers, repeats=repeats, warmup=1)
    label = note or ("parallel_update" if parallel_update else "serial_update")
    _record("T2", "numba", workers, len(X), timed, block=block, note=label)


def _drop_workload(name: str) -> None:
    if not RUNS.exists():
        return
    with RUNS.open(newline="") as f:
        rows = list(csv.DictReader(f))
    kept = [row for row in rows if row.get("workload") != name]
    with RUNS.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(kept)


def _run_t2() -> None:
    import numpy as np

    print("T2 k-means", flush=True)
    prime_numba(8)
    X = np.load(ROOT / "data" / "processed" / "kmeans_X.npy")
    init = np.load(ROOT / "data" / "processed" / "kmeans_init.npy")
    for p in (1, 2, 4, 8):
        bench_t2(X, init, p, repeats=5, block=8192, parallel_update=True)
    for block in (64, 256, 1024, 4096, 16384, 65536, 262144):
        bench_t2(X, init, 4, repeats=3, block=block, parallel_update=True)
    # Weak scaling: 650k rows per worker, p in {1, 2, 4}.
    for p, n in ((1, 650_000), (2, 1_300_000), (4, 2_600_000)):
        bench_t2(X[:n], init, p, repeats=3, block=8192, parallel_update=True, note="weak")


def main(start: str = "all") -> None:
    if start == "all" and RUNS.exists():
        RUNS.unlink()
    info = _write_machine()
    print("machine", info, flush=True)
    if start == "t2":
        _drop_workload("T2")
        _run_t2()
        print("matrix done", RUNS, flush=True)
        return

    files = list_trip_files()
    by_name = {p.name: p for p in files}
    one = [by_name["yellow_tripdata_2023-01.parquet"]]
    year = [by_name[f"yellow_tripdata_2023-{m:02d}.parquet"] for m in range(1, 13)]

    print("T1 2023, 12 files", flush=True)
    for p in (1, 2, 4, 8):
        bench_etl(year, p, "process", repeats=5, note="2023 x12")
    bench_etl(year, 4, "thread", repeats=5, note="2023 x12 thread")
    for chunk in (4, 12):
        bench_etl(year, 4, "process", repeats=3, chunksize=chunk, note=f"2023 x12 chunk={chunk}")
    print("T1 single month, shows one file cannot fill extra workers", flush=True)
    for p in (1, 4):
        bench_etl(one, p, "process", repeats=3, note="2023-01 only")

    print("GIL microbenchmark", flush=True)
    records = _python_records(one[0])
    for mode in ("thread", "process"):
        for p in (1, 4):
            bench_gil(records, p, mode, repeats=5)

    print("T3 FP-Growth", flush=True)
    import numpy as np
    tx_path = ROOT / "data" / "processed" / "transactions_timing.npy"
    transactions = np.load(tx_path, allow_pickle=False).tolist()
    transactions = [list(map(int, row)) for row in transactions]
    for p in (1, 2, 4, 8):
        bench_t3(transactions, p, repeats=5)

    _run_t2()
    print("matrix done", RUNS, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", default="all", choices=["all", "t2"])
    args = parser.parse_args()
    main(args.start)
