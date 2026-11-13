"""Map each monthly parquet to small aggregates, then reduce them.

Workers return histogram counts and zone-hour arrays. They do not return the
raw trips. A systematic sample is written per file and concatenated by the
parent. That split is the map/reduce boundary: communication is O(zones),
not O(rows).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline.clean import RULES, audit_frame, clean_frame

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXT = ROOT / "data" / "external"
OUT = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"

N_ZONES = 266
N_HOURS = 24
N_DOW = 7
SAMPLE_PER_FILE = 50_000

HIST_SPEC = {
    "distance": np.linspace(0, 40, 401),
    "duration_min": np.linspace(0, 180, 361),
    "fare": np.linspace(0, 150, 301),
    "tip_ratio": np.linspace(0, 1, 201),
}

_VALID: set[int] | None = None


def load_valid_zones(path: Path | None = None) -> set[int]:
    path = path or (EXT / "taxi_zone_lookup.csv")
    zones = pd.read_csv(path)
    keep = zones[zones["Borough"].astype(str).str.lower() != "unknown"]
    return set(int(v) for v in keep["LocationID"])


def _rename(table):
    lookup = {name.lower(): name for name in table.column_names}
    wanted = {
        "tpep_pickup_datetime": "pickup",
        "tpep_dropoff_datetime": "dropoff",
        "passenger_count": "passenger_count",
        "trip_distance": "trip_distance",
        "ratecodeid": "RatecodeID",
        "pulocationid": "PULocationID",
        "dolocationid": "DOLocationID",
        "payment_type": "payment_type",
        "fare_amount": "fare_amount",
        "tip_amount": "tip_amount",
        "total_amount": "total_amount",
    }
    present = {}
    for src, dst in wanted.items():
        if src in lookup:
            present[lookup[src]] = dst
    df = table.select(list(present)).to_pandas()
    df = df.rename(columns=present)
    for col in wanted.values():
        if col not in df.columns:
            df[col] = np.nan
    df["pickup"] = pd.to_datetime(df["pickup"], errors="coerce")
    df["dropoff"] = pd.to_datetime(df["dropoff"], errors="coerce")
    for col in ("passenger_count", "trip_distance", "RatecodeID", "fare_amount", "tip_amount", "total_amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("PULocationID", "DOLocationID", "payment_type"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _histograms(clean: pd.DataFrame) -> dict[str, np.ndarray]:
    duration_min = clean["duration_s"].to_numpy() / 60.0
    tip_ratio = np.divide(
        clean["tip_amount"].to_numpy(),
        clean["fare_amount"].to_numpy(),
        out=np.zeros(len(clean), dtype=np.float64),
        where=clean["fare_amount"].to_numpy() > 0,
    )
    tip_ratio = np.clip(tip_ratio, 0, 1)
    values = {
        "distance": clean["trip_distance"].to_numpy(),
        "duration_min": duration_min,
        "fare": clean["fare_amount"].to_numpy(),
        "tip_ratio": tip_ratio,
    }
    out = {}
    for name, edges in HIST_SPEC.items():
        counts, _ = np.histogram(values[name], bins=edges)
        out[name] = counts.astype(np.int64)
    return out


def _zone_arrays(clean: pd.DataFrame):
    shape = (N_ZONES, N_HOURS, N_DOW)
    count = np.zeros(shape, dtype=np.int32)
    sum_distance = np.zeros(shape, dtype=np.float64)
    sum_fare = np.zeros(shape, dtype=np.float64)
    sum_tip = np.zeros(shape, dtype=np.float64)
    sum_duration = np.zeros(shape, dtype=np.float64)
    if len(clean) == 0:
        return count, sum_distance, sum_fare, sum_tip, sum_duration
    pu = clean["PULocationID"].to_numpy(dtype=np.int32)
    hour = clean["pickup"].dt.hour.to_numpy(dtype=np.int32)
    dow = clean["pickup"].dt.day_of_week.to_numpy(dtype=np.int32)
    ok = (pu >= 0) & (pu < N_ZONES) & (hour >= 0) & (hour < N_HOURS)
    np.add.at(count, (pu[ok], hour[ok], dow[ok]), 1)
    np.add.at(sum_distance, (pu[ok], hour[ok], dow[ok]), clean["trip_distance"].to_numpy()[ok])
    np.add.at(sum_fare, (pu[ok], hour[ok], dow[ok]), clean["fare_amount"].to_numpy()[ok])
    np.add.at(sum_tip, (pu[ok], hour[ok], dow[ok]), clean["tip_amount"].fillna(0).to_numpy()[ok])
    np.add.at(sum_duration, (pu[ok], hour[ok], dow[ok]), clean["duration_s"].to_numpy()[ok])
    return count, sum_distance, sum_fare, sum_tip, sum_duration


def _monthly(clean: pd.DataFrame) -> dict[tuple[int, int], int]:
    if len(clean) == 0:
        return {}
    year = clean["pickup"].dt.year.to_numpy()
    month = clean["pickup"].dt.month.to_numpy()
    keys, counts = np.unique(np.stack([year, month], axis=1), axis=0, return_counts=True)
    return {(int(y), int(m)): int(c) for (y, m), c in zip(keys, counts)}


def _sample(clean: pd.DataFrame, stem: str) -> pd.DataFrame:
    n = len(clean)
    if n == 0:
        return clean
    step = max(1, int(np.ceil(n / SAMPLE_PER_FILE)))
    take = clean.iloc[::step].head(SAMPLE_PER_FILE).copy()
    take["source_file"] = stem
    keep_cols = [
        "pickup",
        "PULocationID",
        "DOLocationID",
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "passenger_count",
        "payment_type",
        "duration_s",
        "speed_mph",
        "source_file",
    ]
    return take[keep_cols]


def _empty_arrays():
    shape = (N_ZONES, N_HOURS, N_DOW)
    return {
        "count": np.zeros(shape, dtype=np.int64),
        "sum_distance": np.zeros(shape, dtype=np.float64),
        "sum_fare": np.zeros(shape, dtype=np.float64),
        "sum_tip": np.zeros(shape, dtype=np.float64),
        "sum_duration": np.zeros(shape, dtype=np.float64),
        "hist": {name: np.zeros(len(edges) - 1, dtype=np.int64) for name, edges in HIST_SPEC.items()},
        "monthly": {},
    }


def _add_monthly(total: dict, extra: dict) -> None:
    for key, value in extra.items():
        total[key] = total.get(key, 0) + int(value)


def process_month(path: str, valid_zones: set[int] | None = None, with_payload: bool = True) -> dict:
    """Clean one file in row batches so a worker does not hold the whole month."""
    zones = valid_zones if valid_zones is not None else _VALID
    if zones is None:
        raise RuntimeError("valid zones were not provided")
    path = str(path)
    wanted = [
        "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count", "trip_distance",
        "RatecodeID", "PULocationID", "DOLocationID", "payment_type", "fare_amount",
        "tip_amount", "total_amount",
    ]
    pf = pq.ParquetFile(path)
    available = {name.lower(): name for name in pf.schema_arrow.names}
    columns = [available[name.lower()] for name in wanted if name.lower() in available]
    read_s = clean_s = agg_s = 0.0
    counts = {name: 0 for name in RULES}
    counts["rows_raw"] = 0
    counts["rows_clean"] = 0
    audit = {}
    acc = _empty_arrays()
    samples = []
    for batch in pf.iter_batches(batch_size=1_000_000, columns=columns):
        t0 = time.perf_counter()
        df = _rename(pa_table_from_batch(batch))
        read_s += time.perf_counter() - t0
        t1 = time.perf_counter()
        batch_audit = audit_frame(df, zones)
        clean, batch_counts = clean_frame(df, zones)
        clean_s += time.perf_counter() - t1
        for key, value in batch_counts.items():
            counts[key] = counts.get(key, 0) + int(value)
        for key, value in batch_audit.items():
            audit[key] = audit.get(key, 0) + int(value)
        t2 = time.perf_counter()
        count, sum_distance, sum_fare, sum_tip, sum_duration = _zone_arrays(clean)
        acc["count"] += count
        acc["sum_distance"] += sum_distance
        acc["sum_fare"] += sum_fare
        acc["sum_tip"] += sum_tip
        acc["sum_duration"] += sum_duration
        for name, values in _histograms(clean).items():
            acc["hist"][name] += values
        _add_monthly(acc["monthly"], _monthly(clean))
        if with_payload and len(clean):
            samples.append(_sample(clean, Path(path).stem))
        agg_s += time.perf_counter() - t2
        del df, clean
    payload = {}
    if with_payload:
        sample = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
        if len(sample) > SAMPLE_PER_FILE:
            sample = sample.iloc[:: max(1, len(sample) // SAMPLE_PER_FILE)].head(SAMPLE_PER_FILE)
        payload = {
            "count": acc["count"],
            "sum_distance": acc["sum_distance"],
            "sum_fare": acc["sum_fare"],
            "sum_tip": acc["sum_tip"],
            "sum_duration": acc["sum_duration"],
            "hist": acc["hist"],
            "monthly": acc["monthly"],
            "sample": sample,
        }
    return {
        "file": Path(path).name,
        "counts": counts,
        "audit": audit,
        "times": {"read_s": read_s, "clean_s": clean_s, "agg_s": agg_s},
        "payload": payload,
    }


def pa_table_from_batch(batch):
    import pyarrow as pa
    return pa.Table.from_batches([batch])


def _init_worker(zones: set[int]) -> None:
    global _VALID
    _VALID = zones


def _task(path: str) -> dict:
    return process_month(path, _VALID, with_payload=True)


def _task_timeonly(path: str) -> dict:
    zones = _VALID if _VALID is not None else load_valid_zones()
    return process_month(path, zones, with_payload=False)


def list_trip_files() -> list[Path]:
    files = sorted(RAW.glob("yellow_tripdata_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {RAW}; run python -m pipeline.download")
    return files


def reduce_results(parts: list[dict]) -> dict:
    t0 = time.perf_counter()
    count = np.zeros((N_ZONES, N_HOURS, N_DOW), dtype=np.int64)
    sum_distance = np.zeros_like(count, dtype=np.float64)
    sum_fare = np.zeros_like(count, dtype=np.float64)
    sum_tip = np.zeros_like(count, dtype=np.float64)
    sum_duration = np.zeros_like(count, dtype=np.float64)
    hist = {name: np.zeros(len(edges) - 1, dtype=np.int64) for name, edges in HIST_SPEC.items()}
    monthly: dict[tuple[int, int], int] = {}
    samples = []
    rule_rows = []
    for part in parts:
        payload = part["payload"]
        count += payload["count"]
        sum_distance += payload["sum_distance"]
        sum_fare += payload["sum_fare"]
        sum_tip += payload["sum_tip"]
        sum_duration += payload["sum_duration"]
        for name in hist:
            hist[name] += payload["hist"][name]
        for key, value in payload["monthly"].items():
            monthly[key] = monthly.get(key, 0) + int(value)
        if len(payload["sample"]):
            samples.append(payload["sample"])
        row = {"file": part["file"], **part["counts"], **part["times"]}
        rule_rows.append(row)
    sample = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    merge_s = time.perf_counter() - t0
    return {
        "count": count,
        "sum_distance": sum_distance,
        "sum_fare": sum_fare,
        "sum_tip": sum_tip,
        "sum_duration": sum_duration,
        "hist": hist,
        "monthly": monthly,
        "sample": sample,
        "rules": pd.DataFrame(rule_rows),
        "merge_s": merge_s,
    }


def run(workers: int = 4, executor: str = "process", files: list[Path] | None = None, chunksize: int = 1) -> dict:
    zones = load_valid_zones()
    files = files or list_trip_files()
    paths = [str(p) for p in files]
    pool_cls = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
    t0 = time.perf_counter()
    kwargs = {"max_workers": workers}
    if executor == "process":
        kwargs["initializer"] = _init_worker
        kwargs["initargs"] = (zones,)
    with pool_cls(**kwargs) as ex:
        if executor == "thread":
            parts = list(ex.map(lambda p: process_month(p, zones, True), paths, chunksize=chunksize))
        else:
            parts = list(ex.map(_task, paths, chunksize=chunksize))
    wall = time.perf_counter() - t0
    reduced = reduce_results(parts)
    reduced["wall_s"] = wall
    reduced["workers"] = workers
    reduced["executor"] = executor
    reduced["n_files"] = len(paths)
    return reduced


def save_reduced(reduced: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "zone_hour_dow.npz",
        count=reduced["count"],
        sum_distance=reduced["sum_distance"],
        sum_fare=reduced["sum_fare"],
        sum_tip=reduced["sum_tip"],
        sum_duration=reduced["sum_duration"],
    )
    hist_kwargs = {name: counts for name, counts in reduced["hist"].items()}
    for name, edges in HIST_SPEC.items():
        hist_kwargs[name + "_edges"] = edges
    np.savez_compressed(OUT / "histograms.npz", **hist_kwargs)
    monthly = (
        pd.DataFrame([{"year": y, "month": m, "n": n} for (y, m), n in sorted(reduced["monthly"].items())])
    )
    monthly.to_csv(OUT / "monthly.csv", index=False)
    reduced["rules"].to_csv(OUT / "rule_stats.csv", index=False)
    sample = reduced["sample"]
    if len(sample):
        sample.to_parquet(OUT / "sample.parquet", index=False)
    print(
        f"saved aggregates files={reduced['n_files']} clean_rows={int(reduced['count'].sum()):,} "
        f"sample={len(sample):,} wall={reduced['wall_s']:.1f}s merge={reduced['merge_s']:.1f}s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--executor", choices=["process", "thread"], default="process")
    parser.add_argument("--limit", type=int, default=0, help="if >0, only the first N files")
    args = parser.parse_args()
    files = list_trip_files()
    if args.limit:
        files = files[: args.limit]
    save_reduced(run(workers=args.workers, executor=args.executor, files=files))


if __name__ == "__main__":
    main()
