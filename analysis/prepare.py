"""Build the fixed k-means matrix and the FP-Growth timing transactions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms.kmeans_par import init_centers
from pipeline.features import add_clock_features, attach_weather, load_weather

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EXT = ROOT / "data" / "external"


def _zones() -> pd.DataFrame:
    zones = pd.read_csv(EXT / "taxi_zone_lookup.csv")
    zones["LocationID"] = zones["LocationID"].astype(int)
    return zones[["LocationID", "Borough", "Zone", "service_zone"]]


def load_enriched() -> pd.DataFrame:
    sample = pd.read_parquet(PROC / "sample.parquet")
    sample["pickup"] = pd.to_datetime(sample["pickup"])
    sample = add_clock_features(sample)
    sample = sample.merge(_zones(), left_on="PULocationID", right_on="LocationID", how="left")
    sample = sample.rename(columns={"Borough": "borough", "Zone": "zone_name"})
    weather = load_weather(EXT / "USW00094728.csv")
    sample = attach_weather(sample, weather)
    return sample


def build_kmeans_matrix(sample: pd.DataFrame, n: int = 2_600_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    take = sample
    if len(take) > n:
        take = take.iloc[rng.choice(len(take), size=n, replace=False)].copy()
    hour = take["hour"].to_numpy(dtype=np.float64) / 23.0
    dow = take["dow"].to_numpy(dtype=np.float64) / 6.0
    distance = np.log1p(take["trip_distance"].to_numpy(dtype=np.float64))
    duration = np.log1p(take["duration_s"].to_numpy(dtype=np.float64) / 60.0)
    fare = np.log1p(take["fare_amount"].to_numpy(dtype=np.float64))
    tip = np.clip(take["tip_ratio"].to_numpy(dtype=np.float64), 0, 1)
    speed = np.log1p(take["speed_mph"].to_numpy(dtype=np.float64))
    passenger = take["passenger_count"].fillna(1).to_numpy(dtype=np.float64) / 6.0
    raw = np.column_stack([hour, dow, distance, duration, fare, tip, speed, passenger])
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std[std == 0] = 1.0
    X = ((raw - mean) / std).astype(np.float32)
    init = init_centers(X, k=24, seed=seed).astype(np.float32)
    np.save(PROC / "kmeans_X.npy", X)
    np.save(PROC / "kmeans_init.npy", init)
    (PROC / "kmeans_meta.json").write_text(json.dumps({
        "n": int(len(X)),
        "k_timing": 24,
        "n_iter": 30,
        "seed": seed,
        "features": ["hour", "dow", "log_distance", "log_duration", "log_fare", "tip_ratio", "log_speed", "passenger"],
        "mean": mean.tolist(),
        "std": std.tolist(),
    }, indent=2))
    return X


def build_timing_transactions(sample: pd.DataFrame, n: int = 1_200_000, seed: int = 7):
    rng = np.random.default_rng(seed)
    use = sample[sample["wx"] != "unknown"]
    if len(use) > n:
        use = use.iloc[rng.choice(len(use), size=n, replace=False)]
    # Integer codes. Zone id keeps the tree irregular; the other fields stay small.
    tod = use["tod"].astype("category")
    pay = use["pay_bin"].astype("category")
    dist = use["dist_bin"].astype("category")
    coded = np.column_stack([
        use["PULocationID"].to_numpy(dtype=np.int16),
        tod.cat.codes.to_numpy(dtype=np.int16) + 1000,
        pay.cat.codes.to_numpy(dtype=np.int16) + 1100,
        dist.cat.codes.to_numpy(dtype=np.int16) + 1200,
    ]).astype(np.int16)
    np.save(PROC / "transactions_timing.npy", coded)
    (PROC / "transactions_meta.json").write_text(json.dumps({
        "n": int(len(coded)),
        "fields": ["PULocationID", "tod", "pay", "dist"],
        "min_support_timing": 0.02,
        "seed": seed,
    }, indent=2))
    return coded


def main():
    sample = load_enriched()
    sample.to_parquet(PROC / "sample_enriched.parquet", index=False)
    X = build_kmeans_matrix(sample)
    tx = build_timing_transactions(sample)
    print(f"enriched={len(sample):,} kmeans={len(X):,} transactions={len(tx):,}", flush=True)


if __name__ == "__main__":
    main()
