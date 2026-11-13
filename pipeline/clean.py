"""Sequential cleaning rules for yellow-taxi trips.

R1–R7 are applied in order. A row is charged to the first rule that rejects
it, so the seven counts sum to the number of removed rows. A separate
non-exclusive audit counts how often each problem occurs in the raw file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RULES = [
    "R1_duration",
    "R2_distance",
    "R3_zone",
    "R4_fare",
    "R5_speed",
    "R6_duplicate",
    "R7_invalid_code",
]

MIN_SECONDS = 60.0
MAX_SECONDS = 3 * 3600.0
MAX_SPEED_MPH = 70.0
MIN_FARE = 2.5
MAX_FARE = 500.0


def _blank_counts() -> dict[str, int]:
    return {name: 0 for name in RULES}


def audit_frame(df: pd.DataFrame, valid_zones: set[int]) -> dict[str, int]:
    """Non-exclusive problem counts on the raw frame."""
    n = len(df)
    pickup = df["pickup"]
    dropoff = df["dropoff"]
    dur = (dropoff - pickup).dt.total_seconds()
    dist = df["trip_distance"]
    fare = df["fare_amount"]
    pu = df["PULocationID"]
    do = df["DOLocationID"]
    speed = dist / (dur / 3600.0)
    valid_speed = dur > 0
    return {
        "rows_raw": int(n),
        "missing_key": int(
            (pickup.isna() | dropoff.isna() | dist.isna() | fare.isna() | pu.isna() | do.isna()).sum()
        ),
        "negative_or_zero_duration": int(((dur <= 0) | dur.isna()).sum()),
        "duration_outside_1min_3h": int((dur.notna() & ((dur < MIN_SECONDS) | (dur > MAX_SECONDS))).sum()),
        "nonpositive_distance": int(((dist <= 0) | dist.isna()).sum()),
        "invalid_zone": int((~pu.isin(valid_zones) | ~do.isin(valid_zones) | pu.isna() | do.isna()).sum()),
        "fare_outside_range": int((fare.notna() & ((fare < MIN_FARE) | (fare > MAX_FARE))).sum() + int(fare.isna().sum())),
        "speed_at_least_70": int((valid_speed & (speed >= MAX_SPEED_MPH)).sum()),
        "duplicate_rows": int(df.duplicated(subset=["pickup", "dropoff", "PULocationID", "DOLocationID", "trip_distance", "fare_amount"]).sum()),
        "invalid_code": int(_invalid_code_mask(df).sum()),
    }


def _invalid_code_mask(df: pd.DataFrame) -> pd.Series:
    passenger = df["passenger_count"]
    rate = df["RatecodeID"]
    pay = df["payment_type"]
    bad_passenger = passenger.notna() & ((passenger < 1) | (passenger > 6))
    bad_rate = rate.notna() & ~rate.isin([1, 2, 3, 4, 5, 6])
    bad_pay = pay.notna() & ~pay.isin([1, 2, 3, 4])
    missing_pay = pay.isna()
    return bad_passenger | bad_rate | bad_pay | missing_pay


def clean_frame(df: pd.DataFrame, valid_zones: set[int]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return surviving rows and exclusive removal counts."""
    counts = _blank_counts()
    n = len(df)
    if n == 0:
        return df.iloc[0:0].copy(), counts

    pickup = df["pickup"]
    dropoff = df["dropoff"]
    dur = (dropoff - pickup).dt.total_seconds()
    dist = df["trip_distance"].to_numpy(dtype=np.float64, copy=False)
    fare = df["fare_amount"].to_numpy(dtype=np.float64, copy=False)
    pu = df["PULocationID"]
    do = df["DOLocationID"]
    keep = np.ones(n, dtype=bool)

    def reject(name: str, bad: np.ndarray) -> None:
        newly = bad & keep
        counts[name] = int(newly.sum())
        keep[newly] = False

    bad_r1 = pickup.isna().to_numpy() | dropoff.isna().to_numpy() | ~np.isfinite(dur.to_numpy())
    bad_r1 |= (dur.to_numpy() < MIN_SECONDS) | (dur.to_numpy() > MAX_SECONDS)
    reject("R1_duration", bad_r1)

    bad_r2 = ~np.isfinite(dist) | (dist <= 0)
    reject("R2_distance", bad_r2)

    pu_ok = pu.isin(valid_zones).to_numpy()
    do_ok = do.isin(valid_zones).to_numpy()
    reject("R3_zone", ~(pu_ok & do_ok))

    bad_r4 = ~np.isfinite(fare) | (fare < MIN_FARE) | (fare > MAX_FARE)
    reject("R4_fare", bad_r4)

    dur_np = dur.to_numpy(dtype=np.float64)
    speed = np.divide(dist, dur_np / 3600.0, out=np.full(n, np.nan), where=dur_np > 0)
    reject("R5_speed", ~(np.isfinite(speed) & (speed > 0) & (speed < MAX_SPEED_MPH)))

    survivors = np.flatnonzero(keep)
    if len(survivors):
        subset = df.iloc[survivors]
        dup = subset.duplicated(
            subset=["pickup", "dropoff", "PULocationID", "DOLocationID", "trip_distance", "fare_amount"],
            keep="first",
        ).to_numpy()
        bad_r6 = np.zeros(n, dtype=bool)
        bad_r6[survivors[dup]] = True
        reject("R6_duplicate", bad_r6)

    reject("R7_invalid_code", _invalid_code_mask(df).to_numpy())

    out = df.loc[keep].copy()
    out["duration_s"] = (out["dropoff"] - out["pickup"]).dt.total_seconds().astype(np.float32)
    out["speed_mph"] = (out["trip_distance"] / (out["duration_s"] / 3600.0)).astype(np.float32)
    counts["rows_raw"] = int(n)
    counts["rows_clean"] = int(len(out))
    return out, counts
