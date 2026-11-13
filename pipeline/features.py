"""Derived fields, weather labels, and the fixed item vocabulary for rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

TOD_BINS = [0, 5, 10, 16, 19, 22, 24]
TOD_LABELS = ["night", "morning", "midday", "afternoon", "evening", "late"]
DIST_BINS = [0, 1, 3, 10, np.inf]
DIST_LABELS = ["short", "medium", "long", "very_long"]


def add_clock_features(df: pd.DataFrame) -> pd.DataFrame:
    pickup = df["pickup"]
    df = df.copy()
    df["hour"] = pickup.dt.hour.astype(np.int8)
    df["dow"] = pickup.dt.day_of_week.astype(np.int8)
    df["year"] = pickup.dt.year.astype(np.int16)
    df["month"] = pickup.dt.month.astype(np.int8)
    df["date"] = pickup.dt.floor("D")
    df["tip_ratio"] = np.where(df["fare_amount"] > 0, df["tip_amount"] / df["fare_amount"], 0.0)
    df["tip_ratio"] = df["tip_ratio"].clip(0, 2).astype(np.float32)
    df["daytype"] = np.where(df["dow"] >= 5, "weekend", "weekday")
    df["tod"] = pd.cut(df["hour"], TOD_BINS, labels=TOD_LABELS, right=False).astype(str)
    df["dist_bin"] = pd.cut(df["trip_distance"], DIST_BINS, labels=DIST_LABELS, right=False).astype(str)
    ratio = df["tip_ratio"]
    df["tip_bin"] = np.where(ratio <= 0, "none", np.where(ratio < 0.15, "low", "high"))
    pay = df["payment_type"].astype(int)
    df["pay_bin"] = np.where(pay == 1, "card", np.where(pay == 2, "cash", "other"))
    return df


def load_weather(path) -> pd.DataFrame:
    weather = pd.read_csv(path, usecols=["DATE", "PRCP", "SNOW", "SNWD", "TMAX"])
    weather["date"] = pd.to_datetime(weather["DATE"])
    weather = weather[(weather["date"] >= "2019-01-01") & (weather["date"] <= "2023-12-31")].copy()
    prcp = weather["PRCP"].fillna(0)
    snow = weather["SNOW"].fillna(0)
    snwd = weather["SNWD"].fillna(0)
    # GHCN access values are tenths of a millimetre / millimetre of snow.
    # The sign of the test does not depend on that scale: any positive record counts.
    kind = np.full(len(weather), "dry", dtype=object)
    kind[(prcp > 0) | (snow > 0)] = "rain"
    kind[(snow > 0) | (snwd > 0)] = "snow"
    weather["wx"] = kind
    return weather[["date", "wx", "PRCP", "SNOW", "TMAX"]]


def attach_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(weather[["date", "wx"]], on="date", how="left")
    out["wx"] = out["wx"].fillna("unknown")
    return out


def transaction_items(df: pd.DataFrame) -> list[list[str]]:
    borough = df["borough"].astype(str)
    rows = []
    for b, tod, day, wx, pay, dist, tip in zip(
        borough,
        df["tod"].astype(str),
        df["daytype"].astype(str),
        df["wx"].astype(str),
        df["pay_bin"].astype(str),
        df["dist_bin"].astype(str),
        df["tip_bin"].astype(str),
    ):
        if wx == "unknown" or b in {"nan", "Unknown", ""}:
            continue
        rows.append(
            [
                f"pu={b}",
                f"tod={tod}",
                f"day={day}",
                f"wx={wx}",
                f"pay={pay}",
                f"dist={dist}",
                f"tip={tip}",
            ]
        )
    return rows
