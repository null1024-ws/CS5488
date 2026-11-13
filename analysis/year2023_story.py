"""2023-only descriptive counts and the interpretive k-means refit.

The timing kernels are unchanged. This script only rebuilds the numbers the
report cites for the twelve files of 2023: cleaning is already in
rule_stats.csv; borough, hour, and percentiles are counted here; the cluster
fit uses pickup-zone coordinates and time of day.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapefile
from sklearn.metrics import silhouette_score

from algorithms.kmeans_par import init_centers, kmeans
from pipeline.clean import clean_frame
from pipeline.etl_parallel import _rename, load_valid_zones

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXT = ROOT / "data" / "external"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _centroids() -> pd.DataFrame:
    dest = EXT / "taxi_zones"
    if not any(dest.rglob("*.shp")):
        import zipfile

        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(EXT / "taxi_zones.zip") as zf:
            zf.extractall(dest)
    shp = next(dest.rglob("*.shp"))
    reader = shapefile.Reader(str(shp))
    fields = [f[0] for f in reader.fields[1:]]
    loc_i = fields.index("LocationID")
    rows = []
    for sr, rec in zip(reader.shapes(), reader.records()):
        pts = sr.points
        if not pts:
            continue
        rows.append((int(rec[loc_i]), float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))))
    # TLC shapefile uses NAD83 State Plane New York Long Island, US survey feet.
    return pd.DataFrame(rows, columns=["LocationID", "easting", "northing"])


def _load_month(path: Path, zones: set[int]) -> pd.DataFrame:
    """Clean in the same 1,000,000-row batches the ETL uses, so R6 matches."""
    frames = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=1_000_000):
        df = _rename(pa.Table.from_batches([batch]))
        clean, _counts = clean_frame(df, zones)
        if len(clean):
            frames.append(clean)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _standardize(raw: np.ndarray) -> np.ndarray:
    std = raw.std(axis=0)
    std[std == 0] = 1.0
    return (raw - raw.mean(axis=0)) / std


def _matrix(frame: pd.DataFrame, with_distance: bool) -> np.ndarray:
    hour = frame["hour"].to_numpy(np.float64)
    ang = 2.0 * np.pi * hour / 24.0
    cols = [
        frame["easting"].to_numpy(np.float64),
        frame["northing"].to_numpy(np.float64),
        np.sin(ang),
        np.cos(ang),
        (frame["dow"].to_numpy(np.float64) >= 5).astype(np.float64),
    ]
    if with_distance:
        cols.append(np.log1p(frame["trip_distance"].to_numpy(np.float64)))
    return _standardize(np.column_stack(cols))


def _fit_k(X: np.ndarray, k: int, n_iter: int, seed: int):
    init = init_centers(X, k, seed=seed)
    return kmeans(X, init, n_iter=n_iter, n_threads=4, block=4096, parallel_update=False)


def _profiles(frame: pd.DataFrame, labels: np.ndarray, k: int) -> list[dict]:
    out = []
    for j in range(k):
        part = frame[labels == j]
        borough = part["borough"].fillna("Unknown")
        counts = borough.value_counts(normalize=True)
        hour = part["hour"].to_numpy()
        out.append({
            "cluster": int(j),
            "n": int(len(part)),
            "share": float(len(part) / len(frame)),
            "mean_distance": float(part.trip_distance.mean()),
            "mean_duration_min": float(part.duration_s.mean() / 60.0),
            "mean_fare": float(part.fare_amount.mean()),
            "mean_tip_ratio": float(part.tip_ratio.mean()),
            "mean_hour": float(part.hour.mean()),
            "weekend_share": float((part.dow >= 5).mean()),
            "night_share": float(((hour >= 0) & (hour < 5)).mean()),
            "morning_share": float(((hour >= 5) & (hour < 10)).mean()),
            "midday_share": float(((hour >= 10) & (hour < 16)).mean()),
            "afternoon_share": float(((hour >= 16) & (hour < 19)).mean()),
            "evening_share": float(((hour >= 19) & (hour < 22)).mean()),
            "late_share": float((hour >= 22).mean()),
            "top_borough": str(counts.index[0]),
            "top_borough_share": float(counts.iloc[0]),
            "manhattan_share": float((borough == "Manhattan").mean()),
            "queens_share": float((borough == "Queens").mean()),
            "mean_easting_ft": float(part.easting.mean()),
            "mean_northing_ft": float(part.northing.mean()),
        })
    return out


def _separates(profiles: list[dict]) -> dict:
    hours = [p["mean_hour"] for p in profiles]
    mans = [p["manhattan_share"] for p in profiles]
    queens = [p["queens_share"] for p in profiles]
    dists = [p["mean_distance"] for p in profiles]
    return {
        "hour_range": float(max(hours) - min(hours)),
        "manhattan_range": float(max(mans) - min(mans)),
        "queens_range": float(max(queens) - min(queens)),
        "distance_range": float(max(dists) - min(dists)),
    }


def main() -> None:
    zones = load_valid_zones()
    lookup = pd.read_csv(EXT / "taxi_zone_lookup.csv")
    lookup["LocationID"] = lookup["LocationID"].astype(int)
    borough_of = dict(zip(lookup.LocationID, lookup.Borough))
    cents = _centroids()
    old = np.load(PROC / "histograms.npz")
    edges = {name: old[name + "_edges"] for name in ("distance", "duration_min", "fare", "tip_ratio")}
    hist = {name: np.zeros(len(edges[name]) - 1, dtype=np.int64) for name in edges}
    count = np.zeros((266, 24, 7), dtype=np.int64)
    borough_n: dict[str, int] = {}
    dist_parts, dur_parts, fare_parts = [], [], []
    samples = []
    rng = np.random.default_rng(19)
    files = sorted(RAW.glob("yellow_tripdata_2023-*.parquet"))
    if len(files) != 12:
        raise SystemExit(f"expected 12 files, found {len(files)}")
    for path in files:
        clean = _load_month(path, zones)
        pickup = clean["pickup"]
        in_year = (pickup.dt.year == 2023).to_numpy()
        # Borough and clock counts use pickups dated in 2023.
        dated = clean.loc[in_year]
        for zone_id, n in dated.groupby("PULocationID").size().items():
            name = str(borough_of.get(int(zone_id), "Unknown"))
            borough_n[name] = borough_n.get(name, 0) + int(n)
        zid = dated["PULocationID"].to_numpy()
        hh = dated["pickup"].dt.hour.to_numpy()
        dd = dated["pickup"].dt.day_of_week.to_numpy()
        ok = (zid >= 0) & (zid < 266)
        np.add.at(count, (zid[ok], hh[ok], dd[ok]), 1)
        dist = clean["trip_distance"].to_numpy(np.float64)
        dur = clean["duration_s"].to_numpy(np.float64) / 60.0
        fare = clean["fare_amount"].to_numpy(np.float64)
        tip = np.divide(
            clean["tip_amount"].to_numpy(np.float64),
            fare,
            out=np.zeros(len(clean)),
            where=fare > 0,
        )
        tip = np.clip(tip, 0, 2)
        hist["distance"] += np.histogram(dist, bins=edges["distance"])[0]
        hist["duration_min"] += np.histogram(dur, bins=edges["duration_min"])[0]
        hist["fare"] += np.histogram(fare, bins=edges["fare"])[0]
        hist["tip_ratio"] += np.histogram(np.clip(tip, 0, 1), bins=edges["tip_ratio"])[0]
        dist_parts.append(dist.astype(np.float32))
        dur_parts.append(dur.astype(np.float32))
        fare_parts.append(fare.astype(np.float32))
        take_n = min(25_000, len(clean))
        idx = rng.choice(len(clean), size=take_n, replace=False)
        part = clean.iloc[idx][["pickup", "trip_distance", "duration_s", "fare_amount", "tip_amount", "PULocationID"]].copy()
        part["tip_ratio"] = tip[idx]
        samples.append(part)
        print(f"{path.name} clean={len(clean):,}", flush=True)

    sample = pd.concat(samples, ignore_index=True)
    sample["hour"] = sample["pickup"].dt.hour.astype(np.int16)
    sample["dow"] = sample["pickup"].dt.day_of_week.astype(np.int16)
    sample = sample.merge(cents, left_on="PULocationID", right_on="LocationID", how="left")
    sample["borough"] = sample["PULocationID"].map(borough_of)
    sample = sample.dropna(subset=["easting", "northing"]).reset_index(drop=True)

    n_fit = min(200_000, len(sample))
    fit_idx = rng.choice(len(sample), size=n_fit, replace=False)
    fit = sample.iloc[fit_idx].reset_index(drop=True)

    candidates = {}
    for with_distance, tag in ((True, "geo_time_distance"), (False, "geo_time")):
        X = _matrix(fit, with_distance=with_distance)
        sil_n = min(8_000, len(X))
        sil_idx = rng.choice(len(X), size=sil_n, replace=False)
        sil_scores = {}
        stored = {}
        for k in range(2, 9):
            _c, labels = _fit_k(X, k, n_iter=20, seed=19 + k)
            score = float(silhouette_score(X[sil_idx], labels[sil_idx], metric="euclidean"))
            sil_scores[str(k)] = score
            if k in (2, 4, 5):
                stored[str(k)] = _profiles(fit, labels, k)
        # Refit the reported k candidates for 30 iterations, same seed rule as before.
        refit = {}
        for k in (2, 4, 5):
            _c, labels = _fit_k(X, k, n_iter=30, seed=100 + k)
            refit[str(k)] = {
                "profiles": _profiles(fit, labels, k),
                "separation": _separates(_profiles(fit, labels, k)),
                "labels": labels,
            }
        candidates[tag] = {"silhouette": sil_scores, "scan_profiles": stored, "refit": {
            k: {"profiles": v["profiles"], "separation": v["separation"]} for k, v in refit.items()
        }}
        # Keep labels only for the feature set we might publish.
        candidates[tag]["_labels"] = {k: v["labels"] for k, v in refit.items()}
        print(tag, {k: round(v, 3) for k, v in sil_scores.items()}, flush=True)
        for k, block in candidates[tag]["refit"].items():
            print(" ", k, block["separation"], flush=True)

    # Prefer geography and time. Keep one distance feature only when the
    # k=4 or k=5 profiles still separate on hour or borough. Otherwise the
    # length feature is doing the split the earlier fit already reported.
    chosen_tag = "geo_time_distance"
    chosen = candidates[chosen_tag]
    def _ok(sep: dict) -> bool:
        return sep["hour_range"] >= 2.0 or sep["manhattan_range"] >= 0.08 or sep["queens_range"] >= 0.05

    if not (_ok(chosen["refit"]["4"]["separation"]) or _ok(chosen["refit"]["5"]["separation"])):
        chosen_tag = "geo_time"
        chosen = candidates[chosen_tag]
    k4 = chosen["refit"]["4"]
    k5 = chosen["refit"]["5"]
    # Among k=4 and k=5, keep the higher silhouette when both separate.
    # If they are within 0.01, keep k=4.
    s4 = chosen["silhouette"]["4"]
    s5 = chosen["silhouette"]["5"]
    if _ok(k4["separation"]) and _ok(k5["separation"]):
        best_k = 5 if (s5 - s4) > 0.01 else 4
    elif _ok(k5["separation"]):
        best_k = 5
    else:
        best_k = 4
    labels = candidates[chosen_tag]["_labels"][str(best_k)]
    profile = pd.DataFrame(chosen["refit"][str(best_k)]["profiles"]).sort_values("n", ascending=False)
    fit = fit.copy()
    fit["cluster"] = labels
    zone_cluster = fit.groupby(["PULocationID", "cluster"]).size().reset_index(name="n")
    profile.to_csv(RES / "cluster_profile.csv", index=False)
    zone_cluster.to_csv(RES / "zone_cluster.csv", index=False)

    dist = np.concatenate(dist_parts)
    dur = np.concatenate(dur_parts)
    fare = np.concatenate(fare_parts)
    def pct(a):
        p50, p99 = np.percentile(a, [50, 99])
        return {"p50": float(p50), "p99": float(p99)}
    hour_dow = count.sum(axis=0)  # hour, dow
    peak = np.unravel_index(int(hour_dow.argmax()), hour_dow.shape)
    dated_n = int(count.sum())
    borough_total = int(sum(borough_n.values()))
    eda = {
        "files": 12,
        "borough_n": borough_n,
        "borough_total_dated": borough_total,
        "dated_pickups": dated_n,
        "weekday_share": float(hour_dow[:, :5].sum() / dated_n),
        "peak_hour": int(peak[0]),
        "peak_dow": DOW_NAMES[int(peak[1])],
        "peak_n": int(hour_dow[peak]),
        "distance": pct(dist),
        "duration_min": pct(dur),
        "fare": pct(fare),
        "n_clean_files": int(len(dist)),
        "sample_rows_drawn": int(len(sample)),
        "n_fit": int(n_fit),
        "silhouette_n": 8000,
        "feature_set": chosen_tag,
        "features": {
            "geo_time_distance": ["zone_centroid_lon", "zone_centroid_lat", "sin_hour", "cos_hour", "weekend", "log_distance"],
            "geo_time": ["zone_centroid_lon", "zone_centroid_lat", "sin_hour", "cos_hour", "weekend"],
        }[chosen_tag],
        "k": int(best_k),
        "silhouette_by_k": chosen["silhouette"],
        "reported_silhouette": float(chosen["silhouette"][str(best_k)]),
        "selection": {
            "rule": "Report k=4 or k=5, not the silhouette maximum, when the profiles separate on hour or borough. Prefer the higher silhouette if the gap exceeds 0.01; otherwise k=4.",
            "k4": k4["separation"],
            "k5": k5["separation"],
        },
    }
    # Drop bulky labels before writing the candidate dump.
    dump = {}
    for tag, block in candidates.items():
        dump[tag] = {"silhouette": block["silhouette"], "refit": block["refit"]}
    (RES / "cluster_candidates.json").write_text(json.dumps(dump, indent=2))
    (RES / "year2023_eda.json").write_text(json.dumps(eda, indent=2))
    np.savez(
        PROC / "histograms_2023.npz",
        distance=hist["distance"], distance_edges=edges["distance"],
        duration_min=hist["duration_min"], duration_min_edges=edges["duration_min"],
        fare=hist["fare"], fare_edges=edges["fare"],
        tip_ratio=hist["tip_ratio"], tip_ratio_edges=edges["tip_ratio"],
    )
    np.savez(PROC / "zone_hour_dow_2023.npz", count=count)
    print(json.dumps({k: eda[k] for k in ("k", "feature_set", "reported_silhouette", "peak_hour", "peak_dow", "peak_n", "weekday_share", "borough_n")}, indent=2))


if __name__ == "__main__":
    main()
