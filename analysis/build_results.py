"""Turn aggregates, the enriched sample, and runs.csv into report numbers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from algorithms.fpgrowth import association_rules, mine_serial
from algorithms.kmeans_par import init_centers, kmeans
from pipeline.features import transaction_items

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
EXT = ROOT / "data" / "external"
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _hist_percentile(counts: np.ndarray, edges: np.ndarray, q: float) -> float:
    total = counts.sum()
    if total <= 0:
        return float("nan")
    cdf = np.cumsum(counts)
    idx = int(np.searchsorted(cdf, q * total, side="left"))
    idx = min(idx, len(edges) - 2)
    span = cdf[idx] - (cdf[idx - 1] if idx else 0)
    frac = 0.0 if span == 0 else (q * total - (cdf[idx - 1] if idx else 0)) / span
    return float(edges[idx] + frac * (edges[idx + 1] - edges[idx]))


def _load_zones() -> pd.DataFrame:
    zones = pd.read_csv(EXT / "taxi_zone_lookup.csv")
    zones["LocationID"] = zones["LocationID"].astype(int)
    return zones


def eda_tables() -> dict:
    zones = _load_zones()
    borough = dict(zip(zones["LocationID"], zones["Borough"]))
    data = np.load(PROC / "zone_hour_dow.npz")
    count = data["count"]
    hist = np.load(PROC / "histograms.npz")
    monthly = pd.read_csv(PROC / "monthly.csv")
    rules = pd.read_csv(PROC / "rule_stats.csv")
    raw = int(rules["rows_raw"].sum())
    clean = int(rules["rows_clean"].sum())
    rule_totals = {col: int(rules[col].sum()) for col in rules.columns if col.startswith("R")}
    by_borough = {}
    for zone_id, total in enumerate(count.sum(axis=(1, 2))):
        if total <= 0:
            continue
        name = str(borough.get(zone_id, "Other"))
        by_borough[name] = by_borough.get(name, 0) + int(total)
    hour_dow = count.sum(axis=0)  # hour, dow
    flat = np.argmax(hour_dow)
    peak_hour, peak_dow = np.unravel_index(flat, hour_dow.shape)
    month_key = monthly.assign(key=monthly.year * 100 + monthly.month)
    def month_n(year, month):
        hit = monthly[(monthly.year == year) & (monthly.month == month)]
        return int(hit["n"].sum()) if len(hit) else 0
    april_2019 = month_n(2019, 4)
    april_2020 = month_n(2020, 4)
    april_2022 = month_n(2022, 4)
    april_2023 = month_n(2023, 4)
    percentiles = {}
    for name in ("distance", "duration_min", "fare", "tip_ratio"):
        percentiles[name] = {
            "p50": _hist_percentile(hist[name], hist[name + "_edges"], 0.50),
            "p99": _hist_percentile(hist[name], hist[name + "_edges"], 0.99),
        }
    out = {
        "rows_raw": raw,
        "rows_clean": clean,
        "drop_rate": (raw - clean) / raw if raw else None,
        "rules": rule_totals,
        "borough_counts": by_borough,
        "manhattan_share": by_borough.get("Manhattan", 0) / clean if clean else None,
        "peak_hour": int(peak_hour),
        "peak_dow": DOW[int(peak_dow)],
        "peak_trips": int(hour_dow[peak_hour, peak_dow]),
        "april": {"2019": april_2019, "2020": april_2020, "2022": april_2022, "2023": april_2023},
        "percentiles": percentiles,
        "hour_dow": hour_dow,
        "count": count,
        "monthly": monthly,
        "zones": zones,
    }
    if april_2019:
        out["april_2020_vs_2019"] = april_2020 / april_2019 - 1
        out["april_2022_vs_2019"] = april_2022 / april_2019 - 1
        out["april_2023_vs_2019"] = april_2023 / april_2019 - 1
    return out


def _story_matrix(df: pd.DataFrame) -> np.ndarray:
    hour = df["hour"].to_numpy(np.float64) / 23.0
    dow = df["dow"].to_numpy(np.float64) / 6.0
    distance = np.log1p(df["trip_distance"].to_numpy(np.float64))
    duration = np.log1p(df["duration_s"].to_numpy(np.float64) / 60.0)
    fare = np.log1p(df["fare_amount"].to_numpy(np.float64))
    tip = np.clip(df["tip_ratio"].to_numpy(np.float64), 0, 1)
    speed = np.log1p(df["speed_mph"].to_numpy(np.float64))
    passenger = df["passenger_count"].fillna(1).to_numpy(np.float64) / 6.0
    raw = np.column_stack([hour, dow, distance, duration, fare, tip, speed, passenger])
    std = raw.std(axis=0)
    std[std == 0] = 1
    return ((raw - raw.mean(axis=0)) / std).astype(np.float64)


def fit_clusters(sample: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(19)
    n_fit = min(200_000, len(sample))
    fit_idx = rng.choice(len(sample), size=n_fit, replace=False)
    fit = sample.iloc[fit_idx].reset_index(drop=True)
    X = _story_matrix(fit)
    sil_n = min(8_000, len(X))
    sil_idx = rng.choice(len(X), size=sil_n, replace=False)
    sil_scores = {}
    best_k, best_s = None, -1.0
    for k in range(2, 9):
        init = init_centers(X, k, seed=19 + k)
        centers, labels = kmeans(X, init, n_iter=20, n_threads=4, block=4096, parallel_update=False)
        score = float(silhouette_score(X[sil_idx], labels[sil_idx], metric="euclidean"))
        sil_scores[str(k)] = score
        if score > best_s:
            best_k, best_s = k, score
    init = init_centers(X, best_k, seed=100 + best_k)
    centers, labels = kmeans(X, init, n_iter=30, n_threads=4, block=4096, parallel_update=False)
    fit["cluster"] = labels
    profiles = []
    for j in range(best_k):
        part = fit[fit.cluster == j]
        profiles.append({
            "cluster": int(j),
            "n": int(len(part)),
            "share": float(len(part) / len(fit)),
            "mean_distance": float(part.trip_distance.mean()),
            "mean_duration_min": float(part.duration_s.mean() / 60.0),
            "mean_fare": float(part.fare_amount.mean()),
            "mean_tip_ratio": float(part.tip_ratio.mean()),
            "mean_hour": float(part.hour.mean()),
            "weekend_share": float((part.dow >= 5).mean()),
            "top_borough": str(part.borough.fillna("Unknown").value_counts().index[0]),
            "top_borough_share": float(part.borough.fillna("Unknown").value_counts(normalize=True).iloc[0]),
        })
    profile = pd.DataFrame(profiles).sort_values("n", ascending=False)
    zone_cluster = (
        fit.groupby(["PULocationID", "cluster"]).size().reset_index(name="n")
    )
    zone_cluster.to_csv(RES / "zone_cluster.csv", index=False)
    profile.to_csv(RES / "cluster_profile.csv", index=False)
    meta = {"k": int(best_k), "silhouette": float(best_s), "silhouette_by_k": sil_scores, "n_fit": int(n_fit), "silhouette_n": int(sil_n)}
    return profile, meta


def mine_rules(sample: pd.DataFrame) -> dict:
    rng = np.random.default_rng(11)
    use = sample[sample["wx"] != "unknown"].copy()
    n = min(250_000, len(use))
    use = use.iloc[rng.choice(len(use), size=n, replace=False)]
    transactions = transaction_items(use)
    itemsets = mine_serial(transactions, min_support=0.01)
    rules = association_rules(itemsets, min_confidence=0.6, min_lift=1.2)
    top = rules[:10]
    table = pd.DataFrame([{
        "antecedent": ", ".join(sorted(r["antecedent"])),
        "consequent": ", ".join(sorted(r["consequent"])),
        "support": r["support"],
        "confidence": r["confidence"],
        "lift": r["lift"],
    } for r in top])
    table.to_csv(RES / "rules_top10.csv", index=False)

    def top_pairs(year: int):
        part = sample[(sample.year == year) & (sample.wx != "unknown")]
        if len(part) < 1000:
            return set()
        part = part.iloc[rng.choice(len(part), size=min(80_000, len(part)), replace=False)]
        found = association_rules(mine_serial(transaction_items(part), 0.01), 0.6, 1.2)[:15]
        return {(tuple(sorted(r["antecedent"])), tuple(sorted(r["consequent"]))) for r in found}

    pairs_2022 = top_pairs(2022)
    pairs_2023 = top_pairs(2023)
    overlap = len(pairs_2022 & pairs_2023)
    return {
        "n_transactions": len(transactions),
        "n_itemsets": len(itemsets),
        "n_rules": len(rules),
        "min_support": 0.01,
        "min_confidence": 0.6,
        "min_lift": 1.2,
        "top": table.to_dict(orient="records"),
        "stability_overlap_top15": overlap,
        "stability_2022": len(pairs_2022),
        "stability_2023": len(pairs_2023),
    }


def _median_table(runs: pd.DataFrame) -> pd.DataFrame:
    runs["wall_s"] = runs["wall_s"].astype(float)
    keys = ["workload", "mode", "p", "chunksize", "block", "n_input", "note"]
    rows = []
    for key, part in runs.groupby(keys, dropna=False):
        walls = np.sort(part["wall_s"].to_numpy())
        rec = dict(zip(keys, key))
        rec["median"] = float(np.median(walls))
        rec["q1"] = float(np.quantile(walls, 0.25))
        rec["q3"] = float(np.quantile(walls, 0.75))
        rec["repeats"] = int(len(walls))
        rows.append(rec)
    return pd.DataFrame(rows)


def timing_summary() -> dict:
    path = RES / "runs.csv"
    if not path.exists():
        return {}
    runs = pd.read_csv(path)
    summary = _median_table(runs)
    summary.to_csv(RES / "run_medians.csv", index=False)
    out = {"medians": summary.to_dict(orient="records")}

    def curve(workload, mode, note_exact, block=None):
        part = summary[(summary.workload == workload) & (summary["mode"] == mode)]
        part = part[part.note.fillna("") == note_exact]
        if block is not None and "block" in part.columns:
            block_text = part.block.fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
            part = part[block_text == str(block)]
        part = part.sort_values("p")
        if part.empty:
            return {}
        base = part[part.p == 1]
        if base.empty:
            return {}
        t1 = float(base.iloc[0]["median"])
        points = []
        for _, row in part.iterrows():
            p = int(row.p)
            tp = float(row["median"])
            speedup = t1 / tp if tp else None
            eff = speedup / p if speedup else None
            karp = None
            if speedup and p > 1:
                karp = (1 / speedup - 1 / p) / (1 - 1 / p)
            points.append({
                "p": p, "median_s": tp, "q1": float(row.q1), "q3": float(row.q3),
                "speedup": speedup, "efficiency": eff, "karp_flatt": karp, "n_input": int(row.n_input),
                "note": str(row.note),
            })
        return {"t1": t1, "points": points}

    out["T1"] = curve("T1", "process", "2023 x12")
    out["T2"] = curve("T2", "numba", "parallel_update", block=8192)
    out["T3"] = curve("T3", "fpgrowth", "min_support=0.02")
    out["GIL_thread"] = curve("GIL", "thread", "pure Python row loop")
    out["GIL_process"] = curve("GIL", "process", "pure Python row loop")
    return out


def _tex_escape(text: str) -> str:
    return (text.replace("\\", "\\textbackslash ")
            .replace("_", "\\_")
            .replace("%", "\\%")
            .replace("&", "\\&")
            .replace("#", "\\#"))


def write_numbers(summary: dict):
    def num(value, digits=1):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "NA"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        return f"{float(value):,.{digits}f}"

    lines = []
    def add(name, value):
        lines.append(f"\\def\\{name}{{{value}}}")

    add("RawRows", num(summary["rows_raw"], 0))
    add("CleanRows", num(summary["rows_clean"], 0))
    add("DropPct", num(100 * summary["drop_rate"], 1))
    add("ManhattanPct", num(100 * summary["manhattan_share"], 1))
    add("PeakHour", str(summary["peak_hour"]))
    add("PeakDow", summary["peak_dow"])
    add("AprilNineteen", num(summary["april"]["2019"], 0))
    add("AprilTwenty", num(summary["april"]["2020"], 0))
    add("AprilTwentytwo", num(summary["april"]["2022"], 0))
    add("AprilTwentythree", num(summary["april"]["2023"], 0))
    if "april_2020_vs_2019" in summary:
        add("AprilDropPct", num(100 * summary["april_2020_vs_2019"], 1))
        add("AprilRecoverPct", num(100 * summary["april_2022_vs_2019"], 1))
        add("AprilTwentythreeVsNineteen", num(100 * summary["april_2023_vs_2019"], 1))
    pct = summary["percentiles"]
    add("DistPfifty", num(pct["distance"]["p50"], 2))
    add("DistPnine", num(pct["distance"]["p99"], 1))
    add("DurPfifty", num(pct["duration_min"]["p50"], 1))
    add("FarePfifty", num(pct["fare"]["p50"], 1))
    cl = summary.get("clusters", {})
    add("ClusterK", str(cl.get("k", "")))
    add("Silhouette", num(cl.get("silhouette"), 3))
    add("ClusterN", num(cl.get("n_fit", 0), 0))
    rl = summary.get("rules", {})
    add("RuleN", num(rl.get("n_transactions", 0), 0))
    add("RuleCount", num(rl.get("n_rules", 0), 0))
    add("RuleOverlap", str(rl.get("stability_overlap_top15", "")))
    machine = summary.get("machine", {})
    add("CpuCount", str(machine.get("logical_cpus", "")))
    add("MemGiB", num((machine.get("memory_bytes") or 0) / 2**30, 1))
    text = "\n".join(lines) + "\n"
    (ROOT / "report" / "numbers.tex").write_text(text)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    (ROOT / "report").mkdir(parents=True, exist_ok=True)
    eda = eda_tables()
    sample = pd.read_parquet(PROC / "sample_enriched.parquet")
    profile, cluster_meta = fit_clusters(sample)
    rules = mine_rules(sample)
    timing = timing_summary()
    machine = {}
    if (RES / "machine.json").exists():
        machine = json.loads((RES / "machine.json").read_text())
    # Arrays are not JSON.
    summary = {k: v for k, v in eda.items() if k not in {"hour_dow", "count", "monthly", "zones"}}
    summary["clusters"] = cluster_meta
    summary["cluster_profiles"] = profile.to_dict(orient="records")
    summary["rules"] = rules
    summary["timing"] = timing
    summary["machine"] = machine
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_numbers(summary)
    print(json.dumps({
        "rows_raw": summary["rows_raw"],
        "rows_clean": summary["rows_clean"],
        "drop_rate": summary["drop_rate"],
        "manhattan_share": summary["manhattan_share"],
        "k": cluster_meta["k"],
        "silhouette": cluster_meta["silhouette"],
        "n_rules": rules["n_rules"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
