"""Report figures in the figures4papers publication style."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pubstyle import (
    DEFAULT_COLORS,
    PALETTE,
    FigureStyle,
    apply_publication_style,
    create_subplots,
    finalize_figure,
    make_grouped_bar,
    make_heatmap,
    make_scatter,
    make_trend,
)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
EXT = ROOT / "data" / "external"
FIG = ROOT / "figures"
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _style(size=15, lw=2.0):
    apply_publication_style(FigureStyle(font_size=size, axes_linewidth=lw))


def _zones():
    zones = pd.read_csv(EXT / "taxi_zone_lookup.csv")
    zones["LocationID"] = zones["LocationID"].astype(int)
    return zones


def fig_histograms():
    _style(14, 2)
    hist = np.load(PROC / "histograms_2023.npz")
    specs = [
        ("distance", "Trip distance (miles)", "Trips"),
        ("duration_min", "Duration (minutes)", "Trips"),
        ("fare", "Fare (USD)", "Trips"),
        ("tip_ratio", "Tip / fare", "Trips"),
    ]
    fig, axes = create_subplots(1, 4, figsize=(16, 4.2))
    for ax, (name, xlabel, ylabel) in zip(axes, specs):
        counts = hist[name]
        edges = hist[name + "_edges"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = np.diff(edges)
        ax.bar(centers, counts, width=width * 0.95, color=PALETTE["blue_main"], edgecolor="black", linewidth=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    finalize_figure(fig, FIG / "fig03_histograms", formats=["png", "pdf"], dpi=300, pad=0.6)


def fig_heatmap():
    _style(14, 2)
    count = np.load(PROC / "zone_hour_dow_2023.npz")["count"].sum(axis=0)  # hour, dow
    # Show dow on rows, hour on columns.
    matrix = count.T
    fig, axes = create_subplots(1, 1, figsize=(12, 4.2))
    make_heatmap(
        axes[0], matrix, x_labels=list(range(24)), y_labels=DOW,
        cmap="magma", cbar_label="Pickups",
    )
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Day of week")
    finalize_figure(fig, FIG / "fig04_hour_dow", formats=["png", "pdf"], dpi=300, pad=0.4)


def fig_monthly_and_borough():
    _style(14, 2)
    monthly = pd.read_csv(PROC / "monthly.csv").sort_values(["year", "month"])
    monthly = monthly[monthly.year == 2023].reset_index(drop=True)
    monthly["label"] = monthly.month.astype(int).map(lambda m: f"{m:02d}")
    zones = _zones()
    borough = dict(zip(zones.LocationID, zones.Borough))
    count = np.load(PROC / "zone_hour_dow_2023.npz")["count"]
    # borough x hour
    names = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"]
    hour = np.zeros((len(names), 24))
    zone_totals = count.sum(axis=2)
    for zone_id in range(zone_totals.shape[0]):
        name = borough.get(zone_id)
        if name in names:
            hour[names.index(name)] += zone_totals[zone_id]
    fig, axes = create_subplots(1, 2, figsize=(14, 4.6))
    x = np.arange(len(monthly))
    axes[0].plot(x, monthly.n.to_numpy(), color=PALETTE["blue_main"], lw=2.4)
    axes[0].set_ylabel("Cleaned pickups")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(list(monthly.label), rotation=0)
    axes[0].set_xlabel("Month in 2023")
    make_trend(
        axes[1], np.arange(24), [hour[i] for i in range(len(names))], names,
        ylabel="Pickups", xlabel="Hour", show_shadow=False,
        colors=[PALETTE["blue_main"], PALETTE["teal"], PALETTE["green_3"], PALETTE["red_strong"], PALETTE["violet"], PALETTE["neutral"]],
    )
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)
    finalize_figure(fig, FIG / "fig05_monthly_borough", formats=["png", "pdf"], dpi=300, pad=0.8)


def _centroids():
    zip_path = EXT / "taxi_zones.zip"
    if not zip_path.exists():
        return None
    dest = EXT / "taxi_zones"
    if not any(dest.rglob("*.shp")):
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
    shp = next(dest.rglob("*.shp"), None)
    if shp is None:
        return None
    import shapefile
    reader = shapefile.Reader(str(shp))
    fields = [f[0] for f in reader.fields[1:]]
    if "LocationID" not in fields:
        return None
    loc_i = fields.index("LocationID")
    rows = []
    for sr, rec in zip(reader.shapes(), reader.records()):
        pts = sr.points
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        rows.append((int(rec[loc_i]), float(np.mean(xs)), float(np.mean(ys))))
    return pd.DataFrame(rows, columns=["LocationID", "x", "y"])


def fig_clusters():
    _style(14, 2)
    profile = pd.read_csv(RES / "cluster_profile.csv").sort_values("cluster")
    zone_cluster = pd.read_csv(RES / "zone_cluster.csv")
    majority = (
        zone_cluster.sort_values("n", ascending=False)
        .groupby("PULocationID", as_index=False)
        .first()
    )
    cents = _centroids()
    fig, axes = create_subplots(1, 2, figsize=(13, 5.2))
    shares = profile["share"].to_numpy() * 100
    labels = [f"C{int(c)}" for c in profile.cluster]
    make_grouped_bar(axes[0], labels, [shares], ["Share of sample (%)"], ylabel="Share (%)", colors=[PALETTE["blue_main"]])
    axes[0].legend().remove()
    if cents is not None:
        merged = majority.merge(cents, left_on="PULocationID", right_on="LocationID", how="inner")
        for i, cluster in enumerate(sorted(merged.cluster.unique())):
            part = merged[merged.cluster == cluster]
            make_scatter(
                axes[1], part.x, part.y, label=f"C{int(cluster)}",
                color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)], size=18, alpha=0.85,
            )
        axes[1].set_xlabel("Zone centroid x")
        axes[1].set_ylabel("Zone centroid y")
        axes[1].set_aspect("equal", adjustable="datalim")
        axes[1].tick_params(labelbottom=False, labelleft=False)
    else:
        axes[1].axis("off")
    finalize_figure(fig, FIG / "fig07_clusters", formats=["png", "pdf"], dpi=300, pad=0.6)


def fig_rules():
    _style(14, 2)
    rules = pd.read_csv(RES / "rules_top10.csv")
    if rules.empty:
        return
    fig, axes = create_subplots(1, 1, figsize=(7.2, 5.2))
    sizes = 40 + 80 * (rules.lift - rules.lift.min()) / max(rules.lift.max() - rules.lift.min(), 1e-6)
    make_scatter(axes[0], rules.support, rules.confidence, color=PALETTE["blue_main"], size=sizes, alpha=0.9)
    for _, row in rules.iterrows():
        axes[0].annotate(
            f"{row.lift:.2f}",
            (row.support, row.confidence),
            textcoords="offset points", xytext=(4, 4), fontsize=9, color=PALETTE["gray_dark"],
        )
    axes[0].set_xlabel("Support")
    axes[0].set_ylabel("Confidence")
    finalize_figure(fig, FIG / "fig08_rules", formats=["png", "pdf"], dpi=300, pad=0.5)


def _curve(summary_medians, workload, mode, note, block=None):
    part = summary_medians[(summary_medians.workload == workload) & (summary_medians["mode"] == mode)]
    part = part[part.note.fillna("") == note]
    if block is not None:
        block_text = part.block.fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
        part = part[block_text == str(block)]
    return part.sort_values("p")


def fig_scaling():
    medians = pd.read_csv(RES / "run_medians.csv")
    _style(14, 2)
    fig, axes = create_subplots(1, 1, figsize=(7.4, 5.2))
    series = []
    labels = []
    xs = None
    yerr = []
    colors = [PALETTE["blue_main"], PALETTE["teal"], PALETTE["red_strong"]]
    specs = [("T1", "process", "2023 x12", None, "T1 ETL"),
             ("T2", "numba", "parallel_update", 8192, "T2 k-means"),
             ("T3", "fpgrowth", "min_support=0.02", None, "T3 FP-Growth")]
    for workload, mode, note, block, label in specs:
        part = _curve(medians, workload, mode, note, block)
        part = part[part.p.isin([1, 2, 4, 8])]
        if part.empty or (part.p == 1).sum() == 0:
            continue
        t1 = float(part.loc[part.p == 1, "median"].iloc[0])
        order = part.sort_values("p")
        xs = order.p.to_numpy()
        speed = t1 / order["median"].to_numpy()
        low = t1 / order.q3.to_numpy()
        high = t1 / order.q1.to_numpy()
        series.append(speed)
        labels.append(label)
        yerr.append(np.vstack([speed - low, high - speed]))
    if xs is not None:
        make_trend(axes[0], xs, series, labels, colors=colors, ylabel="Speedup S(p)", xlabel="Workers p", show_shadow=True, yerr=yerr)
        axes[0].plot(xs, xs, color=PALETTE["neutral"], lw=1.5, ls="--", label="Ideal")
        axes[0].legend()
    finalize_figure(fig, FIG / "fig09_speedup", formats=["png", "pdf"], dpi=300, pad=0.5)


def fig_block():
    runs = pd.read_csv(RES / "runs.csv")
    part = runs[(runs.workload == "T2") & (runs.note.fillna("") == "parallel_update") & (runs.p == 4)].copy()
    if part.empty:
        return
    part["block"] = part.block.astype(float).astype(int)
    part["wall_s"] = part.wall_s.astype(float)
    grouped = part.groupby("block").wall_s.median().reset_index().sort_values("block")
    _style(14, 2)
    fig, axes = create_subplots(1, 1, figsize=(7.2, 4.8))
    make_trend(axes[0], grouped.block, [grouped.wall_s], ["p = 4"], colors=[PALETTE["blue_main"]],
               ylabel="Median wall time (s)", xlabel="Rows per block", show_shadow=False)
    axes[0].set_xscale("log", base=2)
    finalize_figure(fig, FIG / "fig11_blocksize", formats=["png", "pdf"], dpi=300, pad=0.5)


def fig_gil():
    medians = pd.read_csv(RES / "run_medians.csv")
    _style(15, 2.2)
    fig, axes = create_subplots(1, 1, figsize=(7.2, 4.8))
    thread = _curve(medians, "GIL", "thread", "pure Python row loop")
    proc = _curve(medians, "GIL", "process", "pure Python row loop")
    if thread.empty or proc.empty:
        return
    ps = sorted(set(thread.p.astype(int)) & set(proc.p.astype(int)))
    tvals, pvals = [], []
    for p in ps:
        tvals.append(float(thread.loc[thread.p == p, "median"].iloc[0]))
        pvals.append(float(proc.loc[proc.p == p, "median"].iloc[0]))
    make_grouped_bar(
        axes[0], [str(p) for p in ps], [tvals, pvals], ["Threads", "Processes"],
        ylabel="Median wall time (s)",
        colors=[PALETTE["red_strong"], PALETTE["blue_main"]],
        hatch=["//", None],
    )
    axes[0].set_xlabel("Workers")
    finalize_figure(fig, FIG / "fig12_gil", formats=["png", "pdf"], dpi=300, pad=0.5)


def fig_phases():
    runs = pd.read_csv(RES / "runs.csv")
    part = runs[(runs.workload == "T1") & (runs["mode"] == "process") & (runs.note.fillna("") == "2023 x12")].copy()
    if part.empty:
        return
    for col in ("wall_s", "read_s", "clean_s", "agg_s"):
        part[col] = pd.to_numeric(part[col], errors="coerce")
    rows = []
    for p, group in part.groupby("p"):
        rows.append({
            "p": int(p),
            "read": np.median(group.read_s) / int(p),
            "clean": np.median(group.clean_s) / int(p),
            "agg": np.median(group.agg_s) / int(p),
            "wall": np.median(group.wall_s),
        })
    table = pd.DataFrame(rows).sort_values("p")
    _style(14, 2)
    fig, axes = create_subplots(1, 1, figsize=(8, 4.8))
    x = np.arange(len(table))
    width = 0.62
    read = table["read"].to_numpy()
    clean = table["clean"].to_numpy()
    agg = table["agg"].to_numpy()
    axes[0].bar(x, read, width=width, color=PALETTE["blue_secondary"], edgecolor="black", linewidth=1.0, label="Read / p")
    axes[0].bar(x, clean, width=width, bottom=read, color=PALETTE["teal"], edgecolor="black", linewidth=1.0, label="Clean / p")
    axes[0].bar(x, agg, width=width, bottom=read + clean, color=PALETTE["green_3"], edgecolor="black", linewidth=1.0, label="Aggregate / p")
    axes[0].plot(x, table.wall, color=PALETTE["red_strong"], marker="o", lw=2.2, label="Wall time")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(p) for p in table.p])
    axes[0].set_ylabel("Seconds")
    axes[0].legend()
    axes[0].set_xlabel("Workers p")
    finalize_figure(fig, FIG / "fig10_phases", formats=["png", "pdf"], dpi=300, pad=0.5)


def fig_diagrams():
    _style(13, 2)
    fig, ax = create_subplots(1, 1, figsize=(12, 3.2))
    axis = ax[0]
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 3)
    axis.axis("off")
    boxes = [
        (0.3, "Monthly\nParquet"),
        (3.2, "Map\nclean + aggregate"),
        (6.4, "Shuffle\nby zone, hour"),
        (9.3, "Reduce\nsmall tables"),
    ]
    colors = [PALETTE["neutral"], PALETTE["blue_main"], PALETTE["teal"], PALETTE["green_3"]]
    for (x, text), color in zip(boxes, colors):
        rect = plt_fancy(axis, x, 0.8, 2.4, 1.4, color)
        axis.text(x + 1.2, 1.5, text, ha="center", va="center", color="white" if color != PALETTE["neutral"] else "black", fontsize=12)
    for x in (2.7, 5.8, 8.8):
        axis.annotate("", xy=(x + 0.45, 1.5), xytext=(x, 1.5), arrowprops=dict(arrowstyle="-|>", color=PALETTE["gray_dark"], lw=1.8))
    finalize_figure(fig, FIG / "fig02_pipeline", formats=["png", "pdf"], dpi=300, pad=0.3)

    fig, ax = create_subplots(1, 1, figsize=(12, 3.4))
    axis = ax[0]
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 3.4)
    axis.axis("off")
    layers = [
        (0.4, 2.0, "Data parallel\nprocess pool over months", PALETTE["blue_main"]),
        (4.2, 2.0, "Kernel parallel\nnumba prange assign", PALETTE["teal"]),
        (8.0, 2.0, "Task parallel\nFP-Growth conditionals", PALETTE["red_strong"]),
    ]
    for x, y, text, color in layers:
        plt_fancy(axis, x, y, 3.4, 1.1, color)
        axis.text(x + 1.7, y + 0.55, text, ha="center", va="center", color="white", fontsize=11)
    axis.text(6, 0.7, "Same inputs, p = 1 compared with p = 2, 4, and 8", ha="center", fontsize=12, color=PALETTE["gray_dark"])
    finalize_figure(fig, FIG / "fig06_tasks", formats=["png", "pdf"], dpi=300, pad=0.3)


def plt_fancy(ax, x, y, w, h, color):
    from matplotlib.patches import FancyBboxPatch
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.15",
                           facecolor=color, edgecolor="black", linewidth=1.2)
    ax.add_patch(patch)
    return patch


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_histograms()
    fig_heatmap()
    fig_monthly_and_borough()
    fig_diagrams()
    if (RES / "cluster_profile.csv").exists():
        fig_clusters()
    if (RES / "rules_top10.csv").exists():
        fig_rules()
    if (RES / "run_medians.csv").exists():
        fig_scaling()
        fig_block()
        fig_gil()
        fig_phases()
    print("figures written to", FIG, flush=True)


if __name__ == "__main__":
    main()
