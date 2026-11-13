"""Check that the report quotes the medians in results/runs.csv.

The script fails if a rounded median is missing from the LaTeX, or if a
template figure from the outline (a 4.90x speedup, a 0.42 silhouette, or a
placeholder student name) has been copied into the report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs.csv"
TEXT = (ROOT / "report" / "report.tex").read_text() + (ROOT / "report" / "scaling.tex").read_text()


def _median(mask) -> float:
    walls = pd.read_csv(RUNS).loc[mask, "wall_s"].astype(float)
    return float(np.median(walls.to_numpy()))


def main() -> int:
    df = pd.read_csv(RUNS)
    df["wall_s"] = df["wall_s"].astype(float)
    wanted = {
        "52.135": df[(df.workload == "T1") & (df.note == "2023 x12") & (df.p == 1)].wall_s,
        "26.717": df[(df.workload == "T1") & (df.note == "2023 x12") & (df.p == 2)].wall_s,
        "13.658": df[(df.workload == "T1") & (df.note == "2023 x12") & (df.p == 4)].wall_s,
        "14.266": df[(df.workload == "T1") & (df.note == "2023 x12") & (df.p == 8)].wall_s,
        "6.428": df[(df.workload == "T2") & (df.note == "parallel_update") & (df.block == 8192) & (df.p == 1)].wall_s,
        "3.208": df[(df.workload == "T2") & (df.note == "parallel_update") & (df.block == 8192) & (df.p == 2)].wall_s,
        "1.611": df[(df.workload == "T2") & (df.note == "parallel_update") & (df.block == 8192) & (df.p == 4) & (df.n_input == 2_600_000)].wall_s,
        "1.659": df[(df.workload == "T2") & (df.note == "parallel_update") & (df.block == 8192) & (df.p == 8)].wall_s,
        "1.497": df[(df.workload == "T3") & (df.p == 1)].wall_s,
        "1.681": df[(df.workload == "T3") & (df.p == 4)].wall_s,
        "27.423": df[(df.note == "2023 x12 thread")].wall_s,
        "17.038": df[(df.note == "2023 x12 chunk=4")].wall_s,
        "49.668": df[(df.note == "2023 x12 chunk=12")].wall_s,
        "38,310,226": None,
        "34,893,980": None,
        "0.550": None,
        "0.312": None,
    }
    failed = False
    for label, series in wanted.items():
        if series is not None:
            rounded = f"{float(np.median(series.to_numpy())):.3f}"
            if rounded != label:
                print(f"median {label} does not match runs.csv ({rounded})")
                failed = True
        if label not in TEXT:
            print(f"report is missing {label}")
            failed = True
    for banned in (
        "4.90", "CHAN Tai Man", "LEE Siu Ming", "WONG Ka Yan", "NG Chi Keung",
        "Karp", "P-core", "E-core", "hyperthread", "218,118,168", "210 million",
    ):
        if banned in TEXT:
            print(f"template string present: {banned}")
            failed = True
    if "0.420" in TEXT or "silhouette 0.42" in TEXT:
        print("template silhouette present")
        failed = True
    if failed:
        return 1
    print("report numbers match runs.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
