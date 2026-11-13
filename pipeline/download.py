"""Download TLC yellow-taxi months, zone lookup, and Central Park GHCN-Daily."""

from __future__ import annotations

import csv
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXT = ROOT / "data" / "external"
MANIFEST = ROOT / "data" / "manifest.csv"

TLC_PAGE = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{ym}.parquet"
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
SHAPE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
NOAA_URL = (
    "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/"
    "access/USW00094728.csv"
)

TLC_LICENSE = (
    "NYC Taxi and Limousine Commission Trip Record Data, "
    "public dataset, " + TLC_PAGE
)
ZONE_LICENSE = "NYC TLC taxi zone lookup and shapefile, public dataset, " + TLC_PAGE
NOAA_LICENSE = (
    "NOAA GHCN-Daily station USW00094728 (Central Park, NY), "
    "U.S. government public domain, https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily"
)


def months(start=(2019, 1), end=(2023, 12)):
    y, m = start
    ey, em = end
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


def _fetch(url: str, dest: Path, retries: int = 4) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "cs5488-parallel-taxi/1.0"})
            with urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
            tmp.replace(dest)
            return dest.stat().st_size
        except (URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}: {last}")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _one_month(y: int, m: int) -> dict:
    ym = f"{y:04d}-{m:02d}"
    url = TRIP_URL.format(ym=ym)
    dest = RAW / f"yellow_tripdata_{ym}.parquet"
    if not dest.exists() or dest.stat().st_size < 1000:
        _fetch(url, dest)
    return {
        "kind": "yellow_trip",
        "name": dest.name,
        "url": url,
        "year": y,
        "month": m,
        "bytes": dest.stat().st_size,
        "rows": _parquet_rows(dest),
        "md5": _md5(dest),
        "path": str(dest.relative_to(ROOT)),
        "license": TLC_LICENSE,
    }


def _static(kind: str, url: str, dest: Path, license_text: str) -> dict:
    if not dest.exists() or dest.stat().st_size < 100:
        _fetch(url, dest)
    rows = ""
    if dest.suffix == ".csv":
        with dest.open("rb") as f:
            rows = max(sum(1 for _ in f) - 1, 0)
    return {
        "kind": kind,
        "name": dest.name,
        "url": url,
        "year": "",
        "month": "",
        "bytes": dest.stat().st_size,
        "rows": rows,
        "md5": _md5(dest),
        "path": str(dest.relative_to(ROOT)),
        "license": license_text,
    }


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    EXT.mkdir(parents=True, exist_ok=True)
    jobs = list(months())
    records = []
    print(f"downloading {len(jobs)} monthly files into {RAW}", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_one_month, y, m): (y, m) for y, m in jobs}
        for fut in as_completed(futs):
            y, m = futs[fut]
            rec = fut.result()
            records.append(rec)
            print(f"  {y}-{m:02d} rows={rec['rows']:,} bytes={rec['bytes']:,}", flush=True)
    records.sort(key=lambda r: (r["year"], r["month"]))
    records.append(_static("taxi_zone_lookup", ZONE_URL, EXT / "taxi_zone_lookup.csv", ZONE_LICENSE))
    records.append(_static("taxi_zones_shapefile", SHAPE_URL, EXT / "taxi_zones.zip", ZONE_LICENSE))
    records.append(_static("noaa_ghcn_daily", NOAA_URL, EXT / "USW00094728.csv", NOAA_LICENSE))
    downloaded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fields = ["kind", "name", "year", "month", "rows", "bytes", "md5", "url", "path", "license", "downloaded_at_utc"]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in records:
            rec["downloaded_at_utc"] = downloaded_at
            w.writerow({k: rec.get(k, "") for k in fields})
    trip_rows = sum(int(r["rows"]) for r in records if r["kind"] == "yellow_trip")
    print(f"manifest {MANIFEST} trip_rows={trip_rows:,}", flush=True)


if __name__ == "__main__":
    main()
