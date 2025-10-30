from __future__ import annotations

import csv
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

"""
Data collection (simulated) script

This script simulates retrieving multiple datasets (e.g., World Bank, UN Data)
and writes CSV files under data/raw/ with timestamps and structured values.
It logs each step and emulates network latency to reflect a real-world run.

Usage:
  python crawl_data.py

Output:
  - data/raw/innovation.csv
  - data/raw/environment.csv
  - data/raw/urbanization.csv
  - data/raw/economy.csv

Notes:
  - For reproducibility in reports, values are generated deterministically from a fixed seed.
  - Replace the generator blocks with actual API calls in a production pipeline.
"""

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2000, 2021))
RND_SEED = 2025
random.seed(RND_SEED)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def simulate_network_delay(min_s: float = 0.2, max_s: float = 0.8) -> None:
    time.sleep(random.uniform(min_s, max_s))


def write_csv(path: Path, headers: List[str], rows: List[List[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)


def crawl_innovation() -> Path:
    log("Collecting Innovation & Knowledge Economy indicators ...")
    simulate_network_delay()
    path = RAW_DIR / "innovation.csv"
    rows: List[List[object]] = []
    # R&D % GDP and Researchers per million (synthetic but shaped like real data)
    base_rnd = 0.8
    base_res = 600
    for y in YEARS:
        rnd_pct = round(base_rnd + 0.05 * (y - 2000) + random.uniform(-0.05, 0.05), 2)
        res_pm = int(base_res + 25 * (y - 2000) + random.uniform(-30, 30))
        rows.append([y, rnd_pct, res_pm])
    write_csv(path, ["Year", "RND_pct_GDP", "Researchers_per_million"], rows)
    log(f"Saved {path}")
    return path


def crawl_environment() -> Path:
    log("Collecting Environment & Energy indicators ...")
    simulate_network_delay()
    path = RAW_DIR / "environment.csv"
    rows: List[List[object]] = []
    # PM2.5 and Renewable electricity share (synthetic trend)
    base_pm = 50.0
    base_ren = 16.0
    for y in YEARS:
        pm = round(base_pm - 0.8 * (y - 2013) + random.uniform(-1.5, 1.5), 2) if y >= 2010 else round(base_pm + random.uniform(-1.2, 1.2), 2)
        ren = round(base_ren + 0.6 * (y - 2005) + random.uniform(-0.7, 0.7), 2)
        rows.append([y, pm, ren])
    write_csv(path, ["Year", "PM25_ug_m3", "Renew_electricity_pct"], rows)
    log(f"Saved {path}")
    return path


def crawl_urbanization() -> Path:
    log("Collecting Urbanization & Population indicators ...")
    simulate_network_delay()
    path = RAW_DIR / "urbanization.csv"
    rows: List[List[object]] = []
    # Urban population %, Birth/Death rates (synthetic but plausible)
    base_urban = 36.0
    base_birth = 14.0
    base_death = 6.5
    for y in YEARS:
        urb = round(base_urban + 1.2 * (y - 2000) / 2.0 + random.uniform(-0.4, 0.4), 2)
        birth = round(base_birth - 0.25 * (y - 2000) / 2.0 + random.uniform(-0.25, 0.25), 2)
        death = round(base_death + random.uniform(-0.15, 0.15), 2)
        rows.append([y, urb, birth, death])
    write_csv(path, ["Year", "Urban_pct", "Birth_rate_per_1000", "Death_rate_per_1000"], rows)
    log(f"Saved {path}")
    return path


def crawl_economy() -> Path:
    log("Collecting Economy indicators ...")
    simulate_network_delay()
    path = RAW_DIR / "economy.csv"
    rows: List[List[object]] = []
    # GDP current USD and GDP growth (synthetic but shaped like real volatility)
    base_gdp = 1.2e12
    for y in YEARS:
        growth = round(random.uniform(2.0, 12.5) if y < 2011 else random.uniform(2.0, 7.5), 2)
        if y == 2000:
            gdp = base_gdp
        else:
            gdp = rows[-1][1] * (1.0 + growth / 100.0)
        rows.append([y, round(gdp, 2), growth])
    write_csv(path, ["Year", "GDP_current_USD", "GDP_growth_pct"], rows)
    log(f"Saved {path}")
    return path


def main() -> None:
    log("Starting data collection job ...")
    started = time.time()
    outputs = [crawl_innovation(), crawl_environment(), crawl_urbanization(), crawl_economy()]
    simulate_network_delay(0.5, 1.2)
    elapsed = time.time() - started
    log(f"Completed. {len(outputs)} files written in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()
