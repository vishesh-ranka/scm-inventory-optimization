"""
Generate synthetic daily demand history for every store in the network.

Run:
    python src/generate_demand.py

Writes `data/demand_history.parquet` with one row per (store, day) and prints a
summary table of each store's region, serving DC, average daily demand and
demand variability.

Demand model (deliberately simple and transparent)
--------------------------------------------------
For store *s* on day *t*:

    mu[s, t] = base[s] * dow[t] * season[s, t] * trend[s, t]
    demand[s, t] ~ NegativeBinomial(mean=mu[s, t], cv=cv[s])

* `dow`    - day-of-week pattern, with a per-store uplift on Sat/Sun.
* `season` - one annual sine wave; amplitude and peak day vary by store, so the
             Southern-hemisphere store peaks mid-year rather than in December.
* `trend`  - gentle per-store year-over-year growth (or decline).
* The negative binomial is used because it produces non-negative integer counts
  that can be *overdispersed* relative to Poisson. That is what lets us give
  low-volume stores genuinely lumpy demand (CV ~0.9) while high-volume stores
  stay steady (CV ~0.2), instead of every node looking the same.

Everything is driven by a single seeded numpy Generator, so re-running the
script reproduces byte-identical output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from network_config import (
    DEMAND_CONFIG,
    DISTRIBUTION_CENTERS,
    STORES,
    Store,
)

# Project layout: this file lives in src/, data lands in data/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "demand_history.parquet"

# Per-store annual growth rate applied on top of the base level. Kept here
# rather than in the network config because it is a property of the synthetic
# history, not of the physical network.
ANNUAL_GROWTH_BY_STORE: dict[str, float] = {
    "S_NA_01": 0.04,
    "S_NA_02": 0.02,
    "S_NA_03": 0.08,
    "S_NA_04": -0.03,
    "S_EU_01": 0.03,
    "S_EU_02": 0.00,
    "S_EU_03": 0.06,
    "S_AP_01": 0.05,
    "S_AP_02": 0.02,
    "S_AP_03": 0.10,
}


def _expected_demand(store: Store, dates: pd.DatetimeIndex) -> np.ndarray:
    """Deterministic mean demand curve mu[t] for one store (no noise yet)."""
    # 1. Day-of-week pattern. Monday == 0 ... Sunday == 6.
    dow_factors = np.asarray(DEMAND_CONFIG.day_of_week_factors)
    factor = dow_factors[dates.dayofweek.to_numpy()]

    # Stores with a strong weekend skew (malls, outlets, tourist spots) get an
    # extra multiplier on Sat/Sun only.
    is_weekend = dates.dayofweek.to_numpy() >= 5
    factor = np.where(is_weekend, factor * store.weekend_uplift, factor)

    # 2. Annual seasonality: a single sine wave peaking on seasonal_peak_day.
    day_of_year = dates.dayofyear.to_numpy()
    phase = 2 * np.pi * (day_of_year - store.seasonal_peak_day) / 365.25
    seasonal = 1.0 + store.seasonal_amplitude * np.cos(phase)

    # 3. Gentle linear year-over-year trend across the history.
    years_elapsed = np.arange(len(dates)) / 365.25
    trend = 1.0 + ANNUAL_GROWTH_BY_STORE.get(store.id, 0.0) * years_elapsed

    mu = store.base_daily_demand * factor * seasonal * trend
    # Guard against a pathological config driving the mean to zero.
    return np.maximum(mu, 0.1)


def _sample_demand(
    mu: np.ndarray, cv: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw integer demand with mean `mu` and coefficient of variation `cv`.

    Negative binomial with mean m and dispersion r has variance m + m^2 / r.
    We want variance = (cv * m)^2, which gives r = m / (cv^2 * m - 1).

    When cv^2 * m <= 1 the target variance is below the Poisson floor (the NB
    cannot be *under*-dispersed), so we fall back to a Poisson draw. That only
    happens for small means combined with a small CV, which this config avoids,
    but the guard keeps the function safe if the parameters are ever retuned.
    """
    target_var = (cv * mu) ** 2
    overdispersed = target_var > mu + 1e-9

    demand = np.empty_like(mu, dtype=np.int64)

    # Poisson branch (variance == mean) for any days that are not overdispersed.
    if (~overdispersed).any():
        demand[~overdispersed] = rng.poisson(mu[~overdispersed])

    # Negative binomial branch for the rest.
    if overdispersed.any():
        m = mu[overdispersed]
        r = m / (cv**2 * m - 1.0)
        p = r / (r + m)  # numpy's `n, p` parameterization
        demand[overdispersed] = rng.negative_binomial(r, p)

    return demand


def generate_demand_history() -> pd.DataFrame:
    """Build the full (store x day) demand history as a tidy DataFrame."""
    rng = np.random.default_rng(DEMAND_CONFIG.random_seed)

    dates = pd.date_range(
        start=DEMAND_CONFIG.start_date,
        periods=DEMAND_CONFIG.n_days,
        freq="D",
    )

    frames = []
    # Iterate in a fixed order so the seeded draws stay reproducible.
    for store_id in sorted(STORES):
        store = STORES[store_id]
        mu = _expected_demand(store, dates)
        demand = _sample_demand(mu, store.demand_cv, rng)

        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "store_id": store.id,
                    "store_name": store.name,
                    "region": store.region,
                    "dc_id": store.dc_id,
                    "demand": demand,
                    # Keep the noiseless mean around: useful later as the
                    # "perfect forecast" benchmark when evaluating policies.
                    "expected_demand": mu,
                }
            )
        )

    history = pd.concat(frames, ignore_index=True)
    history["date"] = history["date"].astype("datetime64[ns]")
    return history


def summarize(history: pd.DataFrame) -> pd.DataFrame:
    """Per-store summary: region, serving DC, mean demand and variability.

    Note: the CV reported here is the *realized* CV of the finished history, so
    it sits above each store's configured `demand_cv`. The configured value is
    the day-to-day noise around the mean curve; the realized value also picks up
    the day-of-week swing, the annual season and the trend. The relative
    ordering across stores is preserved, which is what matters for the network
    being non-uniform.
    """
    summary = (
        history.groupby(["store_id", "store_name", "region", "dc_id"], as_index=False)
        .agg(
            avg_daily_demand=("demand", "mean"),
            std_daily_demand=("demand", "std"),
            min_daily=("demand", "min"),
            max_daily=("demand", "max"),
            zero_demand_days=("demand", lambda s: int((s == 0).sum())),
        )
        .assign(cv=lambda df: df["std_daily_demand"] / df["avg_daily_demand"])
    )

    # Label each store so the high-volume/low-variability vs low-volume/lumpy
    # split in the network is obvious at a glance.
    def _profile(row: pd.Series) -> str:
        volume = "high-volume" if row["avg_daily_demand"] >= 50 else "low-volume"
        variability = "low-var" if row["cv"] < 0.5 else "high-var"
        return f"{volume}/{variability}"

    summary["profile"] = summary.apply(_profile, axis=1)
    return summary.sort_values(["region", "avg_daily_demand"], ascending=[True, False])


def print_summary(history: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Print the quick sanity-check table described in the project brief."""
    start = history["date"].min().date()
    end = history["date"].max().date()
    n_days = history["date"].nunique()

    print()
    print("=" * 104)
    print(
        f"SYNTHETIC DEMAND HISTORY · {n_days} days ({start} -> {end}) · "
        f"{summary.shape[0]} stores · seed {DEMAND_CONFIG.random_seed}"
    )
    print("=" * 104)

    header = (
        f"{'STORE':<9} {'NAME':<20} {'REGION':<15} {'DC':<9} "
        f"{'AVG/DAY':>9} {'STD':>8} {'CV':>6} {'MAX':>6} {'PROFILE':>22}"
    )
    print(header)
    print("-" * 104)

    current_region = None
    for _, row in summary.iterrows():
        if current_region is not None and row["region"] != current_region:
            print("-" * 104)
        current_region = row["region"]
        print(
            f"{row['store_id']:<9} {row['store_name'][:20]:<20} "
            f"{row['region']:<15} {row['dc_id']:<9} "
            f"{row['avg_daily_demand']:>9.1f} {row['std_daily_demand']:>8.1f} "
            f"{row['cv']:>6.2f} {row['max_daily']:>6.0f} {row['profile']:>22}"
        )

    print("-" * 104)

    # Roll the store demand up to the DC that serves it: this is the demand the
    # DC echelon actually has to cover, and it is what the long ocean lead times
    # will be sized against later.
    print()
    print("DC-level aggregate demand (sum of the stores each DC serves):")
    dc_totals = (
        summary.groupby("dc_id")
        .agg(stores=("store_id", "count"), avg_daily_demand=("avg_daily_demand", "sum"))
        .reindex(DISTRIBUTION_CENTERS.keys())
    )
    for dc_id, row in dc_totals.iterrows():
        dc = DISTRIBUTION_CENTERS[dc_id]
        print(
            f"  {dc_id:<9} {dc.name:<16} {dc.region:<15} "
            f"{int(row['stores'])} stores · {row['avg_daily_demand']:>7.1f} units/day · "
            f"supplier lead time {dc.lead_time_mean_days:.0f}d "
            f"(sd {dc.lead_time_std_days:.0f}d) · "
            f"holding {dc.holding_cost_per_unit_per_day:.3f}/unit/day"
        )
    print()


def main() -> None:
    history = generate_demand_history()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(OUTPUT_PATH, index=False)

    summary = summarize(history)
    print_summary(history, summary)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(
        f"Wrote {len(history):,} rows to {OUTPUT_PATH.relative_to(PROJECT_ROOT)} "
        f"({size_kb:.0f} KB)"
    )


if __name__ == "__main__":
    main()
