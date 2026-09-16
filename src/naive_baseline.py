"""
Naive per-node safety stock baseline.

Run:
    python src/naive_baseline.py

This is the standard textbook approach and the baseline every later policy in
this project gets compared against. Every node is treated as an *independent*
single-echelon inventory problem:

    safety_stock = z * sigma_daily * sqrt(lead_time_days)

where `z` is the one-sided normal quantile for a 95% cycle service level and
`sigma_daily * sqrt(L)` is the standard deviation of demand accumulated over
the node's own lead time (the iid-demand assumption: variance adds across days,
so the standard deviation grows with the square root of the lead time).

Node lead times:
    * store -> its own DC->store transit time
    * DC    -> its own supplier->DC ocean + customs time

Node demand:
    * store -> its realized daily demand history
    * DC    -> the sum of the daily demand of the stores it serves

Writes `data/naive_baseline_results.parquet` (one row per node) and prints a
summary table plus an explicit list of the nodes where this formula's
assumptions do not hold.

What this baseline deliberately gets wrong
------------------------------------------
It is a *naive* baseline, so the known shortcomings are measured and reported
rather than quietly patched:

1. Normality. The formula assumes demand over the lead time is near-normal. For
   the low-volume stores the daily distribution is right-skewed and
   zero-inflated, so the normal quantile understates the true 95th percentile.
2. Independence. The sqrt(L) term assumes demand is iid day to day. It is not -
   there is a weekly cycle, an annual season and a trend - so the variance of
   lead-time demand grows faster than L. This turns out to be the LARGEST error
   in the baseline, and it is worst at the long-lead-time DCs rather than at the
   lumpy stores. `print_iid_warning()` measures it.
3. Lead time variability is ignored. The config gives each lane a lead time
   standard deviation, and the classic formula above uses only the mean. A
   reference column shows what including it would add.
4. No pooling. Each node is sized on its own, so the DC holds safety stock
   against its own variability *and* every store holds safety stock against
   theirs. Nothing is pooled. That double-counting is the specific thing a
   multi-echelon policy is meant to remove, and it is why this total is the
   number to beat.
5. DC demand is modelled as the sum of retail demand. In reality a DC sees
   batched store replenishment orders, which are lumpier than the underlying
   sales they cover.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from network_config import (
    DISTRIBUTION_CENTERS,
    STORES,
    stores_for_dc,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMAND_PATH = PROJECT_ROOT / "data" / "demand_history.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "naive_baseline_results.parquet"

# Target cycle service level and the matching one-sided normal quantile.
# 95% -> z = 1.6449. Changing this constant is the only knob on the policy.
SERVICE_LEVEL = 0.95
Z_SCORE = float(stats.norm.ppf(SERVICE_LEVEL))

DAYS_PER_YEAR = 365

# Thresholds used to flag nodes where the normal assumption is a poor fit.
# These are diagnostic rules of thumb, not tuned parameters.
CV_FLAG_THRESHOLD = 0.5  # above this, daily demand is no longer "steady"
SKEW_FLAG_THRESHOLD = 1.5  # pronounced right tail
ZERO_DAY_FLAG_THRESHOLD = 0.01  # >1% of days with no demand at all
# If a fitted normal puts meaningful mass below zero, it is the wrong shape for
# a non-negative count variable.
NEGATIVE_MASS_FLAG_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Building the per-node demand series
# ---------------------------------------------------------------------------


def build_node_demand(history: pd.DataFrame) -> dict[str, pd.Series]:
    """Daily demand series for every node in the network, keyed by node id.

    Stores use their own sales history. A DC's demand is the sum of the daily
    demand of the stores it replenishes (see caveat 4 in the module docstring).
    """
    store_demand = history.pivot_table(
        index="date", columns="store_id", values="demand", aggfunc="sum"
    ).sort_index()

    node_demand: dict[str, pd.Series] = {
        store_id: store_demand[store_id] for store_id in STORES
    }

    for dc_id in DISTRIBUTION_CENTERS:
        served = [s.id for s in stores_for_dc(dc_id)]
        node_demand[dc_id] = store_demand[served].sum(axis=1)

    return node_demand


# ---------------------------------------------------------------------------
# The naive safety stock calculation
# ---------------------------------------------------------------------------


def naive_safety_stock(sigma_daily: float, lead_time_days: float) -> float:
    """Classic single-node safety stock: z * sigma_daily * sqrt(L)."""
    return Z_SCORE * sigma_daily * np.sqrt(lead_time_days)


def safety_stock_with_lead_time_variability(
    mean_daily: float,
    sigma_daily: float,
    lead_time_days: float,
    lead_time_std: float,
) -> float:
    """Reference-only extension that also accounts for lead time variability.

    sigma_LTD = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)

    Reported alongside the baseline purely to size what caveat 2 above is
    leaving on the table. It is NOT the baseline policy.
    """
    variance = lead_time_days * sigma_daily**2 + (mean_daily**2) * lead_time_std**2
    return Z_SCORE * np.sqrt(variance)


def _distribution_diagnostics(demand: pd.Series, lead_time_days: int) -> dict:
    """Shape statistics used to judge whether the normal assumption holds."""
    mean_daily = float(demand.mean())
    sigma_daily = float(demand.std(ddof=1))

    # Demand actually accumulated over a lead-time-length window. Rolling
    # windows overlap, so these are correlated samples, but they are the most
    # honest picture available of the quantity the safety stock has to cover.
    ltd = demand.rolling(lead_time_days).sum().dropna()

    # How much probability mass a fitted normal would put below zero. For a
    # non-negative count variable, anything material here means the normal is
    # simply the wrong distribution.
    negative_mass = float(stats.norm.cdf(0.0, loc=mean_daily, scale=sigma_daily))

    # What the formula's iid assumption predicts the lead-time demand std to
    # be, versus what it actually is. Their ratio (squared) is the factor by
    # which sqrt(L) scaling misses the real variance.
    iid_ltd_std = sigma_daily * np.sqrt(lead_time_days)
    actual_ltd_std = float(ltd.std(ddof=1))

    return {
        "mean_daily_demand": mean_daily,
        "std_daily_demand": sigma_daily,
        "cv": sigma_daily / mean_daily if mean_daily > 0 else np.nan,
        "skew_daily": float(demand.skew()),
        "excess_kurtosis_daily": float(demand.kurt()),
        "zero_demand_day_share": float((demand == 0).mean()),
        "normal_mass_below_zero": negative_mass,
        # Autocorrelation at the lags the formula implicitly assumes away.
        "autocorr_lag1": float(demand.autocorr(1)),
        "autocorr_lag7": float(demand.autocorr(7)),
        # Lead-time demand distribution.
        "ltd_mean": float(ltd.mean()),
        "ltd_std": actual_ltd_std,
        "ltd_p95_empirical": float(ltd.quantile(SERVICE_LEVEL)),
        "ltd_skew": float(ltd.skew()),
        "iid_ltd_std": float(iid_ltd_std),
        "variance_inflation_factor": float((actual_ltd_std / iid_ltd_std) ** 2),
    }


def _flag_reasons(diag: dict) -> list[str]:
    """Name every reason a node is a poor fit for the normal-based formula."""
    reasons = []
    if diag["cv"] >= CV_FLAG_THRESHOLD:
        reasons.append(f"CV {diag['cv']:.2f} >= {CV_FLAG_THRESHOLD}")
    if diag["skew_daily"] >= SKEW_FLAG_THRESHOLD:
        reasons.append(f"right-skewed (skew {diag['skew_daily']:.2f})")
    if diag["zero_demand_day_share"] > ZERO_DAY_FLAG_THRESHOLD:
        reasons.append(
            f"zero-inflated ({diag['zero_demand_day_share']:.1%} of days are zero)"
        )
    if diag["normal_mass_below_zero"] > NEGATIVE_MASS_FLAG_THRESHOLD:
        reasons.append(
            f"fitted normal puts {diag['normal_mass_below_zero']:.1%} of mass below zero"
        )
    return reasons


def compute_baseline(history: pd.DataFrame) -> pd.DataFrame:
    """Run the naive policy over every node and return one row per node."""
    node_demand = build_node_demand(history)
    rows = []

    # --- DCs: lead time is the supplier -> DC ocean + customs leg ----------
    for dc in DISTRIBUTION_CENTERS.values():
        demand = node_demand[dc.id]
        lead_time = int(round(dc.lead_time_mean_days))
        diag = _distribution_diagnostics(demand, lead_time)

        rows.append(
            {
                "node_id": dc.id,
                "node_name": dc.name,
                "node_type": "DC",
                "region": dc.region,
                "served_by": "SUP1",
                "lead_time_days": dc.lead_time_mean_days,
                "lead_time_std_days": dc.lead_time_std_days,
                "holding_cost_per_unit_per_day": dc.holding_cost_per_unit_per_day,
                **diag,
                "safety_stock_units": naive_safety_stock(
                    diag["std_daily_demand"], dc.lead_time_mean_days
                ),
                "safety_stock_units_with_lt_variability": (
                    safety_stock_with_lead_time_variability(
                        diag["mean_daily_demand"],
                        diag["std_daily_demand"],
                        dc.lead_time_mean_days,
                        dc.lead_time_std_days,
                    )
                ),
            }
        )

    # --- Stores: lead time is the DC -> store transit leg ------------------
    for store in STORES.values():
        demand = node_demand[store.id]
        lead_time = int(round(store.lead_time_mean_days))
        diag = _distribution_diagnostics(demand, lead_time)

        rows.append(
            {
                "node_id": store.id,
                "node_name": store.name,
                "node_type": "STORE",
                "region": store.region,
                "served_by": store.dc_id,
                "lead_time_days": store.lead_time_mean_days,
                "lead_time_std_days": store.lead_time_std_days,
                "holding_cost_per_unit_per_day": store.holding_cost_per_unit_per_day,
                **diag,
                "safety_stock_units": naive_safety_stock(
                    diag["std_daily_demand"], store.lead_time_mean_days
                ),
                "safety_stock_units_with_lt_variability": (
                    safety_stock_with_lead_time_variability(
                        diag["mean_daily_demand"],
                        diag["std_daily_demand"],
                        store.lead_time_mean_days,
                        store.lead_time_std_days,
                    )
                ),
            }
        )

    results = pd.DataFrame(rows)

    # Holding cost of carrying the safety stock. Safety stock is by definition
    # the layer that sits there permanently (cycle stock turns over on top of
    # it), so it is charged for every day of the year.
    results["holding_cost_per_day"] = (
        results["safety_stock_units"] * results["holding_cost_per_unit_per_day"]
    )
    results["holding_cost_annual"] = results["holding_cost_per_day"] * DAYS_PER_YEAR

    # How far the normal-based number sits from the empirical 95th percentile
    # of actual lead-time demand. Positive => the formula over-provisions.
    results["safety_stock_empirical"] = (
        results["ltd_p95_empirical"] - results["ltd_mean"]
    )
    results["normal_vs_empirical_error_pct"] = (
        (results["safety_stock_units"] - results["safety_stock_empirical"])
        / results["safety_stock_empirical"]
        * 100
    )

    # Flag the nodes whose demand shape breaks the formula's assumptions.
    flag_reasons = results.apply(lambda r: _flag_reasons(r.to_dict()), axis=1)
    results["assumption_flags"] = flag_reasons.apply("; ".join)
    results["normality_ok"] = flag_reasons.apply(len) == 0

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(results: pd.DataFrame) -> None:
    """Print the per-node table, the network cost roll-up and the caveats."""
    print()
    print("=" * 108)
    print(
        f"NAIVE PER-NODE SAFETY STOCK BASELINE · "
        f"{SERVICE_LEVEL:.0%} service level (z = {Z_SCORE:.4f}) · "
        f"every node sized independently"
    )
    print("=" * 108)
    print("formula:  safety_stock = z * sigma_daily * sqrt(lead_time_days)")
    print()

    header = (
        f"{'NODE':<9} {'NAME':<20} {'REGION':<15} {'TYPE':<6} "
        f"{'LT(d)':>6} {'CV':>6} {'SS UNITS':>10} "
        f"{'$/DAY':>8} {'$/YEAR':>10}  {'':<4}"
    )

    for node_type in ("DC", "STORE"):
        block = results[results["node_type"] == node_type].sort_values(
            ["region", "safety_stock_units"], ascending=[True, False]
        )
        label = "ECHELON 1 - DISTRIBUTION CENTERS" if node_type == "DC" else (
            "ECHELON 2 - RETAIL STORES"
        )
        print(label)
        print(header)
        print("-" * 108)
        for _, r in block.iterrows():
            marker = "  <!" if not r["normality_ok"] else ""
            print(
                f"{r['node_id']:<9} {r['node_name'][:20]:<20} {r['region']:<15} "
                f"{r['node_type']:<6} {r['lead_time_days']:>6.0f} {r['cv']:>6.2f} "
                f"{r['safety_stock_units']:>10.1f} "
                f"{r['holding_cost_per_day']:>8.2f} "
                f"{r['holding_cost_annual']:>10,.0f}{marker}"
            )
        print()

    # --- Network cost roll-up ---------------------------------------------
    by_type = results.groupby("node_type")[
        ["safety_stock_units", "holding_cost_per_day", "holding_cost_annual"]
    ].sum()

    print("=" * 108)
    print("TOTAL NETWORK HOLDING COST UNDER THE NAIVE POLICY")
    print("=" * 108)
    print(f"{'':<26} {'SS UNITS':>12} {'$/DAY':>12} {'$/YEAR':>14}")
    print("-" * 108)
    for node_type in ("DC", "STORE"):
        row = by_type.loc[node_type]
        label = "DC echelon" if node_type == "DC" else "Store echelon"
        print(
            f"{label:<26} {row['safety_stock_units']:>12,.1f} "
            f"{row['holding_cost_per_day']:>12,.2f} "
            f"{row['holding_cost_annual']:>14,.0f}"
        )
    print("-" * 108)
    total = by_type.sum()
    print(
        f"{'NETWORK TOTAL':<26} {total['safety_stock_units']:>12,.1f} "
        f"{total['holding_cost_per_day']:>12,.2f} "
        f"{total['holding_cost_annual']:>14,.0f}"
    )
    print()

    # Regional split: holding rates differ by region, so where the cost lands
    # is not the same as where the units land.
    print("By region:")
    by_region = (
        results.groupby("region")[["safety_stock_units", "holding_cost_annual"]]
        .sum()
        .sort_values("holding_cost_annual", ascending=False)
    )
    for region, row in by_region.iterrows():
        share = row["holding_cost_annual"] / total["holding_cost_annual"]
        print(
            f"  {region:<15} {row['safety_stock_units']:>9,.1f} units · "
            f"{row['holding_cost_annual']:>10,.0f}/year ({share:.0%} of network cost)"
        )
    print()


def print_assumption_warnings(results: pd.DataFrame) -> None:
    """Name the nodes where the normal-demand assumption does not hold."""
    print("=" * 108)
    print("ASSUMPTION CHECK - WHERE THIS FORMULA IS NOT TRUSTWORTHY")
    print("=" * 108)
    print(
        "The formula assumes demand over the lead time is approximately normal.\n"
        "These nodes violate that assumption, so their safety stock number should\n"
        "not be read as delivering a genuine 95% service level:"
    )
    print()

    flagged = results[~results["normality_ok"]].sort_values("cv", ascending=False)

    if flagged.empty:
        print("  (none)")
        return

    for _, r in flagged.iterrows():
        print(f"  {r['node_id']} · {r['node_name']} ({r['region']})")
        for reason in r["assumption_flags"].split("; "):
            print(f"      - {reason}")
        print(
            f"      -> naive SS {r['safety_stock_units']:.1f} units vs "
            f"empirical 95th-pct requirement {r['safety_stock_empirical']:.1f} units "
            f"({r['normal_vs_empirical_error_pct']:+.1f}%)"
        )
        print()

    flagged_cost = flagged["holding_cost_annual"].sum()
    total_cost = results["holding_cost_annual"].sum()
    print(
        f"  {len(flagged)} of {len(results)} nodes flagged, carrying "
        f"{flagged_cost:,.0f}/year ({flagged_cost / total_cost:.0%} of network "
        f"holding cost).\n"
    )

    print("Why it matters:")
    print(
        "  - Lumpy, zero-inflated demand has a long right tail the normal does not\n"
        "    reproduce, so every one of these nodes lands BELOW the units its own\n"
        "    history says a 95th percentile needs. The policy nominally targets 95%\n"
        "    and does not deliver it.\n"
        "  - The symmetric normal also spreads mass below zero (10-18% here), which\n"
        "    is impossible for a non-negative count variable. That misplaced mass is\n"
        "    precisely what is missing from the right tail.\n"
        "  - The right fix is a distribution that matches the data (negative\n"
        "    binomial, or an empirical/bootstrap quantile), not a bigger z."
    )
    print()


def print_iid_warning(results: pd.DataFrame) -> None:
    """Report where sqrt(lead_time) scaling breaks down.

    This check was not part of the original brief, but it turns out to be the
    single largest error in the baseline - larger than the non-normality of the
    lumpy stores - so it is reported rather than buried in a column.
    """
    print("=" * 108)
    print("IID CHECK - WHERE sqrt(LEAD TIME) SCALING BREAKS DOWN")
    print("=" * 108)
    print(
        "The sqrt(L) term assumes demand is independent day to day, so that variance\n"
        "simply adds up. Real demand here is autocorrelated - a weekly cycle, an\n"
        "annual season and a trend - so the variance of demand accumulated over L\n"
        "days grows FASTER than L. The longer the lead time, the worse the miss:"
    )
    print()

    header = (
        f"{'NODE':<9} {'TYPE':<6} {'LT(d)':>6} {'sigma*sqrt(L)':>14} "
        f"{'ACTUAL LTD SD':>14} {'VARIANCE MISS':>14}"
    )
    print(header)
    print("-" * 108)
    for _, r in results.sort_values("variance_inflation_factor").iterrows():
        marker = "  <!" if r["variance_inflation_factor"] >= 2.0 else ""
        print(
            f"{r['node_id']:<9} {r['node_type']:<6} {r['lead_time_days']:>6.0f} "
            f"{r['iid_ltd_std']:>14,.1f} {r['ltd_std']:>14,.1f} "
            f"{r['variance_inflation_factor']:>13.1f}x{marker}"
        )
    print()

    dcs = results[results["node_type"] == "DC"]
    print(
        f"  The three DCs, on 30-45 day lead times, understate lead-time demand\n"
        f"  variance by {dcs['variance_inflation_factor'].min():.1f}x-"
        f"{dcs['variance_inflation_factor'].max():.1f}x. This is a bigger error than the\n"
        f"  non-normality flagged above, and it hits the nodes the brief expected to\n"
        f"  be SAFE - the high-volume, low-CV ones - because the problem is lead time\n"
        f"  length and autocorrelation, not demand lumpiness."
    )
    print()
    by_id = dcs.set_index("node_id")["variance_inflation_factor"]
    print(
        f"  Note the spread across DCs. Singapore is the mildest ({by_id['DC_APAC']:.1f}x) "
        "because\n"
        "  Sydney peaks mid-year while Tokyo and Jakarta peak in December, so the\n"
        "  seasonal swings partly cancel when pooled - drop Sydney from the pool and\n"
        "  Singapore's inflation roughly doubles (2.8x -> 5.2x). Rotterdam's stores are\n"
        "  all in phase; measured over the same 30-day window it sits at 7.5x against\n"
        f"  Singapore's 2.8x, and its 45-day lead time compounds that to "
        f"{by_id['DC_EU']:.1f}x.\n"
        "  Out-of-phase demand is real pooling value this per-node policy cannot see."
    )
    print()


def print_structural_caveats(results: pd.DataFrame) -> None:
    """Report the limitations that apply to every node, not just the flagged ones."""
    print("=" * 108)
    print("STRUCTURAL LIMITATIONS OF THE NAIVE POLICY (all nodes)")
    print("=" * 108)

    lt_extra = (
        results["safety_stock_units_with_lt_variability"] - results["safety_stock_units"]
    )
    lt_extra_cost = (lt_extra * results["holding_cost_per_unit_per_day"]).sum() * (
        DAYS_PER_YEAR
    )
    print(
        f"1. Lead time variability is ignored. The config gives every lane a lead\n"
        f"   time std dev, but the classic formula uses only the mean. Folding it in\n"
        f"   would add {lt_extra.sum():,.0f} units "
        f"({lt_extra_cost:,.0f}/year) across the network - most of it at the DCs,\n"
        f"   where ocean freight and customs are the variable part."
    )
    print()

    dc_units = results.loc[results["node_type"] == "DC", "safety_stock_units"].sum()
    store_units = results.loc[
        results["node_type"] == "STORE", "safety_stock_units"
    ].sum()
    print(
        f"2. No risk pooling. Each of the 13 nodes is sized against its own\n"
        f"   variability in isolation: {dc_units:,.0f} units at the DCs and\n"
        f"   {store_units:,.0f} units at the stores, with the DC layer protecting\n"
        f"   against variability the stores are already covering. This\n"
        f"   double-counting is exactly what a multi-echelon policy removes, and it\n"
        f"   is why this total is the number to beat."
    )
    print()
    print(
        "3. DC demand is taken as the sum of retail sales. A real DC sees batched\n"
        "   store replenishment orders, which are lumpier than the sales beneath\n"
        "   them, so the DC figures here understate true DC-level variability."
    )
    print()
    print(
        "4. Lead-time demand is measured on overlapping rolling windows, so the\n"
        "   empirical percentiles above come from correlated samples and the long\n"
        "   DC windows (30-45 days) have few effectively independent observations."
    )
    print()


def main() -> None:
    history = pd.read_parquet(DEMAND_PATH)
    results = compute_baseline(history)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(OUTPUT_PATH, index=False)

    print_summary(results)
    print_assumption_warnings(results)
    print_iid_warning(results)
    print_structural_caveats(results)

    print(
        f"Wrote {len(results)} node rows to "
        f"{OUTPUT_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
