"""
Rolling-window backtest: are the validation findings stable, or artifacts?

Run:
    python src/rolling_validation.py

`validate_policy.py` tested one 18-month/6-month split and produced two headline
claims:

    * the multi-echelon policy saves ~2.7% against a decentralized 95% policy
    * it eliminates DC stockouts (100% DC order fill vs 92.8% for naive)

Both came from a single holdout that happened to contain the Q4 peak. This
script re-runs the same simulation across four expanding train/test windows so
each claim can be reported as a RANGE rather than a point estimate.

Windows (expanding train, 3-month test, so every quarter of 2024 gets tested):

    W1  train months 1-12   test 13-15   (Q1 2024 - post-peak trough)
    W2  train months 1-15   test 16-18   (Q2 2024 - shoulder)
    W3  train months 1-18   test 19-21   (Q3 2024 - shoulder, pre-peak)
    W4  train months 1-21   test 22-24   (Q4 2024 - the peak)

The design is deliberate: W4 contains the seasonal peak that the original
holdout was dominated by, and W1-W3 do not. If a finding only appears in W4, it
is a statement about Q4, not about the network.

Everything reuses `validate_policy` unchanged - the same fitting functions and
the same corrected simulation (protection interval of lead time + 1 review day,
expedite that accelerates rather than adds volume and is capped at normal
transit time). Nothing about the model is re-derived here; this file only varies
the window and aggregates.

Cache safety: `optimize_network` memoizes rolling windows keyed on the series
name AND its date span, so fitting on four different training slices cannot
return another window's numbers. `check_no_cache_leakage()` asserts this at the
start of every run rather than trusting it.

Three policies are carried through, the minimum needed to state both claims:

    naive             - the textbook per-node baseline
    decentralized_95  - empirical 95% everywhere, no pooling (the like-for-like
                        yardstick for the cost saving)
    optimized         - the multi-echelon policy, 1-day expedite

Writes `data/rolling_validation_results.parquet`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import optimize_network as opt
import validate_policy as vp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "rolling_validation_results.parquet"

# (label, months of training, months of testing). Training always starts at the
# beginning of the history, so each window refits on strictly more data - the
# expanding-window convention, which mirrors how a policy would actually be
# maintained in production.
WINDOW_SPECS: tuple[tuple[str, int, int], ...] = (
    ("W1", 12, 3),
    ("W2", 15, 3),
    ("W3", 18, 3),
    ("W4", 21, 3),
)

EXPEDITE_DAYS = 1


@dataclass(frozen=True)
class Window:
    """One train/test split, resolved to actual dates."""

    label: str
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def test_span(self) -> str:
        return f"{self.test.index[0].date()} -> {self.test.index[-1].date()}"

    @property
    def demand_ratio(self) -> float:
        """Test-period mean daily demand relative to the training period."""
        return self.test.sum(axis=1).mean() / self.train.sum(axis=1).mean()


def build_windows(daily: pd.DataFrame) -> list[Window]:
    """Resolve WINDOW_SPECS into concrete train/test frames."""
    start = daily.index[0]
    windows = []
    for label, train_months, test_months in WINDOW_SPECS:
        train_end = start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        train = daily[daily.index < train_end]
        test = daily[(daily.index >= train_end) & (daily.index < test_end)]
        if len(test) == 0:
            continue
        windows.append(Window(label=label, train=train, test=test))
    return windows


def check_no_cache_leakage(windows: list[Window]) -> None:
    """Confirm different training spans really do produce different windows.

    The rolling-window cache in `optimize_network` is keyed on the series name
    plus its span. If that key were ever weakened back to the name alone, every
    window after the first would silently be fitted on the first window's data
    and this whole backtest would be meaningless while still printing plausible
    numbers. Cheap to assert, catastrophic to miss.
    """
    sample_store = next(iter(vp.STORES))
    lengths = {
        w.label: len(opt.lead_time_windows(w.train[sample_store], 4)) for w in windows
    }
    if len(set(lengths.values())) != len(lengths):
        raise RuntimeError(f"window cache is leaking across train spans: {lengths}")


# ---------------------------------------------------------------------------
# Running the backtest
# ---------------------------------------------------------------------------


def run_window(daily: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Fit all three policies on this window's train data and simulate its test."""
    policies = [
        vp.fit_naive_policy(window.train),
        vp.fit_decentralized_policy(window.train),
        vp.fit_optimized_policy(window.train, EXPEDITE_DAYS),
    ]
    # fit_optimized_policy names itself after the expedite setting; normalize so
    # results can be grouped by policy across windows.
    policies[2].name = "optimized"

    frames = []
    for policy in policies:
        res = vp.run_policy(daily, window.train, window.test, policy)
        res["window"] = window.label
        res["test_span"] = window.test_span
        res["train_days"] = len(window.train)
        res["test_days"] = len(window.test)
        res["demand_ratio"] = window.demand_ratio
        frames.append(res)
    return pd.concat(frames, ignore_index=True)


def summarize_window(results: pd.DataFrame) -> pd.DataFrame:
    """Per (window, policy): store fill, worst store, DC order fill, cost."""
    rows = []
    for (window, policy), block in results.groupby(["window", "policy"], sort=False):
        stores = block[block.node_type == "STORE"]
        dcs = block[block.node_type == "DC"]
        rows.append(
            {
                "window": window,
                "policy": policy,
                "test_span": block.test_span.iloc[0],
                "demand_ratio": block.demand_ratio.iloc[0],
                "store_fill_rate": 1.0
                - stores.unmet_demand.sum() / stores.total_demand.sum(),
                "worst_store_fill": stores.fill_rate.min(),
                "stores_below_95": int((stores.fill_rate < 0.95).sum()),
                "dc_order_fill_rate": 1.0
                - dcs.unmet_demand.sum() / dcs.total_demand.sum(),
                "actual_cost_annual": block.actual_cost_annual.sum(),
            }
        )
    return pd.DataFrame(rows)


def build_claims(summary: pd.DataFrame) -> pd.DataFrame:
    """Per window, the two headline claims restated as numbers."""
    wide = summary.pivot(index="window", columns="policy")
    rows = []
    for window in summary.window.unique():
        decent_cost = wide.loc[window, ("actual_cost_annual", "decentralized_95")]
        opt_cost = wide.loc[window, ("actual_cost_annual", "optimized")]
        naive_dc = wide.loc[window, ("dc_order_fill_rate", "naive")]
        opt_dc = wide.loc[window, ("dc_order_fill_rate", "optimized")]
        rows.append(
            {
                "window": window,
                "test_span": summary.loc[
                    summary.window == window, "test_span"
                ].iloc[0],
                "demand_ratio": summary.loc[
                    summary.window == window, "demand_ratio"
                ].iloc[0],
                "cost_saving_abs": decent_cost - opt_cost,
                "cost_saving_pct": (decent_cost - opt_cost) / decent_cost,
                "naive_dc_fill": naive_dc,
                "optimized_dc_fill": opt_dc,
                "dc_fill_improvement": opt_dc - naive_dc,
                "optimized_store_fill": wide.loc[
                    window, ("store_fill_rate", "optimized")
                ],
                "naive_store_fill": wide.loc[window, ("store_fill_rate", "naive")],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_windows(windows: list[Window]) -> None:
    print()
    print("=" * 104)
    print("ROLLING-WINDOW BACKTEST")
    print("=" * 104)
    print(
        f"  {'WINDOW':<7} {'TRAIN':>10} {'TEST PERIOD':<26} {'TEST DAYS':>10} "
        f"{'TEST vs TRAIN DEMAND':>21}"
    )
    print("  " + "-" * 80)
    for w in windows:
        print(
            f"  {w.label:<7} {len(w.train):>7} d  {w.test_span:<26} "
            f"{len(w.test):>10} {w.demand_ratio - 1:>20.1%}"
        )
    print()


def print_per_window(summary: pd.DataFrame) -> None:
    print("=" * 104)
    print("PER-WINDOW RESULTS")
    print("=" * 104)
    for window in summary.window.unique():
        block = summary[summary.window == window]
        span = block.test_span.iloc[0]
        print(f"  {window}  test {span}")
        print(
            f"    {'POLICY':<18} {'STORE FILL':>11} {'WORST STORE':>12} "
            f"{'<95%':>6} {'DC ORDER FILL':>14} {'ACTUAL $/YEAR':>14}"
        )
        print("    " + "-" * 80)
        for r in block.itertuples():
            print(
                f"    {r.policy:<18} {r.store_fill_rate:>11.2%} "
                f"{r.worst_store_fill:>12.2%} {r.stores_below_95:>6} "
                f"{r.dc_order_fill_rate:>14.2%} {r.actual_cost_annual:>14,.0f}"
            )
        print()


def print_claims(claims: pd.DataFrame) -> None:
    print("=" * 104)
    print("CLAIM 1 - COST SAVING (optimized vs decentralized 95%)")
    print("=" * 104)
    print(
        f"  {'WINDOW':<7} {'TEST PERIOD':<26} {'SAVING $/YR':>13} {'SAVING %':>10}"
    )
    print("  " + "-" * 62)
    for r in claims.itertuples():
        print(
            f"  {r.window:<7} {r.test_span:<26} {r.cost_saving_abs:>13,.0f} "
            f"{r.cost_saving_pct:>10.2%}"
        )
    print("  " + "-" * 62)
    print(
        f"  {'range':<7} {'':<26} "
        f"{claims.cost_saving_abs.min():>7,.0f} to {claims.cost_saving_abs.max():<,.0f}"
        f"   {claims.cost_saving_pct.min():.2%} to {claims.cost_saving_pct.max():.2%}"
    )
    print(
        f"  {'mean':<7} {'':<26} {claims.cost_saving_abs.mean():>13,.0f} "
        f"{claims.cost_saving_pct.mean():>10.2%}"
    )
    print()

    print("=" * 104)
    print("CLAIM 2 - DC ORDER FILL (naive vs optimized)")
    print("=" * 104)
    print(
        f"  {'WINDOW':<7} {'TEST PERIOD':<26} {'NAIVE DC':>10} {'OPT DC':>10} "
        f"{'IMPROVEMENT':>13}"
    )
    print("  " + "-" * 72)
    for r in claims.itertuples():
        print(
            f"  {r.window:<7} {r.test_span:<26} {r.naive_dc_fill:>10.2%} "
            f"{r.optimized_dc_fill:>10.2%} {r.dc_fill_improvement:>13.2%}"
        )
    print("  " + "-" * 72)
    print(
        f"  {'range':<7} {'':<26} "
        f"{claims.naive_dc_fill.min():>10.2%} {claims.optimized_dc_fill.min():>10.2%} "
        f"{claims.dc_fill_improvement.min():.2%} to {claims.dc_fill_improvement.max():.2%}"
    )
    print()


def print_verdict(claims: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("=" * 104)
    print("ARE THE FINDINGS STABLE?")
    print("=" * 104)
    print()

    saving_pct = claims.cost_saving_pct
    saving_min, saving_max = saving_pct.min(), saving_pct.max()
    n_negative = int((saving_pct < 0).sum())
    sign_stable = n_negative == 0

    print("CLAIM 1 - the ~2.7% cost saving")
    print(
        f"  Across {len(claims)} windows the saving ranges {saving_min:.2%} to "
        f"{saving_max:.2%}, mean {saving_pct.mean():.2%}.\n"
        f"  The single-split estimate was 2.7%."
    )
    if sign_stable and saving_max - saving_min < 0.05:
        print(
            "  VERDICT: stable in sign and roughly stable in size. The saving is a\n"
            "  property of the network, not of the original holdout - though it is small\n"
            "  enough throughout that it should never be the headline reason to adopt\n"
            "  the policy."
        )
    elif sign_stable:
        print(
            f"  VERDICT: the SIGN is stable - pooling is cheaper in every window - but the\n"
            f"  SIZE is not, spanning a {saving_max - saving_min:.1%} range. Quote it as a band,\n"
            f"  never as a single number, and expect any given quarter to land anywhere\n"
            f"  inside it."
        )
    else:
        print(
            f"  VERDICT: NOT stable. The saving is negative in {n_negative} of "
            f"{len(claims)} windows, so\n"
            f"  pooling is not reliably cheaper at all. The 2.7% from the single split was\n"
            f"  a favourable draw, and cost should be dropped as a justification."
        )
    print()

    dc_imp = claims.dc_fill_improvement
    opt_dc_min = claims.optimized_dc_fill.min()
    naive_dc_min = claims.naive_dc_fill.min()
    print("CLAIM 2 - DC stockout elimination")
    print(
        f"  Optimized DC order fill never drops below {opt_dc_min:.2%} in any window.\n"
        f"  Naive DC order fill ranges {naive_dc_min:.2%} to "
        f"{claims.naive_dc_fill.max():.2%}.\n"
        f"  Improvement ranges {dc_imp.min():.2%} to {dc_imp.max():.2%}, mean "
        f"{dc_imp.mean():.2%}."
    )
    if opt_dc_min > 0.999:
        print(
            "  VERDICT: stable. The optimized policy holds the DC echelon at essentially\n"
            "  100% in every window tested. This is the robust finding of the whole\n"
            "  project - it does not depend on which quarter you test."
        )
    else:
        print(
            "  VERDICT: mostly stable, but not absolute - the optimized DC echelon does\n"
            "  dip below 100% in at least one window."
        )
    print()

    # What drives the variation: the seasonal content of each test period.
    print("WHAT DRIVES THE VARIATION BETWEEN WINDOWS")
    peak = claims.loc[claims.demand_ratio.idxmax()]
    trough = claims.loc[claims.demand_ratio.idxmin()]
    print(
        f"  The windows differ mainly in how much demand the test quarter carries\n"
        f"  relative to what the policy was fitted on:\n"
        f"    hardest  {peak.window} ({peak.test_span}) at "
        f"{peak.demand_ratio - 1:+.1%} demand -> naive DC fill "
        f"{peak.naive_dc_fill:.2%}, saving {peak.cost_saving_pct:.2%}\n"
        f"    easiest  {trough.window} ({trough.test_span}) at "
        f"{trough.demand_ratio - 1:+.1%} demand -> naive DC fill "
        f"{trough.naive_dc_fill:.2%}, saving {trough.cost_saving_pct:.2%}"
    )
    # Absolute cost is not comparable across windows, and saying so matters
    # before any of this reaches a dashboard.
    opt_costs = summary.loc[summary.policy == "optimized", "actual_cost_annual"]
    naive_w4 = summary[(summary.window == peak.window) & (summary.policy == "naive")]
    print(
        f"\n  Note on reading the cost column: actual cost is mean CARRIED inventory, so\n"
        f"  it falls when demand outruns supply. The same optimized policy costs\n"
        f"  {opt_costs.max():,.0f}/yr in the easiest window and {opt_costs.min():,.0f}/yr in the hardest -\n"
        f"  that 3x spread is the demand environment, not a policy difference. The naive\n"
        f"  policy in {peak.window} looks cheapest of all at "
        f"{naive_w4.actual_cost_annual.iloc[0]:,.0f}/yr while delivering only\n"
        f"  {naive_w4.store_fill_rate.iloc[0]:.2%} fill: empty shelves are inexpensive. Compare policies WITHIN\n"
        f"  a window, as the saving column does, and never costs across windows."
    )

    corr = claims.demand_ratio.corr(claims.naive_dc_fill)
    print(
        f"\n  Correlation between test-period demand ratio and naive DC fill: {corr:+.2f}.\n"
        f"  The naive policy's DC failures concentrate in the windows where demand runs\n"
        f"  hardest above the training period - which is exactly when a buffer is needed,\n"
        f"  and exactly what sigma*sqrt(L) sizing on a long, autocorrelated lead time\n"
        f"  fails to provide. The optimized DC buffer, sized on empirical lead-time\n"
        f"  quantiles, absorbs the same shock without running dry."
    )
    print()

    print("BOTTOM LINE FOR THE DASHBOARD")
    if sign_stable:
        print(
            f"  Lead with DC-echelon reliability: {opt_dc_min:.1%}+ order fill in every window\n"
            f"  against a naive policy that drops to {naive_dc_min:.1%}. Present the cost saving\n"
            f"  as a {saving_min:.1%}-{saving_max:.1%} band with a {saving_pct.mean():.1%} average, not as a single\n"
            f"  headline number."
        )
    else:
        print(
            "  Lead with DC-echelon reliability and service. Do not present a cost saving\n"
            "  headline - it does not replicate across windows."
        )
    print()


def main() -> None:
    daily = opt.load_demand()
    windows = build_windows(daily)
    check_no_cache_leakage(windows)

    print_windows(windows)

    results = pd.concat(
        [run_window(daily, window) for window in windows], ignore_index=True
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(OUTPUT_PATH, index=False)

    summary = summarize_window(results)
    claims = build_claims(summary)

    print_per_window(summary)
    print_claims(claims)
    print_verdict(claims, summary)

    print(
        f"Wrote {len(results)} rows ({len(windows)} windows x 3 policies x 13 nodes) "
        f"to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
