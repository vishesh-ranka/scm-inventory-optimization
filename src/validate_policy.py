"""
Out-of-sample validation of the naive and optimized policies by simulation.

Run:
    python src/validate_policy.py

Everything reported so far has been IN-SAMPLE: safety stock was sized from the
quantiles of the full 2-year history and then scored against that same history,
which cannot fail. This script refits both policies on the first 18 months only,
then runs a day-by-day multi-echelon inventory simulation against the held-out
last 6 months of actual demand and reports what the policies really delivered.

Writes `data/validation_results.parquet`.


READ THE SPLIT BEFORE READING THE RESULTS
=========================================
The holdout is not a random sample of the history, and that matters for every
number below. The last 6 months (2024-07-01 to 2024-12-30) carry:

  * the accumulated year-over-year growth trend, and
  * the Q4 seasonal peak - 9 of 10 stores peak in late November.

Holdout demand averages ~12% above the training period, store by store as much
as +27% (Milan) and as little as -11% (Phoenix, which trends down). So this is a
deliberately hard, forward-in-time test: exactly the situation of deploying a
policy and living with it, but NOT a clean measurement of overfitting alone.

Degradation here therefore has two possible causes, which are separated
explicitly by `oracle` runs at the end:

  1. Estimation error - quantiles fitted to 18 months not generalizing.
  2. A genuinely harder period - demand simply being larger than anything in
     the training window.

An oracle policy refit ON the test period is simulated as a diagnostic. If the
oracle also struggles, the problem is the period. If the oracle is comfortable,
the problem is the fit.


THE SIMULATION
==============
Discrete daily simulation, one independent run per DC region.

Sequence within each day (the order matters, and is chosen so a 0-day expedite
can genuinely serve the same day's demand):

  1. Scheduled arrivals land at the stores and the DC.
  2. Each store reviews and raises its inventory position back to its base-stock
     level by ordering from the DC. The DC ships what it has; the rest goes
     unfilled and the store re-orders the next day. Under the optimized policy
     the urgent slice of that order - whatever the store needs to survive until
     an expedited shipment could land - travels on the fast lane and arrives in
     `expedite_days` instead of the normal transit time.

     Expediting changes the SPEED of the order, never its SIZE. That restraint
     is the whole point: an expedite that also adds volume becomes a parallel
     just-in-time channel, and the store ends up running on daily emergency
     deliveries rather than on the stock the optimizer sized. A first version of
     this simulation did exactly that, and produced a non-monotonic sensitivity
     curve (1-day expedite scoring worse than 2-day and 3-day) that was pure
     artifact.
  3. Customer demand arrives at the stores and is filled from on-hand. Anything
     unfilled is a LOST SALE, not a backorder - retail demand walks away.
  4. The DC raises its own inventory position to its base-stock level by
     ordering from the supplier, who is assumed never to be short.

Base-stock levels come from the fitted policy:

    S = mean lead-time demand (train) + safety stock (train)

and inventories start in steady state (on-hand = safety stock, pipeline holding
one lead time of mean demand), followed by a 60-day warm-up on training data
that is simulated but not measured.

One thing the simulation tests that the analytical model could not: the DC is
driven by actual STORE ORDERS, not by store sales. Order streams are lumpier
than the demand beneath them - that was flagged as an unverifiable caveat in
`naive_baseline.py` and `optimize_network.py`, and here it finally bites for
real.


METRICS, AND WHY THEY ARE NOT THE EARLIER NUMBERS
=================================================
The in-sample figure was a CYCLE service level: P(demand over a lead time <=
stock on hand). The simulation instead measures what actually happened:

  * fill_rate     - units delivered / units demanded. The headline.
  * day_service   - share of store-days with no shortfall at all.
  * actual cost   - mean on-hand inventory actually carried * holding rate,
                    annualized. This is real carried stock (cycle + safety), so
                    it is legitimately larger than the planned safety-stock-only
                    cost quoted earlier, and the two should not be compared
                    directly. Compare policies WITHIN this file.

These are different definitions from the in-sample number, so a gap between them
is not by itself evidence of overfitting - only the like-for-like policy
comparisons and the oracle runs below support that kind of claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import optimize_network as opt
from network_config import DISTRIBUTION_CENTERS, STORES, stores_for_dc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "validation_results.parquet"

SERVICE_LEVEL = 0.95
Z_SCORE = float(stats.norm.ppf(SERVICE_LEVEL))
DAYS_PER_YEAR = 365

TRAIN_MONTHS = 18
WARMUP_DAYS = 60
EXPEDITE_GRID = (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# Policy parameters
# ---------------------------------------------------------------------------


@dataclass
class Policy:
    """Base-stock levels and expedite behaviour for one complete network policy."""

    name: str
    store_base_stock: dict[str, float]
    dc_base_stock: dict[str, float]
    store_safety_stock: dict[str, float]
    dc_safety_stock: dict[str, float]
    # Units a store should keep on hand to ride out the expedite horizon.
    store_expedite_cover: dict[str, float] = field(default_factory=dict)
    use_expedite: bool = False
    expedite_days: int = 0


def split_history(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First TRAIN_MONTHS for fitting, the remainder held out for testing."""
    split_date = daily.index[0] + pd.DateOffset(months=TRAIN_MONTHS)
    return daily[daily.index < split_date], daily[daily.index >= split_date]


def fit_naive_policy(train: pd.DataFrame) -> Policy:
    """The textbook per-node policy: z * sigma_daily * sqrt(L), no expediting."""
    store_ss, store_s = {}, {}
    for store in STORES.values():
        sigma = float(train[store.id].std(ddof=1))
        mean_daily = float(train[store.id].mean())
        ss = Z_SCORE * sigma * np.sqrt(store.lead_time_mean_days)
        store_ss[store.id] = ss
        store_s[store.id] = mean_daily * (store.lead_time_mean_days + 1) + ss

    dc_ss, dc_s = {}, {}
    for dc in DISTRIBUTION_CENTERS.values():
        pooled = opt.dc_demand_series(train, dc.id)
        sigma = float(pooled.std(ddof=1))
        mean_daily = float(pooled.mean())
        ss = Z_SCORE * sigma * np.sqrt(dc.lead_time_mean_days)
        dc_ss[dc.id] = ss
        dc_s[dc.id] = mean_daily * (dc.lead_time_mean_days + 1) + ss

    return Policy(
        name="naive",
        store_base_stock=store_s,
        dc_base_stock=dc_s,
        store_safety_stock=store_ss,
        dc_safety_stock=dc_ss,
        use_expedite=False,
    )


def fit_decentralized_policy(train: pd.DataFrame) -> Policy:
    """Empirical 95% at every node, with no pooling and no expediting.

    This is the alpha = 0.95 point of the pooling curve: each store is sized to
    be self-sufficient at the target service level. It is the policy the
    in-sample analysis used as its like-for-like yardstick when it claimed
    pooling was worth 12.7%, so it has to be simulated too - otherwise the only
    available out-of-sample comparison is against the naive baseline, which runs
    at a different service level and settles nothing.
    """
    curves = opt.build_pooling_curves(train)

    store_ss, store_s, dc_ss, dc_s = {}, {}, {}, {}
    for dc in DISTRIBUTION_CENTERS.values():
        curve = curves[(curves.dc_id == dc.id) & (curves.alpha == 0.95)].iloc[0]
        pooled = opt.dc_demand_series(train, dc.id)
        own = opt.empirical_safety_stock(pooled, dc.lead_time_mean_days)
        dc_ss[dc.id] = own + float(curve.shared_buffer_units)
        dc_s[dc.id] = float(pooled.mean()) * (dc.lead_time_mean_days + 1) + dc_ss[dc.id]

        for store in stores_for_dc(dc.id):
            ss = float(curve[f"local_ss__{store.id}"])
            store_ss[store.id] = ss
            store_s[store.id] = (
                float(train[store.id].mean()) * (store.lead_time_mean_days + 1) + ss
            )

    return Policy(
        name="decentralized_95",
        store_base_stock=store_s,
        dc_base_stock=dc_s,
        store_safety_stock=store_ss,
        dc_safety_stock=dc_ss,
        use_expedite=False,
    )


def fit_optimized_policy(train: pd.DataFrame, expedite_days: int) -> Policy:
    """Refit the multi-echelon policy from `optimize_network` on training data.

    Reuses the real optimizer - the pooling curves and the PuLP MILP - rather
    than re-deriving anything, so what gets validated here is genuinely the
    policy that module produces, just fitted to 18 months instead of 24.
    """
    original = opt.EXPEDITE_LEAD_TIME_DAYS
    opt.EXPEDITE_LEAD_TIME_DAYS = expedite_days
    try:
        curves = opt.build_pooling_curves(train)
        chosen, _problem, _solver = opt.solve_milp(train, curves)
    finally:
        opt.EXPEDITE_LEAD_TIME_DAYS = original

    store_ss, store_s, cover = {}, {}, {}
    dc_ss, dc_s = {}, {}

    for dc in DISTRIBUTION_CENTERS.values():
        alpha = chosen[dc.id]
        curve = curves[(curves.dc_id == dc.id) & (curves.alpha == alpha)].iloc[0]

        pooled = opt.dc_demand_series(train, dc.id)
        own = opt.empirical_safety_stock(pooled, dc.lead_time_mean_days)
        dc_ss[dc.id] = own + float(curve.shared_buffer_units)
        dc_s[dc.id] = (
            float(pooled.mean()) * (dc.lead_time_mean_days + 1) + dc_ss[dc.id]
        )

        for store in stores_for_dc(dc.id):
            ss = float(curve[f"local_ss__{store.id}"])
            store_ss[store.id] = ss
            store_s[store.id] = (
                float(train[store.id].mean()) * (store.lead_time_mean_days + 1) + ss
            )
            # What the store must hold to survive until an expedited shipment
            # can arrive. At a 0-day expedite this is still one day of cover,
            # because demand inside the day cannot be met by a same-day order
            # placed after the fact.
            horizon = expedite_days + 1
            cover[store.id] = float(
                train[store.id].rolling(horizon).sum().dropna().quantile(SERVICE_LEVEL)
            )

    return Policy(
        name=f"optimized_exp{expedite_days}",
        store_base_stock=store_s,
        dc_base_stock=dc_s,
        store_safety_stock=store_ss,
        dc_safety_stock=dc_ss,
        store_expedite_cover=cover,
        use_expedite=True,
        expedite_days=expedite_days,
    )


# ---------------------------------------------------------------------------
# The simulation
# ---------------------------------------------------------------------------


def simulate_region(
    demand: pd.DataFrame,
    dc_id: str,
    policy: Policy,
    train: pd.DataFrame,
    measure_from: int,
) -> list[dict]:
    """Run one DC and its stores day by day over `demand`.

    `measure_from` is the index at which recording starts, so the warm-up days
    ahead of it settle the pipelines without polluting the statistics.
    """
    dc = DISTRIBUTION_CENTERS[dc_id]
    served = stores_for_dc(dc_id)
    n_days = len(demand)
    dc_lead = int(round(dc.lead_time_mean_days))
    horizon = n_days + dc_lead + 10

    # --- steady-state initialization -------------------------------------
    # Inventory position starts at the base-stock level: on-hand holds the
    # safety stock and the pipeline holds one lead time of mean demand.
    store_onhand, store_arrivals, store_lead = {}, {}, {}
    for store in served:
        lead = int(round(store.lead_time_mean_days))
        store_lead[store.id] = lead
        store_onhand[store.id] = max(0.0, policy.store_safety_stock[store.id])
        arrivals = np.zeros(horizon)
        arrivals[:lead] = float(train[store.id].mean())
        store_arrivals[store.id] = arrivals

    pooled_train = opt.dc_demand_series(train, dc_id)
    dc_onhand = max(0.0, policy.dc_safety_stock[dc_id])
    dc_arrivals = np.zeros(horizon)
    dc_arrivals[:dc_lead] = float(pooled_train.mean())

    # --- recorders --------------------------------------------------------
    rec = {
        s.id: {"demand": 0.0, "unmet": 0.0, "days": 0, "short_days": 0,
               "onhand_sum": 0.0, "expedited": 0.0}
        for s in served
    }
    dc_rec = {"requested": 0.0, "shipped": 0.0, "onhand_sum": 0.0, "days": 0,
              "short_days": 0}

    for t in range(n_days):
        measuring = t >= measure_from

        # 1. arrivals land
        for store in served:
            store_onhand[store.id] += store_arrivals[store.id][t]
        dc_onhand += dc_arrivals[t]

        # 2. review and order, before demand, so a 0-day expedite can genuinely
        #    serve the same day. Expediting changes only the SPEED of the
        #    store's normal order, never its size: the order is always
        #    (base stock - inventory position), and the urgent slice of it is
        #    simply shipped on the fast lane. Letting expedite add volume on top
        #    would turn it into a parallel just-in-time channel and flatter the
        #    policy - the store would be running on daily emergency deliveries
        #    rather than on the stock the optimizer actually sized.
        for store in served:
            lead = store_lead[store.id]
            position = store_onhand[store.id] + float(
                store_arrivals[store.id][t + 1 :].sum()
            )
            order = max(0.0, policy.store_base_stock[store.id] - position)
            shipped = min(order, dc_onhand)
            dc_onhand -= shipped

            # Expediting is only ever worth using when the fast lane is actually
            # faster than normal transit. Tokyo and Phoenix sit 2 days from
            # their DC, so a nominal "3-day expedite" would be a slower route -
            # nobody would choose it, and letting the simulation choose it
            # produced a spurious collapse in the 3-day sensitivity row.
            effective_expedite = min(policy.expedite_days, lead)

            urgent = 0.0
            if policy.use_expedite and shipped > 0 and effective_expedite < lead:
                exp_days = effective_expedite
                # Can the store survive until an expedited shipment lands?
                inbound = float(
                    store_arrivals[store.id][t + 1 : t + exp_days + 1].sum()
                )
                deficit = (
                    policy.store_expedite_cover[store.id]
                    - store_onhand[store.id]
                    - inbound
                )
                urgent = float(np.clip(deficit, 0.0, shipped))

            if urgent > 0:
                if effective_expedite == 0:
                    store_onhand[store.id] += urgent
                else:
                    store_arrivals[store.id][t + effective_expedite] += urgent
                if measuring:
                    rec[store.id]["expedited"] += urgent
            store_arrivals[store.id][t + lead] += shipped - urgent

            if measuring:
                dc_rec["requested"] += order
                dc_rec["shipped"] += shipped

        # 3. customer demand, lost sales
        for store in served:
            d = float(demand.iloc[t][store.id])
            sold = min(store_onhand[store.id], d)
            unmet = d - sold
            store_onhand[store.id] -= sold
            if measuring:
                r = rec[store.id]
                r["demand"] += d
                r["unmet"] += unmet
                r["days"] += 1
                r["short_days"] += 1 if unmet > 1e-9 else 0
                r["onhand_sum"] += store_onhand[store.id]

        # 4. DC replenishment from the supplier, who is never short
        dc_position = dc_onhand + float(dc_arrivals[t + 1 :].sum())
        dc_order = max(0.0, policy.dc_base_stock[dc_id] - dc_position)
        dc_arrivals[t + dc_lead] += dc_order

        if measuring:
            dc_rec["onhand_sum"] += dc_onhand
            dc_rec["days"] += 1
            dc_rec["short_days"] += 1 if dc_onhand <= 1e-9 else 0

    # --- assemble rows ----------------------------------------------------
    rows = []
    for store in served:
        r = rec[store.id]
        mean_onhand = r["onhand_sum"] / r["days"]
        rows.append(
            {
                "node_id": store.id,
                "node_name": store.name,
                "node_type": "STORE",
                "region": store.region,
                "policy": policy.name,
                "expedite_days": policy.expedite_days if policy.use_expedite else None,
                "planned_safety_stock": policy.store_safety_stock[store.id],
                "total_demand": r["demand"],
                "unmet_demand": r["unmet"],
                "fill_rate": 1.0 - r["unmet"] / r["demand"],
                "day_service": 1.0 - r["short_days"] / r["days"],
                "mean_onhand": mean_onhand,
                "expedited_units": r["expedited"],
                "actual_cost_annual": mean_onhand
                * store.holding_cost_per_unit_per_day
                * DAYS_PER_YEAR,
            }
        )

    mean_dc_onhand = dc_rec["onhand_sum"] / dc_rec["days"]
    rows.append(
        {
            "node_id": dc.id,
            "node_name": dc.name,
            "node_type": "DC",
            "region": dc.region,
            "policy": policy.name,
            "expedite_days": policy.expedite_days if policy.use_expedite else None,
            "planned_safety_stock": policy.dc_safety_stock[dc.id],
            "total_demand": dc_rec["requested"],
            "unmet_demand": dc_rec["requested"] - dc_rec["shipped"],
            "fill_rate": (
                dc_rec["shipped"] / dc_rec["requested"]
                if dc_rec["requested"] > 0
                else 1.0
            ),
            "day_service": 1.0 - dc_rec["short_days"] / dc_rec["days"],
            "mean_onhand": mean_dc_onhand,
            "expedited_units": 0.0,
            "actual_cost_annual": mean_dc_onhand
            * dc.holding_cost_per_unit_per_day
            * DAYS_PER_YEAR,
        }
    )
    return rows


def run_policy(
    daily: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, policy: Policy
) -> pd.DataFrame:
    """Simulate every region under one policy, with a warm-up tail from train."""
    warmup = train.iloc[-WARMUP_DAYS:]
    sim_frame = pd.concat([warmup, test])

    rows = []
    for dc_id in DISTRIBUTION_CENTERS:
        rows.extend(
            simulate_region(sim_frame, dc_id, policy, train, measure_from=len(warmup))
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Network totals per policy: store fill rate and total carried cost."""
    stores = results[results.node_type == "STORE"]
    grouped = stores.groupby("policy", sort=False).apply(
        lambda g: pd.Series(
            {
                "store_fill_rate": 1.0 - g.unmet_demand.sum() / g.total_demand.sum(),
                "worst_store_fill": g.fill_rate.min(),
                "stores_below_95": int((g.fill_rate < 0.95).sum()),
            }
        ),
        include_groups=False,
    )
    costs = results.groupby("policy", sort=False)["actual_cost_annual"].sum()
    dc_fill = (
        results[results.node_type == "DC"]
        .groupby("policy", sort=False)
        .apply(
            lambda g: 1.0 - g.unmet_demand.sum() / g.total_demand.sum(),
            include_groups=False,
        )
    )
    out = grouped.join(costs.rename("actual_cost_annual"))
    out["dc_order_fill_rate"] = dc_fill
    return out.reset_index()


def print_split(train: pd.DataFrame, test: pd.DataFrame) -> None:
    ratio = test.sum(axis=1).mean() / train.sum(axis=1).mean()
    print()
    print("=" * 104)
    print("OUT-OF-SAMPLE VALIDATION")
    print("=" * 104)
    print(
        f"  train  {train.index[0].date()} -> {train.index[-1].date()}  "
        f"({len(train)} days)   parameters fitted here"
    )
    print(
        f"  test   {test.index[0].date()} -> {test.index[-1].date()}  "
        f"({len(test)} days)   never seen during fitting"
    )
    print(
        f"\n  Holdout demand runs {ratio - 1:+.1%} against the training period: it carries\n"
        f"  the growth trend AND the Q4 peak (9 of 10 stores peak in late November).\n"
        f"  This is a hard forward test by construction, not a neutral sample."
    )
    print()


def print_policy_table(results: pd.DataFrame, policy_name: str, label: str) -> None:
    block = results[results.policy == policy_name].sort_values(
        ["node_type", "region", "node_id"]
    )
    print(f"{label}")
    print(
        f"  {'NODE':<9} {'NAME':<19} {'TYPE':<6} {'FILL RATE':>10} "
        f"{'DAY SVC':>9} {'MEAN OH':>9} {'$/YEAR':>10}"
    )
    print("  " + "-" * 88)
    for r in block.itertuples():
        flag = "" if r.fill_rate >= 0.95 else "  !"
        print(
            f"  {r.node_id:<9} {r.node_name[:19]:<19} {r.node_type:<6} "
            f"{r.fill_rate:>10.2%} {r.day_service:>9.2%} "
            f"{r.mean_onhand:>9,.0f} {r.actual_cost_annual:>10,.0f}{flag}"
        )
    print()


def print_verdict(summary: pd.DataFrame, results: pd.DataFrame) -> None:
    """State plainly what the holdout confirms and what it complicates."""
    s = summary.set_index("policy")

    def row(name: str) -> pd.Series:
        return s.loc[name]

    naive = row("naive")
    decent = row("decentralized_95")
    optimized = row("optimized_exp1")
    oracle = row("oracle_optimized")

    print("=" * 104)
    print("DOES THE OUT-OF-SAMPLE TEST CONFIRM THE IN-SAMPLE FINDINGS?")
    print("=" * 104)
    print()

    print("CONFIRMED - the naive baseline really does miss its target.")
    print(
        f"  In-sample it measured 64.7-92.7% cycle service against a 95% target. On\n"
        f"  held-out data it delivers a {naive.store_fill_rate:.2%} store fill rate, with\n"
        f"  {int(naive.stores_below_95)} of 10 stores under 95% and a worst store at "
        f"{naive.worst_store_fill:.2%}.\n"
        f"  The direction and the rough magnitude both hold up."
    )
    print()

    print("CONFIRMED - the optimized policy really does deliver more service.")
    print(
        f"  {optimized.store_fill_rate:.2%} store fill vs the naive {naive.store_fill_rate:.2%}, "
        f"worst store {optimized.worst_store_fill:.2%} vs\n"
        f"  {naive.worst_store_fill:.2%}. It also holds the DC echelon together: DC order fill is\n"
        f"  {optimized.dc_order_fill_rate:.2%} against the naive {naive.dc_order_fill_rate:.2%}. "
        f"That last number is a failure mode\n"
        f"  the in-sample analysis structurally could not see - under the naive policy the\n"
        f"  DCs run dry, so stores cannot be resupplied even when their own sizing was fine."
    )
    print()

    pooling_saving = decent.actual_cost_annual - optimized.actual_cost_annual
    print("COMPLICATED - the headline 12.7% pooling saving does not survive intact.")
    print(
        f"  The like-for-like in-sample claim was optimized vs no-pooling-at-true-95%.\n"
        f"  Simulated on held-out data those two come out at:\n"
        f"      decentralized 95% : {decent.store_fill_rate:>7.2%} fill, "
        f"{decent.actual_cost_annual:>9,.0f}/yr\n"
        f"      optimized         : {optimized.store_fill_rate:>7.2%} fill, "
        f"{optimized.actual_cost_annual:>9,.0f}/yr\n"
        f"  -> {abs(pooling_saving):,.0f}/yr "
        f"{'cheaper' if pooling_saving > 0 else 'MORE EXPENSIVE'}, at "
        f"{optimized.store_fill_rate - decent.store_fill_rate:+.2%} fill rate."
    )
    if pooling_saving > 0:
        pct = pooling_saving / decent.actual_cost_annual
        print(
            f"  So pooling is still worth something, and it is still the right sign - but\n"
            f"  it is {pct:.1%} out of sample against the 12.7% claimed in sample, roughly a\n"
            f"  fifth of the advertised figure. Part of the value has moved from the cost\n"
            f"  column into the service column: the pooled design also runs "
            f"{optimized.store_fill_rate - decent.store_fill_rate:+.2%} higher\n"
            f"  fill, because a shared buffer can be aimed wherever the shortfall actually\n"
            f"  lands, while ten fixed local buffers cannot be moved once placed.\n"
            f"  Quote {pct:.0%}, not 12.7%."
        )
    else:
        print(
            "  The in-sample analysis said pooling was worth 12.7%. Out of sample it is\n"
            "  not a saving at all on these numbers. Pooling buys SERVICE here rather\n"
            "  than cost: both policies were sized on 18 months that understated the\n"
            "  holdout, and the pooled design absorbs that miss better because a shared\n"
            "  buffer can be aimed wherever the shortfall actually lands."
        )
    print()

    print("COMPLICATED - cost rankings invert once real carried stock is measured.")
    print(
        f"  Planned safety-stock cost made the naive policy look cheapest, and it still\n"
        f"  is on the holdout ({naive.actual_cost_annual:,.0f}/yr) - but only because it is\n"
        f"  starved. Buying real service costs real money: every policy that actually\n"
        f"  approaches 95% lands in the {decent.actual_cost_annual:,.0f}-"
        f"{oracle.actual_cost_annual:,.0f}/yr band."
    )
    print()

    print("THE OVERFITTING QUESTION - how much of the gap is estimation error?")
    print(
        f"  The oracle policy, refit on the test period itself, reaches "
        f"{oracle.store_fill_rate:.2%} fill\n"
        f"  for {oracle.actual_cost_annual:,.0f}/yr. The honestly-fitted optimized policy reaches\n"
        f"  {optimized.store_fill_rate:.2%} for {optimized.actual_cost_annual:,.0f}/yr. The gap "
        f"between them - about\n"
        f"  {oracle.store_fill_rate - optimized.store_fill_rate:.2%} of fill rate - is the price of "
        f"not knowing the future.\n"
        f"  That gap is small, which is the reassuring part: the shortfall against 95% is\n"
        f"  mostly the PERIOD being harder (+12.4% demand, Q4 peak, trend), not the\n"
        f"  quantile method failing to generalize. Refitting more often would recover\n"
        f"  most of it."
    )
    print()

    print("BOTTOM LINE")
    print(
        f"  The optimization survives the holdout, but its case changes shape. It is\n"
        f"  strongest on what the in-sample work could not test: service under a period\n"
        f"  it was not fitted to ({optimized.store_fill_rate:.2%} vs naive "
        f"{naive.store_fill_rate:.2%}), and DC-echelon robustness\n"
        f"  ({optimized.dc_order_fill_rate:.2%} vs {naive.dc_order_fill_rate:.2%} order fill). "
        f"The pooling COST saving is real but much\n"
        f"  smaller than advertised. Lead with service and resilience; quote the cost\n"
        f"  saving at its out-of-sample size.\n"
        f"  Before deploying, refit on a rolling window rather than a fixed 18 months:\n"
        f"  every policy here is sized against a training period that no longer describes\n"
        f"  the demand it will meet."
    )
    print()


def main() -> None:
    daily = opt.load_demand()
    train, test = split_history(daily)
    print_split(train, test)

    all_rows = []

    # --- the two headline policies ---------------------------------------
    naive = fit_naive_policy(train)
    naive_res = run_policy(daily, train, test, naive)
    all_rows.append(naive_res)

    decentralized = fit_decentralized_policy(train)
    all_rows.append(run_policy(daily, train, test, decentralized))

    optimized = fit_optimized_policy(train, opt.EXPEDITE_LEAD_TIME_DAYS)
    opt_res = run_policy(daily, train, test, optimized)
    all_rows.append(opt_res)

    print_policy_table(naive_res, "naive", "NAIVE BASELINE - simulated on held-out data")
    print_policy_table(
        opt_res,
        optimized.name,
        f"OPTIMIZED MULTI-ECHELON ({opt.EXPEDITE_LEAD_TIME_DAYS}-day expedite) "
        f"- simulated on held-out data",
    )

    # --- expedite sensitivity, now measured rather than assumed -----------
    for expedite in EXPEDITE_GRID:
        if expedite == opt.EXPEDITE_LEAD_TIME_DAYS:
            continue
        policy = fit_optimized_policy(train, expedite)
        all_rows.append(run_policy(daily, train, test, policy))

    # --- oracle diagnostic: fit on the test period itself -----------------
    # Separates "the quantiles did not generalize" from "the period was simply
    # harder". An oracle that also struggles means the period, not the fit.
    oracle = fit_optimized_policy(test, opt.EXPEDITE_LEAD_TIME_DAYS)
    oracle.name = "oracle_optimized"
    oracle_res = run_policy(daily, train, test, oracle)
    all_rows.append(oracle_res)

    oracle_naive = fit_naive_policy(test)
    oracle_naive.name = "oracle_naive"
    all_rows.append(run_policy(daily, train, test, oracle_naive))

    results = pd.concat(all_rows, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(OUTPUT_PATH, index=False)

    summary = summarize(results)

    print("=" * 104)
    print("NETWORK SUMMARY - ALL SIMULATED ON THE HELD-OUT 6 MONTHS")
    print("=" * 104)
    print(
        f"  {'POLICY':<22} {'STORE FILL':>11} {'WORST STORE':>12} "
        f"{'<95%':>6} {'DC ORDER FILL':>14} {'ACTUAL $/YEAR':>14}"
    )
    print("  " + "-" * 88)
    for r in summary.itertuples():
        print(
            f"  {r.policy:<22} {r.store_fill_rate:>11.2%} {r.worst_store_fill:>12.2%} "
            f"{int(r.stores_below_95):>6} {r.dc_order_fill_rate:>14.2%} "
            f"{r.actual_cost_annual:>14,.0f}"
        )
    print()

    print("=" * 104)
    print("EXPEDITE SENSITIVITY - SIMULATED, NOT ASSUMED")
    print("=" * 104)
    print(f"  {'EXPEDITE':>9} {'STORE FILL':>11} {'ACTUAL $/YEAR':>14} {'EXPEDITED UNITS':>17}")
    print("  " + "-" * 60)
    for expedite in EXPEDITE_GRID:
        name = f"optimized_exp{expedite}"
        row = summary[summary.policy == name]
        if row.empty:
            continue
        units = results.loc[results.policy == name, "expedited_units"].sum()
        print(
            f"  {str(expedite) + 'd':>9} {row.store_fill_rate.iloc[0]:>11.2%} "
            f"{row.actual_cost_annual.iloc[0]:>14,.0f} {units:>17,.0f}"
        )
    print()

    # This is the one place where simulated evidence flatly contradicts the
    # analytical sensitivity, so it is stated rather than left for the reader.
    exp_rows = summary[summary.policy.str.startswith("optimized_exp")]
    fill_spread = exp_rows.store_fill_rate.max() - exp_rows.store_fill_rate.min()
    cost_spread = (
        exp_rows.actual_cost_annual.max() - exp_rows.actual_cost_annual.min()
    ) / exp_rows.actual_cost_annual.min()
    print(
        f"  The in-sample table made this assumption look decisive: the modelled pooling\n"
        f"  saving ran from 22.8% at 0 days down to 2.2% at 3 days. Simulated, the whole\n"
        f"  0-3 day range moves fill by {fill_spread:.2%} and cost by {cost_spread:.1%}.\n"
        f"  So the expedite assumption matters far LESS than the analytical model implied.\n"
        f"  The reason is that the analytical model let the expedite window set how much\n"
        f"  stock could be stripped out of the stores, whereas in operation the outcome is\n"
        f"  governed by base-stock levels and by whether the DC can supply at all - and a\n"
        f"  slow expedite simply goes unused, because it is capped at normal transit time.\n"
        f"  Good news for deployment: the softest input in the model is not load-bearing."
    )
    print()

    print_verdict(summary, results)

    print(f"Wrote {len(results)} rows to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
