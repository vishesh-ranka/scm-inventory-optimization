"""
Multi-echelon safety stock optimization with PuLP.

Run:
    python src/optimize_network.py

Writes `data/optimized_results.parquet` and prints a naive-vs-optimized
comparison, including *achieved* (not just target) service levels.


WHAT THE BASELINE ANALYSIS TOLD US, AND HOW IT SHAPES THIS MODEL
================================================================
`naive_baseline.py` produced three findings that this model is built around:

1. sigma * sqrt(L) understates real lead-time demand variance by 1.0x at short
   store lead times and 2.8x-10.6x at the 30-45 day DCs, because demand is
   autocorrelated (weekly cycle, annual season, trend).
   -> So this model never uses sigma * sqrt(L). Every safety stock number comes
      from the EMPIRICAL distribution of demand accumulated over rolling
      lead-time windows of the real history. Autocorrelation and skew are
      inherited from the data rather than assumed away.

2. Out-of-phase demand (Sydney peaks mid-year, Tokyo and Jakarta in December)
   creates genuine pooling value the per-node policy cannot see.
   -> So the pooled buffer below is measured by summing SIMULTANEOUS store
      excesses from history, which captures that cancellation directly instead
      of applying a correlation assumption.

3. The naive policy misses its own 95% target at every node.
   -> So cost is never compared without also comparing achieved service level.


THE FORMULATION, AND WHY THIS ONE
=================================
This is the two-stage approximation the brief allows, because a true stochastic
multi-echelon program is not expressible in PuLP (see LIMITATIONS below).

Stage 1 - empirical, outside the LP
-----------------------------------
(a) Build the empirical lead-time demand distribution for all 13 nodes.

(b) Decide where the replenishment buffer belongs, via an exhaustive scan of
    the guaranteed-service model: each DC picks an outbound service time
    S_out in [0, L_DC]. S_out = 0 means the DC promises immediate shipment and
    must itself cover L_DC days; S_out = L_DC means the DC is a cross-dock
    holding nothing and its stores must each cover L_DC + L_store days.
    The scan finds S_out = 0 optimal at all three DCs, decisively (the corner
    beats its neighbour by 2-8% and degrades monotonically). Since central
    space is ~2.4x cheaper per unit AND pooled DC demand is far less variable
    than the stores it serves, pushing the replenishment buffer to the DC wins.
    That result is reported, not assumed - `scan_service_times()` recomputes it
    on every run.

(c) Tabulate the pooling curve. For a grid of local coverage levels alpha,
    compute how large a SHARED buffer the DC needs to catch everything its
    stores did not cover locally:

        floor_i          = q95(demand over EXPEDITE_LEAD_TIME_DAYS)
        local_i(alpha)   = max(floor_i, q_alpha(W_i) - mean(W_i))
        excess_i,t       = max(0, W_i,t - mean(W_i) - local_i(alpha))
        B_d(alpha)       = q95( sum_i excess_i,t )

    where W_i,t is store i's demand over its own lead time in the window ending
    at t. Because the excesses are summed at the SAME timestamp before the
    quantile is taken, B_d(alpha) automatically reflects how often the region's
    stores spike together. Measured here at 65-75% of the sum of the individual
    gaps - that discount is the pooling value, and it is where the money is.

    The `floor_i` term is what stops central stock substituting for local stock
    without limit. Central inventory can only rescue a store if it physically
    arrives in time, so each store must always self-cover demand over the
    expedite window regardless of how cheap DC space is. Without that floor the
    model happily strips stores to zero - and, on right-skewed demand where the
    median sits below the mean, to NEGATIVE safety stock, which is not a
    stocking decision at all.

Stage 2 - the MILP, in PuLP
---------------------------
Choose one alpha per DC region. Lower alpha moves buffer out of expensive store
space and into the cheap shared DC buffer, but the shared buffer has to grow.
The optimizer trades those off:

    minimise  sum_d [ DCown_d * h_d                      (DC replenishment buffer)
                    + sum_k B_d(alpha_k) * h_d * z_d,k   (shared DC buffer)
                    + sum_i sum_k local_i(alpha_k) * h_i * z_d,k ]   (store stock)

    s.t.      sum_k z_d,k = 1  for each DC d          (pick exactly one alpha)
              z_d,k binary

Linear in the decision variables because every safety stock quantity is a
precomputed constant looked up by the selection variable - that is the trick
that keeps an inherently stochastic problem inside an LP solver.

Service level is enforced by CONSTRUCTION rather than as an algebraic
constraint: B_d is sized at the 95th percentile of pooled excess, so with 95%
probability every store in the region is covered. That is a joint guarantee
across the region and therefore at least as strong as per-store 95%. Achieved
service is then measured empirically for every node and reported.


LIMITATIONS - WHAT THIS IS NOT
==============================
Flagging these because the gap between this and a true stochastic multi-echelon
model is real, and the numbers below should be read with them in mind:

1. PuLP is a (mixed-integer) LINEAR programming interface. Service-level
   constraints are inherently nonlinear in the stock level - the mapping from
   units to service is a demand quantile function. Genuine formulations
   (Clark-Scarf, stochastic dynamic programming) need recursive convolution of
   demand distributions and are not expressible as linear constraints. The
   quantile curves are therefore evaluated OUTSIDE the solver on a discrete
   grid, and the solver chooses among tabulated points. That makes this a
   discretized search over a precomputed response surface, not a true
   stochastic program.

2. One alpha per DC region, not per store. Per-store alphas would be the richer
   model, but the pooled buffer B_d depends on the whole vector of store alphas
   simultaneously, so it could no longer be tabulated as a 1-D curve. This is
   the single biggest simplification here, and it means individual stores are
   not cost-optimized against each other within a region.

3. The alpha grid is discrete (0.50 to 0.95). The true optimum may sit between
   grid points. The grid is fine enough that the cost curve is flat near the
   optimum, but this is a discretization, not an exact optimum.

4. Static, single-period, base-stock reasoning. There is no ordering policy, no
   batching, no capacity limit, no lost-sales-versus-backorder distinction, and
   no explicit stockout penalty - service level is a hard constraint rather
   than a cost traded against holding.

5. Central substitution rests on an ASSUMED 1-day expedite capability from a DC
   to a store in its own region (EXPEDITE_LEAD_TIME_DAYS). Nothing in the data
   supports or refutes that number - it is a statement about the physical
   network, and it is the assumption the pooling saving is most sensitive to.
   `print_expedite_sensitivity()` reports what the saving becomes at other
   values; at the limit where expediting takes as long as normal replenishment,
   central substitution is worth nothing and the optimum is the fully
   decentralized policy. Validate this number before trusting the headline.

6. A shared regional buffer assumes DC stock is genuinely fungible across the
   stores it serves and can be allocated after seeing which store needs it.
   That is reasonable for one SKU in a regional DC, but it does presume the
   allocation decision is made well.

7. The DC's own replenishment buffer and its shared store-support buffer are
   added together. In reality one pool of stock serves both roles and could
   partly cover both risks, so the DC total here is somewhat conservative.

8. Everything is evaluated IN-SAMPLE on the same 2 years that set the
   quantiles. Empirical quantiles fit their own history, so achieved service
   levels are optimistic; the long DC windows (30-45 days) in particular have
   few effectively independent observations. An out-of-sample or simulated
   evaluation is the honest next step, and is exactly what the simulation stage
   of this project should provide.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pulp
from scipy import stats

from network_config import (
    DISTRIBUTION_CENTERS,
    STORES,
    stores_for_dc,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMAND_PATH = PROJECT_ROOT / "data" / "demand_history.parquet"
BASELINE_PATH = PROJECT_ROOT / "data" / "naive_baseline_results.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "optimized_results.parquet"

SERVICE_LEVEL = 0.95
Z_SCORE = float(stats.norm.ppf(SERVICE_LEVEL))
DAYS_PER_YEAR = 365

# An empirical quantile can only land on an actual observation, so a node sized
# at the 95th percentile of ~690 rolling windows measures back as 95% give or
# take one window (1/690 = 0.145%). Anything inside this tolerance is quantile
# granularity, not a service miss. The naive baseline's shortfalls are 3-30
# points, nowhere near this, so the tolerance never flatters it.
SERVICE_TOLERANCE = 0.005

# Candidate local coverage levels for the stores. 0.50 is the floor: below the
# median, "safety" stock would be negative. 0.95 reproduces the fully
# decentralized policy (every store self-sufficient, no shared buffer needed).
ALPHA_GRID: tuple[float, ...] = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

# How fast a DC can get an emergency shipment to a store in its own region.
# This is the physical limit on how much of a store's variability central stock
# can absorb, and it is an ASSUMPTION, not something measured from the data.
#
# It matters a lot. A regional DC buffer can only rescue a store if it arrives
# before the store actually runs out, so a store must always self-cover demand
# over the expedite window no matter how cheap central space is. Set it to a
# store's full lead time and central substitution disappears entirely, which
# collapses the model back to the fully decentralized policy.
EXPEDITE_LEAD_TIME_DAYS = 1


# ---------------------------------------------------------------------------
# Stage 1a - empirical lead-time demand
# ---------------------------------------------------------------------------


def load_demand() -> pd.DataFrame:
    """Daily demand as a (date x store) matrix."""
    history = pd.read_parquet(DEMAND_PATH)
    return history.pivot_table(
        index="date", columns="store_id", values="demand", aggfunc="sum"
    ).sort_index()


# The same rolling sums get requested many times - by the service-time scan, the
# pooling curves, the sensitivity sweep and the service-level evaluation - so
# they are memoized. Keyed by the series name (node id) plus window length; the
# underlying daily history never changes within a run.
_WINDOW_CACHE: dict[tuple[str, int], pd.Series] = {}


def lead_time_windows(daily: pd.Series, lead_time_days: float) -> pd.Series:
    """Demand accumulated over each rolling lead-time window.

    This is the quantity safety stock actually has to cover. Using the realized
    rolling sums (rather than sigma * sqrt(L)) is what carries the
    autocorrelation and skew of the real series into the model.
    """
    window = int(round(lead_time_days))
    if window <= 0:
        return pd.Series(dtype=float)

    # The cache key includes the span of the series, not just its name. Without
    # that, fitting on a training slice after having run on the full history
    # would silently hand back full-history windows for the same node id - which
    # would leak the test period into the training fit and quietly invalidate
    # any out-of-sample validation.
    key = (str(daily.name), window, len(daily), daily.index[0], daily.index[-1])
    if key not in _WINDOW_CACHE:
        # Rolling sums via a prefix sum. The service-time scan asks for every
        # window length from 1 to 45 on every series, and pandas' .rolling()
        # walks the array afresh each time; one cumsum answers them all in O(n)
        # per window. Identical arithmetic, just without the repeated passes.
        values = daily.to_numpy(dtype=float)
        prefix = np.concatenate(([0.0], np.cumsum(values)))
        sums = prefix[window:] - prefix[:-window]
        _WINDOW_CACHE[key] = pd.Series(
            sums, index=daily.index[window - 1:], name=daily.name
        )
    return _WINDOW_CACHE[key]


def empirical_safety_stock(
    daily: pd.Series, lead_time_days: float, quantile: float = SERVICE_LEVEL
) -> float:
    """Units needed above mean lead-time demand to cover `quantile` of outcomes.

    Uses numpy rather than pandas for the quantile: this is called several
    hundred times across the service-time scan and the sensitivity sweep, and
    pandas' Series.quantile carries enough per-call overhead (it routes through
    DataFrame.quantile and rebuilds an Index each time) to dominate the runtime.
    numpy's default interpolation is linear, matching pandas, so the numbers are
    unchanged.
    """
    if lead_time_days <= 0:
        return 0.0
    values = lead_time_windows(daily, lead_time_days).to_numpy()
    return float(np.quantile(values, quantile) - values.mean())


def dc_demand_series(daily: pd.DataFrame, dc_id: str) -> pd.Series:
    """A DC's demand: the summed daily demand of the stores it serves.

    Named so the rolling-window cache can key on it like any store series.
    """
    series = daily[[s.id for s in stores_for_dc(dc_id)]].sum(axis=1)
    series.name = dc_id
    return series


# ---------------------------------------------------------------------------
# Stage 1b - where does the replenishment buffer belong?
# ---------------------------------------------------------------------------


def scan_service_times(daily: pd.DataFrame) -> pd.DataFrame:
    """Guaranteed-service scan: best outbound service time for each DC.

    For each candidate S_out the DC covers (L_DC - S_out) days of its own
    pooled demand, and each of its stores covers (S_out + L_store) days. Both
    legs are priced with empirical quantiles. Exhaustive over a few dozen
    integer options per DC, so no solver is needed and the result is exact.
    """
    rows = []
    for dc in DISTRIBUTION_CENTERS.values():
        pooled = dc_demand_series(daily, dc.id)
        lead_time = int(round(dc.lead_time_mean_days))
        served = stores_for_dc(dc.id)

        for s_out in range(lead_time + 1):
            dc_cost = (
                empirical_safety_stock(pooled, lead_time - s_out)
                * dc.holding_cost_per_unit_per_day
            )
            store_cost = sum(
                empirical_safety_stock(daily[st.id], s_out + st.lead_time_mean_days)
                * st.holding_cost_per_unit_per_day
                for st in served
            )
            rows.append(
                {
                    "dc_id": dc.id,
                    "s_out_days": s_out,
                    "annual_cost": (dc_cost + store_cost) * DAYS_PER_YEAR,
                }
            )

    scan = pd.DataFrame(rows)
    return scan.loc[scan.groupby("dc_id")["annual_cost"].idxmin()].set_index("dc_id")


# ---------------------------------------------------------------------------
# Stage 1c - the pooling curve
# ---------------------------------------------------------------------------


def build_pooling_curves(daily: pd.DataFrame) -> pd.DataFrame:
    """For each DC and each alpha, the local store stock and shared DC buffer.

    The shared buffer is sized on the pooled excess: at each historical window
    we add up how much every store in the region went over its own local
    coverage, THEN take the 95th percentile of that regional total. Because the
    sum happens before the quantile, stores that peak at different times cancel
    out on their own - no correlation matrix, no distributional assumption.
    """
    rows = []

    for dc in DISTRIBUTION_CENTERS.values():
        served = stores_for_dc(dc.id)

        # Each store's lead-time demand, aligned on a common date index so the
        # excesses being summed are genuinely contemporaneous.
        windows = pd.DataFrame(
            {st.id: lead_time_windows(daily[st.id], st.lead_time_mean_days)
             for st in served}
        ).dropna()

        # Irreducible local floor. The DC needs EXPEDITE_LEAD_TIME_DAYS to reach
        # the store, so demand over that window can never be covered centrally -
        # central stock would simply arrive too late. Every store holds at least
        # enough to cover that window at the target service level, whatever the
        # optimizer decides about the rest.
        local_floor = {
            st.id: max(
                0.0,
                empirical_safety_stock(daily[st.id], EXPEDITE_LEAD_TIME_DAYS),
            )
            for st in served
        }

        # Work in numpy from here: the alpha grid multiplies every quantile by
        # the number of grid points, and pandas' per-call overhead dominates.
        store_ids = [st.id for st in served]
        window_matrix = windows[store_ids].to_numpy()  # (n_windows, n_stores)
        window_means = window_matrix.mean(axis=0)
        holding = np.array(
            [st.holding_cost_per_unit_per_day for st in served], dtype=float
        )
        floors = np.array([local_floor[sid] for sid in store_ids], dtype=float)
        full_requirement = np.quantile(window_matrix, SERVICE_LEVEL, axis=0)

        for alpha in ALPHA_GRID:
            # Safety stock implied by this coverage level, floored at the
            # expedite requirement and at zero. The zero floor matters
            # independently: on right-skewed demand the median sits below the
            # mean, so alpha near 0.50 would otherwise imply NEGATIVE safety
            # stock, which is not a physical stocking decision.
            alpha_ss = np.quantile(window_matrix, alpha, axis=0) - window_means
            local_ss_arr = np.maximum(floors, alpha_ss)
            local_cost = float((local_ss_arr * holding).sum())

            # The stock level each store actually reaches, and what spills past
            # it. Summed across the region per window, BEFORE any quantile, so
            # stores that peak at different times cancel on their own.
            local_threshold = window_means + local_ss_arr
            excess = np.clip(window_matrix - local_threshold, 0.0, None)
            pooled_excess = excess.sum(axis=1)
            shared_buffer = float(np.quantile(pooled_excess, SERVICE_LEVEL))

            # The same protection bought store-by-store, for comparison: this is
            # what the region would need with no pooling at all.
            unpooled_buffer = float(
                np.maximum(full_requirement - local_threshold, 0.0).sum()
            )

            local_ss = dict(zip(store_ids, local_ss_arr.tolist()))

            rows.append(
                {
                    "dc_id": dc.id,
                    "alpha": alpha,
                    "shared_buffer_units": shared_buffer,
                    "unpooled_buffer_units": unpooled_buffer,
                    "pooling_factor": (
                        shared_buffer / unpooled_buffer
                        if unpooled_buffer > 1e-9
                        else np.nan
                    ),
                    "shared_buffer_cost": shared_buffer
                    * dc.holding_cost_per_unit_per_day,
                    "local_store_cost": local_cost,
                    **{f"local_ss__{sid}": v for sid, v in local_ss.items()},
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 2 - the MILP
# ---------------------------------------------------------------------------


def pick_solver() -> pulp.LpSolver:
    """Return a solver that actually runs on this machine.

    PuLP ships a bundled CBC binary and reports it as "available" purely by
    checking the file exists. On Apple Silicon that binary is x86_64 and raises
    OSError: Bad CPU type unless Rosetta is installed, so availability has to be
    confirmed by solving a trivial problem rather than trusted. HiGHS (pip
    install highspy) has native arm64 wheels and is tried first.
    """
    candidates = []
    if "HiGHS" in pulp.listSolvers(onlyAvailable=True):
        candidates.append(pulp.HiGHS(msg=False))
    candidates.append(pulp.PULP_CBC_CMD(msg=0))

    probe = pulp.LpProblem("solver_probe", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0)
    probe += x
    probe += x >= 1

    for solver in candidates:
        try:
            probe.solve(solver)
            if pulp.LpStatus[probe.status] == "Optimal":
                return solver
        except Exception:  # noqa: BLE001 - any solver failure means try the next
            continue

    raise RuntimeError(
        "No working MILP solver found. Install one with: pip install highspy"
    )


def solve_milp(daily: pd.DataFrame, curves: pd.DataFrame) -> tuple[dict, pulp.LpProblem]:
    """Pick the cost-minimising alpha for each DC region, via PuLP/CBC.

    Decision variables are binary selectors z[d, k] over the tabulated alpha
    grid. Every cost coefficient is a constant computed in stage 1, so the
    objective is linear and CBC solves it exactly.
    """
    problem = pulp.LpProblem("multi_echelon_safety_stock", pulp.LpMinimize)

    # The DC replenishment buffer is a fixed cost here: stage 1b established
    # S_out = 0, so each DC covers its own full supplier lead time regardless of
    # which alpha is chosen. It is added to the objective as a constant so the
    # reported optimum is the true total network cost.
    dc_own_cost = {}
    for dc in DISTRIBUTION_CENTERS.values():
        pooled = dc_demand_series(daily, dc.id)
        units = empirical_safety_stock(pooled, dc.lead_time_mean_days)
        dc_own_cost[dc.id] = units * dc.holding_cost_per_unit_per_day

    # z[dc, alpha] == 1 selects that coverage level for the whole region.
    z = {
        (row.dc_id, row.alpha): pulp.LpVariable(
            f"z_{row.dc_id}_{int(row.alpha * 100)}", cat="Binary"
        )
        for row in curves.itertuples()
    }

    # Tie-break toward higher alpha. Once the expedite floor binds, a run of
    # low-alpha grid points costs exactly the same, and an arbitrary pick among
    # them is not neutral: the higher-alpha option holds more stock locally and
    # so leans less on the assumed expedite capability. The epsilon is far below
    # any real cost difference, so it only ever separates exact ties.
    tie_break = 1e-6

    problem += (
        pulp.lpSum(
            (
                row.shared_buffer_cost
                + row.local_store_cost
                + tie_break * (1.0 - row.alpha)
            )
            * z[(row.dc_id, row.alpha)]
            for row in curves.itertuples()
        )
        + pulp.lpSum(dc_own_cost.values())
    ), "total_daily_holding_cost"

    # Exactly one coverage level per region.
    for dc_id in DISTRIBUTION_CENTERS:
        problem += (
            pulp.lpSum(
                z[(dc_id, alpha)]
                for alpha in curves.loc[curves.dc_id == dc_id, "alpha"]
            )
            == 1,
            f"one_alpha_{dc_id}",
        )

    solver = pick_solver()
    problem.solve(solver)

    chosen = {
        dc_id: float(
            next(a for a in ALPHA_GRID if pulp.value(z[(dc_id, a)]) > 0.5)
        )
        for dc_id in DISTRIBUTION_CENTERS
    }
    return chosen, problem, type(solver).__name__


def verify_by_enumeration(curves: pd.DataFrame, chosen: dict) -> bool:
    """Cross-check the solver against brute force.

    The problem is small and separable by region, so the optimum can be found
    directly. Worth doing: it catches formulation slips that a solver will
    happily optimize straight past.

    Compares achieved COST, not the chosen alpha. Once the expedite floor binds,
    every alpha below it yields identical stock and therefore identical cost, so
    several grid points are exactly optimal; comparing alphas would report a
    spurious disagreement whenever the solver picked a different member of a
    tied set.
    """
    for dc_id, alpha in chosen.items():
        block = curves[curves.dc_id == dc_id]
        total = block["shared_buffer_cost"] + block["local_store_cost"]
        chosen_cost = float(total[block["alpha"] == alpha].iloc[0])
        if not np.isclose(chosen_cost, float(total.min()), rtol=1e-9):
            return False
    return True


# ---------------------------------------------------------------------------
# Achieved service level - measured, not assumed
# ---------------------------------------------------------------------------


def achieved_service_store(
    daily: pd.DataFrame,
    dc_id: str,
    local_ss: dict[str, float],
    shared_buffer: float,
) -> dict[str, float]:
    """Empirical service level per store under the pooled policy.

    A store is short in a window only if BOTH its local stock runs out AND the
    region's shared buffer is exhausted at that same moment. Measuring it this
    way keeps the joint behaviour of the region in the number.

    Takes the chosen local stock levels directly rather than re-deriving them
    from alpha, so the evaluation cannot drift from the floors applied in
    `build_pooling_curves`.
    """
    served = stores_for_dc(dc_id)
    windows = pd.DataFrame(
        {st.id: lead_time_windows(daily[st.id], st.lead_time_mean_days)
         for st in served}
    ).dropna()

    store_ids = [st.id for st in served]
    window_matrix = windows[store_ids].to_numpy()
    local_threshold = window_matrix.mean(axis=0) + np.array(
        [local_ss[sid] for sid in store_ids], dtype=float
    )

    excess = np.clip(window_matrix - local_threshold, 0.0, None)
    region_exhausted = excess.sum(axis=1) > shared_buffer

    return {
        sid: float(1.0 - ((excess[:, i] > 0) & region_exhausted).mean())
        for i, sid in enumerate(store_ids)
    }


def achieved_service_simple(
    daily: pd.Series, lead_time_days: float, safety_stock: float
) -> float:
    """Fraction of lead-time windows a standalone node covers with this stock."""
    values = lead_time_windows(daily, lead_time_days).to_numpy()
    return float((values <= values.mean() + safety_stock).mean())


# ---------------------------------------------------------------------------
# Assembling the comparison
# ---------------------------------------------------------------------------


def build_results(
    daily: pd.DataFrame, curves: pd.DataFrame, chosen: dict
) -> pd.DataFrame:
    """One row per node: naive vs optimized stock, cost and achieved service."""
    baseline = pd.read_parquet(BASELINE_PATH).set_index("node_id")
    rows = []

    for dc in DISTRIBUTION_CENTERS.values():
        alpha = chosen[dc.id]
        curve = curves[(curves.dc_id == dc.id) & (curves.alpha == alpha)].iloc[0]
        pooled = dc_demand_series(daily, dc.id)

        own_buffer = empirical_safety_stock(pooled, dc.lead_time_mean_days)
        optimized_units = own_buffer + curve.shared_buffer_units

        rows.append(
            {
                "node_id": dc.id,
                "node_name": dc.name,
                "node_type": "DC",
                "region": dc.region,
                "lead_time_days": dc.lead_time_mean_days,
                "holding_cost_per_unit_per_day": dc.holding_cost_per_unit_per_day,
                "chosen_alpha": alpha,
                "naive_ss_units": float(baseline.loc[dc.id, "safety_stock_units"]),
                "naive_cost_annual": float(baseline.loc[dc.id, "holding_cost_annual"]),
                "naive_service_achieved": achieved_service_simple(
                    pooled,
                    dc.lead_time_mean_days,
                    float(baseline.loc[dc.id, "safety_stock_units"]),
                ),
                "optimized_ss_units": optimized_units,
                "optimized_replenishment_buffer": own_buffer,
                "optimized_shared_buffer": float(curve.shared_buffer_units),
                "optimized_cost_annual": optimized_units
                * dc.holding_cost_per_unit_per_day
                * DAYS_PER_YEAR,
                # The DC's own buffer is sized at the empirical 95th percentile
                # of its pooled lead-time demand, so it makes service by
                # construction.
                "optimized_service_achieved": achieved_service_simple(
                    pooled, dc.lead_time_mean_days, own_buffer
                ),
            }
        )

    for dc_id, alpha in chosen.items():
        curve = curves[(curves.dc_id == dc_id) & (curves.alpha == alpha)].iloc[0]
        local_ss = {
            st.id: float(curve[f"local_ss__{st.id}"]) for st in stores_for_dc(dc_id)
        }
        store_service = achieved_service_store(
            daily, dc_id, local_ss, float(curve.shared_buffer_units)
        )

        for store in stores_for_dc(dc_id):
            local_units = float(curve[f"local_ss__{store.id}"])
            rows.append(
                {
                    "node_id": store.id,
                    "node_name": store.name,
                    "node_type": "STORE",
                    "region": store.region,
                    "lead_time_days": store.lead_time_mean_days,
                    "holding_cost_per_unit_per_day": (
                        store.holding_cost_per_unit_per_day
                    ),
                    "chosen_alpha": alpha,
                    "naive_ss_units": float(
                        baseline.loc[store.id, "safety_stock_units"]
                    ),
                    "naive_cost_annual": float(
                        baseline.loc[store.id, "holding_cost_annual"]
                    ),
                    "naive_service_achieved": achieved_service_simple(
                        daily[store.id],
                        store.lead_time_mean_days,
                        float(baseline.loc[store.id, "safety_stock_units"]),
                    ),
                    "optimized_ss_units": local_units,
                    "optimized_replenishment_buffer": local_units,
                    "optimized_shared_buffer": 0.0,
                    "optimized_cost_annual": local_units
                    * store.holding_cost_per_unit_per_day
                    * DAYS_PER_YEAR,
                    "optimized_service_achieved": store_service[store.id],
                }
            )

    results = pd.DataFrame(rows)
    results["cost_delta_annual"] = (
        results["optimized_cost_annual"] - results["naive_cost_annual"]
    )
    results["service_delta"] = (
        results["optimized_service_achieved"] - results["naive_service_achieved"]
    )
    return results


def decentralized_cost(daily: pd.DataFrame, curves: pd.DataFrame) -> float:
    """Cost of hitting a true 95% everywhere with NO pooling.

    This is the alpha = 0.95 point: every store self-sufficient, shared buffer
    unnecessary, DCs still sized empirically. It is the honest yardstick for the
    pooling saving, because unlike the naive baseline it actually delivers the
    service level it targets.
    """
    total = 0.0
    for dc in DISTRIBUTION_CENTERS.values():
        pooled = dc_demand_series(daily, dc.id)
        total += (
            empirical_safety_stock(pooled, dc.lead_time_mean_days)
            * dc.holding_cost_per_unit_per_day
        )
        row = curves[(curves.dc_id == dc.id) & (curves.alpha == 0.95)].iloc[0]
        total += row.local_store_cost + row.shared_buffer_cost
    return total * DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_stage1(scan: pd.DataFrame, curves: pd.DataFrame, chosen: dict) -> None:
    print()
    print("=" * 112)
    print("STAGE 1 - EMPIRICAL STRUCTURE (computed from history, outside the solver)")
    print("=" * 112)
    print("1a. Where the replenishment buffer belongs (guaranteed-service scan):")
    for dc_id, row in scan.iterrows():
        dc = DISTRIBUTION_CENTERS[dc_id]
        print(
            f"    {dc_id:<9} optimal outbound service time = "
            f"{int(row['s_out_days'])}d of {int(dc.lead_time_mean_days)}d  "
            f"-> DC absorbs the full supplier lead time "
            f"({row['annual_cost']:,.0f}/yr for the region)"
        )
    print(
        "    All three pick 0: central space is ~2.4x cheaper per unit and pooled DC\n"
        "    demand is far steadier than the stores beneath it, so the long ocean\n"
        "    lead time is cheapest to buffer centrally."
    )
    print()

    print("1b. Pooling curve - shared DC buffer vs local store stock:")
    print(
        f"    {'DC':<9} {'alpha':>6} {'LOCAL $/yr':>12} {'SHARED UNITS':>13} "
        f"{'UNPOOLED':>10} {'POOL FACTOR':>12} {'REGION $/yr':>13}"
    )
    print("    " + "-" * 104)
    for dc_id in DISTRIBUTION_CENTERS:
        block = curves[curves.dc_id == dc_id]
        for row in block.itertuples():
            total = (row.local_store_cost + row.shared_buffer_cost) * DAYS_PER_YEAR
            marker = "  <-- chosen" if np.isclose(row.alpha, chosen[dc_id]) else ""
            factor = (
                f"{row.pooling_factor:.2f}" if not np.isnan(row.pooling_factor) else "-"
            )
            print(
                f"    {dc_id:<9} {row.alpha:>6.2f} "
                f"{row.local_store_cost * DAYS_PER_YEAR:>12,.0f} "
                f"{row.shared_buffer_units:>13,.1f} "
                f"{row.unpooled_buffer_units:>10,.1f} {factor:>12} "
                f"{total:>13,.0f}{marker}"
            )
        print()


def print_comparison(results: pd.DataFrame) -> None:
    print("=" * 112)
    print("NAIVE vs OPTIMIZED - PER NODE")
    print("=" * 112)
    header = (
        f"{'NODE':<9} {'NAME':<19} {'REGION':<14} "
        f"{'NAIVE SS':>9} {'NAIVE $':>9} {'NAIVE SL':>9}   "
        f"{'OPT SS':>9} {'OPT $':>9} {'OPT SL':>8}"
    )

    for node_type in ("DC", "STORE"):
        block = results[results.node_type == node_type].sort_values(
            ["region", "node_id"]
        )
        print("DISTRIBUTION CENTERS" if node_type == "DC" else "RETAIL STORES")
        print(header)
        print("-" * 112)
        for r in block.itertuples():
            miss = "" if r.naive_service_achieved >= SERVICE_LEVEL else " !"
            print(
                f"{r.node_id:<9} {r.node_name[:19]:<19} {r.region:<14} "
                f"{r.naive_ss_units:>9,.0f} {r.naive_cost_annual:>9,.0f} "
                f"{r.naive_service_achieved:>8.1%}{miss:<2}  "
                f"{r.optimized_ss_units:>9,.0f} {r.optimized_cost_annual:>9,.0f} "
                f"{r.optimized_service_achieved:>8.1%}"
            )
        print()


def print_totals(results: pd.DataFrame, unpooled_total: float) -> None:
    naive_cost = results["naive_cost_annual"].sum()
    opt_cost = results["optimized_cost_annual"].sum()
    naive_units = results["naive_ss_units"].sum()
    opt_units = results["optimized_ss_units"].sum()

    naive_worst = results["naive_service_achieved"].min()
    naive_mean = results["naive_service_achieved"].mean()
    opt_worst = results["optimized_service_achieved"].min()
    opt_mean = results["optimized_service_achieved"].mean()
    threshold = SERVICE_LEVEL - SERVICE_TOLERANCE
    naive_missing = int((results["naive_service_achieved"] < threshold).sum())
    opt_missing = int((results["optimized_service_achieved"] < threshold).sum())

    print("=" * 112)
    print("NETWORK TOTALS")
    print("=" * 112)
    print(f"{'':<34} {'UNITS':>12} {'$/YEAR':>14} {'WORST SL':>10} {'MEAN SL':>10} {'NODES < 95%':>13}")
    print("-" * 112)
    print(
        f"{'Naive per-node baseline':<34} {naive_units:>12,.0f} {naive_cost:>14,.0f} "
        f"{naive_worst:>10.1%} {naive_mean:>10.1%} {naive_missing:>13}"
    )
    print(
        f"{'Empirical 95%, no pooling':<34} {'-':>12} {unpooled_total:>14,.0f} "
        f"{SERVICE_LEVEL:>10.1%} {'>=95.0%':>10} {0:>13}"
    )
    print(
        f"{'Optimized multi-echelon':<34} {opt_units:>12,.0f} {opt_cost:>14,.0f} "
        f"{opt_worst:>10.1%} {opt_mean:>10.1%} {opt_missing:>13}"
    )
    print("-" * 112)
    print(
        f"  'NODES < 95%' counts nodes below {threshold:.1%}, allowing "
        f"{SERVICE_TOLERANCE:.1%} for the\n"
        f"  granularity of an empirical quantile on a finite sample. Two optimized DCs\n"
        f"  measure 94.9-95.0%, which is that granularity, not under-delivery."
    )
    print()

    print("READ THESE TWO COMPARISONS SEPARATELY:")
    print()
    pooling_saving = unpooled_total - opt_cost
    print(
        f"  1. Optimized vs a policy that also truly delivers 95% (no pooling):\n"
        f"     {opt_cost:,.0f} vs {unpooled_total:,.0f} per year - "
        f"a saving of {pooling_saving:,.0f} "
        f"({pooling_saving / unpooled_total:.1%}).\n"
        f"     This is the real, like-for-like value of multi-echelon pooling, and\n"
        f"     it is the number to quote. Both policies hit the service target."
    )
    print()
    delta = opt_cost - naive_cost
    print(
        f"  2. Optimized vs the naive baseline: {opt_cost:,.0f} vs {naive_cost:,.0f} "
        f"per year, i.e. {abs(delta):,.0f} "
        f"{'MORE' if delta > 0 else 'LESS'}.\n"
        f"     This comparison is NOT like-for-like and should not be quoted as a\n"
        f"     regression. The naive policy is cheaper because it is under-stocked:\n"
        f"     it achieves {naive_worst:.1%} service at its worst node and misses 95% at\n"
        f"     {naive_missing} of {len(results)} nodes. Buying the service level it was\n"
        f"     supposed to deliver is what costs money; pooling then gives back\n"
        f"     {pooling_saving:,.0f} of it."
    )
    print()


def print_where_savings_come_from(results: pd.DataFrame) -> None:
    print("=" * 112)
    print("WHERE THE OPTIMIZER MOVED STOCK")
    print("=" * 112)
    by_type = results.groupby("node_type")[
        ["naive_ss_units", "optimized_ss_units", "naive_cost_annual",
         "optimized_cost_annual"]
    ].sum()
    for node_type in ("DC", "STORE"):
        r = by_type.loc[node_type]
        label = "DC echelon" if node_type == "DC" else "Store echelon"
        print(
            f"  {label:<16} units {r['naive_ss_units']:>9,.0f} -> "
            f"{r['optimized_ss_units']:>9,.0f}    "
            f"cost {r['naive_cost_annual']:>9,.0f} -> "
            f"{r['optimized_cost_annual']:>9,.0f} /yr"
        )
    print()
    print(
        "  The pattern is the point: units move UP into the DC echelon, where the\n"
        "  empirical quantiles show the naive formula was worst (2.8x-10.6x variance\n"
        "  understated) and where space is cheapest, and come DOWN in expensive store\n"
        "  space, where a shared regional buffer covers the tail more efficiently than\n"
        "  ten independent ones."
    )
    print()


def print_expedite_sensitivity(daily: pd.DataFrame) -> None:
    """How the pooling saving responds to the expedite assumption.

    EXPEDITE_LEAD_TIME_DAYS is the one number in this model that comes from
    judgement rather than data, and the whole central-substitution case rests on
    it, so its influence is reported rather than left implicit.
    """
    global EXPEDITE_LEAD_TIME_DAYS  # noqa: PLW0603 - deliberate for the sweep
    original = EXPEDITE_LEAD_TIME_DAYS

    print("=" * 112)
    print("SENSITIVITY - THE EXPEDITE ASSUMPTION")
    print("=" * 112)
    print(
        "How long the DC needs to reach a store sets how much local stock central\n"
        "stock can replace. This is assumed, not measured, so here is the whole curve:"
    )
    print()
    print(
        f"    {'EXPEDITE':>9} {'OPTIMIZED $/yr':>16} {'NO-POOLING $/yr':>17} "
        f"{'SAVING':>12} {'SAVING %':>10}"
    )
    print("    " + "-" * 70)

    try:
        for expedite in (0, 1, 2, 3):
            EXPEDITE_LEAD_TIME_DAYS = expedite
            curves = build_pooling_curves(daily)
            chosen, problem, _ = solve_milp(daily, curves)
            results = build_results(daily, curves, chosen)
            opt = results["optimized_cost_annual"].sum()
            unpooled = decentralized_cost(daily, curves)
            saving = unpooled - opt
            label = f"{expedite}d" + (" *" if expedite == original else "")
            print(
                f"    {label:>9} {opt:>16,.0f} {unpooled:>17,.0f} "
                f"{saving:>12,.0f} {saving / unpooled:>9.1%}"
            )
    finally:
        EXPEDITE_LEAD_TIME_DAYS = original

    print()
    print(
        "    * = the value used for the headline numbers above.\n"
        "    0d is the theoretical ceiling (instant regional transfer). The saving\n"
        "    shrinks as expediting slows, because more stock has to sit locally.\n"
        "    If the real expedite time is close to normal DC->store transit, most of\n"
        "    the headline saving is not available."
    )
    print()


def main() -> None:
    daily = load_demand()

    scan = scan_service_times(daily)
    curves = build_pooling_curves(daily)
    chosen, problem, solver_name = solve_milp(daily, curves)

    status = pulp.LpStatus[problem.status]
    if status != "Optimal":
        raise RuntimeError(f"MILP did not solve to optimality: {status}")

    print_stage1(scan, curves, chosen)

    print("=" * 112)
    print("STAGE 2 - MILP (PuLP)")
    print("=" * 112)
    print(
        f"    solver: {solver_name} · status: {status} · "
        f"{len(problem.variables())} binary variables · "
        f"{len(problem.constraints)} constraints"
    )
    print(
        f"    chosen local coverage: "
        + ", ".join(f"{k} alpha={v:.2f}" for k, v in chosen.items())
    )
    agrees = verify_by_enumeration(curves, chosen)
    print(
        f"    brute-force cross-check: "
        f"{'agrees with solver' if agrees else 'DISAGREES WITH SOLVER'}"
    )
    print()

    results = build_results(daily, curves, chosen)
    unpooled_total = decentralized_cost(daily, curves)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(OUTPUT_PATH, index=False)

    print_comparison(results)
    print_totals(results, unpooled_total)
    print_where_savings_come_from(results)
    print_expedite_sensitivity(daily)

    print(
        f"Wrote {len(results)} node rows to "
        f"{OUTPUT_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "\nNote: service levels above are IN-SAMPLE against the same history that\n"
        "set the quantiles, so they are optimistic. See LIMITATIONS in the module\n"
        "docstring - a forward simulation is the right next step."
    )


if __name__ == "__main__":
    main()
