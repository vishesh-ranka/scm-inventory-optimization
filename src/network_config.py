"""
Network structure for the multi-echelon inventory optimization project.

Three echelons:

    supplier (overseas)  ->  3 regional DCs  ->  10 retail stores

Everything about the physical network lives here as plain data (dataclasses +
module-level constants). No logic beyond a few small lookup helpers, so that
downstream modules (demand generation, policy optimization, simulation) all
read the same single source of truth.

Units / conventions
-------------------
* Lead times are in DAYS.
* Demand is in UNITS per day (a single generic SKU for now).
* Holding costs are in CURRENCY per unit per DAY.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Echelon 0: the overseas supplier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Supplier:
    """Single overseas source of supply for the whole network."""

    id: str
    name: str
    # Assumed effectively unlimited upstream capacity for now; the interesting
    # constraint in this project is the long ocean lead time downstream.
    daily_capacity: float = float("inf")


SUPPLIER = Supplier(
    id="SUP1",
    name="Overseas Supplier (Asia manufacturing hub)",
)


# ---------------------------------------------------------------------------
# Echelon 1: regional distribution centers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistributionCenter:
    """A regional DC that buys from the supplier and serves local stores.

    Attributes
    ----------
    lead_time_mean_days / lead_time_std_days:
        Supplier -> DC replenishment lead time. Long and variable because it
        bundles ocean freight, port congestion and customs clearance. Means sit
        in the 30-45 day band; std dev captures the customs/port variability.
    holding_cost_per_unit_per_day:
        Cost of holding one unit for one day at the DC. DC storage is cheaper
        per unit than store storage (bulk racking, cheaper land, scale).
    """

    id: str
    name: str
    region: str
    lead_time_mean_days: float
    lead_time_std_days: float
    holding_cost_per_unit_per_day: float


# Regional cost ordering requested for this project:
#   Asia-Pacific (cheapest) < North America < Europe (most expensive).
# Store-level holding costs below follow the same regional ordering.
DISTRIBUTION_CENTERS: dict[str, DistributionCenter] = {
    "DC_NA": DistributionCenter(
        id="DC_NA",
        name="Los Angeles DC",
        region="North America",
        # Trans-Pacific ocean freight + US customs.
        lead_time_mean_days=35.0,
        lead_time_std_days=4.0,
        holding_cost_per_unit_per_day=0.030,
    ),
    "DC_EU": DistributionCenter(
        id="DC_EU",
        name="Rotterdam DC",
        region="Europe",
        # Longest route (Asia -> Suez -> Northern Europe) plus EU customs.
        lead_time_mean_days=45.0,
        lead_time_std_days=6.0,
        holding_cost_per_unit_per_day=0.045,
    ),
    "DC_APAC": DistributionCenter(
        id="DC_APAC",
        name="Singapore DC",
        region="Asia-Pacific",
        # Short intra-Asia shipping leg, so both mean and variability are low.
        lead_time_mean_days=30.0,
        lead_time_std_days=3.0,
        holding_cost_per_unit_per_day=0.018,
    ),
}


# ---------------------------------------------------------------------------
# Echelon 2: retail stores
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Store:
    """A retail store served by exactly one regional DC.

    Demand parameters
    -----------------
    base_daily_demand:
        Long-run average units sold per day, before weekly/annual seasonality.
    demand_cv:
        Coefficient of variation (std / mean) of daily demand. This is the knob
        that makes the network non-uniform: high-volume stores are given a low
        CV (steady, predictable flagship traffic) while low-volume stores are
        given a high CV (lumpy, intermittent demand). Mirrors the spread of
        series behaviour seen in the M5 project.
    weekend_uplift:
        Multiplier applied to Sat/Sun demand. Mall/tourist locations swing more
        than commuter locations.
    seasonal_amplitude:
        Amplitude of a single annual sine wave, as a fraction of the base level
        (e.g. 0.20 => demand ranges roughly +/-20% over the year).
    seasonal_peak_day:
        Day-of-year at which that annual wave peaks. Northern-hemisphere stores
        peak late in the year (holiday season); the Southern-hemisphere store
        peaks mid-year.
    """

    id: str
    name: str
    region: str
    dc_id: str
    # DC -> store transit, 2-5 days depending on distance from the DC.
    lead_time_mean_days: float
    lead_time_std_days: float
    holding_cost_per_unit_per_day: float
    base_daily_demand: float
    demand_cv: float
    weekend_uplift: float = 1.0
    seasonal_amplitude: float = 0.15
    seasonal_peak_day: int = 330  # ~end of November


STORES: dict[str, Store] = {
    # -- North America (served by DC_NA, 4 stores) --------------------------
    "S_NA_01": Store(
        id="S_NA_01",
        name="New York Flagship",
        region="North America",
        dc_id="DC_NA",
        lead_time_mean_days=4.0,
        lead_time_std_days=0.8,
        holding_cost_per_unit_per_day=0.075,
        base_daily_demand=180.0,  # high volume
        demand_cv=0.22,           # low variability
        weekend_uplift=1.15,
        seasonal_amplitude=0.20,
    ),
    "S_NA_02": Store(
        id="S_NA_02",
        name="Chicago Metro",
        region="North America",
        dc_id="DC_NA",
        lead_time_mean_days=4.0,
        lead_time_std_days=0.8,
        holding_cost_per_unit_per_day=0.070,
        base_daily_demand=95.0,   # mid volume
        demand_cv=0.35,
        weekend_uplift=1.10,
        seasonal_amplitude=0.18,
    ),
    "S_NA_03": Store(
        id="S_NA_03",
        name="Denver Suburban",
        region="North America",
        dc_id="DC_NA",
        lead_time_mean_days=3.0,
        lead_time_std_days=0.6,
        holding_cost_per_unit_per_day=0.065,
        base_daily_demand=28.0,   # low volume
        demand_cv=0.75,           # high variability / lumpy
        weekend_uplift=1.25,
        seasonal_amplitude=0.22,
    ),
    "S_NA_04": Store(
        id="S_NA_04",
        name="Phoenix Outlet",
        region="North America",
        dc_id="DC_NA",
        lead_time_mean_days=2.0,
        lead_time_std_days=0.5,
        holding_cost_per_unit_per_day=0.062,
        base_daily_demand=14.0,   # very low volume
        demand_cv=0.95,           # very lumpy
        weekend_uplift=1.35,
        seasonal_amplitude=0.28,
    ),
    # -- Europe (served by DC_EU, 3 stores) ---------------------------------
    "S_EU_01": Store(
        id="S_EU_01",
        name="London Flagship",
        region="Europe",
        dc_id="DC_EU",
        lead_time_mean_days=3.0,
        lead_time_std_days=0.6,
        holding_cost_per_unit_per_day=0.110,
        base_daily_demand=165.0,  # high volume
        demand_cv=0.20,           # low variability
        weekend_uplift=1.12,
        seasonal_amplitude=0.22,
    ),
    "S_EU_02": Store(
        id="S_EU_02",
        name="Berlin City",
        region="Europe",
        dc_id="DC_EU",
        lead_time_mean_days=4.0,
        lead_time_std_days=0.9,
        holding_cost_per_unit_per_day=0.105,
        base_daily_demand=72.0,   # mid volume
        demand_cv=0.40,
        weekend_uplift=1.05,
        seasonal_amplitude=0.16,
    ),
    "S_EU_03": Store(
        id="S_EU_03",
        name="Milan Boutique",
        region="Europe",
        dc_id="DC_EU",
        lead_time_mean_days=5.0,
        lead_time_std_days=1.0,
        holding_cost_per_unit_per_day=0.098,
        base_daily_demand=22.0,   # low volume
        demand_cv=0.85,           # high variability
        weekend_uplift=1.30,
        seasonal_amplitude=0.30,
    ),
    # -- Asia-Pacific (served by DC_APAC, 3 stores) -------------------------
    "S_AP_01": Store(
        id="S_AP_01",
        name="Tokyo Flagship",
        region="Asia-Pacific",
        dc_id="DC_APAC",
        lead_time_mean_days=2.0,
        lead_time_std_days=0.4,
        holding_cost_per_unit_per_day=0.045,
        base_daily_demand=210.0,  # highest volume in the network
        demand_cv=0.18,           # steadiest
        weekend_uplift=1.08,
        seasonal_amplitude=0.14,
    ),
    "S_AP_02": Store(
        id="S_AP_02",
        name="Sydney Harbour",
        region="Asia-Pacific",
        dc_id="DC_APAC",
        lead_time_mean_days=5.0,
        lead_time_std_days=1.0,
        holding_cost_per_unit_per_day=0.040,
        base_daily_demand=60.0,   # mid volume
        demand_cv=0.45,
        weekend_uplift=1.20,
        # Southern hemisphere: annual peak lands mid-year instead of December.
        seasonal_amplitude=0.18,
        seasonal_peak_day=180,
    ),
    "S_AP_03": Store(
        id="S_AP_03",
        name="Jakarta Mall",
        region="Asia-Pacific",
        dc_id="DC_APAC",
        lead_time_mean_days=4.0,
        lead_time_std_days=0.9,
        holding_cost_per_unit_per_day=0.038,
        base_daily_demand=18.0,   # low volume
        demand_cv=0.90,           # very lumpy
        weekend_uplift=1.40,
        seasonal_amplitude=0.25,
    ),
}


# ---------------------------------------------------------------------------
# Demand-history generation settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemandConfig:
    """Settings for the synthetic demand history."""

    start_date: str = "2023-01-01"
    n_days: int = 730  # 2 full years of daily history
    random_seed: int = 42
    # Day-of-week multipliers (Mon..Sun) applied on top of the base level.
    # Sat/Sun entries are further scaled by each store's `weekend_uplift`.
    day_of_week_factors: tuple[float, ...] = (
        0.92,  # Mon
        0.90,  # Tue
        0.95,  # Wed
        1.00,  # Thu
        1.15,  # Fri
        1.30,  # Sat
        1.10,  # Sun
    )


DEMAND_CONFIG = DemandConfig()


# ---------------------------------------------------------------------------
# Small convenience lookups (no business logic lives here)
# ---------------------------------------------------------------------------


REGIONS: tuple[str, ...] = ("North America", "Europe", "Asia-Pacific")


def stores_for_dc(dc_id: str) -> list[Store]:
    """All stores replenished by the given DC."""
    return [s for s in STORES.values() if s.dc_id == dc_id]


def dc_for_store(store_id: str) -> DistributionCenter:
    """The DC that replenishes the given store."""
    return DISTRIBUTION_CENTERS[STORES[store_id].dc_id]


def total_lead_time_days(store_id: str) -> float:
    """End-to-end supplier -> DC -> store lead time, in days.

    Useful as a first sanity check on how much pipeline inventory the network
    carries before any optimization is applied.
    """
    store = STORES[store_id]
    dc = DISTRIBUTION_CENTERS[store.dc_id]
    return dc.lead_time_mean_days + store.lead_time_mean_days


def describe_network() -> str:
    """Human-readable dump of the network, handy for a quick sanity check."""
    lines = [f"Supplier: {SUPPLIER.name} ({SUPPLIER.id})"]
    for dc in DISTRIBUTION_CENTERS.values():
        lines.append(
            f"  {dc.id} · {dc.name} ({dc.region}) · "
            f"supplier lead time {dc.lead_time_mean_days:.0f}d "
            f"(sd {dc.lead_time_std_days:.0f}d) · "
            f"holding {dc.holding_cost_per_unit_per_day:.3f}/unit/day"
        )
        for store in stores_for_dc(dc.id):
            lines.append(
                f"      {store.id} · {store.name} · "
                f"DC lead time {store.lead_time_mean_days:.0f}d · "
                f"holding {store.holding_cost_per_unit_per_day:.3f}/unit/day · "
                f"base demand {store.base_daily_demand:.0f}/day "
                f"(CV {store.demand_cv:.2f})"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_network())
