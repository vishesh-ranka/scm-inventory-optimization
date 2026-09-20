"""
Build a self-contained static version of the dashboard for GitHub Pages.

Run:
    python3 src/build_static_dashboard.py

Writes `docs/index.html`: one file, no server, no Python at view time.

WHY A STATIC BUILD IS POSSIBLE HERE
-----------------------------------
`dashboard.py` performs no computation at view time. Every figure it shows is
already precomputed in `data/*.parquet` by the pipeline; the slider, the quarter
selector and the raw/explained toggle only SELECT between values that already
exist. So the whole page can be rendered client-side with the data baked in,
which is what lets GitHub Pages host it. Pages serves static files only and
cannot run a Python server.

The Streamlit app in `dashboard.py` remains the local/interactive version. This
is the shareable one. Both read the same Parquet files, so they cannot disagree:
regenerate the data and re-run this script and the page updates.

Charts use Plotly.js from a CDN, pinned to an exact version. Colours are the
same validated palette as the Streamlit app (see `dashboard.py` docstring for
the colourblind-safety validation).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from network_config import DISTRIBUTION_CENTERS, STORES, stores_for_dc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "docs" / "index.html"

PLOTLY_VERSION = "2.35.2"

# Same palette as dashboard.py. Validated for OKLCH lightness band, chroma
# floor, protan/deuteranopia separation (Machado 2009 sev 1.0) and contrast
# against the #faf6ee surface, on the all-pairs pairlist.
C = {
    "cream": "#faf6ee",
    "card": "#f3ece0",
    "plot": "#fffdf7",
    "ink": "#2b2620",
    "muted": "#6b6155",
    "terra": "#9a3f1b",
    "grid": "#e3dccd",
    "teal": "#2c9a88",   # optimized
    "blue": "#2961ce",   # decentralized 95%
    "rust": "#ab4400",   # naive
}

POLICY_LABEL = {
    "naive": "Textbook Formula (Naive Baseline)",
    "decentralized_95": "Every Store On Its Own (Decentralized 95%)",
    "optimized": "Shared Warehouse Buffer (Optimized Multi-Echelon)",
}
POLICY_SHORT = {
    "naive": "Textbook Formula",
    "decentralized_95": "Every Store On Its Own",
    "optimized": "Shared Warehouse Buffer",
}


# ---------------------------------------------------------------------------
# Data assembly, mirroring the loaders in dashboard.py
# ---------------------------------------------------------------------------


def quarter_label(span: str) -> str:
    start = pd.Timestamp(span.split(" -> ")[0])
    return f"Q{start.quarter} {start.year}"


def pretty_span(span: str) -> str:
    """'2024-01-01 -> 2024-03-31' -> 'Jan 1, Mar 31, 2024'."""
    a, b = span.split(" -> ")
    s, e = pd.Timestamp(a), pd.Timestamp(b)
    if s.year == e.year:
        return f"{s:%b} {s.day} to {e:%b} {e.day}, {e.year}"
    return f"{s:%b} {s.day}, {s.year} to {e:%b} {e.day}, {e.year}"


def build_payload() -> dict:
    raw = pd.read_parquet(DATA / "rolling_validation_results.parquet")

    rows = []
    for (window, policy), block in raw.groupby(["window", "policy"]):
        stores = block[block.node_type == "STORE"]
        dcs = block[block.node_type == "DC"]
        rows.append(
            {
                "window": window,
                "policy": policy,
                "quarter": quarter_label(block.test_span.iloc[0]),
                "dates": pretty_span(block.test_span.iloc[0]),
                "demand_ratio": float(block.demand_ratio.iloc[0]),
                "store_fill": float(
                    1 - stores.unmet_demand.sum() / stores.total_demand.sum()
                ),
                "worst_store": float(stores.fill_rate.min()),
                "dc_fill": float(1 - dcs.unmet_demand.sum() / dcs.total_demand.sum()),
                "cost": float(block.actual_cost_annual.sum()),
            }
        )
    summary = pd.DataFrame(rows)

    # Expedite sensitivity: the SIMULATED figures, not the analytical ones.
    val = pd.read_parquet(DATA / "validation_results.parquet")
    expedite = []
    for days in (0, 1, 2, 3):
        block = val[val.policy == f"optimized_exp{days}"]
        stores = block[block.node_type == "STORE"]
        dcs = block[block.node_type == "DC"]
        expedite.append(
            {
                "days": days,
                "store_fill": float(
                    1 - stores.unmet_demand.sum() / stores.total_demand.sum()
                ),
                "dc_fill": float(1 - dcs.unmet_demand.sum() / dcs.total_demand.sum()),
                "cost": float(block.actual_cost_annual.sum()),
                "expedited": float(block.expedited_units.sum()),
            }
        )

    # Per-location detail table.
    detail = raw[
        ["window", "policy", "node_name", "node_type", "region",
         "fill_rate", "day_service", "mean_onhand", "actual_cost_annual"]
    ].copy()
    detail["policy"] = detail["policy"].map(POLICY_SHORT)
    detail["node_type"] = detail["node_type"].map(
        {"STORE": "Store", "DC": "Regional Warehouse"}
    )

    windows = list(summary.window.unique())
    wide_cost = summary.pivot(index="window", columns="policy", values="cost")
    saving = {
        w: float(
            (wide_cost.loc[w, "decentralized_95"] - wide_cost.loc[w, "optimized"])
            / wide_cost.loc[w, "decentralized_95"]
        )
        for w in windows
    }

    naive = summary[summary.policy == "naive"]
    return {
        "summary": summary.to_dict("records"),
        "expedite": expedite,
        "detail": detail.to_dict("records"),
        "windows": windows,
        "saving": saving,
        "corr": float(naive.demand_ratio.corr(naive.dc_fill)),
        "policy_label": POLICY_LABEL,
        "policy_short": POLICY_SHORT,
        "colors": C,
    }


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


GLOSSARY = [
    ("Regional warehouse (DC)",
     "A building that receives big sea shipments from the factory and sends "
     "smaller deliveries to nearby stores. There are three, one per region."),
    ("Echelon",
     "One level in the chain. Here there are two levels that hold stock: the "
     "warehouses, and the stores they supply. “Multi-echelon” means "
     "planning both levels together instead of separately."),
    ("Lead time",
     "How long you wait for a delivery after ordering it. Factory to warehouse "
     "is 30 to 45 days (a ship crossing an ocean, plus customs). Warehouse to "
     "store is 2 to 5 days (a truck)."),
    ("Safety stock (backup stock)",
     "Extra inventory kept on hand beyond what you normally expect to sell, so "
     "that an unusually busy week doesn’t empty the shelves. Deciding how "
     "much to keep, and where, is the whole problem."),
    ("Holding cost",
     "What it costs to keep one unit sitting on a shelf for a day: space, money "
     "tied up, insurance, spoilage risk. Warehouse space is roughly 2.4× "
     "cheaper per unit than store space, which is why <em>where</em> you keep "
     "backup stock matters financially."),
    ("Service level",
     "The share of customer demand you manage to supply. The target throughout "
     "this project is 95%."),
    ("Fill rate (% of demand met on time)",
     "Of everything customers actually asked for, the share they actually got. "
     "This is the main scoreboard."),
    ("Warehouse order fill rate",
     "When stores ask their regional warehouse to resupply them, this is the "
     "share of that request the warehouse could actually ship. When it drops, "
     "stores are stranded: they cannot restock no matter how well their own "
     "shelves were planned."),
    ("Variability (standard deviation)",
     "How much demand bounces around from day to day. A store selling 100 units "
     "every day and a store averaging 100 but swinging between 20 and 300 need "
     "very different amounts of backup stock."),
    ("Pooling",
     "Keeping one shared pile of backup stock centrally instead of many small "
     "piles locally. It works because stores rarely have their busiest week at "
     "the same time, so a shared pile can be sent wherever the shortage "
     "actually happens. Ten separate piles cannot be moved once placed."),
    ("Out-of-sample validation",
     "Testing a plan on data it has never seen. If you check a plan against the "
     "same history you used to build it, it will always look good; that proves "
     "nothing. Every result here is out-of-sample."),
    ("Rolling-window backtest",
     "Repeating that honest test several times. Build the plan using everything "
     "up to a date, test it on the next three months, then move the date forward "
     "and repeat. Four such tests were run here, so we can see whether a finding "
     "is consistent or just luck from one period."),
    ("Correlation",
     "A single number from −1 to +1 summarising whether two things move "
     "together. Near +1 they rise together; near −1 one rises as the other "
     "falls; near 0 there is no relationship."),
    ("Expedite window (emergency delivery speed)",
     "How fast a warehouse can rush an emergency delivery to a nearby store "
     "when it is about to run out. The faster this is, the less backup stock "
     "each store needs to hold itself."),
]


def glossary_html() -> str:
    items = "".join(
        f"<div class='gterm'><dt>{t}</dt><dd>{d}</dd></div>" for t, d in GLOSSARY
    )
    return f"<dl class='glossary'>{items}</dl>"


# ---------------------------------------------------------------------------
# Network diagram
# ---------------------------------------------------------------------------
#
# Geometry is generated from network_config rather than hand placed, so the
# picture cannot drift from the network the pipeline actually simulates. The
# real split is 4 stores on DC-1 and 3 each on DC-2 and DC-3.
#
# The SVG is emitted as static markup, so it renders in full with JavaScript
# disabled. The script only adds hover and focus highlighting on top.

SVG_W, SVG_H = 680, 360
X_SUP, X_DC, X_STORE = 61, 300, 580
STORE_Y0, STORE_STEP = 28, 32


def _network_layout():
    """Positions plus the label and tooltip text for every node."""
    store_list = list(STORES.values())
    dc_ids = list(DISTRIBUTION_CENTERS)

    stores = {}
    for i, st in enumerate(store_list):
        stores[st.id] = {
            "key": f"S{i + 1:02d}",
            "y": STORE_Y0 + i * STORE_STEP,
            "store": st,
        }

    dcs = {}
    for i, dc_id in enumerate(dc_ids, start=1):
        served = stores_for_dc(dc_id)
        ys = [stores[s.id]["y"] for s in served]
        dcs[dc_id] = {
            "key": f"DC{i}",
            "label": f"DC-{i} · R{i}",
            "y": sum(ys) / len(ys),
            "dc": DISTRIBUTION_CENTERS[dc_id],
            "served": served,
            "index": i,
        }
    return dcs, stores


def network_svg() -> str:
    dcs, stores = _network_layout()
    y_sup = sum(d["y"] for d in dcs.values()) / len(dcs)
    parts = []

    lead_times = [d["dc"].lead_time_mean_days for d in dcs.values()]
    store_leads = [s["store"].lead_time_mean_days for s in stores.values()]

    # edges, drawn first so nodes sit on top
    for d in dcs.values():
        parts.append(
            f'<path class="edge" data-edge="SUP|{d["key"]}" '
            f'd="M74,{y_sup:.0f} H180 V{d["y"]:.0f} H288"/>'
        )
    for d in dcs.values():
        mid = 424 if d["index"] != 2 else 448
        for st in d["served"]:
            s = stores[st.id]
            parts.append(
                f'<path class="edge" data-edge="{d["key"]}|{s["key"]}" '
                f'd="M312,{d["y"]:.0f} H{mid} V{s["y"]} H571"/>'
            )

    # supplier
    sup_meta = (
        f"Overseas factory · feeds all {len(dcs)} regional warehouses · "
        f"{min(lead_times):.0f} to {max(lead_times):.0f} day sea lead times "
        f"including customs"
    )
    parts.append(
        f'<g class="node supplier" data-node="SUP" tabindex="0" role="button" '
        f'data-title="Supplier" data-meta="{sup_meta}">'
        f'<rect class="shape" x="{X_SUP - 13}" y="{y_sup - 13:.0f}" width="26" height="26"/>'
        f'<rect class="focus-ring" x="{X_SUP - 19}" y="{y_sup - 19:.0f}" '
        f'width="38" height="38"/>'
        f'<text class="label" x="{X_SUP}" y="{y_sup + 31:.0f}" '
        f'text-anchor="middle">Supplier</text></g>'
    )

    # distribution centers
    for d in dcs.values():
        dc = d["dc"]
        keys = ", ".join(stores[s.id]["key"] for s in d["served"])
        meta = (
            f"{dc.name} · {dc.region} · {dc.lead_time_mean_days:.0f} day lead time "
            f"from the factory · serves {len(d['served'])} stores ({keys}) · "
            f"holding cost {dc.holding_cost_per_unit_per_day:.3f} per unit per day"
        )
        y = d["y"]
        parts.append(
            f'<g class="node dc" data-node="{d["key"]}" tabindex="0" role="button" '
            f'data-title="{d["label"]}" data-meta="{meta}">'
            f'<rect class="shape" x="{X_DC - 12}" y="{y - 12:.0f}" width="24" height="24" '
            f'transform="rotate(45 {X_DC} {y:.0f})"/>'
            f'<rect class="focus-ring" x="{X_DC - 19}" y="{y - 19:.0f}" '
            f'width="38" height="38"/>'
            f'<text class="label" x="{X_DC}" y="{y + 38:.0f}" '
            f'text-anchor="middle">{d["label"]}</text></g>'
        )

    # stores
    for d in dcs.values():
        for st in d["served"]:
            s = stores[st.id]
            meta = (
                f"{st.name} · {st.region} · replenished from {d['label']} · "
                f"{st.lead_time_mean_days:.0f} day delivery · 95% service target"
            )
            parts.append(
                f'<g class="node store" data-node="{s["key"]}" tabindex="0" '
                f'role="button" data-title="Store {s["key"]}" data-meta="{meta}">'
                f'<circle class="shape" cx="{X_STORE}" cy="{s["y"]}" r="8"/>'
                f'<circle class="focus-ring" cx="{X_STORE}" cy="{s["y"]}" r="14"/>'
                f'<text class="label" x="{X_STORE + 16}" y="{s["y"] + 4}">'
                f'{s["key"]}</text></g>'
            )

    aria = (
        "Interactive diagram of the supply network: one overseas factory feeding "
        "three regional warehouses, which between them serve ten stores."
    )
    return (
        f'<svg class="diagram" id="net" viewBox="0 0 {SVG_W} {SVG_H}" '
        f'role="group" aria-label="{aria}">' + "".join(parts) + "</svg>"
    )


def network_sr_text() -> str:
    """Text equivalent of the diagram, for screen readers and for no JavaScript."""
    dcs, stores = _network_layout()
    lead_times = [d["dc"].lead_time_mean_days for d in dcs.values()]
    store_leads = [s["store"].lead_time_mean_days for s in stores.values()]

    bits = [
        "Network structure in text: a single overseas factory feeds three "
        "regional warehouses."
    ]
    for d in dcs.values():
        keys = [stores[s.id]["key"] for s in d["served"]]
        listed = ", ".join(keys[:-1]) + " and " + keys[-1]
        bits.append(
            f"{d['label']}, the {d['dc'].name} in {d['dc'].region}, sits "
            f"{d['dc'].lead_time_mean_days:.0f} days from the factory and serves "
            f"{len(keys)} stores: {listed}."
        )
    bits.append(
        f"Factory to warehouse lead times run {min(lead_times):.0f} to "
        f"{max(lead_times):.0f} days by sea including customs. Warehouse to store "
        f"deliveries take {min(store_leads):.0f} to {max(store_leads):.0f} days by "
        f"road. Every store is held to a 95% service target."
    )
    return " ".join(bits)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multi-Echelon Inventory Optimization</title>
<meta name="description" content="Where should a retailer keep its backup stock?
Validated with rolling-window backtests on a simulated 13-location network.">
<script src="https://cdn.plot.ly/plotly-__PLOTLY_VERSION__.min.js" charset="utf-8"></script>
<style>
  :root {
__ROOTVARS__
  }
  * { box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; }
  body {
    margin: 0; background: var(--cream); color: var(--ink);
    font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
          Helvetica, Arial, sans-serif;
    padding-block: 32px;
  }
  .wrap { max-width: 1120px; margin: 0 auto; padding-inline: 20px; }
  h1 { color: var(--terra); font-size: 2rem; margin: 0 0 .4rem; line-height: 1.2; }
  h2 {
    color: var(--terra); font-size: 1.3rem; margin: 2.6rem 0 .6rem;
    border-bottom: 2px solid var(--grid); padding-bottom: .4rem;
  }
  p { margin: .7rem 0; }
  .lede { color: var(--muted); font-size: 1rem; max-width: 80ch; }
  .caption { color: var(--muted); font-size: .92rem; max-width: 85ch; }
  .chartnote {
    background: var(--card); border-radius: 6px; padding: 11px 15px;
    margin: 10px 0 14px; font-size: .94rem;
  }
  .note {
    background: var(--card); border-left: 4px solid var(--teal);
    padding: 14px 18px; border-radius: 4px; margin: 14px 0 18px; font-size: .94rem;
  }
  .callout {
    background: rgba(171,68,0,.08); border-left: 4px solid var(--rust);
    padding: 14px 18px; border-radius: 4px; margin: 14px 0 18px; font-size: .94rem;
  }
  .metrics {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 14px; margin: 22px 0;
  }
  .metric {
    background: var(--card); border: 1px solid var(--grid);
    border-radius: 8px; padding: 14px 16px;
  }
  .metric .label {
    color: var(--muted); font-size: .8rem; text-transform: uppercase;
    letter-spacing: .04em;
  }
  .metric .value { font-size: 1.65rem; font-weight: 600; margin-top: 4px; }
  .metric .sub { color: var(--muted); font-size: .82rem; margin-top: 2px; }
  .metric .sub.bad { color: var(--rust); font-weight: 600; }
  .chart {
    background: var(--plot); border: 1px solid var(--grid);
    border-radius: 8px; padding: 8px; margin: 6px 0 4px;
  }
  .controls {
    display: flex; flex-wrap: wrap; gap: 18px; align-items: center;
    background: var(--card); border: 1px solid var(--grid);
    border-radius: 8px; padding: 12px 16px; margin: 14px 0;
  }
  .controls label { font-size: .9rem; font-weight: 600; }
  select, input[type=range] {
    font: inherit; font-size: .92rem; color: var(--ink); background: var(--plot);
    border: 1px solid var(--grid); border-radius: 5px; padding: 6px 9px;
  }
  input[type=range] { padding: 0; accent-color: var(--terra); min-width: 190px; }
  .seg { display: inline-flex; border: 1px solid var(--grid); border-radius: 5px;
         overflow: hidden; }
  .seg button {
    font: inherit; font-size: .9rem; border: 0; background: var(--plot);
    color: var(--ink); padding: 7px 14px; cursor: pointer;
  }
  .seg button[aria-pressed="true"] { background: var(--terra); color: #fff; }
  details {
    background: var(--card); border: 1px solid var(--grid);
    border-radius: 6px; padding: 10px 16px; margin: 12px 0;
  }
  details > summary {
    cursor: pointer; color: var(--terra); font-weight: 600; font-size: .95rem;
  }
  details[open] > summary { margin-bottom: .6rem; }
  .glossary { margin: 0; display: grid;
               grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 4px 30px; }
  .gterm { padding: 7px 0; border-bottom: 1px solid var(--grid); }
  .glossary dt { font-weight: 700; color: var(--ink); }
  .glossary dd { margin: 3px 0 0; color: var(--muted); font-size: .92rem; }
  .tablewrap { overflow-x: auto; margin: 10px 0; border: 1px solid var(--grid);
                border-radius: 6px; }
  table { border-collapse: collapse; width: 100%; font-size: .88rem;
           background: var(--plot); }
  th, td { padding: 8px 11px; text-align: right; white-space: nowrap;
            border-bottom: 1px solid var(--grid); }
  th { background: var(--card); color: var(--ink); font-weight: 700;
        text-align: right; position: sticky; top: 0; }
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2),
  th:nth-child(3), td:nth-child(3) { text-align: left; }
  .foot { color: var(--muted); font-size: .85rem; margin-top: 2.5rem;
           border-top: 1px solid var(--grid); padding-top: 1rem; }
  code { background: var(--card); color: var(--terra); padding: 1px 5px;
          border-radius: 3px; font-size: .9em; }
  a { color: var(--terra); }
  /* Network diagram ------------------------------------------------- */
  .diagram-block { border: 1px solid var(--grid); border-radius: 8px;
                   background: var(--plot); margin: 12px 0 8px; }
  .diagram-cap { display: flex; flex-wrap: wrap; gap: 6px 22px;
                 padding: 11px 15px; border-bottom: 1px solid var(--grid);
                 color: var(--muted); font-size: .88rem; }
  .diagram-cap b { color: var(--ink); font-weight: 700; }
  .diagram-wrap { position: relative; padding: 16px 12px; }
  .diagram { display: block; width: 100%; height: auto; touch-action: pan-y; }
  .edge { fill: none; stroke: rgba(107,97,85,.34); stroke-width: 1.2;
          transition: stroke .18s ease, stroke-width .18s ease, opacity .18s ease; }
  .node { cursor: pointer; }
  .node .shape { fill: var(--plot); stroke: var(--ink); stroke-width: 1.3;
                 transition: fill .18s ease, stroke .18s ease; }
  .node .label { font-size: 10px; fill: var(--muted); transition: fill .18s ease; }
  .node.supplier .shape { fill: var(--terra); stroke: var(--terra); }
  .node:hover .shape, .node:focus-visible .shape { fill: var(--teal); stroke: var(--teal); }
  .focus-ring { fill: none; stroke: transparent; opacity: 0; }
  .node:focus-visible .focus-ring { stroke: var(--terra); stroke-width: 1.5; opacity: 1; }
  .node:focus { outline: none; }
  .diagram.is-active .edge { opacity: .2; }
  .diagram.is-active .node { opacity: .32; }
  .diagram.is-active .edge.hl { opacity: 1; stroke: var(--teal); stroke-width: 2.1; }
  .diagram.is-active .node.hl { opacity: 1; }
  .diagram.is-active .node.hl .shape { fill: var(--teal); stroke: var(--teal); }
  .diagram.is-active .node.hl.supplier .shape { fill: var(--terra); stroke: var(--terra); }
  .diagram.is-active .node.hl .label { fill: var(--ink); }
  .tip { position: absolute; z-index: 5; pointer-events: none; opacity: 0;
         transform: translate(-50%, -112%); background: var(--ink); color: #fff;
         padding: 9px 12px; max-width: 280px; border-radius: 5px;
         font-size: .78rem; line-height: 1.5; transition: opacity .14s ease; }
  .tip.show { opacity: 1; }
  .tip b { display: block; font-weight: 700; color: #fff; margin-bottom: 3px; }
  .tip span { color: rgba(255,255,255,.8); }
  .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
             overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0; }

  @media (max-width: 640px) {
    h1 { font-size: 1.5rem; } body { font-size: 15px; }
    .controls { flex-direction: column; align-items: flex-start; }
  }
</style>
</head>
<body>
<div class="wrap">

  <h1>Multi-Echelon Inventory Optimization</h1>
  <p class="lede">
    This project asks a simple question: <strong>where should a retailer keep its
    backup stock?</strong> The network here has one overseas factory, three
    regional warehouses (North America, Europe, Asia-Pacific), and ten stores.
    Every number on this page comes from a simulation that was given only past
    data to plan with, then tested against a later period it had never seen -
    the same way a real policy would have to work.
  </p>

  <div class="metrics" id="topMetrics"></div>

  <h2>Start Here: What The Words Mean</h2>
  <p class="caption">Every technical term used anywhere on this page is defined
  below. Nothing further down assumes you have read a supply-chain textbook.</p>
  <details open>
    <summary>Glossary, Plain-Language Definitions</summary>
    __GLOSSARY__
  </details>
  <div class="note">
    <strong>How this page is ordered.</strong> Findings appear in order of how
    well they hold up under testing, not how impressive they sound. Warehouse
    reliability comes first because it was consistent in all four test periods.
    The money saved comes second and is shown as a range, because it was always
    positive but its size changed a lot depending on how busy the period was.
    Dollar costs appear last and only one period at a time, comparing them
    across periods is genuinely misleading, for a reason explained in that section.
  </div>

  <h2>How The Network Is Put Together</h2>
  <p class="caption">
    Two levels hold stock: three regional warehouses, and the ten stores they
    supply. Everything starts at a single overseas factory. The shape matters
    because the two legs of the journey are wildly different lengths: a month or
    more by sea to reach a warehouse, then a couple of days by road to reach a
    shop. That mismatch is what the rest of this page is about.
  </p>

  <div class="diagram-block">
    <div class="diagram-cap">
      <span><b>Network structure:</b> 1 factory to 3 regional warehouses to 10 stores</span>
      <span>Hover or tab to a node to see its connections</span>
    </div>
    <div class="diagram-wrap">
      __NETWORK_SVG__
      <div class="tip" id="tip" role="status" aria-live="polite"></div>
    </div>
  </div>

  <p class="sr-only">__NETWORK_SR__</p>
  <noscript>
    <div class="note">__NETWORK_SR__</div>
  </noscript>

  <p class="caption">
    Note that this is a simple tree: every store is supplied by exactly one
    warehouse, so there is a single path from the factory to any shop. That is
    what makes the pooling question tractable here, and it is a different shape
    from a hierarchy where product and geography cross one another.
  </p>

  <details><summary>Why does this matter?</summary>
    <p>Where a network branches is where the money and the risk sit. A warehouse
    that serves four stores is carrying risk for all four at once, so getting its
    stock level wrong has four times the blast radius of getting one shop wrong.
    That is the argument for planning the levels together rather than one at a
    time.</p>
    <p>The long sea leg is the other half of the picture. An order placed today
    for a warehouse arrives more than a month from now, against a forecast made
    today. The shops, by contrast, can be corrected within days. Most of the
    difficulty in this project comes from that asymmetry.</p>
  </details>

  <h2>Main Finding: Keeping Regional Warehouses In Stock</h2>
  <div class="chartnote">
    <strong>What this shows:</strong> for each three-month test period, how much
    of what stores asked their regional warehouse for was actually shipped.
    <strong>What to look for:</strong> the green dots stay pinned at 100% every
    time, while the orange dots slip, and slip furthest in Q4, the busiest
    quarter.
  </div>
  <div class="chart"><div id="chartDC"></div></div>
  <p class="caption" id="dcCaption"></p>
  <details><summary>Why does this matter?</summary>
    <p>When a regional warehouse runs empty, the damage is not limited to one
    shop. Every store that depends on that warehouse is stranded at once -
    they place their restock orders as normal and simply do not receive the
    goods, however well their own shelves were planned. A single warehouse
    shortage in this network therefore turns into simultaneous empty shelves
    across three or four stores, during the busiest trading weeks of the year,
    which is precisely when lost sales are most expensive and most visible to
    customers.</p>
    <p>This is also the failure that is hardest to see coming. The textbook
    method reports healthy numbers for three quarters out of four. Anyone
    reviewing it in a calm period would reasonably conclude it was working
    fine.</p>
  </details>

  <h2>Why The Textbook Method Breaks Down Under Pressure</h2>
  <div class="controls">
    <label>Chart view</label>
    <span class="seg" id="viewToggle">
      <button data-view="explained" aria-pressed="true">Show Explained</button>
      <button data-view="raw" aria-pressed="false">Show Raw Data</button>
    </span>
    <span class="caption" style="margin:0">
      &ldquo;Explained&rdquo; writes the interpretation onto the chart.
    </span>
  </div>
  <div class="chartnote">
    <strong>What this shows:</strong> each dot is one three-month test period.
    Left-to-right is how much busier (or quieter) that period turned out to be
    than the history the plan was built from. Up-and-down is how well the
    textbook method&rsquo;s warehouses coped.
    <strong>What to look for:</strong> the dots fall as you move right, the
    busier it got, the worse the warehouses performed.
  </div>
  <div class="chart"><div id="chartScatter"></div></div>
  <p class="caption" id="corrCaption"></p>
  <p class="caption">
    <strong>Why it happens.</strong> The textbook method sizes each
    warehouse&rsquo;s backup stock with a standard formula that assumes each
    day&rsquo;s demand is unrelated to the last. Real demand is not like that: it
    has a weekly rhythm, a Christmas season, and a growth trend, so busy days
    arrive in clusters. Across a 30to45 day sea crossing those clusters pile
    up, and the true swing in demand turns out to be between
    <strong>2.8&times; and 10.6&times;</strong> larger than the formula assumes.
    The formula therefore buys a backup stock sized for a calm world. The new
    method sizes the same backup stock from what demand actually did over that
    same 30to45 day window, so the clustering is already accounted for.
  </p>
  <details><summary>Why does this matter?</summary>
    <p>This is a warning about how inventory plans are usually reviewed. The
    textbook formula is not broken in an obvious way, it is the standard
    approach taught everywhere, and it performs perfectly well for most of the
    year. Its weakness only appears under demand pressure, which is the one
    condition where being wrong is expensive.</p>
    <p>The practical implication is that a policy should never be judged on an
    average or a quiet period. It should be judged on its worst quarter, because
    that is where the lost sales, the emergency freight bills, and the
    disappointed customers actually occur. Any review process that looks at
    annual averages will systematically miss this.</p>
  </details>

  <h2>Money Saved: A Range, Not A Single Number</h2>
  <div class="chartnote">
    <strong>What this shows:</strong> how much cheaper the shared-warehouse
    method was than giving every store its own independent backup stock, in each
    test period. <strong>What to look for:</strong> every bar is above zero,
    it saved money every time, but the bars are very different
    heights, so the size of the saving is not something you can promise in
    advance.
  </div>
  <div class="chart"><div id="chartSaving"></div></div>
  <p class="caption" id="savingCaption"></p>
  <details><summary>Why does this matter?</summary>
    <p>A business case built on &ldquo;this saves 2.7%&rdquo; would be on shaky
    ground: in a quiet quarter the real figure is nearer 1.5%, and anyone
    checking the books three months later would find the promised saving missing.
    A business case built on &ldquo;this saves between 1.5% and 8.3%, averaging
    under 4%, and it holds our warehouses at full reliability even at
    Christmas&rdquo; is both more honest and more persuasive, because the
    reliability part is the finding that actually replicated every time.</p>
    <p>It is also worth being clear that this is a modest saving on its own. The
    reason to adopt the approach is that shelves stay stocked in peak season; the
    cost reduction is a bonus, not the headline.</p>
  </details>

  <h2>Try It Yourself: How Fast Must Emergency Deliveries Be?</h2>
  <p class="caption">
    The shared-warehouse method assumes a regional warehouse can rush an
    emergency delivery to a nearby store quickly. The faster that is, the less
    backup stock each store has to hold itself. This was the single softest
    assumption in the whole model, so it was tested directly. Move the
    slider to see what the simulation actually produced at each speed.
  </p>
  <div class="controls">
    <label for="expSlider">Emergency delivery takes</label>
    <input type="range" id="expSlider" min="0" max="3" step="1" value="1">
    <strong id="expLabel" style="min-width:8ch"></strong>
  </div>
  <div class="metrics" id="expMetrics"></div>
  <div class="chartnote">
    <strong>What this shows:</strong> the simulated result at all four emergency
    delivery speeds, with your current choice highlighted.
    <strong>What to look for:</strong> the bars are almost the same height -
    this assumption barely changes the outcome.
  </div>
  <div class="chart"><div id="chartExp"></div></div>
  <div class="note" id="expNote"></div>
  <details><summary>Why does this matter?</summary>
    <p>Models usually rest on a handful of assumptions nobody can verify, and the
    honest question is always &ldquo;how much would it matter if this were
    wrong?&rdquo; Here that question has an answer, and the answer is reassuring:
    even if the assumed emergency delivery speed is off by a factor of three, the
    recommendation does not change.</p>
    <p>That is worth knowing before committing money to a change. It means the
    case does not depend on negotiating a fast internal courier arrangement, and
    it removes an obvious line of objection from anyone reviewing the proposal.</p>
  </details>

  <h2>Cost In Dollars, One Period At A Time</h2>
  <div class="callout" id="costWarning"></div>
  <div class="controls">
    <label for="qSelect">Test period</label>
    <select id="qSelect"></select>
  </div>
  <div class="chartnote">
    <strong>What this shows:</strong> the yearly cost of stock held under each of
    the three methods, within the single period you selected. The share of
    customer demand each one met is printed on its bar.
    <strong>What to look for:</strong> read the cost and the demand-met figure
    together, never the cost on its own.
  </div>
  <div class="chart"><div id="chartCost"></div></div>
  <details><summary>Why does this matter?</summary>
    <p>This is the most common way an inventory review goes wrong. Cost is easy
    to measure and lands straight on a financial report; stockouts are diffuse,
    show up as lost sales nobody logged, and rarely get attributed back to an
    inventory decision. So the policy that quietly runs out of stock can look
    like the responsible, frugal choice on paper.</p>
    <p>Putting the service figure directly on the cost bar is a deliberate guard
    against that. Any time someone presents an inventory saving, the first
    question should be what happened to availability over the same period.</p>
  </details>

  <h2>Share Of Customer Demand Met, By Period</h2>
  <div class="chartnote">
    <strong>What this shows:</strong> of everything customers asked for across
    all ten stores, how much they actually got, under each method. The dotted
    line is the 95% target. <strong>What to look for:</strong> the two methods
    are close for most of the year, then separate in Q4.
  </div>
  <div class="chart"><div id="chartService"></div></div>
  <p class="caption">These bars start at zero, so the differences look modest,
  and they genuinely are modest in ordinary quarters. The gap opens up in
  Q4. The larger operational difference between the two methods is in the
  warehouses, shown at the top of this page.</p>

  <h2>All The Numbers</h2>
  <div class="chartnote">
    <strong>What this shows:</strong> every figure behind the charts above, one
    row per method per test period. <strong>What to look for:</strong> the
    &ldquo;Demand vs Plan&rdquo; column explains most of the variation, the
    higher it is, the worse every method performs.
  </div>
  <div class="tablewrap"><table id="mainTable"></table></div>
  <details>
    <summary>Store-By-Store And Warehouse-By-Warehouse Detail</summary>
    <p class="caption">One row per location per method per test period.</p>
    <div class="tablewrap"><table id="detailTable"></table></div>
  </details>

  <h2>How This Was Worked Out, And What It Cannot Tell You</h2>
  <details>
    <summary>Method And Known Limitations</summary>
    <p><strong>The test.</strong> Four separate tests were run. Each one builds
    the stocking plan using only the history up to a cut-off date, then simulates
    the following three months day by day, stores selling, ordering from
    their warehouse, warehouses ordering from the factory, deliveries arriving
    after realistic delays. Demand the shops could not supply is treated as a
    lost sale, not a delayed one, which is how retail actually works. The
    simulation runs for two months before measurement starts, so it is not being
    judged on its opening conditions.</p>
    <p><strong>Why the standard formula was replaced.</strong> The textbook
    approach understates how much demand really swings over a 30to45 day sea
    crossing, by between 2.8&times; and 10.6&times; at the warehouses,
    because it assumes each day is unrelated to the one before. Every
    backup stock figure here is instead taken from what demand actually did over
    windows of that same length in the real history.</p>
    <p><strong>Where the saving comes from.</strong> The stores do not all get
    busy at the same time, Sydney peaks mid-year while Tokyo and Jakarta
    peak in December. That means one shared pile of backup stock at the warehouse
    only needs 65to75% of what ten separate piles would. This was measured by
    adding up genuine simultaneous shortfalls in the history, not assumed.</p>
    <p><strong>What this cannot tell you.</strong></p>
    <ul>
      <li>The demand history is synthetic, generated data, not a real
      retailer&rsquo;s sales. The structure of the findings is sound, but the
      specific dollar figures describe this simulated network only.</li>
      <li>Only four test periods, drawn from two years of history. The ranges
      shown indicate spread; they are not statistical confidence intervals.</li>
      <li>The planner chooses one stocking level per region rather than per
      individual store, a simplification made so the shared buffer could be
      calculated.</li>
      <li>Emergency delivery speed is assumed rather than measured, though
      as the interactive section above shows, the results barely move across the
      plausible range.</li>
      <li>Every result assumes the factory itself never runs short.</li>
    </ul>
  </details>

  <p class="foot">
    Source: rolling-window backtest of 4 test periods &times; 3 methods &times;
    13 locations. Generated by <code>src/rolling_validation.py</code>, with the
    emergency-delivery comparison from <code>src/validate_policy.py</code>.
    This page is a static build of the Streamlit dashboard
    (<code>dashboard.py</code>), same data, no server required.
    &nbsp;&middot;&nbsp;
    <a href="https://github.com/vishesh-ranka/scm-inventory-optimization">Source
    code on GitHub</a>
  </p>
</div>

<script>
const D = __DATA__;
const C = D.colors;
const PCT = v => (v * 100).toFixed(2) + '%';
const PCT1 = v => (v * 100).toFixed(1) + '%';
const USD = v => '$' + Math.round(v).toLocaleString('en-US');

const byWin = {};
D.summary.forEach(r => { (byWin[r.window] = byWin[r.window] || {})[r.policy] = r; });
const WINS = D.windows;
const QLAB = WINS.map(w => byWin[w].naive.quarter);

const LAYOUT = () => ({
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: C.plot,
  font: { color: C.ink, size: 13,
          family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' },
  margin: { l: 90, r: 30, t: 80, b: 66 },
  legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, x: 0,
            bgcolor: 'rgba(0,0,0,0)' },
  hoverlabel: { bgcolor: '#ffffff', bordercolor: C.grid,
                font: { color: C.ink, size: 13 } },
  title: { font: { size: 15, color: C.terra }, x: 0, xref: 'paper',
           xanchor: 'left', y: 0.97, yanchor: 'top' },
  xaxis: { gridcolor: C.grid, zerolinecolor: C.grid, linecolor: C.grid,
           automargin: true, tickfont: { color: C.muted },
           title: { font: { color: C.muted }, standoff: 14 } },
  yaxis: { gridcolor: C.grid, zerolinecolor: C.grid, linecolor: C.grid,
           automargin: true, tickfont: { color: C.muted },
           title: { font: { color: C.muted }, standoff: 14 } }
});
const CFG = { displayModeBar: false, responsive: true };

/* ---------- top metrics ---------- */
(function () {
  const naiveDC = WINS.map(w => byWin[w].naive.dc_fill);
  const worst = Math.min(...naiveDC);
  const sav = WINS.map(w => D.saving[w]);
  const mean = sav.reduce((a, b) => a + b, 0) / sav.length;
  const cards = [
    ['Warehouse Reliability, New Method', '100%', 'held in all 4 test periods', false],
    ['Warehouse Reliability, Textbook Method', PCT1(worst),
     PCT1(worst - 1) + ' at its worst', true],
    ['Money Saved (Range)', PCT1(Math.min(...sav)) + 'to' + PCT1(Math.max(...sav)),
     'average ' + PCT1(mean), false],
    ['Test Periods Run', String(WINS.length), '3 months each', false]
  ];
  document.getElementById('topMetrics').innerHTML = cards.map(
    ([l, v, s, bad]) => `<div class="metric"><div class="label">${l}</div>` +
      `<div class="value">${v}</div>` +
      `<div class="sub${bad ? ' bad' : ''}">${s}</div></div>`).join('');

  document.getElementById('dcCaption').innerHTML =
    `The new method keeps every regional warehouse able to fill <strong>100%</strong>` +
    ` of store restock orders in all four test periods. The textbook method looks` +
    ` perfectly healthy when trade is calm. It also hits 100% in Q2 2024, the` +
    ` one quarter where demand came in <em>below</em> what it had planned for -` +
    ` but falls to <strong>${PCT1(worst)}</strong> in Q4 2024, the Christmas peak.` +
    ` That is the whole point: it fails exactly when it is needed most, and looks` +
    ` fine the rest of the year.`;

  document.getElementById('savingCaption').innerHTML =
    `Sharing backup stock centrally was cheaper in <strong>every single test` +
    ` period</strong>, so the direction of this finding is solid. The size is not:` +
    ` it ranged from <strong>${PCT1(Math.min(...sav))}</strong> to` +
    ` <strong>${PCT1(Math.max(...sav))}</strong>, averaging` +
    ` <strong>${PCT1(mean)}</strong>. An earlier version of this analysis, based on` +
    ` a single test period, reported 2.7%, which turned out to be one draw` +
    ` from a wide spread rather than a dependable figure. Quote the range, not a` +
    ` single number.`;

  const c = D.corr;
  document.getElementById('corrCaption').innerHTML =
    `The statistical summary of that pattern is a correlation of` +
    ` <strong>${c.toFixed(2)}</strong>, which in plain terms means:` +
    ` <strong>as demand rises above what the plan plans for, the textbook` +
    ` method&rsquo;s warehouse reliability drops, consistently and steeply.</strong>` +
    ` A correlation of &minus;1.00 would be a perfect straight-line relationship, so` +
    ` ${c.toFixed(2)} is a strong one. With only four test periods this is an` +
    ` indication rather than proof, but it matches the mechanism exactly.`;

  const oc = WINS.map(w => byWin[w].optimized.cost);
  const w4 = byWin[WINS[WINS.length - 1]].naive;
  document.getElementById('costWarning').innerHTML =
    `<strong>&#9888; Please read this before reading the chart.</strong> The cost` +
    ` figure here is the value of stock actually sitting on shelves. That means` +
    ` <em>a lower number can signal a failing policy rather than an efficient` +
    ` one</em>. When demand outstrips supply the shelves empty, and empty` +
    ` shelves are cheap to hold.<br><br>This data contains a clear example. In` +
    ` <strong>${w4.quarter} the textbook method produced the lowest cost figure in` +
    ` the entire study, ${USD(w4.cost)} a year, while meeting only` +
    ` ${PCT1(w4.store_fill)} of customer demand</strong> and leaving its warehouses` +
    ` able to fill just ${PCT1(w4.dc_fill)} of store restock orders. That is not a` +
    ` cheap policy; it is an out-of-stock one.<br><br>For the same reason this chart` +
    ` shows <strong>one period at a time</strong>. The identical shared-warehouse` +
    ` method costs ${USD(Math.max(...oc))} a year in the quietest period and` +
    ` ${USD(Math.min(...oc))} in the busiest, a threefold difference that` +
    ` reflects how busy trade was, not how good the policy is.`;
})();

/* ---------- 1. warehouse reliability dumbbell ---------- */
(function () {
  const nv = WINS.map(w => byWin[w].naive.dc_fill * 100);
  const ov = WINS.map(w => byWin[w].optimized.dc_fill * 100);
  const traces = [];
  QLAB.forEach((q, i) => traces.push({
    x: [nv[i], ov[i]], y: [q, q], mode: 'lines', type: 'scatter',
    line: { color: C.grid, width: 4 }, showlegend: false, hoverinfo: 'skip'
  }));
  traces.push({
    x: nv, y: QLAB, mode: 'markers+text', type: 'scatter',
    name: 'Textbook Formula (Naive Baseline)',
    marker: { color: C.rust, size: 15, line: { color: C.plot, width: 2 } },
    text: nv.map(v => v.toFixed(1) + '%'), textposition: 'middle left',
    textfont: { color: C.muted, size: 12 },
    customdata: nv.map(v => 100 - v),
    hovertemplate: '<b>Textbook Formula</b> &middot; %{y}<br>' +
      'The warehouse could fill %{x:.1f}% of what stores asked for<br>' +
      '%{customdata:.1f}% of store restock requests went unfilled<extra></extra>'
  });
  traces.push({
    x: ov, y: QLAB, mode: 'markers+text', type: 'scatter',
    name: 'Shared Warehouse Buffer (Optimized)',
    marker: { color: C.teal, size: 15, line: { color: C.plot, width: 2 } },
    text: ov.map(v => v.toFixed(1) + '%'), textposition: 'middle right',
    textfont: { color: C.muted, size: 12 },
    hovertemplate: '<b>Shared Warehouse Buffer</b> &middot; %{y}<br>' +
      'The warehouse could fill %{x:.1f}% of what stores asked for<br>' +
      'Every store restock request was met in full<extra></extra>'
  });
  const L = LAYOUT();
  L.title.text = 'Share Of Store Restock Orders The Regional Warehouse Could Fill';
  L.xaxis.title.text = '% of store restock orders filled';
  L.xaxis.range = [83.5, 103.5]; L.xaxis.ticksuffix = '%';
  L.height = 400;
  Plotly.newPlot('chartDC', traces, L, CFG);
})();

/* ---------- 2. demand pressure scatter (toggleable) ---------- */
function drawScatter(view) {
  const src = WINS.map(w => byWin[w].naive);
  const x = src.map(r => (r.demand_ratio - 1) * 100);
  const y = src.map(r => r.dc_fill * 100);
  const trace = {
    x, y, mode: 'markers+text', type: 'scatter',
    marker: { color: C.rust, size: 18, line: { color: C.plot, width: 2 } },
    text: src.map(r => r.quarter),
    textposition: src.map(r => r.quarter === 'Q1 2024' ? 'bottom center' : 'top center'),
    textfont: { color: C.ink, size: 12 },
    customdata: src.map(r => [r.quarter, (1 - r.dc_fill) * 100]),
    hovertemplate: '<b>%{customdata[0]}</b><br>' +
      'Demand ran %{x:+.1f}% versus the period the plan was built from<br>' +
      'Warehouses filled %{y:.1f}% of store restock orders<br>' +
      '(%{customdata[1]:.1f% of requests went unfilled)<extra></extra>',
    showlegend: false
  };
  const L = LAYOUT();
  L.title.text = 'Busier Periods vs Warehouse Reliability (Textbook Method)';
  L.xaxis.title.text = 'How much busier the test period was than the plan expected (%)';
  L.yaxis.title.text = '% of store restock orders the warehouse could fill';
  L.xaxis.ticksuffix = '%'; L.xaxis.zeroline = true;
  L.yaxis.ticksuffix = '%'; L.yaxis.range = [85, 104];
  L.height = 450;
  if (view === 'explained') {
    const hardest = src.reduce((a, b) => a.demand_ratio > b.demand_ratio ? a : b);
    const easiest = src.reduce((a, b) => a.demand_ratio < b.demand_ratio ? a : b);
    L.annotations = [
      { x: (hardest.demand_ratio - 1) * 100, y: hardest.dc_fill * 100,
        ax: -120, ay: -55, showarrow: true, arrowhead: 2,
        arrowcolor: C.rust, arrowwidth: 1.5, align: 'left', borderpad: 6,
        bgcolor: 'rgba(255,253,247,0.92)', bordercolor: C.rust, borderwidth: 1,
        font: { color: C.ink, size: 11 },
        text: '<b>Christmas peak</b><br>demand ' +
          Math.round((hardest.demand_ratio - 1) * 100) + '% above plan,<br>warehouses failed ' +
          Math.round((1 - hardest.dc_fill) * 100) + '% of orders' },
      { x: (easiest.demand_ratio - 1) * 100, y: easiest.dc_fill * 100,
        ax: 70, ay: 48, showarrow: true, arrowhead: 2,
        arrowcolor: C.muted, arrowwidth: 1.5, align: 'left', borderpad: 6,
        bgcolor: 'rgba(255,253,247,0.92)', bordercolor: C.grid, borderwidth: 1,
        font: { color: C.ink, size: 11 },
        text: '<b>Quiet quarter</b><br>demand ' +
          Math.abs(Math.round((easiest.demand_ratio - 1) * 100)) +
          '% below plan,<br>everything looks fine' },
      { x: 4.5, y: 92.5, showarrow: false, align: 'left', borderpad: 8,
        bgcolor: 'rgba(243,236,224,0.85)', font: { color: C.terra, size: 12 },
        text: '<b>The pattern:</b> busier period &rarr; worse warehouse reliability.<br>' +
          'Every extra bit of demand makes the textbook plan fail more.' }
    ];
  }
  Plotly.newPlot('chartScatter', [trace], L, CFG);
}
document.querySelectorAll('#viewToggle button').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('#viewToggle button').forEach(o =>
      o.setAttribute('aria-pressed', String(o === b)));
    drawScatter(b.dataset.view);
  });
});
drawScatter('explained');

/* ---------- 3. saving by period ---------- */
(function () {
  const sav = WINS.map(w => D.saving[w] * 100);
  const mean = sav.reduce((a, b) => a + b, 0) / sav.length;
  const abs = WINS.map(w =>
    byWin[w].decentralized_95.cost - byWin[w].optimized.cost);
  const L = LAYOUT();
  L.title.text = 'How Much Cheaper Sharing Backup Stock Was, By Test Period';
  L.yaxis.title.text = 'Money saved (%)';
  L.yaxis.range = [0, 10.5]; L.yaxis.ticksuffix = '%';
  L.height = 400;
  L.shapes = [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: mean, y1: mean,
                line: { color: C.blue, width: 2, dash: 'dash' } }];
  L.annotations = [{ xref: 'paper', x: 0.01, y: mean, yanchor: 'bottom',
    showarrow: false, font: { color: C.blue, size: 12 },
    text: 'average across all four periods: ' + mean.toFixed(1) + '%' }];
  Plotly.newPlot('chartSaving', [{
    x: QLAB, y: sav, type: 'bar',
    marker: { color: C.teal, cornerradius: 4 }, width: 0.5,
    text: sav.map(v => v.toFixed(1) + '%'), textposition: 'outside',
    textfont: { color: C.ink, size: 13 },
    customdata: abs,
    hovertemplate: '<b>%{x}</b><br>Sharing backup stock centrally cost ' +
      '%{y:.1f}% less<br>than giving every store its own, about ' +
      '$%{customdata:,.0f} a year<extra></extra>',
    showlegend: false
  }], L, CFG);
})();

/* ---------- 4. expedite sensitivity ---------- */
const EXP_NAME = d => d === 0 ? 'Same day' : d + ' day' + (d > 1 ? 's' : '');
function drawExpedite(days) {
  const cur = D.expedite.find(e => e.days === days);
  const base = D.expedite.find(e => e.days === 1);
  document.getElementById('expLabel').textContent = EXP_NAME(days);
  const isBase = days === 1;
  const cards = [
    ['% Of Demand Met On Time', PCT(cur.store_fill),
      isBase ? 'the assumption used elsewhere on this page'
             : ((cur.store_fill - base.store_fill) * 100).toFixed(2) +
               '% vs a 1-day assumption'],
    ['Yearly Cost Of Stock Held', USD(cur.cost),
      isBase ? 'the assumption used elsewhere on this page'
             : (((cur.cost - base.cost) / base.cost) * 100).toFixed(1) +
               '% vs a 1-day assumption'],
    ['Warehouse Order Fill', PCT(cur.dc_fill), 'stores got what they asked for']
  ];
  document.getElementById('expMetrics').innerHTML = cards.map(
    ([l, v, s]) => `<div class="metric"><div class="label">${l}</div>` +
      `<div class="value">${v}</div><div class="sub">${s}</div></div>`).join('');

  const L = LAYOUT();
  L.title.text = 'Customer Demand Met On Time, By Emergency Delivery Speed';
  L.yaxis.title.text = '% of demand met on time';
  L.xaxis.title.text = 'Emergency delivery speed';
  /* Zero-based on purpose: bars are read by length, and it also carries the
     message that the four options are indistinguishable. */
  L.yaxis.range = [0, 108]; L.yaxis.ticksuffix = '%';
  L.height = 390;
  Plotly.newPlot('chartExp', [{
    x: D.expedite.map(e => EXP_NAME(e.days)),
    y: D.expedite.map(e => e.store_fill * 100), type: 'bar',
    marker: {
      color: D.expedite.map(e => e.days === days ? C.teal : C.grid),
      cornerradius: 4,
      line: { color: D.expedite.map(e => e.days === days ? C.teal : C.muted), width: 1 }
    },
    width: 0.5,
    text: D.expedite.map(e => (e.store_fill * 100).toFixed(2) + '%'),
    textposition: 'outside', textfont: { color: C.ink, size: 13 },
    customdata: D.expedite.map(e => [e.cost, e.expedited]),
    hovertemplate: '<b>%{x} emergency delivery</b><br>' +
      '%{y:.2f}% of customer demand met on time<br>' +
      'Stock held costs $%{customdata[0]:,.0f a year<br>' +
      '%{customdata[1]:,.0f units moved as emergency deliveries<extra></extra>',
    showlegend: false
  }], L, CFG);
}
(function () {
  const f = D.expedite.map(e => e.store_fill);
  const c = D.expedite.map(e => e.cost);
  const spreadF = (Math.max(...f) - Math.min(...f)) * 100;
  const spreadC = (Math.max(...c) - Math.min(...c)) / Math.min(...c) * 100;
  document.getElementById('expNote').innerHTML =
    `<strong>This barely matters, and that is the interesting part.</strong>` +
    ` Across the whole range from same-day to three-day emergency delivery, the` +
    ` share of demand met on time moves by only` +
    ` <strong>${spreadF.toFixed(2)} percentage points</strong> and the cost of stock` +
    ` held by <strong>${spreadC.toFixed(1)}%</strong>. An earlier, purely theoretical` +
    ` version of this calculation suggested the assumption was critical, it` +
    ` implied the benefit of sharing stock would collapse from 22.8% to 2.2% across` +
    ` this same range. Running the actual simulation showed that was an artefact of` +
    ` the theory, not a real effect. The softest assumption in the model turned out` +
    ` not to be load-bearing, which means the conclusions do not depend on getting` +
    ` it right.`;
})();
document.getElementById('expSlider').addEventListener('input', e =>
  drawExpedite(Number(e.target.value)));
drawExpedite(1);

/* ---------- 5. cost within one period ---------- */
const ORDER = ['naive', 'decentralized_95', 'optimized'];
function drawCost(win) {
  const rows = ORDER.map(p => byWin[win][p]);
  const max = Math.max(...rows.map(r => r.cost));
  const L = LAYOUT();
  L.title.text = 'Yearly Cost Of Stock Held in ' + rows[0].quarter +
    ' (share of demand met shown on each bar)';
  L.yaxis.title.text = 'Cost of stock held ($/year)';
  L.yaxis.range = [0, max * 1.28]; L.yaxis.tickprefix = '$';
  L.height = 430;
  Plotly.newPlot('chartCost', rows.map((r, i) => ({
    x: [D.policy_short[r.policy]], y: [r.cost], type: 'bar',
    marker: { color: [C.rust, C.blue, C.teal][i], cornerradius: 4 },
    width: 0.45, name: D.policy_label[r.policy],
    text: [USD(r.cost) + '<br><span style="font-size:11px">' +
      PCT1(r.store_fill) + ' of demand met</span>'],
    textposition: 'outside', textfont: { color: C.ink, size: 13 },
    hovertemplate: '<b>' + D.policy_label[r.policy] + '</b><br>' +
      'Stock held costs ' + USD(r.cost) + ' a year<br>' +
      'Met ' + PCT(r.store_fill) + ' of customer demand on time<br>' +
      'Warehouses filled ' + PCT(r.dc_fill) + ' of store restock orders<br>' +
      'Worst single store managed ' + PCT(r.worst_store) + '<extra></extra>',
    showlegend: false
  })), L, CFG);
}
(function () {
  const sel = document.getElementById('qSelect');
  sel.innerHTML = WINS.map(w =>
    `<option value="${w}">${byWin[w].naive.quarter} (${byWin[w].naive.dates})</option>`
  ).join('');
  sel.value = WINS[WINS.length - 1];
  sel.addEventListener('change', e => drawCost(e.target.value));
  drawCost(sel.value);
})();

/* ---------- 6. service level by period ---------- */
(function () {
  const L = LAYOUT();
  L.title.text = 'Share Of Customer Demand Met On Time, By Test Period';
  L.yaxis.title.text = '% of demand met on time';
  L.yaxis.range = [0, 105]; L.yaxis.ticksuffix = '%';
  L.bargap = 0.35; L.bargroupgap = 0.08; L.height = 400;
  L.shapes = [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 95, y1: 95,
                line: { color: C.muted, width: 1.5, dash: 'dot' } }];
  L.annotations = [{ xref: 'paper', x: 0.99, y: 95, yanchor: 'bottom',
    showarrow: false, font: { color: C.muted, size: 11 }, text: '95% target' }];
  const traces = ['naive', 'optimized'].map((p, i) => ({
    x: QLAB, y: WINS.map(w => byWin[w][p].store_fill * 100), type: 'bar',
    name: D.policy_label[p],
    marker: { color: [C.rust, C.teal][i], cornerradius: 4 },
    customdata: WINS.map(w => byWin[w][p].worst_store * 100),
    hovertemplate: '<b>' + D.policy_short[p] + '</b> &middot; %{x}<br>' +
      '%{y:.1f}% of all customer demand met on time<br>' +
      'Worst single store managed %{customdata:.1f}%<extra></extra>'
  }));
  Plotly.newPlot('chartService', traces, L, CFG);
})();

/* ---------- network diagram ---------- */
(function () {
  var svg = document.getElementById('net');
  var tip = document.getElementById('tip');
  if (!svg || !tip) return;              // page still fine without the diagram

  var wrap = svg.parentNode;
  var nodes = Array.prototype.slice.call(svg.querySelectorAll('.node'));
  var edges = Array.prototype.slice.call(svg.querySelectorAll('.edge'));

  function clear() {
    svg.classList.remove('is-active');
    nodes.forEach(function (n) { n.classList.remove('hl'); });
    edges.forEach(function (e) { e.classList.remove('hl'); });
    tip.classList.remove('show');
    tip.textContent = '';
  }

  function highlight(node) {
    var id = node.getAttribute('data-node');
    var related = {};
    related[id] = true;

    edges.forEach(function (edge) {
      var pair = edge.getAttribute('data-edge').split('|');
      if (pair[0] === id || pair[1] === id) {
        edge.classList.add('hl');
        related[pair[0]] = true;
        related[pair[1]] = true;
      } else {
        edge.classList.remove('hl');
      }
    });

    nodes.forEach(function (n) {
      n.classList.toggle('hl', !!related[n.getAttribute('data-node')]);
    });
    svg.classList.add('is-active');

    tip.innerHTML = '<b></b><span></span>';
    tip.querySelector('b').textContent = node.getAttribute('data-title');
    tip.querySelector('span').textContent = node.getAttribute('data-meta');

    var box = node.getBoundingClientRect();
    var host = wrap.getBoundingClientRect();
    var x = box.left - host.left + box.width / 2;
    var y = box.top - host.top;

    tip.style.left = '0px';
    tip.style.top = '0px';
    tip.classList.add('show');

    var tw = tip.offsetWidth;
    var pad = 8;
    x = Math.max(tw / 2 + pad, Math.min(x, host.width - tw / 2 - pad));
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }

  nodes.forEach(function (node) {
    node.addEventListener('mouseenter', function () { highlight(node); });
    node.addEventListener('focus', function () { highlight(node); });
    node.addEventListener('blur', clear);
    node.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { clear(); node.blur(); }
    });
  });
  svg.addEventListener('mouseleave', clear);
})();

/* ---------- tables ---------- */
(function () {
  const head = ['Test Period', 'Dates', 'Method', 'Demand vs Plan',
    '% Of Demand Met', 'Worst Single Store', '% Of Store Orders Filled',
    'Yearly Cost', 'Saved vs Own Stock'];
  const rows = [];
  WINS.forEach(w => ORDER.forEach(p => {
    const r = byWin[w][p];
    rows.push([r.quarter, r.dates, D.policy_label[p],
      ((r.demand_ratio - 1) * 100).toFixed(1) + '%',
      PCT(r.store_fill), PCT(r.worst_store), PCT(r.dc_fill), USD(r.cost),
      p === 'optimized' ? PCT(D.saving[w]) : '-']);
  }));
  document.getElementById('mainTable').innerHTML =
    '<thead><tr>' + head.map(h => `<th>${h}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map(r => '<tr>' + r.map(c => `<td>${c}</td>`).join('') + '</tr>').join('') +
    '</tbody>';

  const dh = ['Test Period', 'Method', 'Location', 'Type', 'Region',
    '% Of Demand Met', '% Of Days With No Shortage', 'Avg Stock On Hand', 'Yearly Cost'];
  document.getElementById('detailTable').innerHTML =
    '<thead><tr>' + dh.map(h => `<th>${h}</th>`).join('') + '</tr></thead><tbody>' +
    D.detail.map(r => '<tr>' + [r.window, r.policy, r.node_name, r.node_type,
      r.region, PCT(r.fill_rate), PCT(r.day_service),
      Math.round(r.mean_onhand).toLocaleString('en-US'),
      USD(r.actual_cost_annual)].map(c => `<td>${c}</td>`).join('') + '</tr>').join('') +
    '</tbody>';
})();
</script>
</body>
</html>
"""


def build_html(payload: dict) -> str:
    """Render the page.

    Deliberately NOT an f-string. The template contains thousands of literal
    CSS and JavaScript braces, plus Plotly hover format specs like
    `%{customdata[1]:.1f}`. Inside an f-string every one of those needs
    doubling, and a single missed pair is a SyntaxError at import time (or,
    worse, silently malformed output). Placeholder substitution keeps the
    template byte-for-byte what the browser receives.
    """
    root_vars = "\n".join(f"    --{k}:{v};" for k, v in C.items())
    return (
        TEMPLATE
        .replace("__ROOTVARS__", root_vars)
        .replace("__PLOTLY_VERSION__", PLOTLY_VERSION)
        .replace("__GLOSSARY__", glossary_html())
        .replace("__NETWORK_SVG__", network_svg())
        .replace("__NETWORK_SR__", network_sr_text())
        .replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    )

def main() -> None:
    payload = build_payload()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(payload)
    OUT.write_text(html, encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"Wrote {OUT.relative_to(PROJECT_ROOT)} ({size_kb:.0f} KB)")
    print(f"  {len(payload['summary'])} summary rows, "
          f"{len(payload['detail'])} location rows, "
          f"{len(payload['expedite'])} expedite settings")
    print("  self-contained: data inlined, Plotly from CDN, no server needed")


if __name__ == "__main__":
    main()
