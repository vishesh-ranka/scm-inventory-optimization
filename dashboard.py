"""
Streamlit dashboard for the multi-echelon inventory optimization project.

Run:
    streamlit run dashboard.py

Presents the rolling-window backtest findings in the order the evidence actually
supports them: warehouse (DC) reliability first, because it is stable across
every window tested; the cost saving second and always as a range, because it is
real but swings with demand conditions; and absolute cost only ever one quarter
at a time, because a low cost figure in a busy quarter usually means the policy
ran out of stock rather than saved money.

Written for a reader with no supply-chain or statistics background. Every
technical term is defined in the glossary, every chart carries a plain-language
caption, and every statistic is restated in words next to the number.

PALETTE
-------
Warm cream theme. The series colours were re-validated from scratch against the
new #faf6ee surface in LIGHT mode (the previous dark-theme validation does not
transfer, because the lightness band differs and contrast inverts). Checks run: OKLCH
lightness band, chroma floor, protanopia/deuteranopia separation
(Machado-Oliveira-Fernandes 2009 at severity 1.0) and contrast vs surface, on
the all-pairs pairlist.

    optimized      #2c9a88  teal    L 0.621  C 0.101  contrast 3.20:1
    decentralized  #2961ce  blue    L 0.520  C 0.179  contrast 5.27:1
    naive          #ab4400  rust    L 0.519  C 0.150  contrast 5.47:1

    worst normal-vision dE 21.8 (hard gate >= 15)
    worst CVD dE         14.8 (target >= 8)
    RESULT: PASS

Text colours against cream: charcoal #2b2620 at 13.91:1, terracotta headings
#9a3f1b at 6.28:1, muted ink #6b6155 at 5.62:1, all clear of WCAG AA for body
text, not merely for large text.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"

# --- warm cream theme -----------------------------------------------------
CREAM = "#faf6ee"        # page background
CARD = "#f3ece0"         # raised surface (metric cards, callouts)
PLOT_BG = "#fffdf7"      # chart plotting area, a shade lighter than the page
INK = "#2b2620"          # body text
INK_MUTED = "#6b6155"    # captions, axis labels
TERRACOTTA = "#9a3f1b"   # section headings
GRID = "#e3dccd"         # gridlines, borders

# Validated series palette (see module docstring).
TEAL = "#2c9a88"    # optimized multi-echelon
BLUE = "#2961ce"    # decentralized 95%
RUST = "#ab4400"    # naive baseline

POLICY_COLOR = {"optimized": TEAL, "decentralized_95": BLUE, "naive": RUST}

# Plain-language names. The technical name is kept in parentheses so nothing is
# lost for a reader who already knows the vocabulary.
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

st.set_page_config(
    page_title="Multi-Echelon Inventory Optimization",
    page_icon="📦",
    layout="wide",
)

st.markdown(
    f"""
    <style>
      /* ------------------------------------------------------------------
         Belt and braces. `.streamlit/config.toml` pins the base theme to
         light, which is the real fix for the invisible-text problem. These
         rules repeat the intent at the CSS layer so the page still renders
         correctly if that config is not picked up - running streamlit from a
         different working directory, or a host that supplies its own theme.

         The rule that matters: any surface that can inherit a dark background
         from Streamlit's own theme gets an EXPLICIT light background here, so
         dark text can never land on a dark box.
         ------------------------------------------------------------------ */

      .stApp, .stAppViewContainer, .main, .block-container {{
        background-color: {CREAM}; color: {INK};
      }}
      section[data-testid="stSidebar"] {{ background-color: {CARD}; }}
      section[data-testid="stSidebar"] * {{ color: {INK}; }}

      h1, h2, h3, h4, h5, h6 {{ color: {TERRACOTTA} !important; letter-spacing: .2px; }}
      h2 {{ margin-top: 1.8rem; border-bottom: 2px solid {GRID}; padding-bottom: .4rem; }}
      p, li, label, .stMarkdown, .stMarkdown p, .stMarkdown li {{ color: {INK}; }}
      strong, b {{ color: inherit; }}
      a {{ color: {TERRACOTTA}; }}

      .caption {{ color: {INK_MUTED}; font-size: .9rem; line-height: 1.55; }}
      .chartnote {{
        color: {INK}; font-size: .93rem; line-height: 1.5;
        background-color: {CARD}; border-radius: 6px;
        padding: 10px 14px; margin: 6px 0 12px 0;
      }}
      .callout {{
        background-color: rgba(171, 68, 0, .08);
        border-left: 4px solid {RUST};
        padding: 14px 18px; border-radius: 4px; margin: 10px 0 18px 0;
        color: {INK}; font-size: .93rem; line-height: 1.6;
      }}
      .note {{
        background-color: {CARD}; border-left: 4px solid {TEAL};
        padding: 14px 18px; border-radius: 4px; margin: 10px 0 18px 0;
        color: {INK}; font-size: .93rem; line-height: 1.6;
      }}

      /* Metric cards -------------------------------------------------- */
      div[data-testid="stMetric"] {{
        background-color: {CARD}; border: 1px solid {GRID};
        border-radius: 8px; padding: 14px 16px;
      }}
      div[data-testid="stMetricValue"] {{ color: {INK} !important; font-size: 1.6rem; }}
      div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p {{
        color: {INK_MUTED} !important;
      }}
      /* The "off" (grey) delta inherits a very pale tone in dark base; pin it. */
      div[data-testid="stMetricDelta"] svg {{ vertical-align: middle; }}

      /* Expanders - header bar is the classic dark-on-dark offender ---- */
      div[data-testid="stExpander"] {{
        border: 1px solid {GRID}; border-radius: 6px; background-color: {CARD};
      }}
      div[data-testid="stExpander"] details,
      div[data-testid="stExpander"] summary {{
        background-color: {CARD} !important; color: {INK} !important;
      }}
      div[data-testid="stExpander"] summary p,
      div[data-testid="stExpander"] summary span,
      div[data-testid="stExpander"] summary svg {{
        color: {TERRACOTTA} !important; fill: {TERRACOTTA} !important;
      }}
      div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
        background-color: {CARD} !important; color: {INK} !important;
      }}

      /* Tables --------------------------------------------------------- */
      div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
        background-color: {PLOT_BG}; border: 1px solid {GRID}; border-radius: 6px;
      }}
      div[data-testid="stDataFrame"] * {{ color: {INK}; }}

      /* Form controls: radio, select slider, selectbox ------------------ */
      div[data-testid="stRadio"] label, div[data-testid="stRadio"] p,
      div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label,
      div[data-testid="stSelectSlider"] label {{ color: {INK} !important; }}
      div[data-baseweb="select"] > div {{
        background-color: {PLOT_BG} !important; color: {INK} !important;
        border-color: {GRID} !important;
      }}
      /* Dropdown menus render in a portal at the end of <body>, outside the
         app container, so they never inherit the rules above. Without this the
         open menu is a dark panel with dark text - unreadable. */
      div[data-baseweb="popover"], div[data-baseweb="menu"],
      ul[data-baseweb="menu"], div[data-baseweb="popover"] * {{
        background-color: {PLOT_BG} !important; color: {INK} !important;
      }}
      li[data-baseweb="menu-item"]:hover, div[role="option"]:hover {{
        background-color: {CARD} !important; color: {INK} !important;
      }}
      div[data-testid="stSelectSlider"] div[data-baseweb="slider"] div[role="slider"] {{
        background-color: {TERRACOTTA} !important;
      }}
      /* Slider value bubble and tick labels */
      div[data-testid="stSliderThumbValue"],
      div[data-testid="stTickBar"], div[data-testid="stTickBarMin"],
      div[data-testid="stTickBarMax"] {{ color: {INK_MUTED} !important; }}

      /* Streamlit's own alert boxes, in case any are ever added --------- */
      div[data-testid="stAlert"] {{ background-color: {CARD}; color: {INK}; }}
      div[data-testid="stAlert"] p {{ color: {INK} !important; }}

      /* Tooltips on the `help=` parameters of widgets ------------------- */
      div[data-testid="stTooltipContent"], div[role="tooltip"] {{
        background-color: {PLOT_BG} !important; color: {INK} !important;
        border: 1px solid {GRID} !important;
      }}
      div[data-testid="stTooltipContent"] * {{ color: {INK} !important; }}

      /* Plotly's floating toolbar sits on the transparent chart paper --- */
      .modebar {{ background-color: transparent !important; }}
      .modebar-btn svg {{ fill: {INK_MUTED} !important; }}

      code {{ background-color: {CARD}; color: {TERRACOTTA}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- data -----------------------------------------------------------------


@st.cache_data
def load_rolling() -> pd.DataFrame:
    """Per (window, policy) summary built from the node-level backtest output."""
    raw = pd.read_parquet(DATA / "rolling_validation_results.parquet")
    rows = []
    for (window, policy), block in raw.groupby(["window", "policy"]):
        stores = block[block.node_type == "STORE"]
        dcs = block[block.node_type == "DC"]
        rows.append(
            {
                "window": window,
                "policy": policy,
                "test_span": block.test_span.iloc[0],
                "demand_ratio": block.demand_ratio.iloc[0],
                "store_fill": 1 - stores.unmet_demand.sum() / stores.total_demand.sum(),
                "worst_store": stores.fill_rate.min(),
                "dc_fill": 1 - dcs.unmet_demand.sum() / dcs.total_demand.sum(),
                "cost": block.actual_cost_annual.sum(),
            }
        )
    return pd.DataFrame(rows).sort_values(["window", "policy"])


@st.cache_data
def load_nodes() -> pd.DataFrame:
    return pd.read_parquet(DATA / "rolling_validation_results.parquet")


@st.cache_data
def load_expedite() -> pd.DataFrame:
    """Simulated outcomes at each emergency-delivery speed (0-3 days).

    These come from the held-out simulation in `validate_policy.py`, NOT from
    the earlier analytical sensitivity table, which exaggerated the importance
    of this assumption considerably.
    """
    raw = pd.read_parquet(DATA / "validation_results.parquet")
    rows = []
    for days in (0, 1, 2, 3):
        block = raw[raw.policy == f"optimized_exp{days}"]
        stores = block[block.node_type == "STORE"]
        dcs = block[block.node_type == "DC"]
        rows.append(
            {
                "expedite_days": days,
                "store_fill": 1 - stores.unmet_demand.sum() / stores.total_demand.sum(),
                "worst_store": stores.fill_rate.min(),
                "dc_fill": 1 - dcs.unmet_demand.sum() / dcs.total_demand.sum(),
                "cost": block.actual_cost_annual.sum(),
                "expedited_units": block.expedited_units.sum(),
            }
        )
    return pd.DataFrame(rows)


def quarter_label(span: str) -> str:
    """'2024-01-01 -> 2024-03-31' -> 'Q1 2024'."""
    start = pd.Timestamp(span.split(" -> ")[0])
    return f"Q{start.quarter} {start.year}"


def pretty_span(span: str) -> str:
    """'2024-01-01 -> 2024-03-31' -> 'Jan 1 to Mar 31, 2024'.

    Written month, numeric day, numeric year. The year is printed once when
    both ends share it, which every test period here does, repeating it reads
    as clutter in a table cell.
    """
    start_raw, end_raw = span.split(" -> ")
    start, end = pd.Timestamp(start_raw), pd.Timestamp(end_raw)
    start_txt = f"{start:%b} {start.day}"
    end_txt = f"{end:%b} {end.day}"
    if start.year == end.year:
        return f"{start_txt} – {end_txt}, {end.year}"
    return f"{start_txt}, {start.year} – {end_txt}, {end.year}"


def style_fig(fig: go.Figure, height: int = 400) -> go.Figure:
    """Light chart surface on the cream page, recessive axes, dark ink.

    Margins are generous on purpose: an earlier draft rendered with category
    labels truncated to a single character, tick labels clipped off the bottom,
    and the title sitting on top of the legend.
    """
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        font=dict(color=INK, size=13),
        margin=dict(l=90, r=40, t=95, b=70),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, x=0, xanchor="left",
            bgcolor="rgba(0,0,0,0)", font=dict(color=INK, size=12),
        ),
        hoverlabel=dict(
            bgcolor="#ffffff", font=dict(color=INK, size=13), bordercolor=GRID
        ),
        title=dict(
            font=dict(size=15, color=TERRACOTTA),
            x=0, xref="paper", xanchor="left", y=0.97, yanchor="top",
        ),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                     automargin=True, tickfont=dict(color=INK_MUTED),
                     title_font=dict(color=INK_MUTED), title_standoff=16)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                     automargin=True, tickfont=dict(color=INK_MUTED),
                     title_font=dict(color=INK_MUTED), title_standoff=16)
    return fig


def chart_note(text: str) -> None:
    """The one-line 'what this shows and what to look for' note above a figure."""
    st.markdown(f"<div class='chartnote'>{text}</div>", unsafe_allow_html=True)


def why_it_matters(text: str) -> None:
    """Collapsed business-context note for a non-technical reader."""
    with st.expander("Why does this matter?", expanded=False):
        st.markdown(text)


summary = load_rolling()
summary["quarter"] = summary.test_span.map(quarter_label)
summary["dates"] = summary.test_span.map(pretty_span)
nodes = load_nodes()
expedite = load_expedite()

wide_cost = summary.pivot(index="window", columns="policy", values="cost")
wide_dc = summary.pivot(index="window", columns="policy", values="dc_fill")
saving_pct = (
    (wide_cost["decentralized_95"] - wide_cost["optimized"])
    / wide_cost["decentralized_95"]
)
quarters = summary.drop_duplicates("window").set_index("window")["quarter"]
order = list(quarters.index)
labels = [quarters[w] for w in order]


# --- header ---------------------------------------------------------------

st.title("Multi-Echelon Inventory Optimization")
st.markdown(
    "<p class='caption'>This project asks a simple question: <strong>where should a "
    "retailer keep its backup stock?</strong> The network here has one overseas "
    "factory, three regional warehouses (North America, Europe, Asia-Pacific), and "
    "ten stores. Every number on this page comes from a simulation that was given "
    "only past data to plan with, then tested against a later period it had never "
    "seen, the same way a real policy would have to work.</p>",
    unsafe_allow_html=True,
)

# On the delta colours below: Streamlit paints a delta green or red according to
# its sign, and `delta_color="inverse"` flips that. Only the textbook-method card
# carries a real change figure, and it is bad news, so it takes the default
# ("normal") colouring and renders red. The other three deltas are captions, not
# changes, they take "off" so they stay neutral grey instead of being painted
# green and read as improvements.
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Warehouse Reliability, New Method",
    "100%",
    "held in all 4 test periods",
    delta_color="off",
)
c2.metric(
    "Warehouse Reliability, Textbook Method",
    f"{wide_dc['naive'].min():.1%}",
    f"{wide_dc['naive'].min() - 1:.1%} at its worst",
)
c3.metric(
    "Money Saved (Range)",
    f"{saving_pct.min():.1%}–{saving_pct.max():.1%}",
    f"average {saving_pct.mean():.1%}",
    delta_color="off",
)
c4.metric(
    "Test Periods Run",
    f"{summary.window.nunique()}",
    "3 months each",
    delta_color="off",
)


# --- glossary -------------------------------------------------------------

st.header("Start Here: What The Words Mean")

st.markdown(
    "<p class='caption'>Every technical term used anywhere on this page is defined "
    "below. Nothing further down assumes you have read a supply-chain textbook.</p>",
    unsafe_allow_html=True,
)

with st.expander("Glossary, Plain-Language Definitions", expanded=True):
    g1, g2 = st.columns(2)
    with g1:
        st.markdown(
            """
**Regional warehouse (also called a "DC", short for distribution centre)**: a
large building that receives big shipments from the factory and sends smaller
deliveries out to nearby stores. This network has three, one per region.

**Echelon**: one level in the chain. Here there are two levels that hold
stock: the warehouses, and the stores they supply. "Multi-echelon" just means
planning both levels together instead of separately.

**Lead time**: how long you wait for a delivery after ordering it. Factory to
warehouse is 30 to 45 days (a ship crossing an ocean, plus customs). Warehouse to
store is 2 to 5 days (a truck).

**Safety stock (backup stock)**: extra inventory kept on hand beyond what you
normally expect to sell, so that an unusually busy week doesn't empty the
shelves. Deciding how much to keep, and where, is the whole problem.

**Holding cost**: what it costs to keep one unit sitting on a shelf for a day:
warehouse space, money tied up, insurance, spoilage risk. Warehouse space is
roughly 2.4× cheaper per unit than store space, which is why *where* you keep
backup stock matters financially.

**Service level**: the share of customer demand you manage to supply. The
target throughout this project is 95%.

**Fill rate (% of demand met on time)**: of everything customers actually
asked for, the share they actually got. This is the main scoreboard.
            """
        )
    with g2:
        st.markdown(
            """
**Warehouse order fill rate (% of store restock orders the warehouse could
fill)**, when stores ask their regional warehouse to resupply them, this is
the share of that request the warehouse could actually ship. When it drops,
stores are stranded: they cannot restock no matter how well their own shelves
were planned.

**Variability (standard deviation)**: how much demand bounces around from day
to day. A store selling 100 units every single day and a store averaging 100
but swinging between 20 and 300 need very different amounts of backup stock.

**Pooling**: keeping one shared pile of backup stock centrally instead of
many small piles locally. It works because stores rarely have their busiest
week at the same time, so a shared pile can be sent wherever the shortage
actually happens. Ten separate piles cannot be moved once placed.

**Out-of-sample validation**: testing a plan on data it has never seen. If you
check a plan against the same history you used to build it, it will always look
good; that proves nothing. Every result here is out-of-sample.

**Rolling-window backtest**: repeating that honest test several times. Build
the plan using everything up to a date, test it on the next three months, then
move the date forward and repeat. Four such tests were run here, so we can see
whether a finding is consistent or just luck from one period.

**Correlation**: a single number from −1 to +1 summarising whether two things
move together. Near +1 they rise together; near −1 one rises as the other
falls; near 0 there is no relationship.

**Expedite window (emergency delivery speed)**: how fast a warehouse can rush
an emergency delivery to a nearby store when it is about to run out. The
faster this is, the less backup stock each store needs to hold itself.
            """
        )

st.markdown(
    f"""
    <div class='note'>
    <strong>How this page is ordered.</strong> Findings appear in order of how well
    they hold up under testing, not how impressive they sound. Warehouse reliability
    comes first because it was consistent in all four test periods. The money saved
    comes second and is shown as a range, because it was always positive but its size
    changed a lot depending on how busy the period was. Dollar costs appear last and
    only one period at a time - comparing them across periods is genuinely misleading,
    for a reason explained in that section.
    </div>
    """,
    unsafe_allow_html=True,
)


# --- headline: warehouse reliability --------------------------------------

st.header("Main Finding: Keeping Regional Warehouses In Stock")

chart_note(
    "<strong>What this shows:</strong> for each three-month test period, how much of "
    "what stores asked their regional warehouse for was actually shipped. "
    "<strong>What to look for:</strong> the green dots stay pinned at 100% every "
    "time, while the orange dots slip, and slip furthest in Q4, the busiest quarter."
)

dc_fig = go.Figure()
naive_vals = [wide_dc.loc[w, "naive"] * 100 for w in order]
opt_vals = [wide_dc.loc[w, "optimized"] * 100 for w in order]

# Dumbbell form: values sit between 87% and 100%, where zero-based bars would
# hide the effect entirely and truncated bars would misrepresent it. Dots carry
# no area, so a non-zero axis is honest here.
for lab, nv, ov in zip(labels, naive_vals, opt_vals):
    dc_fig.add_trace(
        go.Scatter(x=[nv, ov], y=[lab, lab], mode="lines",
                   line=dict(color=GRID, width=4),
                   showlegend=False, hoverinfo="skip")
    )
dc_fig.add_trace(
    go.Scatter(
        x=naive_vals, y=labels, mode="markers+text",
        name="Textbook Formula (Naive Baseline)",
        marker=dict(color=RUST, size=15, line=dict(color=PLOT_BG, width=2)),
        text=[f"{v:.1f}%" for v in naive_vals], textposition="middle left",
        textfont=dict(color=INK_MUTED, size=12),
        customdata=[[100 - v] for v in naive_vals],
        hovertemplate=(
            "<b>Textbook Formula</b> · %{y}<br>"
            "The warehouse could fill %{x:.1f}% of what stores asked for<br>"
            "%{customdata[0]:.1f}% of store restock requests went unfilled"
            "<extra></extra>"
        ),
    )
)
dc_fig.add_trace(
    go.Scatter(
        x=opt_vals, y=labels, mode="markers+text",
        name="Shared Warehouse Buffer (Optimized)",
        marker=dict(color=TEAL, size=15, line=dict(color=PLOT_BG, width=2)),
        text=[f"{v:.1f}%" for v in opt_vals], textposition="middle right",
        textfont=dict(color=INK_MUTED, size=12),
        hovertemplate=(
            "<b>Shared Warehouse Buffer</b> · %{y}<br>"
            "The warehouse could fill %{x:.1f}% of what stores asked for<br>"
            "Every store restock request was met in full"
            "<extra></extra>"
        ),
    )
)
dc_fig.update_layout(
    title="Share Of Store Restock Orders The Regional Warehouse Could Fill",
    xaxis_title="% of store restock orders filled",
)
dc_fig.update_xaxes(range=[83.5, 103.5], ticksuffix="%")
st.plotly_chart(style_fig(dc_fig, 420), width="stretch", theme=None)

st.markdown(
    f"""
    <p class='caption'>
    The new method keeps every regional warehouse able to fill <strong>100%</strong>
    of store restock orders in all four test periods. The textbook method looks
    perfectly healthy when trade is calm - it also hits 100% in Q2 2024, the one
    quarter where demand came in <em>below</em> what it had planned for - but falls to
    <strong>{wide_dc['naive'].min():.1%}</strong> in Q4 2024, the Christmas peak.
    That is the whole point: it fails exactly when it is needed most, and looks fine
    the rest of the year.
    </p>
    """,
    unsafe_allow_html=True,
)

why_it_matters(
    """
When a regional warehouse runs empty, the damage is not limited to one shop. Every
store that depends on that warehouse is stranded at once, they place their restock
orders as normal and simply do not receive the goods, however well their own shelves
were planned. A single warehouse shortage in this network therefore turns into
simultaneous empty shelves across three or four stores, during the busiest trading
weeks of the year, which is precisely when lost sales are most expensive and most
visible to customers.

This is also the failure that is hardest to see coming. The textbook method reports
healthy numbers for three quarters out of four. Anyone reviewing it in a calm period
would reasonably conclude it was working fine.
    """
)


# --- demand pressure ------------------------------------------------------

st.header("Why The Textbook Method Breaks Down Under Pressure")

corr = summary[summary.policy == "naive"].demand_ratio.corr(
    summary[summary.policy == "naive"].dc_fill
)

view = st.radio(
    "Choose how much explanation to show on the chart",
    options=["Show Explained", "Show Raw Data"],
    horizontal=True,
    help=(
        "‘Show Explained’ writes the interpretation directly onto the chart. "
        "‘Show Raw Data’ strips it back to just the four measured points."
    ),
)

chart_note(
    "<strong>What this shows:</strong> each dot is one three-month test period. "
    "Left-to-right is how much busier (or quieter) that period turned out to be than "
    "the history the plan was built from. Up-and-down is how well the textbook "
    "method's warehouses coped. <strong>What to look for:</strong> the dots fall as "
    "you move right, the busier it got, the worse the warehouses performed."
)

scatter_src = summary[summary.policy == "naive"].copy()
scatter_fig = go.Figure()
scatter_fig.add_trace(
    go.Scatter(
        x=(scatter_src.demand_ratio - 1) * 100,
        y=scatter_src.dc_fill * 100,
        mode="markers+text",
        marker=dict(color=RUST, size=18, line=dict(color=PLOT_BG, width=2)),
        text=scatter_src.quarter,
        # Q1 and Q3 sit almost on top of each other (+2.5%/98.8% vs +2.9%/99.2%),
        # so their labels go to opposite sides instead of both defaulting above
        # the marker, where they overlapped.
        textposition=[
            "bottom center" if q == "Q1 2024" else "top center"
            for q in scatter_src.quarter
        ],
        textfont=dict(color=INK, size=12),
        name="Textbook Formula",
        customdata=scatter_src[["quarter"]].assign(
            gap=(1 - scatter_src.dc_fill) * 100
        ).values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Demand ran %{x:+.1f}% versus the period the plan was built from<br>"
            "Warehouses filled %{y:.1f}% of store restock orders<br>"
            "(%{customdata[1]:.1f}% of requests went unfilled)"
            "<extra></extra>"
        ),
        showlegend=False,
    )
)

if view == "Show Explained":
    scatter_fig.add_annotation(
        # Up and to the left: placed below the point, this box sat on top of the
        # x-axis tick labels in the bottom-right corner.
        x=21.4, y=87.7, ax=-120, ay=-55, xref="x", yref="y",
        text=(
            "<b>Christmas peak</b><br>demand 21% above plan,<br>"
            "warehouses failed 12% of orders"
        ),
        showarrow=True, arrowhead=2, arrowcolor=RUST, arrowwidth=1.5,
        font=dict(color=INK, size=11), align="left",
        bgcolor="rgba(255,253,247,0.92)", bordercolor=RUST, borderwidth=1,
        borderpad=6,
    )
    scatter_fig.add_annotation(
        x=-12.0, y=100, ax=70, ay=48, xref="x", yref="y",
        text=(
            "<b>Quiet quarter</b><br>demand 12% below plan,<br>"
            "everything looks fine"
        ),
        showarrow=True, arrowhead=2, arrowcolor=INK_MUTED, arrowwidth=1.5,
        font=dict(color=INK, size=11), align="left",
        bgcolor="rgba(255,253,247,0.92)", bordercolor=GRID, borderwidth=1,
        borderpad=6,
    )
    scatter_fig.add_annotation(
        x=4.5, y=92.5, xref="x", yref="y", showarrow=False,
        text=(
            "<b>The pattern:</b> busier period → worse warehouse reliability.<br>"
            "Every extra bit of demand makes the textbook plan fail more."
        ),
        font=dict(color=TERRACOTTA, size=12), align="left",
        bgcolor="rgba(243,236,224,0.85)", borderpad=8,
    )

scatter_fig.update_layout(
    title="Busier Periods vs Warehouse Reliability (Textbook Method)",
    xaxis_title="How much busier the test period was than the plan expected (%)",
    yaxis_title="% of store restock orders the warehouse could fill",
)
scatter_fig.update_xaxes(ticksuffix="%", zeroline=True, zerolinewidth=1)
scatter_fig.update_yaxes(ticksuffix="%", range=[85, 104])
st.plotly_chart(style_fig(scatter_fig, 470), width="stretch", theme=None)

st.markdown(
    f"""
    <p class='caption'>
    The statistical summary of that pattern is a correlation of
    <strong>{corr:+.2f}</strong> - which in plain terms means:
    <strong>as demand rises above what the plan plans for, the textbook method's
    warehouse reliability drops, consistently and steeply.</strong> A correlation of
    −1.00 would be a perfect straight-line relationship, so −0.90 is a strong one.
    With only four test periods this is an indication rather than proof, but it
    matches the mechanism exactly.
    </p>
    <p class='caption'>
    <strong>Why it happens.</strong> The textbook method sizes each warehouse's backup
    stock with a standard formula that assumes each day's demand is unrelated to the
    last. Real demand is not like that: it has a weekly rhythm, a Christmas season,
    and a growth trend, so busy days arrive in clusters. Across a 30–45 day sea
    crossing those clusters pile up, and the true swing in demand turns out to be
    between <strong>2.8× and 10.6×</strong> larger than the formula assumes. The
    formula therefore buys a backup stock sized for a calm world. The new method sizes
    the same backup stock from what demand actually did over that same 30–45 day
    window, so the clustering is already accounted for.
    </p>
    """,
    unsafe_allow_html=True,
)

why_it_matters(
    """
This is a warning about how inventory plans are usually reviewed. The textbook formula
is not broken in an obvious way, it is the standard approach taught everywhere, and it
performs perfectly well for most of the year. Its weakness only appears under demand
pressure, which is the one condition where being wrong is expensive.

The practical implication is that a policy should never be judged on an average or a
quiet period. It should be judged on its worst quarter, because that is where the lost
sales, the emergency freight bills, and the disappointed customers actually occur. Any
review process that looks at annual averages will systematically miss this.
    """
)


# --- cost saving range ----------------------------------------------------

st.header("Money Saved: A Range, Not A Single Number")

chart_note(
    "<strong>What this shows:</strong> how much cheaper the shared-warehouse method "
    "was than giving every store its own independent backup stock, in each test "
    "period. <strong>What to look for:</strong> every bar is above zero, it saved "
    "money every time, but the bars are very different heights, so the size of the "
    "saving is not something you can promise in advance."
)

saving_fig = go.Figure()
saving_fig.add_trace(
    go.Bar(
        x=labels, y=saving_pct.values * 100,
        marker=dict(color=TEAL, cornerradius=4),
        width=0.5,
        text=[f"{v:.1f}%" for v in saving_pct.values * 100],
        textposition="outside", textfont=dict(color=INK, size=13),
        name="Saving vs giving every store its own stock",
        customdata=[
            [wide_cost.loc[w, "decentralized_95"] - wide_cost.loc[w, "optimized"]]
            for w in order
        ],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Sharing backup stock centrally cost %{y:.1f}% less<br>"
            "than giving every store its own, about $%{customdata[0]:,.0f} a year"
            "<extra></extra>"
        ),
        showlegend=False,
    )
)
saving_fig.add_hline(
    y=saving_pct.mean() * 100,
    line=dict(color=BLUE, width=2, dash="dash"),
    annotation_text=f"average across all four periods: {saving_pct.mean():.1%}",
    annotation_position="top left",
    annotation_font=dict(color=BLUE, size=12),
)
saving_fig.update_layout(
    title="How Much Cheaper Sharing Backup Stock Was, By Test Period",
    yaxis_title="Money saved (%)",
)
saving_fig.update_yaxes(range=[0, 10.5], ticksuffix="%")
st.plotly_chart(style_fig(saving_fig, 420), width="stretch", theme=None)

st.markdown(
    f"""
    <p class='caption'>
    Sharing backup stock centrally was cheaper in <strong>every single test
    period</strong>, so the direction of this finding is solid. The size is not: it
    ranged from <strong>{saving_pct.min():.1%}</strong> to
    <strong>{saving_pct.max():.1%}</strong>, averaging
    <strong>{saving_pct.mean():.1%}</strong>. An earlier version of this analysis, based
    on a single test period, reported 2.7% - which turned out to be one draw from a
    wide spread rather than a dependable figure. Quote the range, not a single number.
    The saving is biggest in the busiest periods, because that is when being able to
    send stock wherever the shortage actually is beats having it pre-committed to ten
    separate shops.
    </p>
    """,
    unsafe_allow_html=True,
)

why_it_matters(
    """
A business case built on "this saves 2.7%" would be on shaky ground: in a quiet quarter
the real figure is nearer 1.5%, and anyone checking the books three months later would
find the promised saving missing. A business case built on "this saves between 1.5% and
8.3%, averaging under 4%, and it holds our warehouses at full reliability even at
Christmas" is both more honest and more persuasive, because the reliability part is the
finding that actually replicated every time.

It is also worth being clear that this is a modest saving on its own. The reason to
adopt the approach is that shelves stay stocked in peak season; the cost reduction is a
bonus, not the headline.
    """
)


# --- expedite interactivity ----------------------------------------------

st.header("Try It Yourself: How Fast Must Emergency Deliveries Be?")

st.markdown(
    "<p class='caption'>The shared-warehouse method assumes a regional warehouse can "
    "rush an emergency delivery to a nearby store quickly. The faster that is, the "
    "less backup stock each store has to hold itself. This was the single softest "
    "assumption in the whole model, so it was tested directly. Move the selector to "
    "see what the simulation actually produced at each speed.</p>",
    unsafe_allow_html=True,
)

exp_days = st.select_slider(
    "How many days does an emergency delivery from warehouse to store take?",
    options=[0, 1, 2, 3],
    value=1,
    format_func=lambda d: "Same day" if d == 0 else f"{d} day" + ("s" if d > 1 else ""),
    help=(
        "These are real simulated outcomes from the held-out test period, not "
        "theoretical estimates."
    ),
)

row = expedite[expedite.expedite_days == exp_days].iloc[0]
base = expedite[expedite.expedite_days == 1].iloc[0]

is_baseline = exp_days == 1

e1, e2, e3 = st.columns(3)
e1.metric(
    "% Of Demand Met On Time",
    f"{row.store_fill:.2%}",
    # At 1 day the comparison is against itself, so it would always read
    # "+0.00%" - an arrow pointing at nothing. Say so instead.
    "the assumption used elsewhere on this page" if is_baseline
    else f"{row.store_fill - base.store_fill:+.2%} vs a 1-day assumption",
    delta_color="off" if is_baseline else "normal",
)
e2.metric(
    "Yearly Cost Of Stock Held",
    f"${row.cost:,.0f}",
    "the assumption used elsewhere on this page" if is_baseline
    else f"{(row.cost - base.cost) / base.cost:+.1%} vs a 1-day assumption",
    # Cost going up is bad news, so this one is inverted deliberately.
    delta_color="off" if is_baseline else "inverse",
)
e3.metric(
    "Warehouse Order Fill",
    f"{row.dc_fill:.2%}",
    "stores got what they asked for",
    delta_color="off",
)

chart_note(
    "<strong>What this shows:</strong> the simulated result at all four emergency "
    "delivery speeds, with your current choice highlighted. <strong>What to look "
    "for:</strong> the bars are almost the same height, this assumption barely "
    "changes the outcome."
)

exp_fig = go.Figure()
sel_labels = [
    "Same day" if d == 0 else f"{d} day" + ("s" if d > 1 else "")
    for d in expedite.expedite_days
]
exp_fig.add_trace(
    go.Bar(
        x=sel_labels,
        y=expedite.store_fill * 100,
        marker=dict(
            color=[TEAL if d == exp_days else GRID for d in expedite.expedite_days],
            cornerradius=4,
            line=dict(
                color=[TEAL if d == exp_days else INK_MUTED
                       for d in expedite.expedite_days],
                width=1,
            ),
        ),
        width=0.5,
        text=[f"{v:.2f}%" for v in expedite.store_fill * 100],
        textposition="outside", textfont=dict(color=INK, size=13),
        customdata=expedite[["cost", "expedited_units"]].values,
        hovertemplate=(
            "<b>%{x} emergency delivery</b><br>"
            "%{y:.2f}% of customer demand met on time<br>"
            "Stock held costs $%{customdata[0]:,.0f} a year<br>"
            "%{customdata[1]:,.0f} units moved as emergency deliveries"
            "<extra></extra>"
        ),
        showlegend=False,
    )
)
exp_fig.update_layout(
    title="Customer Demand Met On Time, By Emergency Delivery Speed",
    yaxis_title="% of demand met on time",
    xaxis_title="Emergency delivery speed",
)
# Zero-based on purpose. Bars must start at zero to be read by length, and here
# it also carries the message: the four options are indistinguishable. The exact
# figures are in the metric row above for anyone who needs the decimals.
exp_fig.update_yaxes(range=[0, 108], ticksuffix="%")
st.plotly_chart(style_fig(exp_fig, 400), width="stretch", theme=None)

spread_fill = (expedite.store_fill.max() - expedite.store_fill.min()) * 100
spread_cost = (expedite.cost.max() - expedite.cost.min()) / expedite.cost.min()

st.markdown(
    f"""
    <div class='note'>
    <strong>This barely matters - and that is the interesting part.</strong> Across the
    whole range from same-day to three-day emergency delivery, the share of demand met
    on time moves by only <strong>{spread_fill:.2f} percentage points</strong> and the
    cost of stock held by <strong>{spread_cost:.1%}</strong>. An earlier, purely
    theoretical version of this calculation suggested the assumption was critical -
    it implied the benefit of sharing stock would collapse from 22.8% to 2.2% across
    this same range. Running the actual simulation showed that was an artefact of the
    theory, not a real effect. In practice the outcome is governed by how much stock
    is held and whether the warehouse can supply at all; a slow emergency option
    simply goes unused. The softest assumption in the model turned out not to be
    load-bearing, which means the conclusions do not depend on getting it right.
    </div>
    """,
    unsafe_allow_html=True,
)

why_it_matters(
    """
Models usually rest on a handful of assumptions nobody can verify, and the honest
question is always "how much would it matter if this were wrong?" Here that question
has an answer, and the answer is reassuring: even if the assumed emergency delivery
speed is off by a factor of three, the recommendation does not change.

That is worth knowing before committing money to a change. It means the case does not
depend on negotiating a fast internal courier arrangement, and it removes an obvious
line of objection from anyone reviewing the proposal.
    """
)


# --- absolute cost, one period at a time ---------------------------------

st.header("Cost In Dollars, One Period At A Time")

st.markdown(
    f"""
    <div class='callout'>
    <strong>⚠ Please read this before reading the chart.</strong> The cost figure here
    is the value of stock actually sitting on shelves. That means <em>a lower number
    can signal a failing policy rather than an efficient one</em> - when demand
    outstrips supply the shelves empty, and empty shelves are cheap to hold.
    <br><br>
    This data contains a clear example. In <strong>Q4 2024 the textbook method
    produced the lowest cost figure in the entire study,
    ${wide_cost.loc['W4', 'naive']:,.0f} a year - while meeting only
    {summary[(summary.window == 'W4') & (summary.policy == 'naive')].store_fill.iloc[0]:.1%}
    of customer demand</strong> and leaving its warehouses able to fill just
    {wide_dc.loc['W4', 'naive']:.1%} of store restock orders. That is not a cheap
    policy; it is an out-of-stock one.
    <br><br>
    For the same reason this chart shows <strong>one period at a time</strong>. The
    identical shared-warehouse method costs ${wide_cost['optimized'].max():,.0f} a year
    in the quietest period and ${wide_cost['optimized'].min():,.0f} in the busiest - a
    threefold difference that reflects how busy trade was, not how good the policy is.
    Comparing dollar figures between periods tells you nothing useful, so the page
    does not let you do it.
    </div>
    """,
    unsafe_allow_html=True,
)

window_choice = st.selectbox(
    "Choose a three-month test period to compare the three methods within it",
    options=order,
    format_func=lambda w: (
        f"{quarters[w]}  ({summary.loc[summary.window == w, 'dates'].iloc[0]})"
    ),
    index=3,
)

chart_note(
    "<strong>What this shows:</strong> the yearly cost of stock held under each of "
    "the three methods, within the single period you selected. The share of customer "
    "demand each one met is printed on its bar. <strong>What to look for:</strong> "
    "read the cost and the demand-met figure together, never the cost on its own."
)

block = summary[summary.window == window_choice].set_index("policy")
cost_fig = go.Figure()
for policy in ["naive", "decentralized_95", "optimized"]:
    prow = block.loc[policy]
    cost_fig.add_trace(
        go.Bar(
            x=[POLICY_SHORT[policy]], y=[prow.cost],
            marker=dict(color=POLICY_COLOR[policy], cornerradius=4),
            width=0.45, name=POLICY_LABEL[policy],
            text=[f"${prow.cost:,.0f}<br><span style='font-size:11px'>"
                  f"{prow.store_fill:.1%} of demand met</span>"],
            textposition="outside", textfont=dict(color=INK, size=13),
            hovertemplate=(
                f"<b>{POLICY_LABEL[policy]}</b><br>"
                f"Stock held costs ${prow.cost:,.0f} a year<br>"
                f"Met {prow.store_fill:.1%} of customer demand on time<br>"
                f"Warehouses filled {prow.dc_fill:.1%} of store restock orders<br>"
                f"Worst single store managed {prow.worst_store:.1%}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )
cost_fig.update_layout(
    title=(
        f"Yearly Cost Of Stock Held - {quarters[window_choice]} "
        f"(share of demand met shown on each bar)"
    ),
    yaxis_title="Cost of stock held ($/year)",
)
cost_fig.update_yaxes(range=[0, block.cost.max() * 1.28], tickprefix="$")
st.plotly_chart(style_fig(cost_fig, 450), width="stretch", theme=None)

why_it_matters(
    """
This is the most common way an inventory review goes wrong. Cost is easy to measure and
lands straight on a financial report; stockouts are diffuse, show up as lost sales
nobody logged, and rarely get attributed back to an inventory decision. So the policy
that quietly runs out of stock can look like the responsible, frugal choice on paper.

Putting the service figure directly on the cost bar is a deliberate guard against that.
Any time someone presents an inventory saving, the first question should be what
happened to availability over the same period.
    """
)


# --- service level --------------------------------------------------------

st.header("Share Of Customer Demand Met, By Period")

chart_note(
    "<strong>What this shows:</strong> of everything customers asked for across all "
    "ten stores, how much they actually got, under each method. The dotted line is "
    "the 95% target. <strong>What to look for:</strong> the two methods are close for "
    "most of the year, then separate in Q4, the textbook method drops well below "
    "target while the new method stays near it."
)

svc_fig = go.Figure()
for policy in ["naive", "optimized"]:
    vals = [
        summary[(summary.window == w) & (summary.policy == policy)].store_fill.iloc[0] * 100
        for w in order
    ]
    worst = [
        summary[(summary.window == w) & (summary.policy == policy)].worst_store.iloc[0] * 100
        for w in order
    ]
    svc_fig.add_trace(
        go.Bar(
            x=labels, y=vals, name=POLICY_LABEL[policy],
            marker=dict(color=POLICY_COLOR[policy], cornerradius=4),
            customdata=[[w] for w in worst],
            hovertemplate=(
                f"<b>{POLICY_SHORT[policy]}</b> · %{{x}}<br>"
                "%{y:.1f}% of all customer demand met on time<br>"
                "Worst single store managed %{customdata[0]:.1f}%"
                "<extra></extra>"
            ),
        )
    )
svc_fig.add_hline(
    y=95, line=dict(color=INK_MUTED, width=1.5, dash="dot"),
    annotation_text="95% target", annotation_position="top right",
    annotation_font=dict(color=INK_MUTED, size=11),
)
svc_fig.update_layout(
    title="Share Of Customer Demand Met On Time, By Test Period",
    yaxis_title="% of demand met on time",
    bargap=0.35, bargroupgap=0.08,
)
svc_fig.update_yaxes(range=[0, 105], ticksuffix="%")
st.plotly_chart(style_fig(svc_fig, 420), width="stretch", theme=None)

st.markdown(
    "<p class='caption'>These bars start at zero, so the differences look modest, and "
    "they genuinely are modest in ordinary quarters. The gap opens up in Q4. The "
    "larger operational difference between the two methods is in the warehouses, shown "
    "at the top of this page.</p>",
    unsafe_allow_html=True,
)

why_it_matters(
    """
A few percentage points of demand met sounds small, but in retail it is the difference
between a customer finding what they came for and walking out. Across ten stores in a
peak quarter, the gap shown here for Q4 represents thousands of units of demand that
arrived and could not be served.

It is also worth noting what a 95% target really means: even when a policy is working
exactly as designed, one customer in twenty does not get what they wanted. Choosing that
number is a commercial decision about how much stock the business is willing to fund,
not a technical one.
    """
)


# --- tables ---------------------------------------------------------------

st.header("All The Numbers")

chart_note(
    "<strong>What this shows:</strong> every figure behind the charts above, one row "
    "per method per test period. <strong>What to look for:</strong> the "
    "‘Demand vs Plan’ column explains most of the variation, the higher it is, the "
    "worse every method performs."
)

table = summary.copy()
table["saving"] = table.apply(
    lambda r: (
        (wide_cost.loc[r.window, "decentralized_95"] - r.cost)
        / wide_cost.loc[r.window, "decentralized_95"]
        if r.policy == "optimized" else float("nan")
    ),
    axis=1,
)
display = table[
    ["quarter", "dates", "policy", "demand_ratio", "store_fill",
     "worst_store", "dc_fill", "cost", "saving"]
].copy()
display["policy"] = display["policy"].map(POLICY_LABEL)
display["demand_ratio"] = display["demand_ratio"] - 1
display = display.rename(
    columns={
        "quarter": "Test Period",
        "dates": "Dates",
        "policy": "Method",
        "demand_ratio": "Demand vs Plan",
        "store_fill": "% Of Demand Met",
        "worst_store": "Worst Single Store",
        "dc_fill": "% Of Store Orders Warehouse Filled",
        "cost": "Yearly Cost Of Stock Held",
        "saving": "Saved vs Every Store On Its Own",
    }
)
st.dataframe(
    display.style.format(
        {
            "Demand vs Plan": "{:+.1%}",
            "% Of Demand Met": "{:.2%}",
            "Worst Single Store": "{:.2%}",
            "% Of Store Orders Warehouse Filled": "{:.2%}",
            "Yearly Cost Of Stock Held": "${:,.0f}",
            "Saved vs Every Store On Its Own": "{:.2%}",
        },
        na_rep="-",
    ),
    width="stretch",
    hide_index=True,
)

with st.expander("Store-By-Store And Warehouse-By-Warehouse Detail"):
    st.markdown(
        "<p class='caption'>One row per location per method per test period. "
        "‘% of demand met’ is that single location's own result; ‘average stock on "
        "hand’ is how many units it typically had sitting there.</p>",
        unsafe_allow_html=True,
    )
    detail = nodes[
        ["window", "policy", "node_name", "node_type", "region",
         "fill_rate", "day_service", "mean_onhand", "actual_cost_annual"]
    ].copy()
    detail["policy"] = detail["policy"].map(POLICY_SHORT).fillna(detail["policy"])
    detail["node_type"] = detail["node_type"].map(
        {"STORE": "Store", "DC": "Regional Warehouse"}
    )
    detail = detail.rename(
        columns={
            "window": "Test Period", "policy": "Method", "node_name": "Location",
            "node_type": "Type", "region": "Region", "fill_rate": "% Of Demand Met",
            "day_service": "% Of Days With No Shortage",
            "mean_onhand": "Average Stock On Hand (Units)",
            "actual_cost_annual": "Yearly Cost Of Stock Held",
        }
    )
    st.dataframe(
        detail.style.format(
            {
                "% Of Demand Met": "{:.2%}",
                "% Of Days With No Shortage": "{:.2%}",
                "Average Stock On Hand (Units)": "{:,.0f}",
                "Yearly Cost Of Stock Held": "${:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )


# --- method ---------------------------------------------------------------

st.header("How This Was Worked Out, And What It Cannot Tell You")

with st.expander("Method And Known Limitations"):
    st.markdown(
        """
**The test.** Four separate tests were run. Each one builds the stocking plan using
only the history up to a cut-off date, then simulates the following three months day
by day, stores selling, ordering from their warehouse, warehouses ordering from the
factory, deliveries arriving after realistic delays. Demand the shops could not
supply is treated as a lost sale, not a delayed one, which is how retail actually
works. The simulation runs for two months before measurement starts, so it is not
being judged on its opening conditions.

**Why the standard formula was replaced.** The textbook approach understates how much
demand really swings over a 30 to 45 day sea crossing, by between 2.8× and 10.6× at the
warehouses, because it assumes each day is unrelated to the one before. Every backup
stock figure here is instead taken from what demand actually did over windows of that
same length in the real history.

**Where the saving comes from.** The stores do not all get busy at the same time -
Sydney peaks mid-year while Tokyo and Jakarta peak in December. That means one shared
pile of backup stock at the warehouse only needs 65 to 75% of what ten separate piles
would. This was measured by adding up genuine simultaneous shortfalls in the history,
not assumed.

**What this cannot tell you.**

- The demand history is synthetic, generated data, not a real retailer's sales. The
  structure of the findings is sound, but the specific dollar figures describe this
  simulated network only.
- Only four test periods, drawn from two years of history. The ranges shown indicate
  spread; they are not statistical confidence intervals.
- The planner chooses one stocking level per region rather than per individual store,
  a simplification made so the shared buffer could be calculated.
- Emergency delivery speed is assumed rather than measured, though as the interactive
  section above shows, the results barely move across the plausible range.
- Every result assumes the factory itself never runs short.
        """
    )

st.markdown(
    "<p class='caption' style='margin-top:2rem'>Source: rolling-window backtest of "
    "4 test periods × 3 methods × 13 locations, produced by "
    "<code>src/rolling_validation.py</code>, with the emergency-delivery comparison "
    "from <code>src/validate_policy.py</code>.</p>",
    unsafe_allow_html=True,
)
