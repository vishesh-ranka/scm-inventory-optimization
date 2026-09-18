# Multi-Echelon Inventory Optimization

### ▶ **[Open the live dashboard](https://vishesh-ranka.github.io/scm-inventory-optimization/)**

*Interactive, no install - charts, glossary and all three controls run in your browser.*

---

Deciding **where a retailer should keep its backup stock** - centrally in a few regional warehouses, or locally in each shop - and then testing that decision honestly, against periods the plan was never allowed to see.

A simulated network with two stocking levels:

- **1 overseas factory**, shipping by sea
- **3 regional warehouses** - North America, Europe, Asia-Pacific
- **10 retail stores**, 3–4 per region
- **730 days** of daily demand per store (2 years, fixed random seed)
- **Lead times:** 30–45 days factory → warehouse (ocean freight + customs), 2–5 days warehouse → store
- **4 rolling backtests**, each planning on past data only and tested on the next 3 months

**Stack:** PuLP (MILP) · HiGHS solver · NumPy/pandas · SciPy · Streamlit · Plotly · Parquet

---

## Contents

- [Glossary](#glossary)
- [The problem](#the-problem)
- [What the pipeline does](#what-the-pipeline-does)
- [Results](#results)
- [Why this was hard](#why-this-was-hard)
- [What this cannot tell you](#what-this-cannot-tell-you)
- [Reproducing](#reproducing)
- [Repository layout](#repository-layout)

---

## Glossary

*Seven terms used throughout this page. Each is defined once here and then used plainly.*

| Term | What it means |
|---|---|
| **Regional warehouse (DC)** | A building that receives big sea shipments from the factory and sends smaller deliveries to nearby stores. There are three, one per region. |
| **Safety stock** | Extra inventory kept beyond normal expected sales, so an unusually busy week does not empty the shelves. Deciding how much, and where, is the whole problem. |
| **Lead time** | How long you wait for a delivery after ordering it. |
| **Fill rate** | Of everything customers asked for, the share they actually got. The main scoreboard. |
| **Warehouse order fill rate** | When stores ask their warehouse to restock them, the share of that request the warehouse could actually ship. When it drops, stores are stranded regardless of their own planning. |
| **Pooling** | Keeping one shared pile of backup stock centrally instead of ten small piles locally. It works because stores rarely peak at the same time, so shared stock can be sent wherever the shortage lands. |
| **Out-of-sample test** | Testing a plan on data it has never seen. Checking a plan against the history used to build it will always look good and proves nothing. |

---

## The problem

Warehouse space is roughly **2.4× cheaper per unit** than shop floor space, so keeping backup stock centrally is tempting. But central stock is far away: a shop that runs out cannot be helped in time if the warehouse is a week's drive away.

The standard textbook answer sizes every location independently with one formula:

```
safety stock = z × σ × √(lead time)
```

where `σ` is how much daily demand bounces around. It is taught everywhere, it is one line of code, and on this network **it quietly fails** - for a reason that only becomes visible once you measure it.

That formula assumes each day's demand is unrelated to the day before. Real demand is not like that. It has a weekly rhythm, an annual season, and a growth trend, so busy days **arrive in clusters**. Over a 2-day drive from warehouse to store, that clustering barely matters. Over a **30–45 day sea crossing**, it compounds badly.

---

## What the pipeline does

*Five steps, in the order they run. Each writes a file the next one reads.*

| Step | Script | What it produces |
|---|---|---|
| **1. Network** | `src/network_config.py` | The structure as plain data: which store belongs to which warehouse, every lead time, every holding cost |
| **2. Demand** | `src/generate_demand.py` | 2 years of daily demand for 10 stores, deliberately non-uniform - some high-volume and steady, some low-volume and lumpy with genuine zero-sale days |
| **3. Baseline** | `src/naive_baseline.py` | The textbook policy, plus an audit of every assumption it makes and where each one breaks |
| **4. Optimization** | `src/optimize_network.py` | A two-stage mixed-integer program (PuLP/HiGHS) that chooses how much stock to hold centrally versus locally |
| **5. Validation** | `src/validate_policy.py`, `src/rolling_validation.py` | Day-by-day simulation of each policy against held-out periods it never saw |

`dashboard.py` presents the findings as an interactive Streamlit page.

Crucially, **no safety stock figure in this project comes from the `σ × √L` formula.** Every one is taken from the observed distribution of demand over rolling windows the length of the actual lead time, so the clustering is priced in rather than assumed away.

---

## Results

### Finding 1 - The textbook formula understates real variability by up to 10.6×

Measuring what demand actually did over each location's real lead-time window, against what the formula predicts:

| Location type | Lead time | How much the formula understates true variability |
|---|---|---:|
| **Stores** | 2–5 days | 1.0× – 1.7× |
| **Singapore warehouse** | 30 days | **2.8×** |
| **Los Angeles warehouse** | 35 days | **8.3×** |
| **Rotterdam warehouse** | 45 days | **10.6×** |

The pattern is the point: the error scales with lead time, so it is worst at exactly the locations carrying a month or more of ocean freight. **It is also worst at the locations that look safest** - the warehouses serve pooled demand and have the *steadiest* sales in the network, which is precisely why nobody checks them.

This was not the result I went looking for. The original brief expected the formula to struggle at the lumpy, low-volume stores (Phoenix, Jakarta, Milan). Those stores *are* flagged - 5 of 13 locations fail a normality check, with demand skew up to 2.6 and genuine zero-sale days - but their sizing error is 15–28%, an order of magnitude smaller than the warehouses'.

A useful side effect of measuring rather than assuming: the Singapore warehouse fares best (2.8×) partly because Sydney peaks mid-year while Tokyo and Jakarta peak in December, so the regional swings partly cancel. Remove Sydney from that pool and the figure roughly doubles to 5.2×.

### Finding 2 - Warehouse reliability: 100% versus 87.7%

*Across four rolling backtests, each planning on past data only and tested on the following quarter.*

| Test period | Demand vs what the plan expected | Textbook formula | Optimized policy |
|---|---:|---:|---:|
| Q1 2024 | +2.5% | 98.78% | **100%** |
| Q2 2024 | −12.0% | 100% | **100%** |
| Q3 2024 | +2.9% | 99.18% | **100%** |
| **Q4 2024** | **+21.4%** | **87.71%** | **100%** |

*Share of store restock orders the regional warehouse could fill.*

This is the finding that replicated every time. The optimized policy holds every warehouse at 100% in all four periods. The textbook policy looks perfectly healthy for three quarters - it also reaches 100% in the one quarter where demand came in *below* plan - then drops to **87.71%** at the Christmas peak.

The correlation between how busy a period was and how badly the textbook warehouses performed is **−0.90**: as demand rises above what the plan expected, warehouse reliability falls, consistently and steeply. Four data points make that an indication rather than proof, but it matches the mechanism in Finding 1 exactly.

**Why this matters more than a cost number.** When a warehouse runs empty, every store it serves is stranded simultaneously - they order as normal and receive nothing, however well their own shelves were planned. One warehouse shortage becomes empty shelves across three or four shops, during the most valuable trading weeks of the year.

### Finding 3 - The cost saving is real, but it is a range, not a number

Comparing the optimized policy against giving every store its own independent backup stock, with both actually delivering their service target:

| Test period | Saving |
|---|---:|
| Q1 2024 | 1.47% |
| Q2 2024 | 4.06% |
| Q3 2024 | 1.56% |
| Q4 2024 | 8.30% |
| **Range** | **1.5% – 8.3%** |
| **Average** | **3.8%** |

Pooling was cheaper in **every** period, so the direction is solid. The size is not. An earlier version of this analysis, based on a single held-out period, reported **2.7%** - one draw from a wide spread, presented with more confidence than it deserved. The honest headline is the range.

This is a modest saving. The reason to adopt the approach is Finding 2; the cost reduction is a bonus.

### A trap worth naming: cheaper can mean failing

The textbook policy produces the **lowest cost figure in the entire study** - **\$25,150/year in Q4 2024** - while meeting only **88.62%** of customer demand and stranding its warehouses at 87.71% order fill.

Cost here is the value of stock actually sitting on shelves, so **empty shelves are cheap**. The same optimized policy costs \$190k/year in the calmest quarter and \$60k in the busiest, a threefold swing that reflects trading conditions rather than policy quality.

Two consequences, both applied throughout this repo and its dashboard: absolute cost is only ever compared *within* a single period, and service level is printed next to every cost figure so it cannot be read alone.

---

## Why this was hard

<details>
<summary><strong>Expand technical details</strong> - formulation, solver, simulation mechanics, and the bugs found along the way</summary>

### A stochastic multi-echelon program does not fit in a linear solver

Service-level constraints are nonlinear in the stock level: the map from units to service is a demand quantile function. Proper formulations (Clark–Scarf, stochastic dynamic programming) need recursive convolution of demand distributions and cannot be written as linear constraints.

The approach here is a **two-stage approximation**:

**Stage 1, outside the solver.** Build the empirical lead-time demand distribution for all 13 locations. Then tabulate a *pooling curve*: for a grid of local coverage levels α, compute how large a shared warehouse buffer is needed to catch whatever the stores did not cover locally:

```
local_i(α)   = max(floor_i, q_α(W_i) − mean(W_i))
excess_i,t   = max(0, W_i,t − mean(W_i) − local_i(α))
B_d(α)       = q95( Σ_i excess_i,t )
```

where `W_i,t` is store *i*'s demand over its own lead time in the window ending at *t*. Because the excesses are **summed at the same timestamp before** the quantile is taken, out-of-phase demand cancels on its own - no correlation matrix, no distributional assumption. Measured pooling factor: **0.65–0.78**, i.e. a shared buffer needs 65–78% of what ten separate buffers would.

**Stage 2, the MILP in PuLP.** Binary selectors choose one α per region. Every cost coefficient is a constant from stage 1, so the objective is linear and solves exactly:

```
minimise  Σ_d [ DCown_d·h_d + Σ_k B_d(α_k)·h_d·z_d,k + Σ_i Σ_k local_i(α_k)·h_i·z_d,k ]
s.t.      Σ_k z_d,k = 1  for each region d;  z binary
```

This is a discretized search over a precomputed response surface, not a true stochastic program. It is cross-checked against brute-force enumeration on every run - which caught a tie-handling bug in the verifier when the expedite floor binds and several grid points become exactly optimal.

### A guaranteed-service scan settles where the buffer belongs

Before the pooling model, an exhaustive scan of the Graves–Willems guaranteed-service formulation asks each warehouse to pick an outbound service time from 0 to its full lead time. All three pick **0** - absorb the entire ocean lead time centrally - because central space is cheaper *and* pooled warehouse demand is far steadier than the stores beneath it. That result is recomputed every run rather than assumed.

### Central stock cannot rescue a store it cannot reach in time

The first version of the pooling model let the shared buffer cover any store shortfall, and the optimizer promptly stripped stores down to **negative safety stock** - on right-skewed demand the median sits below the mean. Central substitution needs a physical limit: each store must self-cover demand over however long an emergency delivery takes.

That expedite window is the softest assumption in the model, so it was tested rather than defended. **The analytical sensitivity was badly misleading:** it implied the pooling benefit would collapse from 22.8% to 2.2% across a 0–3 day range. Simulating it moved fill rate by **0.41 percentage points** and cost by **5.4%**. The assumption is not load-bearing - a slow expedite option simply goes unused, because it is capped at normal transit time.

### Three simulation bugs that produced plausible nonsense

The validation simulation is a daily loop: arrivals land, stores review and order, demand arrives and unmet demand is a **lost sale** (not a backorder), warehouses reorder from the factory. Three bugs each produced believable-looking output:

1. **Expedite added volume instead of accelerating it.** A daily top-up to a fixed cover level turned emergency delivery into a parallel just-in-time channel, so stores ran on daily emergency shipments rather than the stock being tested. Symptom: a non-monotonic sensitivity curve where a 1-day expedite scored *worse* than 2-day and 3-day. Expediting now changes the speed of an order, never its size.
2. **The protection interval was off by one.** Base stock covered `L` days, but with daily review the true interval is `L + 1`. This under-stocked *every* policy; the tell was that even an oracle policy fitted on the test data itself failed at 89%.
3. **A "3-day expedite" was being used on 2-day lanes** - a slower route nobody would ever choose. Now capped at normal transit time.

A fourth was latent rather than active: the rolling-window cache was keyed only on `(node_id, window_length)`, so fitting on a training slice after a full-history run would silently return full-history values - leaking the test period into the fit and invalidating the whole backtest while still printing plausible numbers. It is now keyed on the date span too, and the rolling script asserts this on every run.

### Honest evaluation required building the right comparison

Comparing the optimized policy to the naive baseline on cost is not like-for-like, because the naive policy is cheaper partly by under-stocking. The comparison that supports Finding 3 is against a **decentralized policy that also truly delivers 95%** - every store self-sufficient, no pooling. That policy had to be built and simulated purely to have a fair yardstick.

An **oracle** policy, fitted on the test period itself, is also simulated as a diagnostic: it reaches 98.76% fill against the honestly-fitted 97.01%. That 1.76-point gap is the cost of not knowing the future, and it is small - meaning the shortfall against target is mostly the period being genuinely harder, not the method failing to generalize.

### Practical notes

- PuLP's bundled CBC binary is x86-64 and fails on Apple Silicon with `OSError: Bad CPU type`. The solver selector *probes* candidates by solving a trivial problem rather than trusting PuLP's availability list, and prefers HiGHS (`pip install highspy`), which ships native arm64 wheels.
- The first optimizer run took over two minutes: `pandas.Series.quantile` dominated, at ~35 ms per call on a 700-element series, because it routes through `DataFrame.quantile` and rebuilds an index each time. Replacing the rolling sums with a prefix-sum and the quantiles with NumPy brought it to **1.8 seconds**, verified identical to the pandas result.

</details>

---

## What this cannot tell you

- **The demand history is synthetic** - generated from a fixed seed, not a real retailer's sales. The structure of the findings is sound; the specific dollar figures describe this simulated network only.
- **Four backtest windows, from two years of history.** The ranges indicate spread; they are not confidence intervals.
- The optimizer picks **one coverage level per region**, not per store - the shared buffer cannot be tabulated as a 1-D curve otherwise. This is the single biggest simplification.
- Emergency delivery speed is **assumed**, not measured, though the simulated sensitivity shows the results barely move across the plausible range.
- Every result assumes the **factory itself never runs short**.
- Service levels reported by the in-sample analysis are optimistic by construction; only the rolling backtest figures are out-of-sample.

---

## Reproducing

```bash
pip install -r requirements.txt

python3 src/generate_demand.py            # 2 years of seeded synthetic demand
python3 src/naive_baseline.py             # textbook baseline + assumption audit
python3 src/optimize_network.py           # two-stage MILP
python3 src/validate_policy.py            # single held-out simulation
python3 src/rolling_validation.py         # 4 rolling backtests

streamlit run dashboard.py                # interactive dashboard (local)
python3 src/build_static_dashboard.py     # rebuild docs/index.html (the hosted page)
```

The whole pipeline runs in **under 10 seconds**. Demand generation is seeded, so every figure above reproduces exactly - two consecutive runs produce byte-identical Parquet output.

The `data/*.parquet` files are committed (~300 KB total) so the dashboard works straight from a fresh clone without running anything.

---

## Repository layout

```
src/network_config.py           network structure, lead times, holding costs (plain data)
src/generate_demand.py          seeded synthetic daily demand → data/demand_history.parquet
src/naive_baseline.py           textbook z·σ·√L policy + audit of its assumptions
src/optimize_network.py         two-stage MILP (PuLP/HiGHS) over empirical pooling curves
src/validate_policy.py          day-by-day simulation on one held-out period
src/rolling_validation.py       4 expanding train/test windows; stability of the findings
src/build_static_dashboard.py   renders docs/index.html from the same Parquet files
dashboard.py                    Streamlit dashboard (run locally)
docs/index.html                 static build served by GitHub Pages - the live link
.streamlit/config.toml          pinned light theme the dashboard depends on
data/                           results at each stage (Parquet)
requirements.txt                dependencies, split by what the dashboard alone needs
```

### Two versions of the same dashboard

`dashboard.py` (Streamlit) and `docs/index.html` (static) read the **same Parquet
files**, so they cannot disagree. The static build exists because GitHub Pages
serves files but cannot run Python - and it works here only because the dashboard
does no computation at view time: every figure is precomputed by the pipeline, and
the slider, quarter selector and Explained/Raw toggle merely select between values
that already exist. All of that interactivity is preserved in the browser. What the
static version cannot do is compute anything **new**; a control that changed a model
input and re-ran the optimization would need the Streamlit version.
