# Zero-Shot Foundation Model Forecasting of Visitor Flow in Heritage Museums

**with Topology-Constrained Reconciliation**

Shahab Anbarjafari · [shb@3sholding.com](mailto:shb@3sholding.com)

---

Heritage museums need to know how many people will be in each gallery in five, fifteen,
thirty and sixty minutes. They need it to protect the collection, to staff the rooms, and
to keep the building safe when it fills. What they do not have is years of labelled
history to train a model on: the sensors went in last month, the building is one of a
kind, and nobody is going to run a two-year data collection campaign before the first
useful forecast.

This repository asks whether a time-series foundation model can produce that forecast
with **no site-specific training at all**, and whether the building's own topology — the
fact that a person who leaves one room must appear in an adjacent one — can be imposed on
the forecast as a hard constraint to make it both physically coherent and more accurate.

The codebase is a complete experimental apparatus: a discrete-event museum simulator, a
literature-grounded sensor degradation model, wrappers for four foundation models, a
constrained-projection reconciler, a leakage-proof rolling-origin evaluation harness with
proper significance testing, three decision-utility heads, and a reporting layer that
refuses to emit a number no run produced.

---

## Table of contents

- [What is being claimed](#what-is-being-claimed)
- [Ground rules](#ground-rules)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The canonical data contract](#the-canonical-data-contract)
- [Architecture](#architecture)
  - [M1 — schema and graph](#m1--schema-and-graph)
  - [M2 — the simulator](#m2--the-simulator)
  - [M3 — sensor degradation](#m3--sensor-degradation)
  - [M4 — forecasters](#m4--forecasters)
  - [M5 — topology-constrained reconciliation](#m5--topology-constrained-reconciliation)
  - [M6 — evaluation](#m6--evaluation)
  - [M7 — experiments](#m7--experiments)
- [Command line reference](#command-line-reference)
- [Datasets](#datasets)
  - [A limitation worth stating before the results](#a-limitation-worth-stating-before-the-results)
  - [ROBOD](#robod)
  - [HZMetro](#hzmetro)
- [Reproducibility](#reproducibility)
- [Development](#development)
- [Current status](#current-status)
- [Citation and licence](#citation-and-licence)

---

## What is being claimed

Eight experiments, each answering one question. None of their results are in this README,
because writing a number here that no logged run produced would violate the project's
second ground rule. Run them and read `paper/tables/`.

| | Question | Config |
|---|---|---|
| **E1** | Does a zero-shot foundation model beat the baselines a building already has? | `configs/experiments/E1.yaml` |
| **E2** | Do the other series, the opening schedule, or the CO2 proxy add anything? | `configs/experiments/E2.yaml` |
| **E3** | Does the constrained projection improve accuracy as well as coherence? | `configs/experiments/E3.yaml` |
| **E4** | How much local history before a trained model overtakes zero-shot? | `configs/experiments/E4.yaml` |
| **E5** | How fast does each method degrade as the sensors get worse? | `configs/experiments/E5.yaml` |
| **E6** | Does the forecast change a decision an operator would make? | `configs/experiments/E6.yaml` |
| **E7a** | Does zero-shot forecasting transfer to a real building? | `configs/experiments/E7_robod.yaml` |
| **E7b** | Does it transfer to real network flow at scale? | `configs/experiments/E7_hzmetro.yaml` |

E7 is the one that makes the work publishable. Everything before it is measured on a
simulator whose parameters were chosen by the same person making the claim.

---

## Ground rules

These are enforced by the code, not by good intentions.

**1. Every external API is verified before it is used.** The foundation model wrappers
were written against the live repositories and model cards, not against remembered
signatures. Where an API differs from what the specification assumed, the code follows
the API and the docstring records the discrepancy.

**2. No fabricated results.** There is no code path that writes a placeholder number into
a results file, a table or a plot. `mflow report` raises rather than emitting an empty
table, and records in `report.json` exactly which artefacts it could not build and why.
If a run has not happened, the artefact does not exist.

**3. Determinism.** Every experiment takes a seed. Every run writes a manifest with the
git commit, a config hash, the seed, the versions of the fifteen packages that can move a
number, and the hardware. `mflow reproduce results/<run_id>` compares your environment
against a logged one and lists every difference. Runs refuse to start from a dirty
working tree unless you explicitly pass `--allow-dirty`, and the manifest records that
you did.

**4. Strict separation by time.** Train, validation and test are split by time and
separated by a gap of at least one horizon, so a training target can never overlap a test
context. `mflow.eval.protocol.context_panel` is the only function that hands data to a
forecaster during evaluation, and it physically cannot return anything at or after the
origin. Normalisation statistics are never computed from the full series.

**5. Fail loudly.** No bare `except`, no silent fallbacks. A forecaster that raises stops
the run rather than being skipped — a results table with a quietly missing method invites
exactly the wrong conclusion. Where data must be imputed, the run records where: when a
sensor dropout forces the conservation anchor to be carried forward, the number of steps
it was carried appears in `reconciliation.parquet` as `anchor_staleness_steps`.

**6. Typed, linted, tested.** Python 3.11+, type hints on every public function, `ruff`
and `mypy --strict`-adjacent clean, 372 tests.

---

## Installation

Python 3.11 or newer. The project is developed on 3.12.

```bash
git clone <this repository>
cd zero-shot-FMF

python3.12 -m venv myenv
source myenv/bin/activate

pip install -e ".[dev]"
```

The core install pulls numpy, pandas, pyarrow, scipy, cvxpy/OSQP, networkx, torch,
scikit-learn, LightGBM, statsmodels, matplotlib and pydantic. It does **not** pull the
foundation models, because each one downloads hundreds of megabytes of weights. Install
only the ones you intend to run:

```bash
pip install -e ".[timesfm]"      # TimesFM 3.0 and 2.5
pip install -e ".[chronos]"      # Chronos-2
pip install -e ".[toto]"         # Toto 2.0
```

Everything except the foundation models works without them, including the whole
simulator, the reconciler, the harness and all seven experiment configs in `--dry-run`.

### Hugging Face credentials

Create a `.env` in the repository root:

```
HUGGINGFACE_API_KEY=hf_...
```

It is gitignored and must stay that way. `mflow.cli.load_environment()` reads it at
start-up and prints only the *names* of the keys it found — the token is never returned,
logged, or written into a manifest. Gated repositories need it; public ones do not.

---

## Quick start

```bash
# 1. Generate a synthetic museum: 8 rooms, 8 two-way doorways, 30 days at one-minute
#    sampling. The `outside` node and the entrance edges make the graph closed.
mflow simulate configs/sites/house_museum.yaml --days 30 --seed 0

# 2. Look at what is available.
mflow list

# 3. Check the pipeline end to end. Trivial forecasters only, no model weights, ~4s.
mflow run configs/experiments/smoke.yaml --allow-dirty

# 4. Build the tables from whatever has been run.
mflow report --output paper/tables
```

Then plan a real experiment before paying for it:

```bash
mflow run configs/experiments/E1.yaml --dry-run
```

which prints every site, seed, variant and method-by-reconciler cell it will execute.

---

## The canonical data contract

Every site — simulated or real — is a directory with the same seven files. Nothing
downstream knows or cares where a site came from.

```
data/canonical/<site_id>/
├── meta.json                   site_id, interval_seconds, timezone, opening_hours,
│                               source, provenance, has_ground_truth_flow
├── nodes.csv                   node_id, name, kind, area_m2, capacity_persons
├── edges.csv                   edge_id, src_node, dst_node, width_m,
│                               capacity_persons_per_min, reverse_edge_id
├── occupancy.parquet           timestamp, node_id, count    (nullable Int32 on disk)
├── flow.parquet                timestamp, edge_id, count
├── covariates_past.parquet     timestamp, scope, variable, value
└── covariates_future.parquet   the same, for channels known in advance
```

Two conventions are load-bearing:

**The building is closed.** There is exactly one virtual `outside` node. Every entrance
and exit is an edge to it, so the whole site satisfies a conservation identity with no
sources or sinks:

> `o_v(t) − o_v(t−1) − Σ_{e ∈ in(v)} f_e(t) + Σ_{e ∈ out(v)} f_e(t) = 0`

This identity is what M5 turns into `A y = b` and projects onto. `validate_site()` checks
it on load, so a site that violates it cannot enter the pipeline.

**Counts are nullable integers on disk and floats in memory.** People are countable, so a
float column on disk would invite silently fractional occupancy. But a sensor that drops
out has no reading at all, and that gap has to survive the round trip rather than become
a zero — so in memory a missing reading is a plain `NaN` that every consumer already
knows how to see.

A forecaster never sees a `SiteData`. It sees a frozen `Panel`: a `(n_series, T)` float32
array, its series identifiers, a timezone-aware timestamp index, and the past and
known-future covariates. `tests/fixtures/toy_site/` is the reference implementation of
the contract and is deliberately committed.

---

## Architecture

```
src/mflow/
├── schema.py          M1  canonical contract, Panel, validation
├── graph.py           M1  BuildingGraph
├── manifest.py            run manifests, seeding, determinism
├── paths.py               filesystem conventions (MFLOW_ROOT-aware)
├── profiling.py           latency and peak-memory measurement
├── cli.py                 the `mflow` command
├── sim/               M2  discrete-event museum simulator
├── sensors/           M3  counting, environment, faults, pipeline
├── forecast/          M4  ABC + trivial / classical / foundation / graph NN
├── reconcile/         M5  constraints, projection, MinT, quantile lifting
├── eval/              M6  protocol, metrics, significance, harness, report
├── risk/              M6  congestion, anomaly, exposure
├── experiments/       M7  config, runner, risk adapter
└── data/                  external dataset acquisition and adapters
```

### M1 — schema and graph

`schema.py` is the contract and its validator. The validator is not a formality: it
checks the time grid is regular and complete, that every reference resolves, that counts
are non-negative and within capacity, that covariates cover the grid, and that
conservation holds to tolerance. It reports the worst offending node and timestamp rather
than just failing.

### M2 — the simulator

A graph-level discrete-event simulator. Visitors arrive by a non-homogeneous Poisson
process shaped by opening hours and day of week, are assigned a **visiting style** from
the Véron & Levasseur (1983) typology — ant, fish, butterfly, grasshopper, as
operationalised by Zancanaro et al. (2007) — and draw log-normal dwell times per room
(Yoshimura et al., 2014). Groups move together.

The interesting part is the doorway resolution, which took three attempts to get right.
Within one tick, every doorway is served simultaneously: grants are relaxed bottom-up
from available headroom to a fixed point, and then opposing demand on each doorway is
matched so that two full rooms can trade occupants. Without that exchange pass a
saturated building gridlocks — two adjacent full rooms can never swap visitors, and
agents strand permanently. The earlier top-down scheme (cut inflows to over-capacity
nodes) is worse than useless: cutting an inflow to `v` reduces the outflow of `u`, which
then over-fills, and the recursion converges on the all-zero solution.

Three site configurations ship: `house_museum` (8 rooms), `palazzo` (24), `national`
(64). Each declares target ranges for mean visit duration, node visit fraction and
occupancy peak-to-mean, drawn from the visitor-studies literature; `scripts/simulate_site.py`
fails if the simulation falls outside them.

**Tier B (JuPedSim continuous-space) is not implemented.** The specification marks it
optional, and writing it against an unverified API would breach ground rule 1.

### M3 — sensor degradation

Real museum instrumentation is bad in specific, documented ways, and every parameter here
comes from a published error rate cited in the docstring of the module that uses it — not
from invention.

- **Counting** (`counting.py`): a fixed per-sensor multiplicative bias drawn once, plus a
  congestion-saturating miss probability and a double-count probability. Cites Cokbas et
  al. (2020, CVPRW), Gruber et al. (2014, *Energies* 7(3) 1685–1705), Gade et al. (2016,
  *Sensors* 16(1) 62).
- **Environment** (`environment.py`): a single-zone CO2 mass balance
  `V dC/dt = 10⁶·G·n(t) − Q·(C − C_out)`, integrated **exactly** over each interval of
  constant occupancy so the answer does not depend on the interval length. Temperature
  and humidity use Magnus–Tetens saturation vapour pressure (Alduchov & Eskridge, 1996)
  and proper humidity-ratio conversion.
- **Faults** (`faults.py`): two-state Markov chains for bursty dropout and stuck values,
  plus clock skew. Every introduced gap is recorded in a `FaultRecord`, because ground
  rule 5 forbids imputing without saying where.

Two modelling points worth stating. First, a crowded room can see relative humidity
*fall* as moisture rises, because the air warms faster than it moistens — the code gets
this right and a test asserts the humidity *ratio*, not the RH. Second, the profiles carry
an `hvac_rejection_fraction`: a heritage museum controls temperature and humidity tightly
for the collection's sake, which is exactly what makes those two channels weak occupancy
proxies and **CO2 the usable one**. Conditioning removes heat and moisture but not CO2.
That asymmetry is the argument for the environmental covariate.

Four profiles: `clean`, `realistic`, `degraded`, `harsh`.

The known failure mode is documented and tested: in a sparsely occupied room the CO2
signal is sensor noise. The museum shop averages 0.18 occupants and its CO2 correlates at
0.58 with a lag of 8 minutes — which is why the M3 acceptance test uses the **median** lag
across rooms, matching the building-level framing of the occupancy-sensing literature, and
gates the per-room correlation on occupancy spread.

### M4 — forecasters

One abstract base class, one registry, one factory. An experiment config names a method as
a string and the harness never imports a model class. Heavy dependencies are imported
lazily inside the factories, so `mflow list` does not download a checkpoint.

| Family | Methods |
|---|---|
| Trivial | `last_value`, `seasonal_naive`, `historical_average` |
| Classical | `sarima`, `ets`, `global_lightgbm` |
| Foundation | `timesfm3_univariate`, `timesfm3_multivariate`, `timesfm3_multivariate_covariates`, `timesfm25_univariate`, `chronos2_univariate`, `chronos2_multivariate`, `chronos2_multivariate_covariates`, `toto2` |
| Graph NN | `dcrnn`, `stgcn`, `graph_wavenet` |

Weights come from `google/timesfm-3.0-pytorch`, `google/timesfm-2.5-200m-pytorch`,
`amazon/chronos-2` and `Datadog/Toto-2.0-313m`. All four have been loaded from their real
checkpoints and checked against a held-out origin: finite, monotone across quantiles, and
better than `last_value` on MAE.

Two things that check turned up, both of which a mocked test would have missed:

- **Chronos names its series in a column called `target_name`, not `target`.** The
  wrapper used to probe for `target` and, on not finding it, hand every series the first
  series' rows. Nothing complained — the frame had the right number of rows and the
  quantiles were monotone — and the only symptom was a multivariate MAE 2.6× the
  univariate one. The column is now required rather than probed, so the same class of
  mismatch stops the run instead of producing a plausible wrong number.
- **Chronos cannot take a timezone-aware timestamp column**, because it normalises one
  with `.to_numpy().view("int64")`. It is handed naive UTC, which loses nothing: the
  column exists only to establish order and spacing, and nothing it returns is a
  timestamp.

### M5 — topology-constrained reconciliation

A forecaster produces each series independently, so nothing makes its occupancy and flow
predictions agree with the physics of the building. This is the paper's contribution.

The conservation identities across the whole horizon stack into `A y = b`, with `b`
anchored on the occupancy observed at the origin. The proposed reconciler solves an
**uncertainty-weighted constrained QP**:

> minimise `(y − ŷ)ᵀ W (y − ŷ)` subject to `A y = b`, `0 ≤ y ≤ u`

with `W = diag(1/σ²)` and `σ = (q₉₀ − q₁₀) / 2.5631`. The forecast's own predictive spread
decides which series get moved: a room the model is confident about is held near its
prediction, and the correction is absorbed by the series the model was unsure of. It is
solved with cvxpy/OSQP through a DPP-compliant parameterised problem, compiled once and
reused across origins.

Four strategies are compared: `none`, `mint` (MinT(Shrink) with Schäfer–Strimmer
shrinkage, equality constraints only), `uniform_weight` (the same projection with `W = I`,
isolating whether the weighting does any work), and `proposed`. The point correction is
carried through the quantile fan either by `shift` or by `per_quantile` re-solving.

Two differences show up immediately and are worth watching for in E3 and E5: MinT has no
box constraints, so it can push a forecast to a negative headcount or past a room's
capacity, while the proposed projection cannot; and the projection verifies its own
residual after solving, so an inaccurate OSQP status is backstopped rather than trusted.

### M6 — evaluation

**Protocol** (`protocol.py`). Rolling origin, fixed context length, origins every `stride`
steps through the test window, horizons reported as prefixes of a single forecast so the
cross-horizon comparison is within one model call. A frozen `RollingOriginPlan` is built
before any model runs and shared by every method, so no method can evaluate on an easier
subset.

**Metrics** (`metrics.py`). `mae`, `rmse`, `mase`, `wql`, `crps`, `coverage_80` and
interval width, `conservation_residual_mae`, `violation_rate`, `quantile_crossing_rate`,
plus `latency_ms` and `peak_memory_mb` measured *by the harness* rather than self-reported
by each wrapper, so the numbers are comparable across methods.

One decision worth flagging: a series whose seasonal-naive denominator is zero — a store
cupboard that is empty every day — gets `NaN`, not a floored denominator. Flooring at
`1e-6` produced MASE values in the hundreds of thousands and silently dominated the mean.
Such series are excluded from MASE, and if none remain the metric raises.

**Significance** (`significance.py`). Diebold–Mariano (1995) with the Harvey–Leybourne–
Newbold (1997) small-sample correction and a Newey–West long-run variance truncated at
`h−1` with Bartlett weights, and Holm–Bonferroni step-down across each family of
comparisons.

The HLN correction is undefined when there are fewer than roughly `2h + 2` origins, and
rather than silently returning a statistic of `−0.0` the test raises with a message
telling you to shorten the stride. The reporting layer turns that into an explicit `n/a`
in the table with the reason attached, because **"we tested and found nothing" and "we
could not test" must not look the same in a manuscript**.

**Harness** (`harness.py`). Runs the plan, scores per origin (not just aggregated — the
paired series cannot be recovered from an average), and writes tidy frames.

**Risk heads** (`risk/`). Congestion alerting against a reactive baseline; anomaly
detection at a pinned false-alarm budget; and exposure accounting. The exposure head
reports **a formulation and a worked example only** — no dataset links visitor load to a
conservation outcome, so there is no validation claim to make, and the caveat travels in a
column of the output so a table cannot strip it.

**Reporting** (`report.py`). Reads `results/`, writes every table as both CSV and LaTeX and
every figure as both PDF and PNG. Paired tests key on the full occasion — run, site, seed,
variant and origin — so pooling runs cannot collapse origin 34566 of two different sites
into one row.

### M7 — experiments

An experiment is a YAML file naming sites, methods, reconcilers, a protocol, and the
*variants* that make it an experiment: the sensor profiles of E5, the training budgets of
E4, the covariate sets of E2. One run is one site, one seed, one variant.

The protocol block has **no defaults**, deliberately: a context length or stride chosen by
the code rather than by the config would make two experiments incomparable with nothing in
either file saying so. Unknown keys are rejected rather than ignored, so `sensor_profil:
harsh` fails at load instead of running the whole sweep on clean data.

Each run writes:

```
results/<experiment>_<site>_<variant>_s<seed>/
├── manifest.json            written BEFORE the run starts
├── predictions.parquet      per method, origin, series, step: every quantile + actual
├── metrics.parquet          per method, origin, series group, horizon
├── resources.parquet        per method, origin: latency and peak memory
├── reconciliation.parquet   residual and violation before/after, anchor staleness
└── risk.parquet             only when the experiment asks for the decision heads
```

The manifest is written first on purpose. A run that crashes leaves a directory with a
manifest and no metrics, which `load_runs` skips — so the difference between "this
finished" and "this was attempted" survives on disk, and a partial sweep can never be
reported as a complete one.

The covariate ablation works by removing channels from the panel, not by asking wrappers
to ignore them. A wrapper told to ignore a channel could still normalise against it; a
channel that is not in the panel cannot be used at all.

The training-budget sweep blanks the training window before the cutoff rather than
shortening the panel, so every budget shares the same origins, contexts and test window
and only the quantity of history changes.

Two guards exist because real data forced them. `require_observed` drops origins whose
context or target is too empty to evaluate, recording the count and the reason in the
manifest; it is off for simulated sites, where a dropped origin would mean a bug rather
than a hole in the record. And a site that measures no doorway flow cannot be reconciled
at all — the run stops rather than projecting onto a constraint assembled entirely from
the forecaster's own predictions.

---

## Command line reference

```
mflow simulate  <site.yaml> [--days N] [--start DATE] [--seed N] [--warmup-days N]
mflow degrade   <site_id> [--profile NAME] [--seed N]
mflow run       <experiment.yaml> [--dry-run] [--sites ...] [--seeds ...]
                [--variants ...] [--overwrite] [--allow-dirty] [--results DIR]
mflow report    [--output DIR] [--results DIR] [--reference METHOD]
mflow reproduce <results/run_id>
mflow list
```

`--data DIR` before the subcommand overrides `data/canonical/`. Setting `MFLOW_ROOT`
relocates the whole project tree.

---

## Datasets

**Data is never committed.** `data/` and `results/` are gitignored. Each source has a
`scripts/fetch_<name>.py` that downloads into `data/raw/<name>/`, records the download
date and a checksum, and then invokes its adapter to write `data/canonical/<site_id>/`.

| Source | Role | Where |
|---|---|---|
| ROBOD | Real room-level occupancy with CO2, temperature, humidity — the closest public analogue to museum instrumentation | [github.com/ideas-lab-nus/robod](https://github.com/ideas-lab-nus/robod), figshare 19234530 |
| HZMetro / SHMetro | A real closed graph with a conservation identity of exactly the assumed form, at a scale no museum dataset reaches | [github.com/HCPLab-SYSU/PVCGN](https://github.com/HCPLab-SYSU/PVCGN) |
| Melbourne pedestrian counting | Real directional counts, hourly and per-minute | City of Melbourne open data; Zenodo 4656626 snapshot |
| METR-LA / PEMS-BAY | Used **only** to validate the graph baselines against their published numbers | [github.com/liyaguang/DCRNN](https://github.com/liyaguang/DCRNN) |
| ATC indoor tracking | Indoor pedestrian trajectories in a large public building | [dil.atr.jp](https://dil.atr.jp) |
| Museum UWB | Indoor positioning in an actual museum | Zenodo 14918763 |

HZMetro is a transport network, not a museum. It is included for the scale of its real
directional flow; no claim about visitor behaviour in heritage settings rests on it, and
its config says so.

### A limitation worth stating before the results

**No public dataset in this project measures room occupancy and doorway flow at the same
time.** ROBOD counts people in rooms and nothing at the doorways; PVCGN counts fare-gate
crossings and nothing standing in a station. Each supplies exactly one side of the
conservation identity, so on neither of them can a reconciler be evaluated: it would be
projecting onto a constraint assembled partly from the forecaster's own predictions, and
the coherence residual it reported would measure the forecaster's self-consistency rather
than its agreement with the building.

The runner enforces this rather than leaving it to a reader to notice — asking for any
reconciler but `none` on such a site is an error. The consequence is that **the
reconciliation claim rests on simulation (E3, E5) and is not corroborated on real data.**

The specification assumed HZMetro would corroborate it, describing it as "a closed graph
with a conservation identity of exactly the assumed form". Reading the release rather
than assuming it, that is not the case: closing the identity at a station needs the flows
*between* stations, which is the origin–destination matrix, and PVCGN does not publish
one. The obvious workaround — occupancy as the running total of entries minus exits — was
tried and rejected on the evidence: it goes negative at 76 of Hangzhou's 80 stations,
reaching −26,068, because a commuter station discharges far more people in the morning
than it takes in. A quantity that is routinely negative is not a count of people in a
room.

Closing this gap needs a source carrying trajectories, from which occupancy and crossings
can be derived consistently from the same observations. The ATC indoor tracking release is
the candidate, and it is the outstanding item in the status table below.

### ROBOD

```bash
python scripts/fetch_robod.py     # ~20 MB, writes data/canonical/robod_bldg1/
mflow run configs/experiments/E7_robod.yaml
```

Five rooms of the SDE4 building at the National University of Singapore, camera-counted
at five minutes, with CO2, temperature, humidity, illuminance and Wi-Fi association
counts alongside. Three things about the conversion are worth knowing before reading any
number that comes out of it, and all three are recorded in the site's `meta.provenance`:

- **ROBOD measures no doorway flow.** `flow.parquet` is written dense and entirely
  missing, and `has_ground_truth_flow` is false. Flow is *not* derived from successive
  occupancy differences: doing so would manufacture exactly the quantity reconciliation
  is meant to be evaluated on, and every coherence number computed over it would be
  circular.
- **ROBOD publishes no opening hours for SDE4**, so `opening_hours` is empty and no
  `is_open` covariate is emitted. An invented timetable would make the calendar arm of
  the covariate ablation measure a fiction.
- **The record has holes.** Collection ran in weekday blocks with weekends and a
  two-month vacation break absent: two thirds of the (room, step) cells on the regular
  grid are unobserved. Gaps are carried through as missing values, and the protocol
  handles them explicitly through `require_observed`, which drops origins with no usable
  context or no truth to score against and records the count and the reason in the run
  manifest. A run over a gappy record therefore cannot silently evaluate on a fraction of
  the origins its stride implies.

### HZMetro

```bash
python scripts/fetch_pvcgn.py    # ~32 MB, writes data/canonical/hzmetro/
mflow run configs/experiments/E7_hzmetro.yaml
```

Eighty Hangzhou metro stations at fifteen minutes, giving 160 measured flow series and a
real physical adjacency of 84 station pairs, which travels in `meta.provenance` for the
graph baselines. Service runs 05:15–23:30 local, so 23% of the steps on the regular grid
are overnight and unobserved; `require_observed` handles them as it does ROBOD's.

One detail matters more than the rest. PVCGN ships ridership as `(T, 4, N, 2)` and its
README describes the last axis as `(inflow/outflow)`, which reads as channel 0 being
entries. **The data says the reverse**, and since the conservation constraint's sign
depends on it, it was checked rather than assumed. Across the whole Hangzhou record:

- in the first interval of the service day channel 0 sums to 29 and channel 1 to 240 —
  nobody alights before anybody has boarded, so channel 1 is entries;
- in the last interval channel 0 sums to 1,495 and channel 1 to 118 — the last trains
  emptying out after entry has stopped;
- within every service day the running total of channel 1 leads that of channel 0, and
  the two converge to within 0.37% by close of service.

So channel 0 is exits and channel 1 is entries. `mflow.data.pvcgn.EXIT_CHANNEL` and
`ENTRY_CHANNEL` are the single place that mapping is applied, and the finding is repeated
in the site's provenance so it travels with the data.

Shanghai (288 stations, 576 series) converts from the same download with
`python scripts/fetch_pvcgn.py --cities shanghai`. Nothing in the experiment plan needs it
yet, so it is off by default.

---

## Reproducibility

Every logged run is reproducible from its manifest:

```bash
mflow reproduce results/E1_sim_house_museum_default_s0
```

which prints either `this environment reproduces it exactly` or a list of every
difference — config hash, seed, commit, and any of the fifteen tracked package versions.

The seeding discipline: `mflow.manifest.rng(seed, *stream)` derives each component's
generator from the run seed and a stable stream name via BLAKE2b, so adding a component
does not perturb the draws of components that already exist.

---

## Development

```bash
ruff check src tests scripts
ruff format src tests scripts
mypy src
pytest
```

The suite is 372 tests and runs in about twenty-seven seconds. Tests needing downloaded
weights or fetched datasets are opt-in:

```bash
pytest -m requires_weights
pytest -m requires_data
```

`RuntimeWarning` is promoted to an error in the test configuration. That is deliberate: an
`All-NaN slice encountered` buried in numpy is a bug somewhere upstream of it, and it
should stop the suite rather than quietly produce a `NaN` in a results table.

A note on the mypy configuration: `python_version = "3.12"` even though the runtime floor
is 3.11, because the pinned numpy ships stubs using PEP 695 `type` statements that mypy
only parses when told to target 3.12 or later.

---

## Current status

| Milestone | State |
|---|---|
| M1 canonical contract, graph, validation | complete |
| M2 Tier A simulator, three sites | complete |
| M2 Tier B JuPedSim | **not implemented** — optional in the specification; writing it against an unverified API would breach ground rule 1 |
| M3 sensor degradation, four profiles | complete |
| M4 forecaster interface and implementations | complete — all four foundation wrappers verified against their real checkpoints |
| M5 reconciliation | complete |
| M6 protocol, metrics, significance, harness, risk heads, reporting | complete |
| M7 experiment configs E1–E7, runner, CLI | complete |
| Dataset acquisition framework (download, checksums, provenance, adapter base) | complete |
| ROBOD adapter and `scripts/fetch_robod.py` | complete — `E7_robod` runs end to end |
| PVCGN adapter and `scripts/fetch_pvcgn.py` | complete — `E7_hzmetro` runs end to end |
| A real site measuring occupancy **and** flow | **not started** — the gap that leaves the reconciliation claim resting on simulation; ATC indoor tracking is the candidate |
| Melbourne, UWB, DCRNN adapters | **not started** |

The simulated sites in `data/canonical/` are generated, not committed. Build them with:

```bash
mflow simulate configs/sites/house_museum.yaml --days 480 --seed 0
mflow simulate configs/sites/palazzo.yaml      --days 120 --seed 0
mflow simulate configs/sites/national.yaml     --days 120 --seed 0
```

The house museum gets 480 days because E4 needs it: its 30- and 90-day training budgets
require windows at least that long, which under the default 0.6/0.2/0.2 split means a
record of 450 days. The runner refuses a budget the record cannot meet rather than
truncating it, so a short site fails at the first run instead of producing a curve that
appears to flatten for reasons that are an artefact of the data.

### Cost of a full run

Measured on this machine (Apple silicon, MPS), per forecast origin over 24 series with a
1440-step context: Chronos-2 multivariate 0.8 s, TimesFM 3 multivariate 1.1 s, TimesFM 3
univariate 2.5 s, Chronos-2 univariate 3.3 s, **Toto 2.0 25.5 s**.

Toto at 313m parameters dominates everything else by more than an order of magnitude, and
E1's test window at `stride: 60` holds a few thousand origins. A full sweep is therefore
not an overnight job at that stride, and the honest options are to cap origins with
`max_origins` — the protocol thins evenly, so a capped run is a uniform subsample of the
same window rather than a different one — or to drop to a smaller Toto checkpoint.
Whichever is chosen has to be recorded in the config before the runs start, not discovered
afterwards; `ProtocolConfig.check_origin_budget` warns when a cap leaves too few origins
for a Diebold-Mariano test at the longest horizon.

---

## Citation and licence

Licence: Apache-2.0. See `LICENSE`.

The paper is in preparation. Until it appears, cite the software:

```bibtex
@software{anbarjafari_zeroshot_fmf,
  author  = {Anbarjafari, Shahab},
  title   = {Zero-Shot Foundation Model Forecasting of Visitor Flow in Heritage
             Museums with Topology-Constrained Reconciliation},
  year     = {2026},
  url      = {https://github.com/001shahab/zero-shot-FMF}
}
```

Third-party datasets and model weights carry their own licences, recorded in the fetch
script and the adapter docstring for each source. Nothing in this repository relicenses
them.
