# Topology-Constrained Reconciliation for Zero-Shot Forecasting of Building Occupancy and Doorway Flow

Code, configurations and analysis scripts for the manuscript of the same title.

Shahab Anbarjafari · [shb@3sholding.com](mailto:shb@3sholding.com)

---

A building that has just been instrumented has no history on which to train a
forecaster, and it still needs to know how many people will be in each room in five,
fifteen, thirty and sixty minutes. This repository asks two questions about that cold
start. Can a pre-trained time series foundation model produce the forecast with **no
site-specific training at all**? And can the building's own topology, the fact that a
person who leaves one room must appear in an adjacent one, be imposed on the forecast as
a hard constraint, so that the forecast is physically coherent and no less accurate?

The codebase is the experimental apparatus behind the answers: a discrete-event simulator
of visitor movement through a building, a literature-grounded sensor degradation model,
wrappers for four foundation models, a constrained-projection reconciler, a leakage-proof
rolling-origin evaluation harness with significance testing, and a reporting layer that
refuses to emit a number that no run produced.

The project began as a study of heritage museums, and the simulator and several module
names still say so. No real museum data was available, so the manuscript makes no claim
about museums: its real-data evidence is room occupancy in an office and laboratory
building and station flow in a metro network, and its reconciliation evidence is
simulated. The title of the manuscript says what the evidence covers.

---

## Table of contents

- [Version of record](#version-of-record)
- [What is being claimed](#what-is-being-claimed)
  - [Which experiments have been run](#which-experiments-have-been-run)
  - [Seeds are not replicates](#seeds-are-not-replicates)
  - [Results](#results)
  - [The configuration behind every table](#the-configuration-behind-every-table)
  - [Analyses added in the second revision](#analyses-added-in-the-second-revision)
- [Ground rules](#ground-rules)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The canonical data contract](#the-canonical-data-contract)
- [Architecture](#architecture)
  - [M1: schema and graph](#m1-schema-and-graph)
  - [M2: the simulator](#m2-the-simulator)
  - [M3: sensor degradation](#m3-sensor-degradation)
  - [M4: forecasters](#m4-forecasters)
  - [M5: topology-constrained reconciliation](#m5-topology-constrained-reconciliation)
  - [M6: evaluation](#m6-evaluation)
  - [M7: experiments](#m7-experiments)
- [Command line reference](#command-line-reference)
- [Datasets](#datasets)
  - [A limitation that bounds the reconciliation claim](#a-limitation-that-bounds-the-reconciliation-claim)
  - [ROBOD](#robod)
  - [HZMetro](#hzmetro)
- [Reproducibility](#reproducibility)
- [Development](#development)
- [Current status](#current-status)
- [Citation and licence](#citation-and-licence)

---

## Version of record

| | |
|---|---|
| Manuscript | *Topology-Constrained Reconciliation for Zero-Shot Forecasting of Building Occupancy and Doorway Flow*, second revision |
| Git tag | `manuscript-r2` |
| Manuscript source | `paper/Zero_Shot_FMF_v3/MainText.tex` |
| Analysis scripts and logged summaries | `paper/Topology_Constrained_Reconciliation_for_Zero_Shot_Forecastingof_Occupancy_and_Flow/` |
| Experiment configurations | `configs/experiments/` |

Everything in this file describes the repository at that tag, and the tag exists so that
the version a reader checks is unambiguous. Results of earlier versions are not repeated
here; they remain in the commit history.

`paper/Topology_Constrained_Reconciliation_for_Zero_Shot_Forecastingof_Occupancy_and_Flow/main.tex`
is the preprint-format source of an earlier version and is kept only for the record. The
analysis scripts live beside it because that is where they were first written.

---

## What is being claimed

The manuscript states five hypotheses, each with the observation that would falsify it,
and reports the outcome of each. Two were not borne out and one holds only in part. The
table gives the outcome as the manuscript states it.

| | Hypothesis | Outcome |
|---|---|---|
| **H1** | A zero-shot foundation model beats every local baseline, including a trained one, on real occupancy and real flow | **Falsified as stated.** It beats every untrained baseline on both sites. Against a properly specified gradient-boosted model it is ahead on room occupancy and behind on metro flow |
| **H2** | The zero-shot advantage declines with the seasonal strength of the target, within a dataset | **Not supported by the within-dataset analysis.** The slope has the predicted sign and its confidence interval includes zero, so the dependence is not statistically established. It is not shown to be absent |
| **H3** | Foundation models are better calibrated and sharper at matched coverage, with lower CRPS | **Partly refuted.** The calibration claim does not hold. The CRPS claim holds on both sites. The sharpness claim holds against the seasonal baseline on room occupancy and is not established out of sample on metro flow |
| **H4** | Raw forecasts violate conservation materially, and the projection removes the violation while respecting bounds that MinT breaks | **Supported, on the simulator only** |
| **H5** | Weighting the projection by the model's own quantile spread is what makes reconciliation improve accuracy | **Supported for the internal design choices, on the simulator only.** Against the raw forecast the gain is in the mean and not at the typical origin |

Three boundaries apply to everything below. The reconciliation evidence is simulated,
because no public dataset reports occupancy and doorway flow for the same building. The
coherence guarantee covers the projected median path and not the whole predictive
distribution. And no operational or decision benefit is claimed, because the
decision-utility experiment was not run.

### Which experiments have been run

This table is the registry of record and agrees with the limitations section of the
manuscript. Every row marked *run* has manifests under `results/`, which is not
committed and is rebuilt from the configurations; `ls results/` is the authoritative
check.

| | Question | Config | Status | Used in the manuscript |
|---|---|---|---|---|
| **E1** | Zero-shot against the baselines, on the simulator | `E1.yaml`, `E1_classical.yaml` | not run | no; the simulator is covered by E11 and E12 |
| **E2** | Do covariates add anything? | `E2.yaml` | **not run** | no; no claim about covariates |
| **E3** | Reconciliation arms on the simulator | `E3.yaml` | not run | no; the reconcilers are applied offline to the forecasts of E11 and E12 |
| **E4** | Training-budget sweep on the simulator | `E4.yaml` | run, superseded | no; replaced by E11, which removes a padding artefact at small budgets |
| **E5** | Sensor degradation | `E5.yaml` | **not run** | no; no claim about robustness to sensor faults |
| **E6** | Decision utility | `E6.yaml` | **not run** | no; no claim about lead time or operational benefit |
| **E7** | Real sites at one daily season of context | `E7_robod.yaml`, `E7_hzmetro.yaml` | run | only as the one-season column of Table 5 and for the determinism check |
| **E8** | Real sites at two daily seasons, with a trained baseline | `E8_robod.yaml`, `E8_hzmetro.yaml` | run | yes; the headline comparison, calibration and dependence analyses |
| **E9** | Context sweep, 1 to 14 daily seasons | `E9_*.yaml` | run | yes; Table 6 |
| **E10** | Gradient-boosted ladder: horizon matched, seasonal lags, tuned, per series | `E10_lightgbm_robod.yaml`, `E10_lightgbm_hzmetro.yaml` | run | yes; Table 4 |
| **E11** | Training-budget sweep on real ROBOD and on the simulator | `E11_efficiency_robod.yaml`, `E11_efficiency_sim.yaml` | run | yes; data efficiency, and the baseline reconciliation study |
| **E12** | Simulator scenario in which capacity binds | `E12_crowded.yaml` | run | yes; the crowded reconciliation study |

All configuration files are in `configs/experiments/`. The three rows in bold are the
experiments that remain unexecuted and on which the manuscript makes no claim: the
covariate ablation, the sensor-degradation study and the decision-utility study. E1 and
E3 were also never executed as separate experiments; the questions they asked are
answered by E11 and E12, with the reconcilers applied offline to the stored quantile
forecasts by the analysis script.

### Seeds are not replicates

Every forecaster here except the gradient-boosted model is deterministic. This was
checked and not assumed: on both real sites the three untrained baselines and all four
foundation model configurations produce forecasts that agree to the bit across seeds.
Running a deterministic method under several seeds does not replicate it, and pooling
the copies would multiply the sample size without adding information and would
understate every p-value.

All inference in the manuscript therefore runs on **distinct forecast origins from a
single seed**, and every count reported here and in the manuscript is a count of
distinct origins: 261 on ROBOD, 181 to 196 on HZMetro depending on horizon, and 300 on
each simulator configuration. Some configurations list more than one seed. Those extra
runs exist to make the determinism check possible, and `analysis.py` asserts it in
`assert_seeds_are_replicates` before discarding them. No result in the manuscript is a
three-seed result.

### Results

These are the results of the manuscript at the tag above, and no others. Relative
improvements use one definition throughout: 100 (1 − L_A / L_B), the reduction in the
loss of the better method A relative to the worse method B, with L the MASE unless
another loss is named.

**Real room occupancy, ROBOD, 261 origins, MASE (manuscript Table 2).** Two daily seasons
of context.

| Method | 5 min | 15 min | 30 min | 60 min |
|---|---|---|---|---|
| Toto-2.0, zero-shot | 0.186 | **0.294** | **0.328** | **0.374** |
| TimesFM-3 multivariate, zero-shot | 0.197 | 0.314 | 0.354 | 0.410 |
| Chronos-2 multivariate, zero-shot | 0.217 | 0.326 | 0.361 | 0.411 |
| TimesFM-3 univariate, zero-shot | 0.206 | 0.318 | 0.359 | 0.417 |
| Last value | **0.164** | 0.361 | 0.449 | 0.538 |
| LightGBM, as first specified | 0.263 | 0.439 | 0.517 | 0.601 |
| LightGBM, per series, the strongest local model | 0.278 | 0.407 | 0.460 | 0.512 |
| Historical average | 0.703 | 0.753 | 0.748 | 0.736 |
| Seasonal naive | 0.758 | 0.819 | 0.811 | 0.796 |

**Real station flow, HZMetro, 181 to 196 origins, MASE (manuscript Tables 3 and 4).**

| Method | 15 min | 30 min | 60 min | 120 min |
|---|---|---|---|---|
| LightGBM, per series, the strongest local model | **0.493** | **0.504** | **0.506** | **0.511** |
| TimesFM-3 univariate, zero-shot | 0.536 | 0.556 | 0.588 | 0.613 |
| TimesFM-3 multivariate, zero-shot | 0.544 | 0.577 | 0.614 | 0.637 |
| Chronos-2 multivariate, zero-shot | 0.594 | 0.627 | 0.691 | 0.779 |
| Toto-2.0, zero-shot | 0.596 | 0.648 | 0.743 | 0.888 |
| Seasonal naive | 0.764 | 0.772 | 0.757 | 0.731 |
| LightGBM, as first specified | 0.792 | 0.873 | 1.006 | 1.135 |
| Historical average | 0.881 | 0.897 | 0.891 | 0.880 |
| Last value | 0.901 | 1.098 | 1.484 | 2.015 |

What these two tables support:

- **Against untrained baselines the foundation models win on both sites**, by 30.5% at
  one hour on room occupancy and by 29.8% at fifteen minutes on metro flow.
- **Against a properly specified local model the result splits by site.** On room
  occupancy the best zero-shot model is ahead by 26.9% at one hour (0.374 against
  0.512). On metro flow the local model is ahead by 16.6% at two hours (0.511 against
  0.613). Each figure is the reduction in MASE of the better method relative to the
  worse.
- **The comparison against a seasonal baseline is decided by context length.** At one
  daily season of context the foundation models lose on metro flow at two hours (MASE
  1.037 against 0.774 on the origins common to the sweep); at two they win, and accuracy keeps improving to about seven
  seasons (0.424) and fourteen (0.414), while the daily seasonal baseline does not move.
- **Under a moving-block bootstrap that respects the overlap between origins**, the
  interval for the foundation model's advantage over the seasonal baseline at the
  longest metro horizon contains zero, at every block length from 5 to 40 origins. It
  excludes zero only if the origins are treated as independent.

**Calibration and sharpness (manuscript Tables 7 and 8).** The foundation models are not
better calibrated than the seasonal baseline. On room
occupancy the best foundation model's nominal 80% interval is half as wide as the
seasonal one while covering more outcomes: 1.17 against 2.35 persons, at coverage 0.901
against 0.839. To reach a common coverage near 90% the seasonal interval must be widened
6.0 times when the factor is fitted on the evaluation observations and 8.4 times when it
is estimated on a preceding calibration period. Exact matching at the nominal 80% is not
attainable on that dataset, because 69.4% of the outcomes are zero and the coverage of
every interval jumps as it reaches its published width. On metro flow a sharpness
advantage appears when the factor is fitted retrospectively and is not established out
of sample. CRPS, which needs no rescaling, favours the
foundation models on both sites: 18.79 against 25.93 at one hour on metro flow.

**Reconciliation, simulator only, 300 origins per configuration (manuscript Table 9).**
MAE in persons; residual and bound are the conservation and the capacity or sign
violations.

| Reconciler | Residual | Bound | MAE | Residual | Bound | MAE |
|---|---|---|---|---|---|---|
| | *baseline* | | | *crowded* | | |
| None (raw) | 1.055 | 0.006 | 0.3495 | 0.960 | 0.451 | 0.3654 |
| MinT (shrink) | < 1e-9 | 2.062 | 0.3516 | < 1e-9 | 2.944 | 0.3676 |
| Unbounded, uncertainty weighted | < 1e-9 | 1.414 | 0.3485 | < 1e-9 | 2.377 | 0.3652 |
| Uniform weight | < 1e-9 | 0.000 | 0.3506 | < 1e-9 | 0.000 | 0.3661 |
| **Proposed** | < 1e-9 | 0.000 | **0.3475** | < 1e-9 | 0.000 | **0.3633** |

What the reconciliation study supports, and what it does not:

- **Coherence and feasibility are exact**, by construction, for the projected median.
  MinT restores conservation and breaks the capacity and sign bounds.
- **The gain in mean error is small and is a statement about the mean.** For TimesFM-3
  the mean MAE falls by 0.98%, with a 95% moving-block bootstrap interval of
  [−1.41, −0.61]. At the typical origin the error rises very slightly: the median change
  is +0.013%, with interval [+0.011, +0.015], and 211 of 300 origins worsen. Both
  statements are established and they do not conflict, because the differences are
  strongly skewed. For LightGBM no effect on the mean is established. The intervals are
  stable to block lengths from 1 to 40 origins.
- **The non-median quantiles are not coherent.** Translating them by the median
  correction leaves a residual of up to 0.085 persons. Projecting each quantile
  separately makes every level coherent and feasible, but the levels then cross on 11.3%
  of adjacent pairs, and the rearrangement that restores their order leaves residuals of
  0.003 to 0.012 persons, the median included. Quantile paths that are individually
  admissible are not a coherent joint distribution.
- **The risk-sensitive evidence is narrow.** The error in the magnitude of the predicted
  peak falls by 6.3% on the baseline configuration and 4.0% on the crowded one, and
  both reductions are statistically established. The detection of near-capacity and
  at-capacity conditions is essentially unchanged, with hit rates moving by 0.2 to 1.3
  percentage points and Brier scores by less than 0.001, and the error at high
  occupancy is not consistently improved. A more accurate peak magnitude is not improved
  identification of unsafe conditions, and neither is a demonstrated operational
  benefit.

### The configuration behind every table

Every table and figure of the manuscript is generated from logged runs by `analysis.py`
or `analysis_round2.py` in the analysis directory named above. The runs are produced by
the configurations below, all in `configs/experiments/`.

| Manuscript item | Configuration | Script |
|---|---|---|
| Table 1, evaluation sites and seasonal strength | the canonical sites built by `scripts/fetch_robod.py`, `scripts/fetch_pvcgn.py` and `configs/sites/house_museum.yaml` | `analysis.py` |
| Table 2, ROBOD | `E8_robod.yaml` | `analysis.py` |
| Table 3, HZMetro | `E8_hzmetro.yaml` | `analysis.py` |
| Table 4, gradient-boosted ladder | `E10_lightgbm_hzmetro.yaml`, `E10_lightgbm_robod.yaml` | `analysis.py`; restated by `analysis_round2.py` |
| Table 5, one against two seasons of context | `E7_hzmetro.yaml` (one season), `E8_hzmetro.yaml` (two) | `analysis.py` |
| Table 6, context sweep | `E9_hzmetro_c{96,192,288,672,1344}.yaml`, `E9_robod_bldg1_c{288,576,864,2016,4032}.yaml` | `analysis.py` |
| Within-dataset test of H2 | `E8_hzmetro.yaml` | `analysis.py` |
| Table 7, coverage and width as published | `E8_robod.yaml`, `E8_hzmetro.yaml` | `analysis.py` |
| Table 8, width at matched coverage | `E8_robod.yaml`, `E8_hzmetro.yaml` | `analysis_round2.py` |
| Table 9, reconciliation | `E11_efficiency_sim.yaml` (variant `d090`), `E12_crowded.yaml` | `analysis.py` |
| Table 10, ablations | the same two | `analysis.py` |
| Tables 11 and 12, mean and typical-origin effects, block length | the same two | `analysis_round2.py` |
| Table 13, risk-sensitive quantities | the same two | `analysis_round2.py` |
| Table 14, coherence by quantile level | `E11_efficiency_sim.yaml` (variant `d090`) | `analysis.py` for the shift rows, `analysis_round2.py` for the per-quantile rows |
| Table 15, sensitivity to the spread estimate | `E11_efficiency_sim.yaml` (variant `d090`) | `analysis.py` |
| Table 16, dependence between origins | `E8_robod.yaml`, `E8_hzmetro.yaml` | `analysis.py` |
| Data-efficiency figure | `E11_efficiency_sim.yaml`, `E11_efficiency_robod.yaml` | `analysis.py` |
| Simulator profile figure | `configs/sites/house_museum.yaml`, `configs/sites/house_museum_crowded.yaml` | `analysis.py` |

The reconcilers of Tables 9 to 15 are not separate runs. The forecasters are run once,
with `reconcilers: [none]`, and every reconciler is then applied offline to the stored
quantile forecasts, so that all arms see identical inputs.

### Analyses added in the second revision

```bash
cd paper/Topology_Constrained_Reconciliation_for_Zero_Shot_Forecastingof_Occupancy_and_Flow
python analysis.py           # first: writes summary.json and the per-origin tables
python analysis_round2.py    # then: writes summary_round2.json and tables/round2_*.csv
python verify_round2.py      # checks the manuscript text against summary_round2.json
```

`analysis_round2.py` runs no forecaster. It recomputes, from the predictions that the
logged runs already wrote, the five analyses that the second review asked for:

1. interval width at matched empirical coverage, with the scale factor and the coverage
   computed on the same observations by the same rule, and repeated with the factor
   estimated on a calibration period that precedes the evaluation period;
2. the gradient-boosted comparison under the single definition of relative improvement
   given above;
3. the reconciliation effect at the mean and at the typical origin separately, with a
   sign test, a bootstrap interval for the median, and the sensitivity of every interval
   to the bootstrap block length, including the interval of the headline real-data test,
   where serial dependence between origins is strong;
4. conservation residuals under per-quantile projection, before and after the monotone
   rearrangement that repairs quantile crossings;
5. peak error, error at high occupancy, and the detection of near-capacity and
   at-capacity conditions, with thresholds and sample sizes.

`verify_round2.py` rebuilds every row of the new tables and every number quoted in the
revised text from `summary_round2.json`, formats each as it is printed, and fails if any
of them is absent from the manuscript source. `verify_numbers.py` does the same for the
numbers of the first revision, against `main.tex`. `summary_round2.json` and `tables/round2_*.csv` are
committed, so the check can be run without the model weights or the datasets.

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
the run rather than being skipped: a results table with a quietly missing method invites
exactly the wrong conclusion. Where data must be imputed, the run records where: when a
sensor dropout forces the conservation anchor to be carried forward, the number of steps
it was carried appears in `reconciliation.parquet` as `anchor_staleness_steps`.

**6. Typed, linted, tested.** Python 3.11+, type hints on every public function, `ruff`
and `mypy --strict`-adjacent clean, 424 tests.

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
simulator, the reconciler, the harness and every experiment config in `--dry-run`.

### Hugging Face credentials

Create a `.env` in the repository root:

```
HUGGINGFACE_API_KEY=hf_...
```

It is gitignored and must stay that way. `mflow.cli.load_environment()` reads it at
start-up and prints only the *names* of the keys it found; the token is never returned,
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

# 4. Build the tables from whatever has been run. Once results/ holds more than one
#    experiment, name the one you want: a horizon is a number of steps, and a step is
#    five minutes on ROBOD and fifteen on HZMetro.
mflow report --experiment E8_robod --output paper/E8_robod
```

Then plan a real experiment before paying for it:

```bash
mflow run configs/experiments/E1.yaml --dry-run
```

which prints every site, seed, variant and method-by-reconciler cell it will execute.

---

## The canonical data contract

Every site, simulated or real, is a directory with the same seven files. Nothing
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
a zero, so in memory a missing reading is a plain `NaN` that every consumer already
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

### M1: schema and graph

`schema.py` is the contract and its validator. The validator is not a formality: it
checks the time grid is regular and complete, that every reference resolves, that counts
are non-negative and within capacity, that covariates cover the grid, and that
conservation holds to tolerance. It reports the worst offending node and timestamp rather
than just failing.

### M2: the simulator

A graph-level discrete-event simulator. Visitors arrive by a non-homogeneous Poisson
process shaped by opening hours and day of week, are assigned a **visiting style** from
the Véron & Levasseur (1983) typology (ant, fish, butterfly, grasshopper, as
operationalised by Zancanaro et al. (2007)) and draw log-normal dwell times per room
(Yoshimura et al., 2014). Groups move together.

The interesting part is the doorway resolution, which took three attempts to get right.
Within one tick, every doorway is served simultaneously: grants are relaxed bottom-up
from available headroom to a fixed point, and then opposing demand on each doorway is
matched so that two full rooms can trade occupants. Without that exchange pass a
saturated building gridlocks: two adjacent full rooms can never swap visitors, and
agents strand permanently. The earlier top-down scheme (cut inflows to over-capacity
nodes) is worse than useless: cutting an inflow to `v` reduces the outflow of `u`, which
then over-fills, and the recursion converges on the all-zero solution.

Three site configurations ship: `house_museum` (8 rooms), `palazzo` (24), `national`
(64). Each declares target ranges for mean visit duration, node visit fraction and
occupancy peak-to-mean, drawn from the visitor-studies literature; `scripts/simulate_site.py`
fails if the simulation falls outside them.

**Tier B (JuPedSim continuous-space) is not implemented.** The specification marks it
optional, and writing it against an unverified API would breach ground rule 1.

### M3: sensor degradation

Real museum instrumentation is bad in specific, documented ways, and every parameter here
comes from a published error rate cited in the docstring of the module that uses it, not
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
*fall* as moisture rises, because the air warms faster than it moistens; the code gets
this right and a test asserts the humidity *ratio*, not the RH. Second, the profiles carry
an `hvac_rejection_fraction`: a heritage museum controls temperature and humidity tightly
for the collection's sake, which is exactly what makes those two channels weak occupancy
proxies and **CO2 the usable one**. Conditioning removes heat and moisture but not CO2.
That asymmetry is the argument for the environmental covariate.

Four profiles: `clean`, `realistic`, `degraded`, `harsh`.

The known failure mode is documented and tested: in a sparsely occupied room the CO2
signal is sensor noise. The museum shop averages 0.18 occupants and its CO2 correlates at
0.58 with a lag of 8 minutes, which is why the M3 acceptance test uses the **median** lag
across rooms, matching the building-level framing of the occupancy-sensing literature, and
gates the per-room correlation on occupancy spread.

### M4: forecasters

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
  series' rows. Nothing complained (the frame had the right number of rows and the
  quantiles were monotone) and the only symptom was a multivariate MAE 2.6× the
  univariate one. The column is now required rather than probed, so the same class of
  mismatch stops the run instead of producing a plausible wrong number.
- **Chronos cannot take a timezone-aware timestamp column**, because it normalises one
  with `.to_numpy().view("int64")`. It is handed naive UTC, which loses nothing: the
  column exists only to establish order and spacing, and nothing it returns is a
  timestamp.

### M5: topology-constrained reconciliation

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

### M6: evaluation

**Protocol** (`protocol.py`). Rolling origin, fixed context length, origins every `stride`
steps through the test window, horizons reported as prefixes of a single forecast so the
cross-horizon comparison is within one model call. A frozen `RollingOriginPlan` is built
before any model runs and shared by every method, so no method can evaluate on an easier
subset.

**Metrics** (`metrics.py`). `mae`, `rmse`, `mase`, `wql`, `crps`, `coverage_80` and
interval width, `conservation_residual_mae`, `violation_rate`, `quantile_crossing_rate`,
plus `latency_ms` and `peak_memory_mb` measured *by the harness* rather than self-reported
by each wrapper, so the numbers are comparable across methods.

One decision worth flagging: a series whose seasonal-naive denominator is zero, such as a store
cupboard that is empty every day, gets `NaN`, not a floored denominator. Flooring at
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

**Harness** (`harness.py`). Runs the plan, scores per origin (not just aggregated; the
paired series cannot be recovered from an average), and writes tidy frames.

**Risk heads** (`risk/`). Congestion alerting against a reactive baseline; anomaly
detection at a pinned false-alarm budget; and exposure accounting. The exposure head
reports **a formulation and a worked example only**: no dataset links visitor load to a
conservation outcome, so there is no validation claim to make, and the caveat travels in a
column of the output so a table cannot strip it.

**Reporting** (`report.py`). Reads `results/`, writes every table as both CSV and LaTeX and
every figure as both PDF and PNG. Paired tests key on the full occasion (run, site, seed,
variant and origin), so pooling runs cannot collapse origin 34566 of two different sites
into one row.

### M7: experiments

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
manifest and no metrics, which `load_runs` skips, so the difference between "this
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
at all: the run stops rather than projecting onto a constraint assembled entirely from
the forecaster's own predictions.

---

## Command line reference

```
mflow simulate  <site.yaml> [--days N] [--start DATE] [--seed N] [--warmup-days N]
mflow degrade   <site_id> [--profile NAME] [--seed N]
mflow run       <experiment.yaml> [--dry-run] [--sites ...] [--seeds ...]
                [--variants ...] [--overwrite] [--allow-dirty] [--results DIR]
mflow report    [--output DIR] [--results DIR] [--reference METHOD]
                [--experiment ID]
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

Only the first two sources below have adapters and are used in the manuscript. The
others were considered in the original design, have no adapter, and supply no result.

| Source | Role | Where |
|---|---|---|
| ROBOD | Real room-level occupancy with CO2, temperature, humidity; the closest public analogue to museum instrumentation | [github.com/ideas-lab-nus/robod](https://github.com/ideas-lab-nus/robod), figshare 19234530 |
| HZMetro / SHMetro | Real directional station flow at a scale no building dataset reaches. It supplies flow only, not occupancy; see the limitation below | [github.com/HCPLab-SYSU/PVCGN](https://github.com/HCPLab-SYSU/PVCGN) |
| Melbourne pedestrian counting | Not used. Real directional counts, hourly and per-minute | City of Melbourne open data; Zenodo 4656626 snapshot |
| METR-LA / PEMS-BAY | Not used. Intended only to validate the graph baselines against their published numbers | [github.com/liyaguang/DCRNN](https://github.com/liyaguang/DCRNN) |
| ATC indoor tracking | Not used. Indoor pedestrian trajectories in a large public building | [dil.atr.jp](https://dil.atr.jp) |
| Museum UWB | Not used. Indoor positioning in an actual museum | Zenodo 14918763 |

HZMetro is a transport network, not a museum. It is included for the scale of its real
directional flow; no claim about visitor behaviour in heritage settings rests on it, and
its config says so.

### A limitation that bounds the reconciliation claim

**No public dataset in this project measures room occupancy and doorway flow at the same
time.** ROBOD counts people in rooms and nothing at the doorways; PVCGN counts fare-gate
crossings and nothing standing in a station. Each supplies exactly one side of the
conservation identity, so on neither of them can a reconciler be evaluated: it would be
projecting onto a constraint assembled partly from the forecaster's own predictions, and
the coherence residual it reported would measure the forecaster's self-consistency rather
than its agreement with the building.

The runner enforces this rather than leaving it to a reader to notice: asking for any
reconciler but `none` on such a site is an error. The consequence is that **the
reconciliation claim rests on simulation (E11 and E12) and is not corroborated on real data.**

The specification assumed HZMetro would corroborate it, describing it as "a closed graph
with a conservation identity of exactly the assumed form". Reading the release rather
than assuming it, that is not the case: closing the identity at a station needs the flows
*between* stations, which is the origin–destination matrix, and PVCGN does not publish
one. The obvious workaround, occupancy as the running total of entries minus exits, was
tried and rejected on the evidence: it goes negative at 76 of Hangzhou's 80 stations,
reaching −26,068, because a commuter station discharges far more people in the morning
than it takes in. A quantity that is routinely negative is not a count of people in a
room.

Closing this gap needs a building in which doorway crossings and room occupancy are
measured independently of each other. The ATC indoor tracking release is the nearest
public candidate, and the status table below explains why it is not sufficient. Validation
on such a site is the manuscript's principal item of future work.

### ROBOD

```bash
python scripts/fetch_robod.py     # ~20 MB, writes data/canonical/robod_bldg1/
mflow run configs/experiments/E8_robod.yaml
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
mflow run configs/experiments/E8_hzmetro.yaml
```

Eighty Hangzhou metro stations at fifteen minutes, giving 160 measured flow series and a
real physical adjacency of 84 station pairs, which travels in `meta.provenance` for the
graph baselines. Service runs 05:15–23:30 local, so 23% of the steps on the regular grid
are overnight and unobserved; `require_observed` handles them as it does ROBOD's.

One detail matters more than the rest. PVCGN ships ridership as `(T, 4, N, 2)` and its
README describes the last axis as `(inflow/outflow)`, which reads as channel 0 being
entries. **The data says the reverse**, and since the conservation constraint's sign
depends on it, it was checked rather than assumed. Across the whole Hangzhou record:

- in the first interval of the service day channel 0 sums to 29 and channel 1 to 240:
  nobody alights before anybody has boarded, so channel 1 is entries;
- in the last interval channel 0 sums to 1,495 and channel 1 to 118: the last trains
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
mflow reproduce results/E8_robod_robod_bldg1_default_s0
```

which prints either `this environment reproduces it exactly` or a list of every
difference: config hash, seed, commit, and any of the fifteen tracked package versions.

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

The suite is 424 tests and runs in about twenty seconds. Tests needing downloaded
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
| M2 Tier B JuPedSim | **not implemented**; optional in the specification, and writing it against an unverified API would breach ground rule 1 |
| M3 sensor degradation, four profiles | complete |
| M4 forecaster interface and implementations | complete; all four foundation wrappers verified against their real checkpoints |
| M5 reconciliation | complete |
| M6 protocol, metrics, significance, harness, risk heads, reporting | complete |
| M7 experiment configs E1–E12, runner, CLI | complete |
| Dataset acquisition framework (download, checksums, provenance, adapter base) | complete |
| ROBOD adapter and `scripts/fetch_robod.py` | complete; `E7_robod` and `E8_robod` run end to end |
| PVCGN adapter and `scripts/fetch_pvcgn.py` | complete; `E7_hzmetro` and `E8_hzmetro` run end to end |
| **Executed** | E4, E7, E8, E9, E10, E11, E12; all logged under `results/`. The manuscript reports E8 to E12, and E7 only for the one-season column of its Table 5; see the registry above |
| E1 and E3 as separate experiments | **not run**; their questions are answered by E11 and E12 |
| E2 covariate ablation executed | **not run**; no claim in the paper depends on it |
| E5 sensor degradation executed | **not run**; no claim in the paper depends on it |
| E6 decision utility executed | **not run**; no claim in the paper depends on it |
| A real site measuring occupancy **and** flow | **not started**; this is the gap that leaves the reconciliation claim resting on simulation, and the manuscript's principal item of future work. ATC indoor tracking is the nearest candidate but supplies no occupancy measurement independent of the tracking, so it cannot test the constraint without circularity. What is needed is synchronised doorway counting plus a separate per-room occupancy estimate. |
| Melbourne, UWB, DCRNN adapters | **not started** |

The simulated sites in `data/canonical/` are generated, not committed. Build them with:

```bash
mflow simulate configs/sites/house_museum.yaml         --days 480 --seed 0
mflow simulate configs/sites/house_museum_crowded.yaml --days 120 --seed 0
mflow simulate configs/sites/palazzo.yaml              --days 120 --seed 0
mflow simulate configs/sites/national.yaml             --days 120 --seed 0
```

The first two are the baseline and the crowded configuration of the reconciliation study
(E11 and E12). The house museum gets 480 days because the training-budget sweep needs
it: its 30- and 90-day training budgets
require windows at least that long, which under the default 0.6/0.2/0.2 split means a
record of 450 days. The runner refuses a budget the record cannot meet rather than
truncating it, so a short site fails at the first run instead of producing a curve that
appears to flatten for reasons that are an artefact of the data.

### Cost of a full run

Measured on this machine (Apple silicon, MPS), seconds per forecast origin over 24 series
with a 1440-step context, after the first call:

| Method | Warm-up | Steady state |
|---|---|---|
| `chronos2_multivariate` | 0.8 s | 0.08 s |
| `chronos2_univariate` | 3.3 s | 0.09 s |
| `toto2` | 4.3 s | 0.13 s |
| `timesfm3_multivariate` | 1.2 s | 0.14 s |
| `timesfm3_univariate` | 3.0 s | 0.18 s |

Warm-up and steady state differ by one to two orders of magnitude, which is worth stating
because timing a single origin gives an answer that is wrong by that much: Toto's first
call takes 4.3 s and its second takes 0.13 s. A sweep amortises the warm-up over thousands
of origins, so only the right-hand column matters for planning.

On those numbers the original plan of E1 to E7 was estimated at roughly 78 hours
uncapped, or 48 hours with `max_origins: 480`. The protocol thins evenly, so a capped run is a uniform subsample of
the same window rather than a different one. `ProtocolConfig.check_origin_budget` warns
when a cap leaves too few origins for a Diebold–Mariano test at the longest horizon,
which at `horizon: 60` needs about 122.

---

## Citation and licence

Licence: Apache-2.0. See `LICENSE`.

The manuscript is under review. Until it appears, cite the software, and name the tag
`manuscript-r2` if you refer to the results reported above:

```bibtex
@software{anbarjafari_zeroshot_fmf,
  author  = {Anbarjafari, Shahab},
  title   = {Topology-Constrained Reconciliation for Zero-Shot Forecasting of
             Building Occupancy and Doorway Flow},
  year    = {2026},
  version = {manuscript-r2},
  url     = {https://github.com/001shahab/zero-shot-FMF}
}
```

Third-party datasets and model weights carry their own licences, recorded in the fetch
script and the adapter docstring for each source. Nothing in this repository relicenses
them.
