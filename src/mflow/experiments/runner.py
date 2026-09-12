"""Turn an experiment configuration into logged runs (M7).

One run is one site, one seed and one variant. The runner resolves the variant, applies
the sensor profile, restricts the covariates, builds the rolling-origin plan, evaluates
every method-by-reconciler cell against it, and writes ``manifest.json`` plus the four
parquet files into ``results/<run_id>/``.

The manifest is written *before* the evaluation starts. A run that crashes therefore
leaves a directory with a manifest and no metrics, which :func:`mflow.eval.report.load_runs`
skips. That is deliberate: the difference between "this run finished" and "this run was
attempted" has to survive on disk, or a partial sweep will be reported as a complete one.

Degradation is applied to a copy of the clean site, and the clean site is kept as the
scoring target. A model evaluated under a sensor profile sees the damaged series and is
scored against the truth, which is the only way to measure what the degradation costs
rather than how well the model predicts its own noise.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mflow.eval.harness import EvaluationResult, MethodSpec, run_evaluation
from mflow.eval.protocol import RollingOriginPlan, build_plan
from mflow.eval.report import (
    METRICS_FILE,
    PREDICTIONS_FILE,
    RECONCILIATION_FILE,
    RESOURCES_FILE,
    RISK_FILE,
)
from mflow.experiments.config import (
    CovariateSet,
    ExperimentConfig,
    ResolvedVariant,
    load_experiment,
)
from mflow.experiments.risk_eval import evaluate_risk
from mflow.forecast import GRAPH_METHODS, build_forecaster
from mflow.graph import BuildingGraph
from mflow.manifest import RunManifest, set_global_seed
from mflow.paths import canonical_dir, configs_dir
from mflow.reconcile import Reconciler, default_registry
from mflow.schema import Panel, SiteData, load_site
from mflow.sensors import degrade, load_sensor_profile

#: Covariate channels a museum knows in advance: opening hours, scheduled groups,
#: calendar effects. Everything scoped to ``global`` is of this kind by construction,
#: because a channel that varies by room is a measurement, not a schedule.
CALENDAR_SCOPE = "global"


class RunnerError(RuntimeError):
    """Raised when an experiment cannot be executed as configured."""


@dataclass(frozen=True)
class RunOutcome:
    """What one run produced.

    Attributes:
        manifest: the manifest that was written first.
        directory: the run directory.
        result: the evaluation result, or None when the run was skipped because its
            directory already held a complete result.
        seconds: wall-clock duration, or 0.0 for a skipped run.
    """

    manifest: RunManifest
    directory: Path
    result: EvaluationResult | None
    seconds: float

    @property
    def skipped(self) -> bool:
        """Whether this run was already on disk and was not recomputed."""
        return self.result is None


def covariate_filter(covariates: CovariateSet) -> Callable[[str], bool] | None:
    """Predicate selecting the covariate channels a condition is allowed to see.

    Returns None for ``all``, so the no-ablation condition goes down exactly the same
    code path as the others rather than round-tripping through a predicate that always
    returns True.
    """
    if covariates == "all":
        return None
    if covariates == "none":
        return lambda _cid: False
    if covariates == "calendar":
        return lambda cid: cid.split("|", 1)[0] == CALENDAR_SCOPE
    if covariates == "environment":
        return lambda cid: cid.split("|", 1)[0] != CALENDAR_SCOPE
    raise RunnerError(f"unknown covariate set {covariates!r}")


def build_method_specs(
    variant: ResolvedVariant, *, graph: BuildingGraph, seed: int
) -> list[MethodSpec]:
    """Instantiate the method-by-reconciler grid for one variant.

    A forecaster is constructed once per cell rather than shared across reconcilers, so
    that a stateful model cannot carry anything from one cell into the next.
    """
    registry = default_registry()
    specs: list[MethodSpec] = []
    for method, reconciler_name in variant.cells():
        forecaster = build_forecaster(
            method.name,
            graph=graph if method.name in GRAPH_METHODS else None,
            seed=seed,
            **method.params,
        )
        reconciler: Reconciler | None = (
            None if reconciler_name == "none" else registry.create(reconciler_name)
        )
        specs.append(
            MethodSpec(
                forecaster=forecaster,
                reconciler=reconciler,
                quantile_strategy=variant.quantile_strategy,
                label=f"{method.display}+{reconciler_name}",
            )
        )
    return specs


def prepare_site(
    site: SiteData, variant: ResolvedVariant, *, seed: int
) -> tuple[SiteData, dict[str, Any]]:
    """Apply the variant's sensor profile, returning the observed site and a provenance note.

    The clean site is never modified. For the ``clean`` profile the observed site *is* the
    clean one, which makes the no-degradation condition a genuine control rather than a
    pass through a measurement model configured to do nothing.
    """
    if variant.sensor_profile == "clean":
        return site, {"sensor_profile": "clean", "degraded": False}
    profile_path = configs_dir("sensors") / f"{variant.sensor_profile}.yaml"
    profile = load_sensor_profile(profile_path)
    observed, report = degrade(site, profile, seed=seed)
    return observed, {
        "sensor_profile": profile.name,
        "degraded": True,
        "degradation": report.summary(),
    }


def _check_reconcilable(site: SiteData, variant: ResolvedVariant) -> None:
    """Refuse to reconcile a site whose flows were never measured.

    Every reconciler works from the conservation identity, which relates a node's change
    in occupancy to the flows across its doors. ROBOD counts people in rooms and nothing
    at the doorways, so on that site the identity has no observed terms: a reconciler
    would project onto a constraint built entirely from the forecaster's own guesses, and
    the resulting coherence residual would measure the forecaster's self-consistency
    rather than its agreement with the building. Running it and reporting the number
    would be worse than not running it, so the run stops here and says why.
    """
    reconcilers = {name for _method, name in variant.cells()} - {"none"}
    if not reconcilers:
        return
    measured = site.flow["count"].notna().any()
    if not measured:
        raise RunnerError(
            f"site {site.meta.site_id!r} records no doorway flow at all, so the "
            f"conservation identity has no observed terms and reconciler(s) "
            f"{sorted(reconcilers)} have nothing to reconcile against. Use "
            "`reconcilers: [none]` for this site, or evaluate the constraint machinery "
            "on a site that measures flow."
        )


def _panels(
    observed_site: SiteData, truth_site: SiteData, variant: ResolvedVariant
) -> tuple[Panel, Panel]:
    """The panel models see and the panel they are scored against."""
    keep = covariate_filter(variant.covariates)
    observed = observed_site.to_panel().select_covariates(keep=keep)
    # The truth panel is only ever indexed for actuals, so its covariates are irrelevant;
    # it is built from the clean site so that scoring is against what really happened.
    truth = truth_site.to_panel()
    return observed, truth


def _plan_for(config: ExperimentConfig, panel: Panel) -> RollingOriginPlan:
    protocol = config.protocol
    return build_plan(
        panel,
        context_length=protocol.context_length,
        horizons=protocol.horizons,
        stride=protocol.stride,
        quantiles=protocol.quantiles,
        fractions=protocol.split,
        max_origins=protocol.max_origins,
        require_observed=protocol.require_observed,
    )


def _tag(frame: pd.DataFrame, columns: dict[str, Any]) -> pd.DataFrame:
    """Write the condition columns onto a results frame."""
    return frame.assign(**columns) if len(frame) else frame


def run_one(
    config: ExperimentConfig,
    site_id: str,
    variant: ResolvedVariant,
    seed: int,
    *,
    data_root: Path | None = None,
    results_root: Path | None = None,
    allow_dirty: bool = False,
    overwrite: bool = False,
) -> RunOutcome:
    """Execute a single site/variant/seed run and write it to ``results/``.

    Args:
        config: the experiment.
        site_id: a directory under ``data/canonical/``.
        variant: the resolved condition.
        seed: the run seed.
        data_root: override ``data/canonical/``.
        results_root: override ``results/``.
        allow_dirty: permit a run from a working tree with uncommitted changes. Never set
            this for a reported result.
        overwrite: recompute a run whose results are already on disk.

    Raises:
        RunnerError: if the site is missing, or the plan cannot be built.
    """
    base = canonical_dir() if data_root is None else Path(data_root)
    site_path = base / site_id
    if not site_path.is_dir():
        raise RunnerError(
            f"site {site_id!r} is not under {base}. Fetch or simulate it first; this "
            "runner does not invent data."
        )

    set_global_seed(seed)
    truth_site = load_site(site_path)
    _check_reconcilable(truth_site, variant)
    observed_site, provenance = prepare_site(truth_site, variant, seed=seed)
    observed, truth = _panels(observed_site, truth_site, variant)
    plan = _plan_for(config, observed)

    run_config: dict[str, Any] = {
        "experiment": config.id,
        "site_id": site_id,
        "variant": variant.model_dump(),
        "protocol": config.protocol.model_dump(),
        "provenance": provenance,
        "n_origins": len(plan),
        # Recorded separately so that a run over a record with holes in it cannot look
        # like a run over a complete one with a longer stride.
        "n_origins_enumerated": len(plan) + len(plan.dropped),
        "n_origins_dropped": len(plan.dropped),
        "dropped_reasons": plan.describe()["dropped_reasons"],
    }
    manifest = RunManifest.create(
        experiment=config.id,
        seed=seed,
        config=run_config,
        run_id=f"{config.id}_{site_id}_{variant.label}_s{seed}",
        allow_dirty=allow_dirty,
        notes={
            "question": config.question,
            "origin_budget_warning": config.protocol.check_origin_budget(len(plan)),
            **config.notes,
        },
    )
    directory = (
        manifest.directory if results_root is None else Path(results_root) / manifest.run_id
    )
    if (directory / METRICS_FILE).is_file() and not overwrite:
        return RunOutcome(manifest=manifest, directory=directory, result=None, seconds=0.0)

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )

    graph = BuildingGraph.from_site(truth_site)
    specs = build_method_specs(variant, graph=graph, seed=seed)

    started = time.perf_counter()
    result = run_evaluation(
        observed_site,
        plan,
        specs,
        panel=_limited(observed, plan, variant, observed_site),
        truth_panel=truth,
        mase_season=config.protocol.mase_season,
    )
    elapsed = time.perf_counter() - started

    columns = dict(variant.condition_columns)
    columns.setdefault("site_id", site_id)
    _tag(result.predictions, columns).to_parquet(directory / PREDICTIONS_FILE, index=False)
    _tag(result.metrics, columns).to_parquet(directory / METRICS_FILE, index=False)
    _tag(result.resources, columns).to_parquet(directory / RESOURCES_FILE, index=False)
    _tag(result.reconciliation, columns).to_parquet(
        directory / RECONCILIATION_FILE, index=False
    )
    if config.risk is not None:
        # Scored against the clean site, because a congestion alert is right or wrong
        # about what really happened in the room, not about what the sensor reported.
        risk = evaluate_risk(result, truth_site, plan, config.risk, seed=seed)
        _tag(risk, columns).to_parquet(directory / RISK_FILE, index=False)
    return RunOutcome(
        manifest=manifest, directory=directory, result=result, seconds=elapsed
    )


def _limited(
    panel: Panel, plan: RollingOriginPlan, variant: ResolvedVariant, site: SiteData
) -> Panel:
    """Blank the training window a trained method is not allowed to fit on.

    The data-efficiency experiment gives a trained model a budget of local history. Rather
    than handing it a shorter panel -- which would also shorten every zero-shot context
    and change the protocol out from under the comparison -- the steps before the budget
    are set to NaN. The plan, the origins and the contexts are identical across budgets;
    only the amount of history that carries a value changes.
    """
    if variant.train_days is None:
        return panel
    steps_per_day = 24 * 3600 // site.meta.interval_seconds
    wanted = variant.train_days * steps_per_day
    start = plan.split.train_end - wanted
    if start <= 0:
        raise RunnerError(
            f"{variant.train_days} days is {wanted} steps but the training window of "
            f"{site.meta.site_id!r} is only {plan.split.train_end} steps. Shorten the "
            "budget or lengthen the record; it will not be quietly truncated."
        )
    series = panel.series.copy()
    series[:, :start] = float("nan")
    return Panel(
        series=series,
        series_ids=list(panel.series_ids),
        timestamps=panel.timestamps,
        past_covariates=panel.past_covariates,
        past_covariate_ids=list(panel.past_covariate_ids),
        future_covariates=panel.future_covariates,
        future_covariate_ids=list(panel.future_covariate_ids),
        interval_seconds=panel.interval_seconds,
        site_id=panel.site_id,
    )


def run_experiment(
    config: ExperimentConfig | str | Path,
    *,
    sites: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    variants: Sequence[str] | None = None,
    data_root: Path | None = None,
    results_root: Path | None = None,
    allow_dirty: bool = False,
    overwrite: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> list[RunOutcome]:
    """Execute every cell of an experiment's site by seed by variant grid.

    Args:
        config: the experiment, or a path to its YAML file.
        sites: restrict to these sites. Every name must be declared by the experiment.
        seeds: restrict to these seeds.
        variants: restrict to these variant labels.
        data_root: override ``data/canonical/``.
        results_root: override ``results/``.
        allow_dirty: permit runs from a dirty working tree.
        overwrite: recompute runs already on disk.
        on_progress: called with a one-line status before each run.

    Raises:
        RunnerError: if a restriction names something the experiment does not declare.
            A typo in ``--sites`` that silently ran nothing would look exactly like a
            sweep that completed.
    """
    experiment = config if isinstance(config, ExperimentConfig) else load_experiment(config)
    chosen_sites = _restrict(experiment.sites, sites, "site")
    chosen_seeds = _restrict(experiment.seeds, seeds, "seed")
    resolved = experiment.resolved_variants()
    if variants is not None:
        labels = _restrict([v.label for v in resolved], variants, "variant")
        resolved = [v for v in resolved if v.label in set(labels)]

    outcomes: list[RunOutcome] = []
    total = len(chosen_sites) * len(chosen_seeds) * len(resolved)
    index = 0
    for site_id in chosen_sites:
        for variant in resolved:
            for seed in chosen_seeds:
                index += 1
                if on_progress is not None:
                    on_progress(
                        f"[{index}/{total}] {experiment.id} {site_id} "
                        f"{variant.label} seed={seed}"
                    )
                outcomes.append(
                    run_one(
                        experiment,
                        site_id,
                        variant,
                        seed,
                        data_root=data_root,
                        results_root=results_root,
                        allow_dirty=allow_dirty,
                        overwrite=overwrite,
                    )
                )
    return outcomes


def _restrict(declared: Sequence[Any], requested: Sequence[Any] | None, kind: str) -> list[Any]:
    """Intersect a requested subset with what the experiment declares, loudly."""
    if requested is None:
        return list(declared)
    unknown = [item for item in requested if item not in set(declared)]
    if unknown:
        raise RunnerError(
            f"{kind}(s) {unknown} are not declared by this experiment; it declares "
            f"{list(declared)}"
        )
    return [item for item in declared if item in set(requested)]


def describe_experiment(config: ExperimentConfig) -> str:
    """A human-readable plan of what an experiment will run, for a dry run."""
    lines = [
        f"{config.id}: {config.question}",
        f"  sites      : {', '.join(config.sites)}",
        f"  seeds      : {config.seeds}",
        f"  horizons   : {config.protocol.horizons} (context {config.protocol.context_length}, "
        f"stride {config.protocol.stride})",
        f"  runs       : {config.n_runs()}",
    ]
    for variant in config.resolved_variants():
        cells = variant.cells()
        lines.append(
            f"  - {variant.label}: {len(cells)} cell(s), profile={variant.sensor_profile}, "
            f"covariates={variant.covariates}, train_days={variant.train_days}"
        )
        for method, reconciler in cells:
            lines.append(f"      {method.display}+{reconciler}")
    return "\n".join(lines)
