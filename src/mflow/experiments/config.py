"""Declarative experiment configuration (M7).

An experiment is a YAML file. It names the sites, the methods, the reconcilers and the
rolling-origin protocol, and it lists the *variants* that make it an experiment rather
than a single run: the sensor profiles of E5, the training budgets of E4, the covariate
sets of E2. The runner takes the cross product of sites, seeds and variants, and every
resulting run is logged separately with its variant recorded in the metrics.

Nothing here has a default that silently changes a result. In particular:

* the protocol block has no defaults at all, because a context length or a stride chosen
    20|  by the code rather than by the config would make two experiments incomparable without
  anything in either config saying so;
* a variant must carry a label, which becomes the value of its column in the metrics,
  so that the reporting module can attribute every row to a condition;
* every field is validated at load time, so a typo in a method name fails before a GPU
  is warmed up rather than three hours in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mflow.experiments.risk_eval import RiskConfig
from mflow.forecast import available_forecasters

#: Covariate sets the ablation can request. ``none`` strips every channel, ``calendar``
#: keeps the ones a museum knows in advance, ``environment`` keeps the sensed ones, and
#: ``all`` keeps whatever the site carries.
CovariateSet = Literal["none", "calendar", "environment", "all"]

#: Reconcilers an experiment can name. ``none`` is the unreconciled forecast.
ReconcilerName = Literal["none", "mint", "uniform_weight", "proposed"]

#: How a point reconciliation is carried through the quantile fan.
QuantileStrategy = Literal["shift", "per_quantile"]


class ExperimentConfigError(ValueError):
    """Raised when an experiment file cannot be resolved into runs."""


class MethodConfig(BaseModel):
    """One forecasting method and its hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    #: Name accepted by :func:`mflow.forecast.build_forecaster`.
    name: str
    #: Label used in the results tables. Defaults to ``name``.
    label: str | None = None
    #: Constructor keyword arguments, passed through verbatim.
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _known_method(cls, value: str) -> str:
        known = available_forecasters()
        if value not in known:
            raise ValueError(f"unknown forecaster {value!r}; known: {known}")
        return value

    @property
    def display(self) -> str:
        """The name this method appears under in a results table."""
        return self.label or self.name


class ProtocolConfig(BaseModel):
    """The rolling-origin protocol, stated in full with no defaults.

    ``stride`` interacts with the significance test: the Diebold-Mariano statistic needs
    roughly more than twice the horizon in origins before the Harvey-Leybourne-Newbold
    correction is defined, so a long horizon evaluated at a coarse stride produces a
    results table with no significance markers in it. :meth:`check_origin_budget` says so
    at load time rather than at reporting time.
    """

    model_config = ConfigDict(extra="forbid")

    context_length: int = Field(gt=0)
    horizons: list[int] = Field(min_length=1)
    stride: int = Field(gt=0)
    quantiles: list[float] = Field(min_length=2)
    split: tuple[float, float, float] = (0.6, 0.2, 0.2)
    max_origins: int | None = Field(default=None, gt=0)
    #: Seasonal lag for the MASE denominator, in steps. ``None`` means one day.
    mase_season: int | None = Field(default=None, gt=0)

    @field_validator("horizons")
    @classmethod
    def _positive_horizons(cls, value: list[int]) -> list[int]:
        if any(h < 1 for h in value):
            raise ValueError(f"horizons must be positive, got {value}")
        return sorted(set(value))

    @field_validator("quantiles")
    @classmethod
    def _proper_quantiles(cls, value: list[float]) -> list[float]:
        if any(not 0.0 < q < 1.0 for q in value):
            raise ValueError(f"quantiles must lie strictly inside (0, 1), got {value}")
        ordered = sorted(set(value))
        if len(ordered) != len(value):
            raise ValueError(f"quantiles contain duplicates: {value}")
        return ordered

    @property
    def horizon(self) -> int:
        """The longest horizon, which is what is actually forecast."""
        return self.horizons[-1]

    def check_origin_budget(self, n_origins: int) -> str | None:
        """Warn if the plan cannot support a significance test at the longest horizon.

        Returns:
            A message when the budget is too small, otherwise None. The runner reports it
            and continues, because an experiment that only needs point estimates is a
            legitimate thing to run; what is not legitimate is discovering the shortfall
            after the run and quietly dropping the markers.
        """
        needed = 2 * self.horizon + 2
        if n_origins < needed:
            return (
                f"{n_origins} origins cannot support a Diebold-Mariano test at horizon "
                f"{self.horizon}, which needs about {needed}. The main table will have no "
                f"significance markers at that horizon. Reduce stride below "
                f"{max(1, self.stride * n_origins // needed)} to fix it."
            )
        return None


class Variant(BaseModel):
    """One condition of the experiment.

    Every field left unset falls back to the experiment-level value, so a variant states
    only what it changes. The set fields are written into the metrics frame as columns,
    which is how :mod:`mflow.eval.report` attributes a row to a condition.
    """

    model_config = ConfigDict(extra="forbid")

    #: Short identifier, unique within the experiment.
    label: str
    #: Sensor profile applied to the clean site, or ``clean`` for no degradation.
    sensor_profile: str | None = None
    #: Days of local training history a trained method may fit on.
    train_days: int | None = None
    #: Which covariate channels the panel carries.
    covariates: CovariateSet | None = None
    #: Methods, overriding the experiment-level list.
    methods: list[MethodConfig] | None = None
    #: Reconcilers, overriding the experiment-level list.
    reconcilers: list[ReconcilerName] | None = None
    #: Quantile reconciliation strategy, overriding the experiment-level value.
    quantile_strategy: QuantileStrategy | None = None

    def condition_columns(self) -> dict[str, Any]:
        """The condition columns this variant contributes to the metrics frame."""
        columns: dict[str, Any] = {"variant": self.label}
        for field_name in ("sensor_profile", "train_days", "covariates"):
            value = getattr(self, field_name)
            if value is not None:
                columns[field_name] = value
        return columns


class ExperimentConfig(BaseModel):
    """A complete experiment, as loaded from ``configs/experiments/<id>.yaml``."""

    model_config = ConfigDict(extra="forbid")

    #: Identifier used in the run id and the manifest, e.g. ``E1``.
    id: str
    #: One-line statement of the question the experiment answers.
    question: str
    #: Site directories under ``data/canonical/``.
    sites: list[str] = Field(min_length=1)
    #: The rolling-origin protocol.
    protocol: ProtocolConfig
    #: Methods every variant runs unless it overrides them.
    methods: list[MethodConfig] = Field(default_factory=list)
    #: Reconcilers every variant runs unless it overrides them.
    reconcilers: list[ReconcilerName] = Field(default=["none"])
    #: Quantile reconciliation strategy.
    quantile_strategy: QuantileStrategy = "shift"
    #: Sensor profile applied unless a variant overrides it.
    sensor_profile: str = "clean"
    #: Covariate set unless a variant overrides it.
    covariates: CovariateSet = "all"
    #: Training budget in days unless a variant overrides it. ``None`` is the whole
    #: training window.
    train_days: int | None = Field(default=None, gt=0)
    #: Run seeds. More than one is how a claim is separated from a lucky draw.
    seeds: list[int] = Field(default=[0], min_length=1)
    #: The conditions. A single unnamed variant is the degenerate one-condition case.
    variants: list[Variant] = Field(default_factory=lambda: [Variant(label="default")])
    #: Score the decision heads as well as the forecast. ``None`` skips them, which is
    #: the right default for every experiment except E6: running them costs little but
    #: writing a risk table for an experiment that was not designed to answer a decision
    #: question invites the table to be read as if it had been.
    risk: RiskConfig | None = None
    #: Free-text notes recorded verbatim in every manifest this experiment produces.
    notes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _every_variant_has_methods(self) -> ExperimentConfig:
        labels = [variant.label for variant in self.variants]
        if len(set(labels)) != len(labels):
            raise ValueError(f"variant labels must be unique, got {labels}")
        for variant in self.variants:
            if not (variant.methods or self.methods):
                raise ValueError(
                    f"variant {variant.label!r} has no methods and the experiment declares "
                    "none either"
                )
        return self

    def resolve(self, variant: Variant) -> ResolvedVariant:
        """Fill a variant's unset fields from the experiment-level values."""
        return ResolvedVariant(
            label=variant.label,
            methods=variant.methods or self.methods,
            reconcilers=variant.reconcilers or self.reconcilers,
            quantile_strategy=variant.quantile_strategy or self.quantile_strategy,
            sensor_profile=variant.sensor_profile or self.sensor_profile,
            covariates=variant.covariates or self.covariates,
            train_days=variant.train_days if variant.train_days is not None else self.train_days,
            condition_columns=variant.condition_columns(),
        )

    def resolved_variants(self) -> list[ResolvedVariant]:
        """Every variant, fully resolved, in declaration order."""
        return [self.resolve(variant) for variant in self.variants]

    def n_runs(self) -> int:
        """How many runs this experiment will produce."""
        return len(self.sites) * len(self.seeds) * len(self.variants)


class ResolvedVariant(BaseModel):
    """A variant with every field filled in, ready to be turned into runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    methods: list[MethodConfig]
    reconcilers: list[ReconcilerName]
    quantile_strategy: QuantileStrategy
    sensor_profile: str
    covariates: CovariateSet
    train_days: int | None
    #: Columns written onto every results frame, so a row can be attributed
    #: to the condition that produced it.
    condition_columns: dict[str, Any]

    def cells(self) -> list[tuple[MethodConfig, ReconcilerName]]:
        """The method-by-reconciler grid this variant runs."""
        return [
            (method, reconciler)
            for method in self.methods
            for reconciler in self.reconcilers
        ]


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Read and validate an experiment file.

    Raises:
        FileNotFoundError: if the file does not exist.
        pydantic.ValidationError: if the experiment is malformed.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"experiment config not found: {file}")
    payload = yaml.safe_load(file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExperimentConfigError(f"{file} does not contain a YAML mapping")
    return ExperimentConfig.model_validate(payload)
