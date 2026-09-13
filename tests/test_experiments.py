"""M7: experiment configuration and the runner.

The configuration tests check that a malformed experiment fails at load time rather than
after a GPU has been warmed up, and the runner tests check the two properties an
experiment layer has to have: a run writes exactly the artefacts it claims to, and a
condition can always be traced back from a row of the results.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from mflow.eval.report import METRICS_FILE, PREDICTIONS_FILE, RISK_FILE, load_runs
from mflow.experiments import (
    ExperimentConfig,
    RunnerError,
    covariate_filter,
    describe_experiment,
    load_experiment,
    run_experiment,
)
from mflow.paths import configs_dir
from mflow.schema import SiteData, load_site, write_site

SHIPPED = sorted(configs_dir("experiments").glob("*.yaml"))


def base_config(**overrides) -> dict:
    """A minimal valid experiment, cheap enough to actually run."""
    config = {
        "id": "T1",
        "question": "does the runner work?",
        "sites": ["toy"],
        "seeds": [0],
        "protocol": {
            "context_length": 48,
            "horizons": [5, 10],
            "stride": 6,
            "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "mase_season": 60,
        },
        "methods": [{"name": "last_value"}],
        "reconcilers": ["none"],
    }
    config.update(overrides)
    return config


@pytest.fixture
def toy_data(toy_site_path, tmp_path):
    """A canonical data root holding one copy of the toy site."""
    root = tmp_path / "canonical"
    shutil.copytree(toy_site_path, root / "toy")
    return root


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", SHIPPED, ids=[p.stem for p in SHIPPED])
def test_every_shipped_experiment_loads(path) -> None:
    config = load_experiment(path)
    assert config.id == path.stem
    assert config.question.strip()
    assert config.n_runs() > 0
    # describe() is what --dry-run prints, so it must not raise on any shipped config.
    assert config.id in describe_experiment(config)


@pytest.mark.parametrize("path", SHIPPED, ids=[p.stem for p in SHIPPED])
def test_every_shipped_experiment_can_support_its_own_test(path) -> None:
    # The origin budget is a property of the protocol, not of the data, so it can be
    # checked without running anything. An experiment configured with a stride that
    # cannot support the significance test at its own longest horizon would produce a
    # main table with no markers and no explanation of why.
    config = load_experiment(path)
    protocol = config.protocol
    if config.id == "smoke":
        pytest.skip("the smoke config is a plumbing check, not an experiment")
    # A 30-day record at the configured interval, 20% of which is the test window.
    test_window = 0.2 * (24 * 3600 // 60) * 30 if protocol.context_length > 100 else 0.2 * 2000
    available = int(test_window // protocol.stride)
    assert protocol.check_origin_budget(available) is None, (
        f"{path.stem} cannot support a Diebold-Mariano test at horizon {protocol.horizon} "
        f"with stride {protocol.stride}"
    )


def test_an_unknown_method_is_rejected_at_load_time() -> None:
    with pytest.raises(ValidationError, match="unknown forecaster"):
        ExperimentConfig.model_validate(base_config(methods=[{"name": "crystal_ball"}]))


def test_an_unknown_reconciler_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(base_config(reconcilers=["wishful_thinking"]))


def test_an_unknown_field_is_rejected_rather_than_ignored() -> None:
    # A typo in a key must not be silently dropped: `sensor_profil: harsh` would
    # otherwise run the whole sweep on clean data.
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(base_config(sensor_profil="harsh"))


def test_duplicate_variant_labels_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        ExperimentConfig.model_validate(
            base_config(variants=[{"label": "a"}, {"label": "a"}])
        )


def test_a_variant_with_no_methods_anywhere_is_rejected() -> None:
    with pytest.raises(ValidationError, match="has no methods"):
        ExperimentConfig.model_validate(
            base_config(methods=[], variants=[{"label": "empty"}])
        )


def test_quantiles_must_be_proper_and_distinct() -> None:
    for bad in ([0.0, 0.5, 0.9], [0.1, 0.1, 0.9], [0.1, 1.0]):
        config = base_config()
        config["protocol"]["quantiles"] = bad
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(config)


def test_the_protocol_has_no_defaults_for_the_things_that_move_a_number() -> None:
    for required in ("context_length", "horizons", "stride", "quantiles"):
        config = base_config()
        del config["protocol"][required]
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(config)


def test_a_variant_inherits_what_it_does_not_override() -> None:
    config = ExperimentConfig.model_validate(
        base_config(
            sensor_profile="realistic",
            covariates="calendar",
            variants=[{"label": "inherit"}, {"label": "override", "sensor_profile": "harsh"}],
        )
    )
    inherit, override = config.resolved_variants()
    assert inherit.sensor_profile == "realistic"
    assert inherit.covariates == "calendar"
    assert override.sensor_profile == "harsh"
    assert override.covariates == "calendar"


def test_a_variant_records_only_what_it_varies() -> None:
    config = ExperimentConfig.model_validate(
        base_config(variants=[{"label": "d7", "train_days": 7}])
    )
    columns = config.resolved_variants()[0].condition_columns
    assert columns == {"variant": "d7", "train_days": 7}
    # sensor_profile is not in the columns because this variant does not vary it; a
    # column that is constant across an experiment tells the reader nothing and invites
    # them to think it was a condition.
    assert "sensor_profile" not in columns


def test_the_cell_grid_is_the_cross_product() -> None:
    config = ExperimentConfig.model_validate(
        base_config(
            methods=[{"name": "last_value"}, {"name": "seasonal_naive"}],
            reconcilers=["none", "proposed"],
        )
    )
    assert len(config.resolved_variants()[0].cells()) == 4


def test_a_method_label_overrides_the_name_in_the_table() -> None:
    config = ExperimentConfig.model_validate(
        base_config(methods=[{"name": "last_value", "label": "persistence"}])
    )
    assert config.resolved_variants()[0].methods[0].display == "persistence"


def test_the_origin_budget_warning_names_the_shortfall() -> None:
    config = ExperimentConfig.model_validate(base_config())
    assert config.protocol.check_origin_budget(1000) is None
    message = config.protocol.check_origin_budget(5)
    assert message is not None
    assert "Diebold-Mariano" in message and "horizon 10" in message


def test_a_missing_experiment_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="experiment config not found"):
        load_experiment(tmp_path / "nope.yaml")


# --------------------------------------------------------------------------- #
# Covariate filtering
# --------------------------------------------------------------------------- #


def test_the_covariate_filter_partitions_the_channels() -> None:
    channels = ["global|is_open", "global|booked_groups", "foyer|co2_ppm", "study|temp_c"]

    assert covariate_filter("all") is None  # no predicate at all, not a trivial one

    none = covariate_filter("none")
    assert none is not None
    assert not any(none(c) for c in channels)

    calendar = covariate_filter("calendar")
    assert calendar is not None
    assert [c for c in channels if calendar(c)] == ["global|is_open", "global|booked_groups"]

    environment = covariate_filter("environment")
    assert environment is not None
    assert [c for c in channels if environment(c)] == ["foyer|co2_ppm", "study|temp_c"]


def test_removing_covariates_removes_them_from_the_panel(toy_site) -> None:
    panel = toy_site.to_panel()
    assert panel.past_covariate_ids  # the fixture carries CO2
    stripped = panel.select_covariates(keep=covariate_filter("none"))
    assert stripped.past_covariates is None
    assert stripped.future_covariates is None
    assert stripped.past_covariate_ids == []
    # The series themselves are untouched: an ablation removes context, not targets.
    np.testing.assert_array_equal(stripped.series, panel.series)


def test_keeping_everything_returns_the_same_panel(toy_site) -> None:
    panel = toy_site.to_panel()
    assert panel.select_covariates(keep=covariate_filter("all")) is panel


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #


def write_config(tmp_path, **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(base_config(**overrides)), encoding="utf-8")
    return path


def test_a_run_writes_every_artefact_it_claims(tmp_path, toy_data) -> None:
    results = tmp_path / "results"
    outcomes = run_experiment(
        write_config(tmp_path),
        data_root=toy_data,
        results_root=results,
        allow_dirty=True,
    )
    assert len(outcomes) == 1
    directory = outcomes[0].directory
    assert {p.name for p in directory.iterdir()} == {
        "manifest.json",
        "predictions.parquet",
        METRICS_FILE,
        "resources.parquet",
        "reconciliation.parquet",
    }
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 0
    assert manifest["config"]["site_id"] == "toy"
    assert manifest["config"]["n_origins"] > 0


def test_the_risk_file_appears_only_when_the_experiment_asks(tmp_path, toy_data) -> None:
    plain = run_experiment(
        write_config(tmp_path),
        data_root=toy_data,
        results_root=tmp_path / "plain",
        allow_dirty=True,
    )
    assert not (plain[0].directory / RISK_FILE).exists()

    with_risk = run_experiment(
        write_config(
            tmp_path / "risk",
            # The toy fixture spans four hours inside opening time, so there is no
            # closed window to place a closed-gallery anomaly in. Asking for one would
            # (correctly) raise; the head's refusal is tested in test_risk.py.
            risk={
                "anomaly_kinds": ["dwell_cluster", "stuck_sensor"],
                "anomaly_per_kind": 1,
                "anomaly_duration": 2,
                "alert_probability": 0.3,
            },
        ),
        data_root=toy_data,
        results_root=tmp_path / "risky",
        allow_dirty=True,
    )
    frame = pd.read_parquet(with_risk[0].directory / RISK_FILE)
    assert set(frame["head"]) == {"congestion", "anomaly", "exposure"}
    # The exposure caveat has to survive into the file, because a table built from it
    # would otherwise present an illustrative budget as a measured limit.
    exposure = frame[frame["head"] == "exposure"]
    assert exposure["caveat"].str.contains("Illustrative only").all()
    assert pd.api.types.is_numeric_dtype(frame["value"])


def test_every_row_can_be_traced_to_its_condition(tmp_path, toy_data) -> None:
    run_experiment(
        write_config(
            tmp_path,
            variants=[
                {"label": "no_cov", "covariates": "none"},
                {"label": "all_cov", "covariates": "all"},
            ],
        ),
        data_root=toy_data,
        results_root=tmp_path / "results",
        allow_dirty=True,
    )
    runs = load_runs(tmp_path / "results")
    assert len(runs) == 2
    for run in runs:
        for filename in (METRICS_FILE, PREDICTIONS_FILE):
            frame = run.frame(filename)
            assert set(frame["variant"].unique()) <= {"no_cov", "all_cov"}
            assert set(frame["covariates"].unique()) <= {"none", "all"}
            assert (frame["site_id"] == "toy").all()


def test_a_completed_run_is_not_recomputed(tmp_path, toy_data) -> None:
    config = write_config(tmp_path)
    first = run_experiment(
        config, data_root=toy_data, results_root=tmp_path / "r", allow_dirty=True
    )
    assert not first[0].skipped
    second = run_experiment(
        config, data_root=toy_data, results_root=tmp_path / "r", allow_dirty=True
    )
    assert second[0].skipped
    third = run_experiment(
        config,
        data_root=toy_data,
        results_root=tmp_path / "r",
        allow_dirty=True,
        overwrite=True,
    )
    assert not third[0].skipped


def test_a_run_is_deterministic(tmp_path, toy_data) -> None:
    config = write_config(tmp_path, sensor_profile="realistic")
    frames = []
    for name in ("a", "b"):
        outcome = run_experiment(
            config, data_root=toy_data, results_root=tmp_path / name, allow_dirty=True
        )[0]
        frames.append(pd.read_parquet(outcome.directory / PREDICTIONS_FILE))
    pd.testing.assert_frame_equal(*frames)


def test_a_missing_site_fails_immediately(tmp_path, toy_data) -> None:
    with pytest.raises(RunnerError, match="does not invent data"):
        run_experiment(
            write_config(tmp_path, sites=["nonexistent"]),
            data_root=toy_data,
            results_root=tmp_path / "r",
            allow_dirty=True,
        )


def test_a_typo_in_a_restriction_fails_rather_than_running_nothing(tmp_path, toy_data) -> None:
    # A sweep that ran zero cells because of a typo looks exactly like a sweep that
    # completed, which is why this raises instead of returning an empty list.
    with pytest.raises(RunnerError, match="not declared by this experiment"):
        run_experiment(
            write_config(tmp_path),
            sites=["toyy"],
            data_root=toy_data,
            results_root=tmp_path / "r",
            allow_dirty=True,
        )


def test_a_training_budget_the_record_cannot_meet_is_refused(tmp_path, toy_data) -> None:
    with pytest.raises(RunnerError, match="will not be quietly truncated"):
        run_experiment(
            write_config(tmp_path, variants=[{"label": "d30", "train_days": 30}]),
            data_root=toy_data,
            results_root=tmp_path / "r",
            allow_dirty=True,
        )


def test_degradation_is_applied_and_scored_against_the_truth(tmp_path, toy_data) -> None:
    clean = run_experiment(
        write_config(tmp_path / "c", sensor_profile="clean"),
        data_root=toy_data,
        results_root=tmp_path / "clean",
        allow_dirty=True,
    )[0]
    harsh = run_experiment(
        write_config(tmp_path / "h", sensor_profile="harsh"),
        data_root=toy_data,
        results_root=tmp_path / "harsh",
        allow_dirty=True,
    )[0]

    clean_predictions = pd.read_parquet(clean.directory / PREDICTIONS_FILE)
    harsh_predictions = pd.read_parquet(harsh.directory / PREDICTIONS_FILE)
    # Same actuals: both are scored against the clean site.
    np.testing.assert_allclose(
        clean_predictions["actual"].to_numpy(),
        harsh_predictions["actual"].to_numpy(),
        equal_nan=True,
    )
    # Different forecasts: only the harsh run saw damaged inputs.
    assert not np.allclose(
        clean_predictions["q50"].to_numpy(),
        harsh_predictions["q50"].to_numpy(),
        equal_nan=True,
    )
    assert pd.read_parquet(harsh.directory / METRICS_FILE)["mae"].mean() > 0


def test_a_dirty_tree_is_refused_by_default(tmp_path, toy_data, monkeypatch) -> None:
    monkeypatch.setattr(
        "mflow.manifest.git_state",
        lambda: {"commit": "abc", "branch": "main", "dirty": True, "dirty_files": ["x.py"]},
    )
    from mflow.manifest import DirtyWorkingTreeError

    with pytest.raises(DirtyWorkingTreeError, match="dirty working tree"):
        run_experiment(
            write_config(tmp_path),
            data_root=toy_data,
            results_root=tmp_path / "r",
            allow_dirty=False,
        )


def test_a_crashed_run_leaves_a_manifest_and_no_metrics(tmp_path, toy_data, monkeypatch) -> None:
    # The manifest is written before the evaluation starts, so a run that dies partway
    # leaves a directory recording the attempt but no metrics. Reporting must skip it:
    # the difference between "finished" and "was attempted" has to survive on disk.
    def explode(*_args, **_kwargs):
        raise RuntimeError("the forecaster fell over")

    monkeypatch.setattr("mflow.experiments.runner.run_evaluation", explode)
    results = tmp_path / "results"
    with pytest.raises(RuntimeError, match="fell over"):
        run_experiment(
            write_config(tmp_path),
            data_root=toy_data,
            results_root=results,
            allow_dirty=True,
        )
    directory = next(results.iterdir())
    assert (directory / "manifest.json").is_file()
    assert not (directory / METRICS_FILE).exists()
    with pytest.raises(Exception, match="no completed run"):
        load_runs(results)


def test_an_unloadable_site_fails_before_any_results_are_created(tmp_path, toy_site_path) -> None:
    # A site that cannot even be validated never gets a manifest, because there is
    # nothing yet to record the provenance of.
    root = tmp_path / "canonical"
    broken = load_site(toy_site_path)
    broken.occupancy.loc[:, "count"] = np.nan
    write_site(broken, root / "toy", validate=False)

    results = tmp_path / "results"
    with pytest.raises(Exception, match="cannot be checked at all"):
        run_experiment(
            write_config(tmp_path),
            data_root=root,
            results_root=results,
            allow_dirty=True,
        )
    assert not results.exists()


def test_a_site_without_measured_flow_cannot_be_reconciled(tmp_path, toy_data) -> None:
    # ROBOD counts people in rooms and nothing at the doorways. Reconciling there would
    # project onto a constraint built entirely from the forecaster's own predicted flows,
    # so the coherence residual would measure self-consistency rather than agreement with
    # the building. The run has to stop and say that, not produce a number.
    site = load_site(toy_data / "toy")
    flowless = SiteData(
        meta=site.meta.model_copy(update={"has_ground_truth_flow": False}),
        nodes=site.nodes,
        edges=site.edges,
        occupancy=site.occupancy,
        flow=site.flow.assign(count=np.nan),
        covariates_past=site.covariates_past,
        covariates_future=site.covariates_future,
    )
    write_site(flowless, toy_data / "toy")

    with pytest.raises(RunnerError, match="records no doorway flow at all"):
        run_experiment(
            write_config(tmp_path, reconcilers=["none", "proposed"]),
            data_root=toy_data,
            results_root=tmp_path / "results",
            allow_dirty=True,
        )

    # The unreconciled arm is still perfectly runnable on such a site.
    outcomes = run_experiment(
        write_config(tmp_path / "b", reconcilers=["none"]),
        data_root=toy_data,
        results_root=tmp_path / "results",
        allow_dirty=True,
    )
    assert len(outcomes) == 1


def test_dropped_origins_reach_the_manifest(tmp_path, toy_data) -> None:
    site = load_site(toy_data / "toy")
    holed = SiteData(
        meta=site.meta,
        nodes=site.nodes,
        edges=site.edges,
        occupancy=site.occupancy.assign(
            count=site.occupancy["count"].where(
                site.occupancy["timestamp"] < site.occupancy["timestamp"].max()
                - pd.Timedelta(minutes=25)
            )
        ),
        flow=site.flow,
        covariates_past=site.covariates_past,
        covariates_future=site.covariates_future,
    )
    write_site(holed, toy_data / "toy", validate=False)

    outcomes = run_experiment(
        write_config(
            tmp_path,
            protocol={
                "context_length": 20,
                "horizons": [4],
                "stride": 1,
                "quantiles": [0.1, 0.5, 0.9],
                "mase_season": 60,
                "require_observed": True,
            },
        ),
        data_root=toy_data,
        results_root=tmp_path / "results",
        allow_dirty=True,
    )
    manifest = json.loads(
        (outcomes[0].directory / "manifest.json").read_text(encoding="utf-8")
    )
    # A run over a gappy record evaluates on fewer origins than its stride implies, and
    # the reader of the results has to be able to see that from the manifest alone.
    assert manifest["config"]["n_origins_dropped"] > 0
    assert manifest["config"]["n_origins"] < manifest["config"]["n_origins_enumerated"]


def test_a_site_without_measured_occupancy_cannot_be_reconciled(tmp_path, toy_data) -> None:
    # PVCGN counts fare-gate crossings and nothing standing in a station. That is the
    # mirror image of ROBOD, and it fails for the same reason: only one side of the
    # conservation identity is observed.
    site = load_site(toy_data / "toy")
    write_site(
        SiteData(
            meta=site.meta.model_copy(update={"has_ground_truth_flow": False}),
            nodes=site.nodes,
            edges=site.edges,
            occupancy=site.occupancy.assign(count=np.nan),
            flow=site.flow,
            covariates_past=site.covariates_past,
            covariates_future=site.covariates_future,
        ),
        toy_data / "toy",
    )
    with pytest.raises(RunnerError, match="records no occupancy at all"):
        run_experiment(
            write_config(tmp_path, reconcilers=["proposed"]),
            data_root=toy_data,
            results_root=tmp_path / "results",
            allow_dirty=True,
        )
