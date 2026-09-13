"""Reporting: tables and figures built only from logged runs.

The tests that matter most here are the refusals. A reporting module that quietly emits an
empty table when a run is missing is the easiest way to get a fabricated number into a
manuscript, so every "no run for this" path is tested explicitly.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mflow.eval.report import (
    METRICS_FILE,
    RECONCILIATION_FILE,
    RESOURCES_FILE,
    Crossover,
    ReportError,
    build_report,
    combine,
    covariate_table,
    data_efficiency,
    degradation_curve,
    degradation_figure,
    latency_table,
    load_runs,
    main_table,
    reconciliation_table,
    write_figure,
    write_table,
)
from mflow.manifest import RunManifest


def make_metrics(
    methods: dict[str, float],
    *,
    n_origins: int = 200,
    horizons=(5, 60),
    seed: int = 0,
    extra: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Synthetic per-origin metrics with a known ordering between methods."""
    generator = np.random.default_rng(seed)
    rows = []
    for method, level in methods.items():
        for group in ("occupancy", "flow", "all"):
            for horizon in horizons:
                errors = level + generator.normal(0.0, 0.05, n_origins)
                for origin, mae in enumerate(errors):
                    rows.append(
                        {
                            "method": method,
                            "origin": origin,
                            "series_group": group,
                            "horizon": horizon,
                            "mae": float(mae),
                            "wql": float(mae * 10),
                            "mase": float(mae),
                            "crps": float(mae * 0.8),
                            "coverage_80": 0.8,
                            "violation_rate": 0.0,
                            **(extra or {}),
                        }
                    )
    return pd.DataFrame(rows)


def write_run(
    directory,
    run_id: str,
    experiment: str,
    metrics: pd.DataFrame,
    *,
    config: dict[str, object] | None = None,
    extra_frames: dict[str, pd.DataFrame] | None = None,
):
    """Lay out a completed run on disk the way the harness does."""
    run_dir = directory / run_id
    run_dir.mkdir(parents=True)
    manifest = RunManifest(
        run_id=run_id,
        experiment=experiment,
        seed=0,
        config=config or {},
        config_hash="0" * 16,
        git={"commit": "abc", "branch": "main", "dirty": False, "dirty_files": []},
        packages={},
        hardware={},
        created_at="2026-01-01T00:00:00+00:00",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest.__dict__, indent=2, default=str), encoding="utf-8"
    )
    metrics.to_parquet(run_dir / METRICS_FILE, index=False)
    for name, frame in (extra_frames or {}).items():
        frame.to_parquet(run_dir / name, index=False)
    return run_dir


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def test_an_empty_results_directory_raises_rather_than_reporting_nothing(tmp_path) -> None:
    with pytest.raises(ReportError, match="no completed run"):
        load_runs(tmp_path)


def test_a_missing_results_directory_raises(tmp_path) -> None:
    with pytest.raises(ReportError, match="no completed run"):
        load_runs(tmp_path / "never_created")


def test_an_unfinished_run_is_skipped_not_half_reported(tmp_path) -> None:
    # A manifest is written before the run starts, so a directory holding only a manifest
    # is a run that crashed. Reporting its (absent) numbers is exactly what must not
    # happen.
    started = tmp_path / "crashed"
    started.mkdir()
    (started / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReportError, match="no completed run"):
        load_runs(tmp_path)


def test_completed_runs_are_found_and_filtered(tmp_path) -> None:
    write_run(tmp_path, "E1_a", "E1", make_metrics({"a": 1.0}, n_origins=5))
    write_run(tmp_path, "E4_b", "E4", make_metrics({"a": 1.0}, n_origins=5))
    assert {run.run_id for run in load_runs(tmp_path)} == {"E1_a", "E4_b"}
    assert [run.run_id for run in load_runs(tmp_path, experiment="E4")] == ["E4_b"]
    with pytest.raises(ReportError, match="experiment 'E9'"):
        load_runs(tmp_path, experiment="E9")


def test_combine_tags_every_row_with_its_run(tmp_path) -> None:
    write_run(
        tmp_path,
        "E1_a",
        "E1",
        make_metrics({"a": 1.0}, n_origins=5),
        config={"site_id": "sim_house_museum"},
    )
    write_run(tmp_path, "E1_b", "E1", make_metrics({"b": 2.0}, n_origins=5))
    combined = combine(load_runs(tmp_path))
    assert set(combined["run_id"]) == {"E1_a", "E1_b"}
    assert set(combined["site_id"].dropna()) == {"sim_house_museum"}


def test_a_run_without_the_requested_frame_raises(tmp_path) -> None:
    write_run(tmp_path, "E1_a", "E1", make_metrics({"a": 1.0}, n_origins=5))
    with pytest.raises(ReportError, match=r"has no resources\.parquet"):
        combine(load_runs(tmp_path), RESOURCES_FILE)


# --------------------------------------------------------------------------- #
# Main table
# --------------------------------------------------------------------------- #


def test_a_clearly_better_method_is_starred() -> None:
    metrics = make_metrics({"baseline": 1.0, "good": 0.5})
    table = main_table(metrics, reference="baseline")
    good = table[table["method"] == "good"]
    assert (good["significant"] == "*").all()
    assert (table[table["method"] == "baseline"]["significant"] == "").all()


def test_a_worse_method_is_never_starred() -> None:
    metrics = make_metrics({"baseline": 1.0, "bad": 2.0})
    table = main_table(metrics, reference="baseline")
    assert (table[table["method"] == "bad"]["significant"] == "").all()


def test_an_insignificant_improvement_is_reported_without_a_claim() -> None:
    # A tiny edge that the test cannot separate from noise. The number still appears --
    # the reader is entitled to see it -- but it carries no marker.
    metrics = make_metrics({"baseline": 1.0, "marginal": 0.999}, n_origins=200, seed=3)
    table = main_table(metrics, reference="baseline")
    marginal = table[table["method"] == "marginal"]
    assert marginal["mae"].notna().all()
    assert (marginal["significant"] == "").all()


def test_the_table_covers_every_group_and_horizon() -> None:
    table = main_table(make_metrics({"a": 1.0, "b": 0.5}), reference="a")
    assert set(table["series_group"]) == {"occupancy", "flow"}
    assert set(table["horizon"]) == {5, 60}
    assert len(table) == 2 * 2 * 2


def test_an_unknown_reference_method_raises() -> None:
    with pytest.raises(ReportError, match="is not in the metrics"):
        main_table(make_metrics({"a": 1.0}), reference="ghost")


def test_a_horizon_that_was_not_run_raises() -> None:
    with pytest.raises(ReportError, match="no metrics for group"):
        main_table(make_metrics({"a": 1.0}), reference="a", horizons=[999])


def test_an_untestable_cell_says_so_rather_than_looking_insignificant() -> None:
    # The Diebold-Mariano test refuses a 60-step horizon on 10 origins. The table still
    # reports the errors -- the reader is entitled to them -- but marks the comparison
    # n/a and records why. Rendering it as an empty marker would be indistinguishable
    # from "we tested and found nothing", which is a different and false claim.
    metrics = make_metrics({"a": 1.0, "b": 0.5}, n_origins=10, horizons=(60,))
    table = main_table(metrics, reference="a")
    tested = table[table["method"] == "b"]
    assert (tested["significant"] == "n/a").all()
    assert tested["significance_note"].str.contains("cannot support").all()
    assert tested["mae"].notna().all()
    # The reference row is labelled as such, not as untestable.
    assert (table[table["method"] == "a"]["significant"] == "").all()


def test_a_tested_and_insignificant_cell_is_blank_not_na() -> None:
    metrics = make_metrics({"a": 1.0, "marginal": 0.999}, n_origins=200, seed=3)
    table = main_table(metrics, reference="a")
    marginal = table[table["method"] == "marginal"]
    assert (marginal["significant"] == "").all()
    assert (marginal["significance_note"] == "").all()


# --------------------------------------------------------------------------- #
# Other tables
# --------------------------------------------------------------------------- #


def test_the_reconciliation_table_shows_both_sides() -> None:
    metrics = make_metrics({"raw": 1.0, "proj": 0.9}, n_origins=20)
    reconciliation = pd.DataFrame(
        {
            "method": ["proj"] * 20,
            "origin": range(20),
            "residual_before": 0.3,
            "residual_after": 1e-11,
            "violation_before": 0.05,
            "violation_after": 0.0,
            "adjustment_norm": 4.2,
            "solve_time_ms": 2.0,
            "anchor_staleness_steps": 0.0,
        }
    )
    table = reconciliation_table(metrics, reconciliation)
    projected = table[table["method"] == "proj"]
    assert (projected["residual_before"] > projected["residual_after"]).all()
    # The unreconciled method is still in the table with its coherence columns empty,
    # because "not reconciled" and "reconciled to zero" are different facts.
    assert table[table["method"] == "raw"]["residual_after"].isna().all()


def test_the_latency_table_reports_the_tail_as_well_as_the_median() -> None:
    resources = pd.DataFrame(
        {
            "method": ["fast"] * 100 + ["slow"] * 100,
            "latency_ms": list(np.full(100, 1.0)) + list(np.linspace(10.0, 200.0, 100)),
            "peak_memory_mb": [10.0] * 100 + [500.0] * 100,
        }
    )
    table = latency_table(resources, make_metrics({"fast": 1.0, "slow": 0.5}, n_origins=5))
    assert list(table["method"]) == ["fast", "slow"]
    slow = table[table["method"] == "slow"].iloc[0]
    assert slow["latency_ms_p95"] > slow["latency_ms_median"]
    assert slow["peak_memory_mb"] == 500.0
    assert slow["n_calls"] == 100


def test_a_covariate_ablation_without_a_recorded_condition_raises() -> None:
    with pytest.raises(ReportError, match="no 'covariates' column"):
        covariate_table(make_metrics({"a": 1.0}, n_origins=5))


def test_a_covariate_ablation_splits_by_condition() -> None:
    none = make_metrics({"a": 1.0}, n_origins=5, extra={"covariates": "none"})
    full = make_metrics({"a": 0.8}, n_origins=5, extra={"covariates": "full"}, seed=1)
    table = covariate_table(pd.concat([none, full], ignore_index=True))
    assert set(table["covariates"]) == {"none", "full"}
    # The "all" group is dropped, since the ablation is reported per series kind.
    assert set(table["series_group"]) == {"occupancy", "flow"}


# --------------------------------------------------------------------------- #
# Data efficiency
# --------------------------------------------------------------------------- #


def efficiency_metrics(levels: dict[int, tuple[float, float]]) -> pd.DataFrame:
    """One budget per entry, mapping days to (zero-shot MAE, trained MAE)."""
    frames = []
    for seed, (days, (zero, trained)) in enumerate(levels.items()):
        frames.append(
            make_metrics(
                {"zeroshot": zero, "trained": trained},
                n_origins=200,
                horizons=(5,),
                seed=seed,
                extra={"train_days": days},
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_the_crossover_is_located_and_marked_significant() -> None:
    metrics = efficiency_metrics(
        {1: (1.0, 3.0), 7: (1.0, 1.5), 30: (1.0, 0.6), 90: (1.0, 0.4)}
    )
    curve, crossover = data_efficiency(
        metrics, zero_shot="zeroshot", trained="trained", horizon=5
    )
    assert list(curve["train_days"]) == [1, 7, 30, 90]
    assert crossover.days_below == 7
    assert crossover.days_above == 30
    assert crossover.significant
    assert "crossover between 7 and 30 days" in crossover.describe()


def test_no_crossover_is_reported_as_no_crossover() -> None:
    metrics = efficiency_metrics({1: (1.0, 3.0), 7: (1.0, 2.0), 30: (1.0, 1.5)})
    _, crossover = data_efficiency(
        metrics, zero_shot="zeroshot", trained="trained", horizon=5
    )
    assert crossover.days_above is None
    assert crossover.describe() == "no crossover within the budgets that were run"


def test_the_confidence_interval_brackets_the_difference() -> None:
    metrics = efficiency_metrics({1: (1.0, 3.0), 30: (1.0, 0.6)})
    curve, _ = data_efficiency(metrics, zero_shot="zeroshot", trained="trained", horizon=5)
    assert (curve["ci_low"] < curve["difference"]).all()
    assert (curve["difference"] < curve["ci_high"]).all()


def test_a_budget_run_for_only_one_method_raises() -> None:
    both = make_metrics(
        {"zeroshot": 1.0, "trained": 2.0}, n_origins=50, horizons=(5,), extra={"train_days": 1}
    )
    one = make_metrics(
        {"zeroshot": 1.0}, n_origins=50, horizons=(5,), extra={"train_days": 30}
    )
    with pytest.raises(ReportError, match="was not run for both"):
        data_efficiency(
            pd.concat([both, one], ignore_index=True),
            zero_shot="zeroshot",
            trained="trained",
            horizon=5,
        )


def test_a_curve_without_a_budget_column_raises() -> None:
    with pytest.raises(ReportError, match="no 'train_days' column"):
        data_efficiency(
            make_metrics({"zeroshot": 1.0, "trained": 2.0}, n_origins=5),
            zero_shot="zeroshot",
            trained="trained",
            horizon=5,
        )


def test_the_crossover_description_names_the_significance() -> None:
    assert "not significant" in Crossover(7, 30, False).describe()
    assert "leads from the smallest budget" in Crossover(None, 1, True).describe()


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #


def degradation_metrics(levels: dict[str, float]) -> pd.DataFrame:
    frames = [
        make_metrics(
            {"m": level}, n_origins=50, horizons=(5,), seed=i, extra={"sensor_profile": profile}
        )
        for i, (profile, level) in enumerate(levels.items())
    ]
    return pd.concat(frames, ignore_index=True)


def test_the_degradation_curve_is_ordered_worst_to_best() -> None:
    metrics = degradation_metrics(
        {"harsh": 2.0, "clean": 0.5, "degraded": 1.5, "realistic": 0.8}
    )
    curve = degradation_curve(metrics, horizon=5)
    assert list(curve["sensor_profile"].astype(str)) == [
        "clean",
        "realistic",
        "degraded",
        "harsh",
    ]
    assert curve["mae"].is_monotonic_increasing


def test_a_missing_profile_leaves_a_gap_and_is_refused() -> None:
    metrics = degradation_metrics({"clean": 0.5, "realistic": 0.8, "harsh": 2.0})
    with pytest.raises(ReportError, match=r"no run for sensor profile\(s\) \['degraded'\]"):
        degradation_curve(metrics, horizon=5)


def test_a_curve_without_a_profile_column_raises() -> None:
    with pytest.raises(ReportError, match="no 'sensor_profile' column"):
        degradation_curve(make_metrics({"a": 1.0}, n_origins=5), horizon=5)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def test_a_table_is_written_as_both_csv_and_latex(tmp_path) -> None:
    table = main_table(make_metrics({"a": 1.0, "b": 0.5}), reference="a")
    paths = write_table(table, tmp_path, "main", caption="Accuracy by method.")
    assert [p.name for p in paths] == ["main.csv", "main.tex"]
    assert pd.read_csv(paths[0]).shape == table.shape
    tex = paths[1].read_text(encoding="utf-8")
    assert "Accuracy by method." in tex
    assert r"\label{tab:main}" in tex


def test_a_figure_is_written_as_both_pdf_and_png(tmp_path) -> None:
    metrics = degradation_metrics(
        {"clean": 0.5, "realistic": 0.8, "degraded": 1.5, "harsh": 2.0}
    )
    figure = degradation_figure(degradation_curve(metrics, horizon=5))
    paths = write_figure(figure, tmp_path, "degradation")
    assert [p.name for p in paths] == ["degradation.pdf", "degradation.png"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_build_report_records_what_it_could_not_build(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    write_run(
        results,
        "E1_x",
        "E1",
        make_metrics({"baseline": 1.0, "good": 0.5}),
        extra_frames={
            RECONCILIATION_FILE: pd.DataFrame(
                {
                    "method": ["good"],
                    "origin": [0],
                    "residual_before": [0.2],
                    "residual_after": [1e-12],
                    "violation_before": [0.0],
                    "violation_after": [0.0],
                    "adjustment_norm": [3.1],
                    "solve_time_ms": [1.0],
                    "anchor_staleness_steps": [0.0],
                }
            )
        },
    )
    output = tmp_path / "report"
    written = build_report(output, root=results, reference="baseline")

    assert "main" in written
    assert "reconciliation" in written
    # No resources frame and no covariate column were logged, so those two artefacts do
    # not exist and are recorded as absent rather than emitted empty.
    assert "latency" not in written
    assert not (output / "latency.csv").exists()

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["runs"] == ["E1_x"]
    assert report["reference_method"] == "baseline"
    assert "resources.parquet" in report["skipped"]["latency"]
    assert "covariates" in report["skipped"]


def test_build_report_picks_the_weakest_baseline_by_default(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    write_run(results, "E1_x", "E1", make_metrics({"weak": 2.0, "strong": 0.5}))
    build_report(tmp_path / "report", root=results)
    report = json.loads((tmp_path / "report" / "report.json").read_text(encoding="utf-8"))
    assert report["reference_method"] == "weak"


def test_build_report_on_an_empty_results_tree_raises(tmp_path) -> None:
    (tmp_path / "results").mkdir()
    with pytest.raises(ReportError, match="no completed run"):
        build_report(tmp_path / "report", root=tmp_path / "results")


# --------------------------------------------------------------------------- #
# Sites that measure only one side of the panel
# --------------------------------------------------------------------------- #
#
# Both real datasets in this project are half-panels: ROBOD counts people in rooms and
# nothing at the doorways, PVCGN counts fare-gate crossings and nothing in the station.
# Reporting a real E7_robod run turned up two places that assumed a full panel, and both
# of them stopped the report rather than degrading. Neither was caught by a suite whose
# fixtures always build occupancy, flow and all.


def half_panel_metrics(methods: dict[str, float], group: str, **kwargs) -> pd.DataFrame:
    """Metrics from a site that measures `group` and nothing else."""
    metrics = make_metrics(methods, **kwargs)
    return metrics[metrics["series_group"].isin([group, "all"])].reset_index(drop=True)


def test_a_site_with_no_flow_still_gets_a_main_table() -> None:
    metrics = half_panel_metrics({"good": 0.5, "weak": 1.0}, "occupancy")
    table = main_table(metrics, reference="weak")
    assert set(table["series_group"]) == {"occupancy"}
    assert not table.empty


def test_a_flow_only_site_still_gets_a_main_table() -> None:
    metrics = half_panel_metrics({"good": 0.5, "weak": 1.0}, "flow")
    assert set(main_table(metrics, reference="weak")["series_group"]) == {"flow"}


def test_naming_a_group_explicitly_still_demands_it() -> None:
    # Defaulting to what is present must not turn a run that was supposed to evaluate
    # flow and did not into a silently smaller table.
    metrics = half_panel_metrics({"good": 0.5, "weak": 1.0}, "occupancy")
    with pytest.raises(ReportError, match="no metrics for group 'flow'"):
        main_table(metrics, reference="weak", groups=("occupancy", "flow"))


def test_metrics_with_neither_group_say_so() -> None:
    metrics = make_metrics({"good": 0.5, "weak": 1.0})
    metrics = metrics[metrics["series_group"] == "all"].reset_index(drop=True)
    with pytest.raises(ReportError, match="none of which is 'occupancy' or 'flow'"):
        main_table(metrics, reference="weak")


def test_a_report_with_no_reconciled_run_skips_that_table_and_says_why() -> None:
    # `reconciler: none` everywhere is the normal state for a half-panel site, not a
    # malformed run, so it is a recorded skip rather than a crash.
    metrics = half_panel_metrics({"good": 0.5, "weak": 1.0}, "occupancy")
    with pytest.raises(ReportError, match="no run applied a reconciler"):
        reconciliation_table(metrics, pd.DataFrame())


def test_build_report_writes_the_main_table_for_a_half_panel_site(tmp_path) -> None:
    metrics = half_panel_metrics({"good": 0.5, "weak": 1.0}, "occupancy")
    results = tmp_path / "results"
    # The harness writes reconciliation.parquet even when nothing was reconciled, so the
    # frame exists and is empty rather than being absent.
    write_run(
        results,
        "E7_half_s0",
        "E7_half",
        metrics,
        extra_frames={RECONCILIATION_FILE: pd.DataFrame()},
    )
    written = build_report(tmp_path / "paper", root=results, reference="weak")
    assert "main" in written
    report = json.loads((tmp_path / "paper" / "report.json").read_text())
    assert "no run applied a reconciler" in report["skipped"]["reconciliation"]


def test_runs_from_two_experiments_cannot_share_a_table(tmp_path) -> None:
    # A horizon is a number of steps, and a step is five minutes on ROBOD and fifteen on
    # HZMetro. Pooling them would average two different questions under one row label.
    results = tmp_path / "results"
    for experiment in ("E7_robod", "E7_hzmetro"):
        write_run(
            results,
            f"{experiment}_s0",
            experiment,
            half_panel_metrics({"good": 0.5, "weak": 1.0}, "occupancy"),
        )
    with pytest.raises(ReportError, match="cannot share a table"):
        build_report(tmp_path / "paper", root=results, reference="weak")


def test_naming_the_experiment_reports_just_that_one(tmp_path) -> None:
    results = tmp_path / "results"
    for experiment in ("E7_robod", "E7_hzmetro"):
        write_run(
            results,
            f"{experiment}_s0",
            experiment,
            half_panel_metrics({"good": 0.5, "weak": 1.0}, "occupancy"),
        )
    build_report(tmp_path / "paper", root=results, reference="weak", experiment="E7_robod")
    report = json.loads((tmp_path / "paper" / "report.json").read_text())
    assert report["runs"] == ["E7_robod_s0"]
