"""M1 acceptance: the canonical data contract.

The acceptance criterion is that ``validate_site`` passes on the committed toy site and
fails with a *specific* error on each deliberately corrupted variant. Each corruption
test therefore asserts on the error text, not merely on the exception type: a validator
that raised a generic message for every fault would satisfy the type check while being
useless in practice.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mflow.schema import (
    OCCUPANCY_FILE,
    Panel,
    SiteValidationError,
    conservation_residual,
    load_site,
    occ_series_id,
    parse_series_id,
    validate_site,
)

Corrupter = Callable[[str, Callable[[pd.DataFrame], pd.DataFrame]], Path]


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


def test_toy_site_validates(toy_site_path: Path) -> None:
    site = validate_site(toy_site_path)
    assert site.meta.site_id == "toy_three_room"
    assert site.meta.interval_seconds == 60


def test_toy_site_is_conservation_exact(toy_site) -> None:
    residual = conservation_residual(toy_site)
    assert np.nanmax(np.abs(residual.to_numpy())) == 0.0


def test_canonical_series_order(toy_site) -> None:
    # Occupancy of interior nodes first, then edge flows, each in file order.
    assert toy_site.series_ids[:3] == ["occ:foyer", "occ:gallery_a", "occ:gallery_b"]
    assert toy_site.series_ids[3:] == [
        "flow:e_in",
        "flow:e_out",
        "flow:e_fa",
        "flow:e_af",
        "flow:e_ab",
        "flow:e_ba",
    ]
    assert "occ:outside" not in toy_site.series_ids


def test_panel_round_trip(toy_site) -> None:
    panel = toy_site.to_panel(horizon=12)
    assert panel.series.dtype == np.float32
    assert panel.series.shape == (9, 240)
    assert panel.timestamps.tz is not None
    assert panel.past_covariates is not None
    assert panel.past_covariates.shape == (3, 240)
    assert panel.future_covariates is not None
    assert panel.future_covariates.shape == (1, 252)
    assert panel.horizon_covered == 12
    assert panel.index_of("occ:foyer") == 0


def test_panel_slice_keeps_alignment(toy_site) -> None:
    panel = toy_site.to_panel(horizon=12)
    window = panel.slice_time(100, 160, horizon=12)
    assert window.n_timesteps == 60
    assert window.future_covariates is not None
    assert window.future_covariates.shape == (1, 72)
    np.testing.assert_array_equal(window.series, panel.series[:, 100:160])


def test_panel_slice_rejects_uncovered_horizon(toy_site) -> None:
    panel = toy_site.to_panel(horizon=0)
    with pytest.raises(ValueError, match="future covariates cover"):
        panel.slice_time(0, 240, horizon=60)


def test_parse_series_id_round_trip() -> None:
    assert parse_series_id(occ_series_id("gallery_a")) == ("occupancy", "gallery_a")
    assert parse_series_id("flow:e_in") == ("flow", "e_in")
    with pytest.raises(ValueError, match="neither"):
        parse_series_id("gallery_a")


def test_panel_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="float32"):
        Panel(
            series=np.zeros((2, 5), dtype=np.float64),
            series_ids=["occ:a", "occ:b"],
            timestamps=pd.date_range("2026-01-01", periods=5, freq="60s", tz="UTC"),
            past_covariates=None,
            past_covariate_ids=[],
            future_covariates=None,
            future_covariate_ids=[],
            interval_seconds=60,
            site_id="x",
        )


# --------------------------------------------------------------------------- #
# Corrupted variants. Each must fail, and fail specifically.
# --------------------------------------------------------------------------- #


def test_rejects_unknown_node_reference(corrupt: Corrupter) -> None:
    site = corrupt(
        OCCUPANCY_FILE,
        lambda df: df.assign(node_id=df["node_id"].replace("gallery_b", "ghost_room")),
    )
    with pytest.raises(SiteValidationError, match="unknown node ids"):
        validate_site(site)


def test_rejects_unknown_edge_reference(corrupt: Corrupter) -> None:
    site = corrupt(
        "flow.parquet", lambda df: df.assign(edge_id=df["edge_id"].replace("e_ab", "e_x"))
    )
    with pytest.raises(SiteValidationError, match="unknown edge ids"):
        validate_site(site)


def test_rejects_edge_to_missing_node(corrupt: Corrupter) -> None:
    site = corrupt(
        "edges.csv",
        lambda df: df.assign(dst_node=df["dst_node"].replace("gallery_b", "nowhere")),
    )
    with pytest.raises(SiteValidationError, match="references unknown nodes"):
        validate_site(site)


def test_rejects_missing_timestamp_in_grid(corrupt: Corrupter) -> None:
    stamps = pd.read_parquet(Path("tests/fixtures/toy_site") / OCCUPANCY_FILE)["timestamp"]
    drop = sorted(stamps.unique())[100]
    site = corrupt(OCCUPANCY_FILE, lambda df: df[df["timestamp"] != drop])
    corrupt("flow.parquet", lambda df: df[df["timestamp"] != drop])
    with pytest.raises(SiteValidationError, match="regular 60s grid"):
        validate_site(site)


def test_rejects_ragged_panel(corrupt: Corrupter) -> None:
    # Remove a single (timestamp, node) cell: the grid is intact but the panel is ragged.
    site = corrupt(
        OCCUPANCY_FILE,
        lambda df: df.drop(df.index[(df["node_id"] == "gallery_a")][5]),
    )
    with pytest.raises(SiteValidationError, match="is ragged"):
        validate_site(site)


def test_rejects_negative_counts(corrupt: Corrupter) -> None:
    def _negate(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.loc[df.index[10], "count"] = -4
        return df

    site = corrupt(OCCUPANCY_FILE, _negate)
    with pytest.raises(SiteValidationError, match="negative counts"):
        validate_site(site)


def test_rejects_occupancy_above_capacity(corrupt: Corrupter) -> None:
    def _overfill(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.loc[df["node_id"] == "gallery_b", "count"] = 999
        return df

    site = corrupt(OCCUPANCY_FILE, _overfill)
    with pytest.raises(SiteValidationError, match="exceeds node capacity"):
        validate_site(site)


def test_rejects_duplicate_rows(corrupt: Corrupter) -> None:
    site = corrupt(OCCUPANCY_FILE, lambda df: pd.concat([df, df.iloc[[0]]], ignore_index=True))
    with pytest.raises(SiteValidationError, match="duplicate \\(timestamp, node_id\\) rows"):
        validate_site(site)


def test_rejects_timezone_naive_timestamps(corrupt: Corrupter) -> None:
    site = corrupt(
        OCCUPANCY_FILE, lambda df: df.assign(timestamp=df["timestamp"].dt.tz_localize(None))
    )
    with pytest.raises(SiteValidationError, match="timezone-naive"):
        validate_site(site)


def test_rejects_conservation_violation(corrupt: Corrupter) -> None:
    def _break_balance(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mask = df["edge_id"] == "e_in"
        df.loc[df.index[mask][50], "count"] = int(df.loc[df.index[mask][50], "count"]) + 7
        return df

    site = corrupt("flow.parquet", _break_balance)
    with pytest.raises(SiteValidationError, match="conservation residual"):
        validate_site(site)


def test_rejects_missing_outside_node(corrupt: Corrupter) -> None:
    site = corrupt("nodes.csv", lambda df: df.assign(kind=df["kind"].replace("outside", "gallery")))
    with pytest.raises(SiteValidationError, match="exactly one node of kind 'outside'"):
        validate_site(site)


def test_rejects_unknown_node_kind(corrupt: Corrupter) -> None:
    site = corrupt("nodes.csv", lambda df: df.assign(kind=df["kind"].replace("gallery", "atrium")))
    with pytest.raises(SiteValidationError, match="unknown node kinds"):
        validate_site(site)


def test_rejects_asymmetric_reverse_edge(corrupt: Corrupter) -> None:
    site = corrupt(
        "edges.csv",
        lambda df: df.assign(reverse_edge_id=df["reverse_edge_id"].replace("e_out", "e_ab")),
    )
    with pytest.raises(SiteValidationError, match="reverse_edge_id is not symmetric"):
        validate_site(site)


def test_rejects_future_covariate_that_is_not_known_in_advance(corrupt: Corrupter) -> None:
    site = corrupt("covariates_future.parquet", lambda df: df.assign(variable="co2_ppm"))
    with pytest.raises(SiteValidationError, match="not known at forecast"):
        validate_site(site)


def test_rejects_covariate_with_unknown_scope(corrupt: Corrupter) -> None:
    site = corrupt(
        "covariates_past.parquet",
        lambda df: df.assign(scope=df["scope"].replace("gallery_b", "basement")),
    )
    with pytest.raises(SiteValidationError, match="unknown scopes"):
        validate_site(site)


def test_rejects_outside_occupancy_series(site_copy: Path) -> None:
    path = site_copy / OCCUPANCY_FILE
    frame = pd.read_parquet(path)
    extra = frame[frame["node_id"] == "foyer"].assign(node_id="outside")
    pd.concat([frame, extra], ignore_index=True).to_parquet(path, index=False)
    with pytest.raises(SiteValidationError, match="virtual outside node"):
        validate_site(site_copy)


def test_rejects_bad_interval_in_meta(site_copy: Path) -> None:
    meta_path = site_copy / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["interval_seconds"] = 300
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(SiteValidationError, match="regular 300s grid"):
        validate_site(site_copy)


def test_rejects_missing_file(site_copy: Path) -> None:
    (site_copy / "flow.parquet").unlink()
    with pytest.raises(SiteValidationError, match="missing required files"):
        load_site(site_copy)
