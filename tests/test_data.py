"""Dataset acquisition and the shared adapter scaffolding.

These tests never touch the network. The download path is exercised against ``file://``
URLs, which go through exactly the same urllib code as an https fetch.
"""

from __future__ import annotations

import json
import tarfile
import zipfile

import pandas as pd
import pytest

from mflow.data import (
    Adapter,
    AdapterError,
    DownloadManifest,
    FetchError,
    FileRecord,
    download,
    extract,
    long_counts,
    long_covariates,
    regular_grid,
    sha256,
    to_utc,
)
from mflow.schema import SiteData, load_site


@pytest.fixture
def served(tmp_path):
    """A file on disk and its ``file://`` URL, standing in for a remote source."""
    source = tmp_path / "remote" / "data.csv"
    source.parent.mkdir(parents=True)
    source.write_text("timestamp,room,count\n2026-01-01T00:00:00,a,3\n", encoding="utf-8")
    return source, source.as_uri()


# --------------------------------------------------------------------------- #
# Download and provenance
# --------------------------------------------------------------------------- #


def test_a_download_records_its_digest_and_size(tmp_path, served) -> None:
    source, url = served
    record = download(url, tmp_path / "data.csv")
    assert record.sha256 == sha256(source)
    assert record.bytes == source.stat().st_size
    assert (tmp_path / "data.csv").read_bytes() == source.read_bytes()


def test_a_known_digest_is_verified(tmp_path, served) -> None:
    source, url = served
    record = download(url, tmp_path / "ok.csv", expected_sha256=sha256(source))
    assert record.sha256 == sha256(source)


def test_a_changed_file_is_rejected_and_kept_for_inspection(tmp_path, served) -> None:
    _, url = served
    with pytest.raises(FetchError, match="do not just delete this"):
        download(url, tmp_path / "bad.csv", expected_sha256="0" * 64)
    # A digest mismatch usually means the publisher re-released the data, which the
    # project needs to see. Deleting the evidence would turn that into a mystery.
    assert (tmp_path / "bad.csv.rejected").is_file()
    assert not (tmp_path / "bad.csv").exists()
    assert not (tmp_path / "bad.csv.part").exists()


def test_a_failed_download_leaves_nothing_behind(tmp_path) -> None:
    with pytest.raises(FetchError, match="could not download"):
        download((tmp_path / "nope.csv").as_uri(), tmp_path / "out.csv")
    assert not (tmp_path / "out.csv").exists()
    assert not (tmp_path / "out.csv.part").exists()


def test_an_existing_file_is_not_refetched(tmp_path, served) -> None:
    _, url = served
    destination = tmp_path / "data.csv"
    download(url, destination)
    destination.write_text("locally edited", encoding="utf-8")
    record = download(url, destination)
    # The digest reported is of what is on disk, not of what the URL now serves, so a
    # stale local copy shows up as a difference rather than being silently refreshed.
    assert record.sha256 == sha256(destination)
    assert destination.read_text(encoding="utf-8") == "locally edited"
    refreshed = download(url, destination, overwrite=True)
    assert refreshed.sha256 != record.sha256


def test_a_manifest_round_trips(tmp_path) -> None:
    manifest = DownloadManifest(
        source="robod",
        downloaded_at="2026-01-01T00:00:00+00:00",
        files=[FileRecord(name="a.csv", url="https://example/a.csv", sha256="ab", bytes=2)],
        licence="CC BY 4.0",
        citation="Tekler et al. (2022)",
        notes={"doi": "10.6084/m9.figshare.19234530"},
    )
    manifest.write(tmp_path)
    loaded = DownloadManifest.read(tmp_path)
    assert loaded == manifest
    assert json.loads((tmp_path / "download.json").read_text())["licence"] == "CC BY 4.0"


def test_files_without_provenance_are_refused(tmp_path) -> None:
    with pytest.raises(FetchError, match="no recorded provenance"):
        DownloadManifest.read(tmp_path)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def test_a_zip_is_unpacked(tmp_path) -> None:
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inner/data.csv", "x")
    destination = extract(archive, tmp_path / "out")
    assert (destination / "inner" / "data.csv").read_text() == "x"


def test_a_tar_is_unpacked(tmp_path) -> None:
    payload = tmp_path / "data.csv"
    payload.write_text("x", encoding="utf-8")
    archive = tmp_path / "a.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(payload, arcname="inner/data.csv")
    destination = extract(archive, tmp_path / "out")
    assert (destination / "inner" / "data.csv").read_text() == "x"


def test_an_archive_that_escapes_its_destination_is_refused(tmp_path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.csv", "x")
    with pytest.raises(FetchError, match="would extract to"):
        extract(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.csv").exists()


def test_an_unknown_archive_format_is_refused(tmp_path) -> None:
    plain = tmp_path / "a.csv"
    plain.write_text("x", encoding="utf-8")
    with pytest.raises(FetchError, match="neither a zip nor a tar"):
        extract(plain, tmp_path / "out")


def test_an_already_unpacked_archive_is_left_alone(tmp_path) -> None:
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("data.csv", "original")
    destination = extract(archive, tmp_path / "out")
    (destination / "data.csv").write_text("edited", encoding="utf-8")
    extract(archive, destination)
    assert (destination / "data.csv").read_text() == "edited"
    extract(archive, destination, overwrite=True)
    assert (destination / "data.csv").read_text() == "original"


# --------------------------------------------------------------------------- #
# Conversion helpers
# --------------------------------------------------------------------------- #


def test_local_time_converts_through_the_named_zone() -> None:
    # Rome is UTC+1 in January and UTC+2 in July. A fixed offset would get one of them
    # wrong, which is the whole reason the conversion goes through the zone name.
    winter = to_utc(pd.Series(["2026-01-15 12:00:00"]), "Europe/Rome")
    summer = to_utc(pd.Series(["2026-07-15 12:00:00"]), "Europe/Rome")
    assert winter.iloc[0] == pd.Timestamp("2026-01-15 11:00:00", tz="UTC")
    assert summer.iloc[0] == pd.Timestamp("2026-07-15 10:00:00", tz="UTC")


def test_already_aware_timestamps_are_converted_not_relocalised() -> None:
    aware = pd.Series(pd.to_datetime(["2026-01-15 12:00:00+05:00"]))
    assert to_utc(aware, "Europe/Rome").iloc[0] == pd.Timestamp("2026-01-15 07:00:00", tz="UTC")


def test_an_ambiguous_local_time_is_refused() -> None:
    # 02:30 on the autumn transition happens twice. Picking one silently would shift an
    # hour of data by an hour, which no downstream check would catch.
    with pytest.raises(AdapterError, match="daylight-saving transition"):
        to_utc(pd.Series(["2026-10-25 02:30:00"]), "Europe/Rome")


def test_a_nonexistent_local_time_is_refused() -> None:
    with pytest.raises(AdapterError, match="daylight-saving transition"):
        to_utc(pd.Series(["2026-03-29 02:30:00"]), "Europe/Rome")


def test_the_grid_spans_the_record_and_counts_the_gaps() -> None:
    stamps = pd.DatetimeIndex(
        ["2026-01-01 00:00", "2026-01-01 00:15", "2026-01-01 01:00"], tz="UTC"
    )
    grid, missing = regular_grid(stamps, 900)
    assert len(grid) == 5  # 00:00, 00:15, 00:30, 00:45, 01:00
    assert missing == 2  # 00:30 and 00:45 were never recorded
    assert grid[0] == stamps.min() and grid[-1] == stamps.max()


def test_an_empty_record_has_no_grid() -> None:
    with pytest.raises(AdapterError, match="empty timestamp index"):
        regular_grid(pd.DatetimeIndex([], tz="UTC"), 900)


def test_counts_are_reshaped_into_the_canonical_columns() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "sensor": ["b", "a"],
            "people": [2, 5],
        }
    )
    out = long_counts(frame, entity_column="sensor", value_column="people", entity_name="node_id")
    assert list(out.columns) == ["timestamp", "node_id", "count"]
    assert list(out["node_id"]) == ["a", "b"]
    assert list(out["count"]) == [5, 2]


def test_a_missing_source_column_is_refused() -> None:
    frame = pd.DataFrame({"timestamp": [], "sensor": []})
    with pytest.raises(AdapterError, match=r"missing column\(s\) \['people'\]"):
        long_counts(frame, entity_column="sensor", value_column="people", entity_name="node_id")


def test_wide_environmental_columns_are_melted_and_renamed() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "room": ["a", "b"],
            "CO2_ppm": [600.0, 800.0],
            "Temp_C": [21.0, None],
            "Unused": [1.0, 2.0],
        }
    )
    out = long_covariates(
        frame, {"CO2_ppm": "co2_ppm", "Temp_C": "temp_c"}, scope_column="room"
    )
    assert list(out.columns) == ["timestamp", "scope", "variable", "value"]
    assert set(out["variable"]) == {"co2_ppm", "temp_c"}
    # The unmapped column is dropped, and the missing temperature reading is dropped
    # rather than carried through as a NaN covariate row.
    assert "Unused" not in set(out["variable"])
    assert len(out) == 3


def test_melting_a_frame_with_no_known_columns_gives_an_empty_frame() -> None:
    frame = pd.DataFrame({"timestamp": [], "room": []})
    out = long_covariates(frame, {"CO2_ppm": "co2_ppm"}, scope_column="room")
    assert out.empty
    assert list(out.columns) == ["timestamp", "scope", "variable", "value"]


# --------------------------------------------------------------------------- #
# The Adapter base class
# --------------------------------------------------------------------------- #


@pytest.fixture
def raw_with_manifest(tmp_path):
    """A raw directory carrying a provenance manifest."""
    raw = tmp_path / "raw" / "toy_source"
    raw.mkdir(parents=True)
    DownloadManifest(
        source="toy_source",
        downloaded_at="2026-01-01T00:00:00+00:00",
        files=[FileRecord(name="a.csv", url="https://example/a.csv", sha256="beef", bytes=4)],
        licence="CC BY 4.0",
        citation="Somebody et al. (2025)",
    ).write(raw)
    return raw


def make_adapter(site: SiteData | None, raw, destination, skipped=None):
    """An adapter that returns a prepared site, for testing the base class."""

    class _Toy(Adapter):
        source = "toy_source"
        version = "7"

        def build(self):
            return ([] if site is None else [site]), (skipped or {})

    return _Toy(raw=raw, destination=destination)


def test_an_adapter_writes_a_validated_site_with_provenance(
    toy_site, raw_with_manifest, tmp_path
) -> None:
    destination = tmp_path / "canonical"
    result = make_adapter(toy_site, raw_with_manifest, destination).run()
    assert len(result.sites) == 1

    written = load_site(result.sites[0])
    provenance = written.meta.provenance
    assert provenance["source"] == "toy_source"
    assert provenance["adapter_version"] == "7"
    assert provenance["licence"] == "CC BY 4.0"
    assert provenance["citation"] == "Somebody et al. (2025)"
    # The digests of the raw files travel with the site, so a canonical directory can
    # always be traced back to the exact bytes it was built from.
    assert provenance["raw_files"] == {"a.csv": "beef"}


def test_an_adapter_reports_what_it_skipped(toy_site, raw_with_manifest, tmp_path) -> None:
    result = make_adapter(
        toy_site,
        raw_with_manifest,
        tmp_path / "canonical",
        skipped={"bldg2": "no occupancy ground truth"},
    ).run()
    # A skipped building is not a building that scored badly, and the distinction has to
    # reach whoever reads the results.
    assert result.skipped == {"bldg2": "no occupancy ground truth"}
    assert "bldg2: no occupancy ground truth" in result.summary()


def test_an_adapter_without_raw_files_says_to_fetch_them(tmp_path) -> None:
    adapter = make_adapter(None, tmp_path / "absent", tmp_path / "canonical")
    with pytest.raises(AdapterError, match="Run the fetch script"):
        adapter.run()


def test_an_adapter_on_unprovenanced_files_is_refused(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    adapter = make_adapter(None, raw, tmp_path / "canonical")
    with pytest.raises(FetchError, match="no recorded provenance"):
        adapter.run()


def test_an_adapter_cannot_write_a_site_that_breaks_the_contract(
    toy_site, raw_with_manifest, tmp_path
) -> None:
    # Conservation is the check standing between a real dataset and a silently wrong
    # graph, so it must fire on the adapter path exactly as it does everywhere else.
    broken = SiteData(
        meta=toy_site.meta,
        nodes=toy_site.nodes,
        edges=toy_site.edges,
        occupancy=toy_site.occupancy,
        # One edge carries five extra people that arrive from nowhere.
        flow=toy_site.flow.assign(
            count=toy_site.flow["count"]
            + 5 * (toy_site.flow["edge_id"] == toy_site.edges["edge_id"].iloc[0])
        ),
        covariates_past=toy_site.covariates_past,
        covariates_future=toy_site.covariates_future,
    )
    adapter = make_adapter(broken, raw_with_manifest, tmp_path / "canonical")
    with pytest.raises(Exception, match="conservation residual"):
        adapter.run()
