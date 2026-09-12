"""Downloading and provenance for external datasets.

Specification section 9: each source gets a ``scripts/fetch_<name>.py`` that downloads
into ``data/raw/<name>/``, records the download date and a checksum, and then invokes its
adapter to write ``data/canonical/<site_id>/``. **Data is never committed.**

The checksum is the point. A dataset that is silently re-released -- a column renamed, a
month of data backfilled, a licence changed -- would otherwise make a logged run
irreproducible with nothing on disk to show why. :func:`verify_download` compares what is
on disk against what the manifest recorded and says exactly which file moved.

Nothing here knows anything about any particular dataset. Format-specific work lives in
the adapter for that source, where it can cite the documentation it was written against.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mflow.paths import ensure_dir, raw_dir

#: Written beside the downloaded files; the record of where they came from.
DOWNLOAD_MANIFEST = "download.json"

#: Read in chunks so that a multi-gigabyte archive does not have to fit in memory.
_CHUNK = 1 << 20


class FetchError(RuntimeError):
    """Raised when a download cannot be completed or verified."""


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileRecord:
    """One downloaded file.

    Attributes:
        name: filename relative to the source's raw directory.
        url: where it came from.
        sha256: hex digest at download time.
        bytes: size at download time.
    """

    name: str
    url: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class DownloadManifest:
    """Provenance for one source's raw files.

    Attributes:
        source: short name, matching ``data/raw/<source>/``.
        downloaded_at: UTC timestamp of the download.
        files: what was fetched.
        licence: the licence the source is published under, as stated by the publisher.
        citation: how the source asks to be cited.
        notes: anything else worth recording, such as a dataset version.
    """

    source: str
    downloaded_at: str
    files: list[FileRecord]
    licence: str
    citation: str
    notes: dict[str, Any] = field(default_factory=dict)

    def write(self, directory: Path) -> Path:
        """Write ``download.json`` into ``directory``."""
        target = ensure_dir(directory) / DOWNLOAD_MANIFEST
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def read(cls, directory: Path) -> DownloadManifest:
        """Load the manifest from a raw directory.

        Raises:
            FetchError: if it is absent, which means the files arrived by some route that
                recorded nothing about them.
        """
        target = Path(directory) / DOWNLOAD_MANIFEST
        if not target.is_file():
            raise FetchError(
                f"{target} does not exist: these files have no recorded provenance. "
                f"Re-run the fetch script for {Path(directory).name}."
            )
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["files"] = [FileRecord(**record) for record in payload["files"]]
        return cls(**payload)


def download(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    overwrite: bool = False,
) -> FileRecord:
    """Fetch one file, verifying it if a digest is known.

    Args:
        url: what to fetch.
        destination: where to put it.
        expected_sha256: the publisher's digest, when they publish one. Checked after the
            download and before the file is moved into place.
        overwrite: re-download a file that is already present.

    Raises:
        FetchError: on a failed download or a digest mismatch. A mismatched file is left
            in place with a ``.rejected`` suffix rather than deleted, so it can be
            inspected: a changed digest usually means the publisher re-released the data,
            which is something the project needs to know about, not silently retry.
    """
    destination = Path(destination)
    if destination.is_file() and not overwrite:
        return FileRecord(
            name=destination.name,
            url=url,
            sha256=sha256(destination),
            bytes=destination.stat().st_size,
        )

    ensure_dir(destination.parent)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, _CHUNK)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise FetchError(f"could not download {url}: {error}") from error

    digest = sha256(partial)
    if expected_sha256 is not None and digest != expected_sha256:
        rejected = destination.with_suffix(destination.suffix + ".rejected")
        partial.replace(rejected)
        raise FetchError(
            f"{url} has digest {digest} but {expected_sha256} was expected. The file is "
            f"at {rejected} for inspection. If the publisher has re-released the data, "
            "update the expected digest in the fetch script and say so in the commit; do "
            "not just delete this."
        )
    partial.replace(destination)
    return FileRecord(
        name=destination.name, url=url, sha256=digest, bytes=destination.stat().st_size
    )


def fetch_source(
    source: str,
    urls: dict[str, str],
    *,
    licence: str,
    citation: str,
    digests: dict[str, str] | None = None,
    notes: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> tuple[Path, DownloadManifest]:
    """Download every file of one source and write its provenance manifest.

    Args:
        source: short name; the files land in ``data/raw/<source>/``.
        urls: filename to URL.
        licence: the licence the publisher states.
        citation: how the publisher asks to be cited.
        digests: known SHA-256 digests, keyed by the same filenames.
        notes: extra provenance, such as a dataset version or a DOI.
        overwrite: re-download files already present.

    Returns:
        The raw directory and the manifest written into it.
    """
    directory = ensure_dir(raw_dir(source))
    known = digests or {}
    records = [
        download(
            url,
            directory / name,
            expected_sha256=known.get(name),
            overwrite=overwrite,
        )
        for name, url in sorted(urls.items())
    ]
    manifest = DownloadManifest(
        source=source,
        downloaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        files=records,
        licence=licence,
        citation=citation,
        notes=notes or {},
    )
    manifest.write(directory)
    return directory, manifest


def verify_download(source: str) -> list[str]:
    """Re-hash a source's raw files and report every difference from its manifest.

    Returns:
        A list of human-readable differences; empty when the files are exactly what was
        recorded.

    Raises:
        FetchError: if the manifest is missing.
    """
    directory = raw_dir(source)
    manifest = DownloadManifest.read(directory)
    differences: list[str] = []
    for record in manifest.files:
        path = directory / record.name
        if not path.is_file():
            differences.append(f"{record.name} is missing")
            continue
        digest = sha256(path)
        if digest != record.sha256:
            differences.append(f"{record.name} hashes to {digest}, manifest says {record.sha256}")
    return differences


def extract(archive: Path, destination: Path, *, overwrite: bool = False) -> Path:
    """Unpack a zip or tar archive, refusing entries that escape the destination.

    Args:
        archive: the archive.
        destination: directory to unpack into.
        overwrite: unpack again even if the destination is already populated.

    Raises:
        FetchError: for an unsupported format, or for an archive containing a path that
            resolves outside ``destination``. Public research datasets are not usually
            hostile, but an absolute or ``..`` member would write outside the data
            directory and that is not something to discover afterwards.
    """
    archive = Path(archive)
    destination = ensure_dir(Path(destination))
    if any(destination.iterdir()) and not overwrite:
        return destination

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            _check_members(zf.namelist(), destination)
            zf.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            _check_members(tf.getnames(), destination)
            tf.extractall(destination, filter="data")
    else:
        raise FetchError(f"{archive} is neither a zip nor a tar archive")
    return destination


def _check_members(names: Iterable[str], destination: Path) -> None:
    """Refuse any archive member that would land outside ``destination``."""
    root = destination.resolve()
    for name in names:
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise FetchError(f"archive member {name!r} would extract to {target}, outside {root}")
