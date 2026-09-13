"""External dataset acquisition and conversion to the canonical contract.

Each source has two halves. ``scripts/fetch_<name>.py`` downloads into
``data/raw/<name>/`` and records provenance; the adapter in this package reads those raw
files and writes ``data/canonical/<site_id>/``. Data is never committed.
"""

from mflow.data.adapter import (
    Adapter,
    AdapterError,
    AdapterResult,
    long_counts,
    long_covariates,
    regular_grid,
    to_utc,
)
from mflow.data.fetch import (
    DOWNLOAD_MANIFEST,
    DownloadManifest,
    FetchError,
    FileRecord,
    download,
    extract,
    fetch_source,
    sha256,
    verify_download,
)
from mflow.data.pvcgn import PvcgnAdapter
from mflow.data.robod import RobodAdapter

__all__ = [
    "DOWNLOAD_MANIFEST",
    "Adapter",
    "AdapterError",
    "AdapterResult",
    "DownloadManifest",
    "FetchError",
    "FileRecord",
    "PvcgnAdapter",
    "RobodAdapter",
    "download",
    "extract",
    "fetch_source",
    "long_counts",
    "long_covariates",
    "regular_grid",
    "sha256",
    "to_utc",
    "verify_download",
]
