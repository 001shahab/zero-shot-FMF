#!/usr/bin/env python
"""Download ROBOD and convert it to a canonical site.

Writes ``data/raw/robod/`` and then ``data/canonical/robod_bldg1/``. Neither is committed.

    python scripts/fetch_robod.py

Source: https://github.com/ideas-lab-nus/robod (Figshare record 19234530).
"""

from __future__ import annotations

import argparse
import sys

from mflow.data.fetch import fetch_source
from mflow.data.robod import (
    ROBOD_CITATION,
    ROBOD_FILES,
    ROBOD_LICENCE,
    RobodAdapter,
)


def main(argv: list[str] | None = None) -> int:
    """Fetch and convert. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite", action="store_true", help="re-download files already on disk"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="convert the files already under data/raw/robod/",
    )
    args = parser.parse_args(argv)

    if not args.skip_download:
        directory, manifest = fetch_source(
            "robod",
            ROBOD_FILES,
            licence=ROBOD_LICENCE,
            citation=ROBOD_CITATION,
            notes={
                "repository": "https://github.com/ideas-lab-nus/robod",
                "figshare": "https://doi.org/10.6084/m9.figshare.19234530",
                "doi": "10.1007/s12273-022-0925-9",
                "collection_period": "2021-09-07 to 2021-12-23, Asia/Singapore",
                "sampling_interval_seconds": 300,
            },
            overwrite=args.overwrite,
        )
        print(f"downloaded {len(manifest.files)} file(s) to {directory}")

    result = RobodAdapter().run()
    print(result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
