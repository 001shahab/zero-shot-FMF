#!/usr/bin/env python
"""Download the PVCGN metro release and convert it to canonical sites.

Writes ``data/raw/pvcgn/`` and then ``data/canonical/hzmetro/``. Neither is committed.

    python scripts/fetch_pvcgn.py                 # Hangzhou, 80 stations
    python scripts/fetch_pvcgn.py --cities hangzhou shanghai

Source: https://github.com/HCPLab-SYSU/PVCGN, archive ``data/data.tar.gz``.
"""

from __future__ import annotations

import argparse
import sys

from mflow.data.fetch import extract, fetch_source
from mflow.data.pvcgn import (
    CITIES,
    EXTRACTED,
    PVCGN_ARCHIVE_URL,
    PVCGN_CITATION,
    PVCGN_LICENCE,
    PvcgnAdapter,
)
from mflow.paths import raw_dir

ARCHIVE = "data.tar.gz"


def main(argv: list[str] | None = None) -> int:
    """Fetch, unpack and convert. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cities",
        nargs="+",
        default=["hangzhou"],
        choices=sorted(CITIES),
        help="which cities to convert (default: hangzhou)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="re-download and re-unpack"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="convert what is already under data/raw/pvcgn/",
    )
    args = parser.parse_args(argv)

    if args.skip_download:
        print(f"skipping download; converting {raw_dir('pvcgn')}")
        result = PvcgnAdapter(cities=tuple(args.cities)).run()
        print(result.summary())
        return 0

    directory, manifest = fetch_source(
        "pvcgn",
        {ARCHIVE: PVCGN_ARCHIVE_URL},
        licence=PVCGN_LICENCE,
        citation=PVCGN_CITATION,
        notes={
            "repository": "https://github.com/HCPLab-SYSU/PVCGN",
            "doi": "10.1109/TITS.2020.3036057",
            "interval_seconds": 900,
            "service_window_local": "05:15-23:30, no data overnight",
            "channel_order": (
                "channel 0 is exits, channel 1 is entries; established from the data, "
                "opposite to the dataset README's wording. See mflow.data.pvcgn."
            ),
        },
        overwrite=args.overwrite,
    )
    print(f"downloaded {len(manifest.files)} file(s) to {directory}")
    # Into a subdirectory rather than alongside the archive, so that "is this already
    # unpacked?" is a question about an empty directory rather than about a directory
    # that always contains at least the archive and its manifest.
    unpacked = extract(directory / ARCHIVE, directory / EXTRACTED, overwrite=args.overwrite)
    print(f"unpacked {ARCHIVE} to {unpacked}")

    result = PvcgnAdapter(cities=tuple(args.cities)).run()
    print(result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
