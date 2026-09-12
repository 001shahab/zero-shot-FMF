"""Shared fixtures.

The committed toy site under ``tests/fixtures/toy_site`` is treated as read-only. Tests
that need to break it copy it into ``tmp_path`` first via the ``site_copy`` fixture.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from mflow.graph import BuildingGraph
from mflow.schema import SiteData, load_site

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
TOY_SITE = FIXTURE_ROOT / "toy_site"


@pytest.fixture(scope="session")
def toy_site_path() -> Path:
    """Path to the committed, valid three-room toy site."""
    if not TOY_SITE.is_dir():
        pytest.fail(
            f"{TOY_SITE} is missing; regenerate it with `python scripts/build_toy_fixture.py`"
        )
    return TOY_SITE


@pytest.fixture(scope="session")
def toy_site(toy_site_path: Path) -> SiteData:
    """The loaded and validated toy site."""
    return load_site(toy_site_path)


@pytest.fixture(scope="session")
def toy_graph(toy_site: SiteData) -> BuildingGraph:
    """Building graph of the toy site."""
    return BuildingGraph.from_site(toy_site)


@pytest.fixture
def site_copy(toy_site_path: Path, tmp_path: Path) -> Path:
    """A writable copy of the toy site."""
    destination = tmp_path / "site"
    shutil.copytree(toy_site_path, destination)
    return destination


@pytest.fixture
def corrupt(site_copy: Path) -> Callable[[str, Callable[[pd.DataFrame], pd.DataFrame]], Path]:
    """Return a helper that rewrites one file of the site copy through a mutation.

    Example::

        corrupt("occupancy.parquet", lambda df: df.assign(count=-1))
    """

    def _apply(filename: str, mutate: Callable[[pd.DataFrame], pd.DataFrame]) -> Path:
        target = site_copy / filename
        frame = pd.read_parquet(target) if target.suffix == ".parquet" else pd.read_csv(target)
        mutated = mutate(frame)
        if target.suffix == ".parquet":
            mutated.to_parquet(target, index=False)
        else:
            mutated.to_csv(target, index=False)
        return site_copy

    return _apply
