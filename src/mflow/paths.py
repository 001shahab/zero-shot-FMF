"""Filesystem conventions.

Every path used by the project is derived here so that a run can be relocated by setting
``MFLOW_ROOT`` rather than by editing code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

_ENV_ROOT: Final[str] = "MFLOW_ROOT"


def repo_root() -> Path:
    """Return the project root.

    Honours ``MFLOW_ROOT`` when set, otherwise walks up from this file to the directory
    containing ``pyproject.toml``.
    """
    override = os.environ.get(_ENV_ROOT)
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(
        "cannot locate the project root: no pyproject.toml above "
        f"{here} and {_ENV_ROOT} is unset"
    )


def data_dir() -> Path:
    """``data/`` -- gitignored, populated by ``scripts/fetch_*.py``."""
    return repo_root() / "data"


def raw_dir(name: str) -> Path:
    """``data/raw/<name>/`` -- untouched downloads for one source."""
    return data_dir() / "raw" / name


def canonical_dir(site_id: str | None = None) -> Path:
    """``data/canonical/[<site_id>/]`` -- sites in the canonical data contract."""
    base = data_dir() / "canonical"
    return base if site_id is None else base / site_id


def results_dir(run_id: str | None = None) -> Path:
    """``results/[<run_id>/]`` -- one directory per logged run, gitignored."""
    base = repo_root() / "results"
    return base if run_id is None else base / run_id


def configs_dir(kind: str | None = None) -> Path:
    """``configs/[<kind>/]`` where kind is ``sites``, ``sensors``, ``experiments`` or ``models``."""
    base = repo_root() / "configs"
    return base if kind is None else base / kind


def ensure_dir(path: Path) -> Path:
    """Create ``path`` and its parents if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
