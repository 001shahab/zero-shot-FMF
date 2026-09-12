"""Run manifests and determinism.

Ground rule 3 of the specification: every experiment takes a seed, and every run writes a
manifest recording the git commit, the config hash, the seed, package versions and the
hardware. Two runs whose manifests agree must produce identical numbers, so the manifest
is the unit of reproducibility and is written before any result file.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import numpy as np

from mflow.paths import ensure_dir, repo_root, results_dir

MANIFEST_FILE: Final[str] = "manifest.json"

#: Packages whose version can change a number. Recorded verbatim in every manifest.
TRACKED_PACKAGES: Final[tuple[str, ...]] = (
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "scipy",
    "cvxpy",
    "osqp",
    "networkx",
    "torch",
    "scikit-learn",
    "lightgbm",
    "statsmodels",
    "timesfm",
    "chronos-forecasting",
    "toto-models",
)


class DirtyWorkingTreeError(RuntimeError):
    """Raised when a run is started from a working tree with uncommitted changes.

    A result whose code cannot be recovered from a commit is not reproducible, so runs
    that write to ``results/`` refuse to start unless the tree is clean or the caller
    explicitly opts out with ``allow_dirty``.
    """


def set_global_seed(seed: int) -> None:
    """Seed every global random number generator this project can reach.

    Model code should prefer an explicit :class:`numpy.random.Generator`; this function
    exists for third-party libraries that only expose global state.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeding legacy global state for third parties
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rng(seed: int, *stream: str | int) -> np.random.Generator:
    """Return a generator for a named sub-stream of ``seed``.

    Deriving each component's generator from the run seed and a stable name means that
    adding a component does not perturb the draws of the components that already exist.
    """
    material = "|".join(str(s) for s in stream).encode("utf-8")
    offset = int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big")
    return np.random.default_rng([seed, offset])


def config_hash(config: Any) -> str:
    """Stable 16-hex-character digest of a JSON-serialisable configuration."""
    payload = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return value


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *args],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return None
    return out.stdout.strip()


def git_state() -> dict[str, Any]:
    """Commit, branch and dirty flag of the working tree."""
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "dirty_files": sorted(line[3:] for line in status.splitlines()) if status else [],
    }


def package_versions() -> dict[str, str]:
    """Installed versions of the packages that can move a number."""
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def hardware() -> dict[str, Any]:
    """Machine description, for the latency and memory claims."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        info["total_memory_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:
        info["total_memory_gb"] = None
    try:
        import torch

        info["torch_device"] = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        info["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError:
        info["torch_device"] = None
        info["cuda_device"] = None
    return info


@dataclass(frozen=True)
class RunManifest:
    """Everything needed to reproduce a run, written before the run produces output."""

    run_id: str
    experiment: str
    seed: int
    config: dict[str, Any]
    config_hash: str
    git: dict[str, Any]
    packages: dict[str, str]
    hardware: dict[str, Any]
    created_at: str
    notes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        experiment: str,
        seed: int,
        config: Any,
        run_id: str | None = None,
        allow_dirty: bool = False,
        notes: dict[str, Any] | None = None,
    ) -> RunManifest:
        """Assemble a manifest for a run that is about to start.

        Args:
            experiment: experiment identifier, e.g. ``E1``.
            seed: the run seed; also applied globally via :func:`set_global_seed`.
            config: the resolved configuration, any JSON-serialisable object.
            run_id: override the generated ``<experiment>_<confighash>_s<seed>`` id.
            allow_dirty: permit a run from a working tree with uncommitted changes.
                Development convenience only; never set it for a reported result.
            notes: free-form extras, recorded verbatim.

        Raises:
            DirtyWorkingTreeError: if the tree is dirty and ``allow_dirty`` is not set.
        """
        resolved = _jsonable(config)
        digest = config_hash(resolved)
        state = git_state()
        if state["dirty"] and not allow_dirty:
            raise DirtyWorkingTreeError(
                "refusing to start a logged run from a dirty working tree; commit "
                f"{state['dirty_files'][:5]} (and {max(0, len(state['dirty_files']) - 5)} more) "
                "or pass allow_dirty=True for a throwaway run"
            )
        return cls(
            run_id=run_id or f"{experiment}_{digest}_s{seed}",
            experiment=experiment,
            seed=seed,
            config=resolved,
            config_hash=digest,
            git=state,
            packages=package_versions(),
            hardware=hardware(),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            notes=notes or {},
        )

    @property
    def directory(self) -> Path:
        """The ``results/<run_id>/`` directory for this run."""
        return results_dir(self.run_id)

    def write(self) -> Path:
        """Create the run directory and write ``manifest.json`` into it."""
        directory = ensure_dir(self.directory)
        target = directory / MANIFEST_FILE
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def read(cls, path: str | Path) -> RunManifest:
        """Load a manifest from a run directory or a ``manifest.json`` path."""
        target = Path(path)
        if target.is_dir():
            target = target / MANIFEST_FILE
        payload = json.loads(target.read_text(encoding="utf-8"))
        return cls(**payload)

    def reproduces(self, other: RunManifest) -> tuple[bool, list[str]]:
        """Compare two manifests on the fields that can change a number.

        Returns:
            ``(True, [])`` when the runs must agree numerically, otherwise ``False`` and
            a list of human-readable differences.
        """
        differences: list[str] = []
        if self.config_hash != other.config_hash:
            differences.append(f"config hash {self.config_hash} != {other.config_hash}")
        if self.seed != other.seed:
            differences.append(f"seed {self.seed} != {other.seed}")
        if self.git.get("commit") != other.git.get("commit"):
            differences.append(f"commit {self.git.get('commit')} != {other.git.get('commit')}")
        for name, version in self.packages.items():
            other_version = other.packages.get(name)
            if version != other_version:
                differences.append(f"{name} {version} != {other_version}")
        return (not differences), differences
