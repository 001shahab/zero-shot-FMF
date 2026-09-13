"""Command line entry point.

Everything the project does from a shell goes through ``mflow <command>``::

    mflow simulate      configs/sites/house_museum.yaml --seed 0
    mflow degrade       sim_house_museum --profile realistic --seed 0
    mflow run           configs/experiments/E1.yaml --dry-run
    mflow report        --output paper/tables
    mflow reproduce     results/E1_sim_house_museum_default_s0

Subcommands that write into ``results/`` refuse to start from a working tree with
uncommitted changes, because a number whose code cannot be recovered from a commit is not
reproducible. ``--allow-dirty`` exists for development and must not be used for a
reported result; every manifest records which of the two it was.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from mflow.manifest import DirtyWorkingTreeError, RunManifest
from mflow.paths import configs_dir, repo_root


def load_environment(path: str | Path | None = None) -> dict[str, str]:
    """Load ``.env`` into the process environment and report which keys it set.

    The Hugging Face token lives here. It is read at start-up so that a gated model
    repository fails at authentication rather than three layers down inside a library,
    and the *values* are never returned, logged or written into a manifest -- only the
    names of the keys that were found.

    Args:
        path: the env file. Defaults to ``.env`` at the project root.

    Returns:
        A mapping from each key that was set to the string ``"set"``. The token itself is
        deliberately not in the return value; anything that needs it reads
        ``os.environ`` directly.
    """
    from dotenv import dotenv_values, load_dotenv

    target = Path(path) if path is not None else repo_root() / ".env"
    if not target.is_file():
        return {}
    load_dotenv(target, override=False)
    found = {key: "set" for key, value in dotenv_values(target).items() if value}

    # huggingface_hub reads HF_TOKEN from the environment for every request it makes on
    # its own account, including the repository resolution that happens before a wrapper
    # gets to pass `token=` to anything. Without the alias a gated checkpoint fails as an
    # anonymous 401 from inside the library rather than as a missing-credentials error
    # here, so mirror the key under the name the library actually looks for.
    token = os.environ.get("HUGGINGFACE_API_KEY")
    if token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = token
        found["HF_TOKEN"] = "set from HUGGINGFACE_API_KEY"
    return found


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def cmd_simulate(args: argparse.Namespace) -> int:
    """Generate a synthetic site from a site configuration."""
    from mflow.schema import write_site
    from mflow.sim import GraphSimulator, aggregate, load_site_config, summarise

    config = load_site_config(args.config)
    simulator = GraphSimulator(config, seed=args.seed)
    result = simulator.run(args.start, args.days)
    site = aggregate(simulator, result, drop_warmup_days=args.warmup_days)

    destination = Path(args.output) if args.output else _canonical(args) / config.site_id
    write_site(site, destination)
    print(f"wrote {config.site_id} to {destination}")
    for key, value in sorted(summarise(result, site).items()):
        print(f"  {key}: {value}")
    return 0


def cmd_degrade(args: argparse.Namespace) -> int:
    """Apply a sensor profile to a clean site and write the observed copy."""
    from mflow.schema import load_site, write_site
    from mflow.sensors import degrade, load_sensor_profile

    base = _canonical(args)
    site = load_site(base / args.site_id)
    profile = load_sensor_profile(configs_dir("sensors") / f"{args.profile}.yaml")
    observed, report = degrade(site, profile, seed=args.seed)
    destination = Path(args.output) if args.output else base / observed.meta.site_id
    write_site(observed, destination)
    print(f"wrote {observed.meta.site_id} to {destination}")
    for key, value in sorted(report.summary().items()):
        print(f"  {key}: {value}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Execute an experiment."""
    from mflow.experiments import describe_experiment, load_experiment, run_experiment

    environment = load_environment()
    if environment:
        print(f"loaded .env keys: {', '.join(sorted(environment))}")

    config = load_experiment(args.config)
    if args.dry_run:
        print(describe_experiment(config))
        return 0

    try:
        outcomes = run_experiment(
            config,
            sites=args.sites,
            seeds=args.seeds,
            variants=args.variants,
            data_root=_canonical(args),
            results_root=Path(args.results) if args.results else None,
            allow_dirty=args.allow_dirty,
            overwrite=args.overwrite,
            on_progress=lambda line: print(line, flush=True),
        )
    except DirtyWorkingTreeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    for outcome in outcomes:
        state = "skipped (already on disk)" if outcome.skipped else f"{outcome.seconds:.1f}s"
        print(f"  {outcome.manifest.run_id}: {state}")
    ran = sum(not outcome.skipped for outcome in outcomes)
    print(f"{ran} run(s) written, {len(outcomes) - ran} skipped")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Build the tables and figures the logged runs support."""
    from mflow.eval.report import ReportError, build_report

    try:
        written = build_report(
            Path(args.output),
            root=Path(args.results) if args.results else None,
            reference=args.reference,
            experiment=args.experiment,
        )
    except ReportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for name, paths in sorted(written.items()):
        print(f"{name}: {', '.join(str(p.name) for p in paths)}")
    print(f"wrote {sum(len(p) for p in written.values())} file(s) to {args.output}")
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Check whether the current environment can reproduce a logged run."""
    manifest = RunManifest.read(args.run)
    current = RunManifest.create(
        experiment=manifest.experiment,
        seed=manifest.seed,
        config=manifest.config,
        allow_dirty=True,
    )
    same, differences = manifest.reproduces(current)
    print(f"run {manifest.run_id} from {manifest.created_at}")
    if same:
        print("this environment reproduces it exactly")
        return 0
    print("this environment differs:")
    for difference in differences:
        print(f"  {difference}")
    return 1


def cmd_list(args: argparse.Namespace) -> int:
    """List the available methods, reconcilers, sites and experiments."""
    from mflow.forecast import ZERO_SHOT_METHODS, available_forecasters
    from mflow.reconcile import default_registry

    print("forecasters:")
    for name in available_forecasters():
        kind = "zero-shot" if name in ZERO_SHOT_METHODS else "trained"
        print(f"  {name:38s} {kind}")
    print("reconcilers:")
    for name in sorted(default_registry().factories):
        print(f"  {name}")
    print("experiments:")
    for path in sorted(configs_dir("experiments").glob("*.yaml")):
        print(f"  {path.stem:38s} {path}")
    print("canonical sites:")
    base = _canonical(args)
    for path in sorted(p for p in base.glob("*") if p.is_dir()):
        print(f"  {path.name:38s} {path}")
    return 0


def _canonical(args: argparse.Namespace) -> Path:
    """The canonical data directory, honouring ``--data``."""
    from mflow.paths import canonical_dir

    return Path(args.data) if getattr(args, "data", None) else canonical_dir()


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser."""
    parser = argparse.ArgumentParser(
        prog="mflow",
        description=(
            "Zero-shot foundation model forecasting of visitor flow in heritage museums "
            "with topology-constrained reconciliation."
        ),
    )
    parser.add_argument(
        "--data", help="override data/canonical/", default=None, metavar="DIR"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    simulate = sub.add_parser("simulate", help="generate a synthetic site")
    simulate.add_argument("config", help="a file under configs/sites/")
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--days", type=int, default=30)
    simulate.add_argument("--start", default="2026-03-02", help="first local date to simulate")
    simulate.add_argument(
        "--warmup-days",
        type=int,
        default=1,
        help="leading days to discard, since the building starts empty",
    )
    simulate.add_argument("--output", default=None, help="write here instead of data/canonical/")
    simulate.set_defaults(func=cmd_simulate)

    degrade = sub.add_parser("degrade", help="apply a sensor profile to a clean site")
    degrade.add_argument("site_id", help="a directory under data/canonical/")
    degrade.add_argument("--profile", default="realistic", help="a file under configs/sensors/")
    degrade.add_argument("--seed", type=int, default=0)
    degrade.add_argument("--output", default=None)
    degrade.set_defaults(func=cmd_degrade)

    run = sub.add_parser("run", help="execute an experiment")
    run.add_argument("config", help="a file under configs/experiments/")
    run.add_argument("--sites", nargs="+", default=None, help="restrict to these sites")
    run.add_argument("--seeds", nargs="+", type=int, default=None)
    run.add_argument("--variants", nargs="+", default=None)
    run.add_argument("--results", default=None, help="override results/")
    run.add_argument(
        "--dry-run", action="store_true", help="print the plan without running anything"
    )
    run.add_argument("--overwrite", action="store_true", help="recompute runs already on disk")
    run.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit a run from a dirty tree; never use this for a reported result",
    )
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="build tables and figures from results/")
    report.add_argument("--output", default="paper/tables")
    report.add_argument("--results", default=None, help="override results/")
    report.add_argument(
        "--reference", default=None, help="baseline for the significance markers"
    )
    report.add_argument(
        "--experiment",
        default=None,
        help="report one experiment; required when results/ holds more than one",
    )
    report.set_defaults(func=cmd_report)

    reproduce = sub.add_parser("reproduce", help="check a logged run against this environment")
    reproduce.add_argument("run", help="a results/<run_id>/ directory")
    reproduce.set_defaults(func=cmd_reproduce)

    listing = sub.add_parser("list", help="list methods, reconcilers, sites and experiments")
    listing.set_defaults(func=cmd_list)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
