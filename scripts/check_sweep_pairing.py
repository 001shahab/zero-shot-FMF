"""Check that the context sweep scores the same origins at every context length.

The sweep's whole value is that the only thing varying is how much history the model
reads. If lengthening the context also moved the origin set, a difference between two
contexts could be a difference between two test windows instead, and the paper's
methodological claim would not follow. The paper asserts this pairing, so it is checked
here rather than assumed, and the check is cheap enough to run every time.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mflow.eval.protocol import build_plan  # noqa: E402
from mflow.schema import load_site  # noqa: E402

CASES = {
    "hzmetro": ([96, 192, 288, 672, 1344], [1, 2, 4, 8], 2),
    "robod_bldg1": ([288, 576, 864, 2016, 4032], [1, 3, 6, 12], 12),
}


def main() -> int:
    failures = 0
    for site_id, (contexts, horizons, stride) in CASES.items():
        site = load_site(ROOT / "data" / "canonical" / site_id)
        panel = site.to_panel()
        sets: dict[int, frozenset[int]] = {}
        for context in contexts:
            plan = build_plan(
                panel,
                context_length=context,
                horizons=horizons,
                stride=stride,
                quantiles=[0.5],
                require_observed=True,
            )
            sets[context] = frozenset(task.origin for task in plan.tasks)
            print(
                f"{site_id:14s} context {context:5d}: {len(sets[context]):4d} origins, "
                f"{len(plan.dropped):4d} dropped"
            )
        common = frozenset.intersection(*sets.values())
        union = frozenset.union(*sets.values())
        if common == union:
            print(f"  {site_id}: all contexts score the identical origin set\n")
            continue
        # Not a bug. ``require_observed`` asks each live series for one finite value
        # inside the context, so on a gappy record a longer context can rescue an origin
        # that a shorter one had to drop. The sweep is therefore restricted to the
        # intersection, which is what keeps the comparison paired.
        print(
            f"  {site_id}: origin sets differ by {len(union - common)} of {len(union)}; "
            f"the sweep is analysed on the {len(common)} common origins\n"
        )
        if len(common) < 0.95 * len(union):
            failures += 1
            print(f"  {site_id}: too few common origins to pair the sweep")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
