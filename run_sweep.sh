#!/bin/bash
# Sequential sweep, one experiment at a time.
#
# The lock matters. This command has been dispatched twice on two occasions, and two
# runners sharing results/ is what destroyed the first E6 attempt: both cleared their own
# outputs on start and the second wiped the first's directory three hours in. Completed
# runs are skipped by the runner itself, which returns early when metrics.parquet already
# exists, so re-running this script is cheap and resumes where it stopped.
#
# mkdir is the lock because macOS has no flock(1): creating a directory is atomic, and it
# fails rather than succeeding a second time.
cd "$(dirname "$0")"
mkdir logs 2>/dev/null
if ! mkdir logs/.sweep.lock 2>/dev/null; then
  echo "a sweep is already running (logs/.sweep.lock exists); refusing to start a second" >&2
  exit 1
fi
trap 'rmdir logs/.sweep.lock 2>/dev/null' EXIT
export MPLCONFIGDIR=/tmp/mpl
for e in E4 E5 E3 E6 E2 E1; do
  echo "=== $e starting $(date +%H:%M:%S)" >> logs/sweep.log
  myenv/bin/python -u -m mflow.cli run "configs/experiments/$e.yaml" >> "logs/$e.log" 2>&1
  echo "=== $e exit=$? $(date +%H:%M:%S)" >> logs/sweep.log
done
echo "=== SWEEP COMPLETE $(date +%H:%M:%S)" >> logs/sweep.log
