#!/bin/bash
# E8a then E8b, one at a time. Same lock discipline as run_sweep.sh: two runners sharing
# results/ destroyed an E6 attempt once already.
cd "$(dirname "$0")"
mkdir logs 2>/dev/null
if ! mkdir logs/.e8.lock 2>/dev/null; then
  echo "an E8 sweep is already running; refusing to start a second" >&2
  exit 1
fi
trap 'rmdir logs/.e8.lock 2>/dev/null' EXIT
export MPLCONFIGDIR=/tmp/mpl
for e in E8_robod E8_hzmetro; do
  echo "=== $e starting $(date +%H:%M:%S)" >> logs/e8.log
  myenv/bin/python -u -m mflow.cli run "configs/experiments/$e.yaml" >> "logs/$e.log" 2>&1
  echo "=== $e exit=$? $(date +%H:%M:%S)" >> logs/e8.log
done
echo "=== E8 COMPLETE $(date +%H:%M:%S)" >> logs/e8.log
