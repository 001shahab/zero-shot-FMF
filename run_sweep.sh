#!/bin/bash
# Sequential sweep. One experiment at a time: two concurrent runners racing on the same
# results/ directory is what destroyed the first E6 attempt.
cd "$(dirname "$0")"
export MPLCONFIGDIR=/tmp/mpl
for e in E4 E5 E3 E6 E2 E1; do
  echo "=== $e starting $(date +%H:%M:%S)" >> logs/sweep.log
  myenv/bin/python -u -m mflow.cli run "configs/experiments/$e.yaml" \
    >> "logs/$e.log" 2>&1
  echo "=== $e exit=$? $(date +%H:%M:%S)" >> logs/sweep.log
done
echo "=== SWEEP COMPLETE $(date +%H:%M:%S)" >> logs/sweep.log
