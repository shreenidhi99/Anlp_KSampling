#!/bin/bash
# Submit all four experiment jobs at once; Slurm runs them in parallel as nodes free up.
#   bash delta/submit_all.sh
# Requires delta/preflight.sh and the GPU check job to have passed.
set -euo pipefail
cd "$(dirname "$0")/.."
for j in nb08 nb07 nb09 nb10; do
    echo "== $j"; bash delta/submit.sh "$j"
done
echo
echo "Watch:   squeue -u \$USER"
echo "Logs:    delta/logs/<job>-<id>.out   (tail -f to follow)"
echo "Cancel:  scancel <id>   or   scancel -u \$USER"
