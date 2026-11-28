#!/bin/bash
# Submit one job with the account and partition from delta/config.sh.
#   bash delta/submit.sh check | nb08 | nb07 | nb09 | nb10
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$REPO_DIR"
job="${1:-}"
[ -f "delta/jobs/$job.sbatch" ] || { echo "usage: bash delta/submit.sh check|nb08|nb07|nb09|nb10"; exit 1; }
[[ "$DELTA_ACCOUNT" != FILL_IN* ]] || { echo "Set DELTA_ACCOUNT in delta/config.sh first (run: accounts)"; exit 1; }
[ -f delta/logs/PREFLIGHT_PASSED ] || { echo "Run bash delta/preflight.sh first (it must pass)."; exit 1; }
# SKIP_CHECK=1: queue jobs while the GPU check is still pending (they stop at the first failing cell anyway)
if [ "$job" != check ] && [ "${SKIP_CHECK:-0}" != 1 ]; then
    [ -f delta/logs/CHECK_PASSED ] || { echo "Run the GPU check first: bash delta/submit.sh check"; exit 1; }
    [ "$(cat delta/logs/CHECK_PASSED)" = "$(git rev-parse HEAD)" ] || \
        echo "note: the GPU check passed on an older commit; rerun it if environments or scripts changed."
fi
extra=()
[ "$job" = nb08 ] && [ "$NB08_EXCLUSIVE" = 1 ] && extra+=(--exclusive --mem=0)
mkdir -p delta/logs
sbatch --account="$DELTA_ACCOUNT" --partition="$DELTA_PARTITION" ${extra[@]+"${extra[@]}"} "delta/jobs/$job.sbatch"
