#!/bin/bash
# Run on YOUR LAPTOP in Git Bash (not on Delta), once `ssh delta` works.
# Copies nb04_logits.tgz (the 6 saved logits steps, ~640 MB) to your Delta home; delta/setup_env.sh
# unpacks it into runs/nb04 (or, if setup already ran, this script unpacks it directly).
#   bash delta/upload_logits.sh [path/to/nb04_logits.tgz]
set -euo pipefail
TGZ="${1:-$HOME/Downloads/nb04_logits.tgz}"
[ -f "$TGZ" ] || { echo "not found: $TGZ"; exit 1; }
scp "$TGZ" delta:nb04_logits.tgz
ssh delta 'if [ -d ~/Anlp_KSampling/runs ]; then cd ~/Anlp_KSampling && tar -xzf ~/nb04_logits.tgz && rm ~/nb04_logits.tgz && ls -la runs/nb04/; else echo "uploaded to ~/nb04_logits.tgz; setup_env.sh will unpack it"; fi'
