#!/bin/bash
# One-time setup on a Delta LOGIN node (no GPU needed). Safe to rerun; finished steps are skipped.
#   cd ~/Anlp_KSampling && bash delta/setup_env.sh
# Builds both Python environments, registers them as Jupyter kernels, puts runs/ on large storage,
# and unpacks the saved logits if they were uploaded.
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$REPO_DIR"
echo "repo:  $REPO_DIR"
echo "large: $BIG_DIR"
mkdir -p "$BIG_DIR" "$HF_HOME" "$UV_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" delta/logs

# runs/ holds outputs and profiler traces (GBs): keep it on large storage, linked into the repo.
if [ ! -e runs ]; then
    mkdir -p "$BIG_DIR/runs" && ln -s "$BIG_DIR/runs" runs && echo "runs/ -> $BIG_DIR/runs"
fi

# uv downloads its own Python, so nothing depends on which modules Delta offers.
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

# ---- env 1: ksampling (notebooks 07, 08) -------------------------------------------------------------
echo; echo "== ksampling env: $VENV_KS"
[ -x "$VENV_KS/bin/python" ] || uv venv --python 3.10 "$VENV_KS"    # the notebooks were run on 3.10.12
# requirements.txt is the exact package list of the working environment on the shared machine.
# Its CUDA wheels (torch, torchvision +cu130) come from the PyTorch index.
VIRTUAL_ENV="$VENV_KS" uv pip install --python "$VENV_KS/bin/python" \
    --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match -r requirements.txt
VIRTUAL_ENV="$VENV_KS" uv pip install --python "$VENV_KS/bin/python" nbconvert nbclient ipykernel
"$VENV_KS/bin/python" -m ipykernel install --user --name anlp-ksampling --display-name "anlp-ksampling (Delta)"

# ---- env 2: vLLM (notebooks 09a, 09, 10) -------------------------------------------------------------
echo; echo "== vLLM env: $VENV_VLLM"
[ -x "$VENV_VLLM/bin/python" ] || uv venv --python 3.12 "$VENV_VLLM"
VIRTUAL_ENV="$VENV_VLLM" uv pip install --python "$VENV_VLLM/bin/python" "vllm==0.31.0"
VIRTUAL_ENV="$VENV_VLLM" uv pip install --python "$VENV_VLLM/bin/python" -e ./dg_topk_plugin
VIRTUAL_ENV="$VENV_VLLM" uv pip install --python "$VENV_VLLM/bin/python" \
    nbconvert nbclient ipykernel pandas matplotlib datasets openai
"$VENV_VLLM/bin/python" -m ipykernel install --user --name anlp-vllm --display-name "anlp-vllm (Delta)"

# ---- saved logits (uploaded by delta/upload_logits.sh) -----------------------------------------------
echo
n=$( (ls runs/nb04/step_*.pt 2>/dev/null || true) | wc -l)
if [ "$n" -eq 0 ] && [ -f "$HOME/nb04_logits.tgz" ]; then
    tar -xzf "$HOME/nb04_logits.tgz" -C "$REPO_DIR" && rm "$HOME/nb04_logits.tgz"
    n=$(ls runs/nb04/step_*.pt | wc -l)
fi
echo "saved logits in runs/nb04: $n of 6"

cat <<EOF

Setup done. Next steps:
  1. Hugging Face login (once; paste your token when asked):
       $VENV_KS/bin/hf auth login
  2. Download models and dataset (about 110 GB, 30-60 min; runs in the background, safe to log out):
       nohup bash delta/prefetch.sh > delta/logs/prefetch.log 2>&1 &
  3. When it finishes:  bash delta/preflight.sh
EOF
