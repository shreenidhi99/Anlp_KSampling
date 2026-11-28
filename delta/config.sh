# Settings shared by every Delta script. Source it:  source delta/config.sh
# Only DELTA_ACCOUNT must be filled in (run `accounts` on Delta to see it).

# ---- FILL IN -------------------------------------------------------------------------------
# Slurm account of the GPU allocation, as printed by `accounts` (looks like  abcd-delta-gpu).
export DELTA_ACCOUNT="${DELTA_ACCOUNT:-FILL_IN-delta-gpu}"
# GPU partition. gpuA100x4 = nodes with 4x A100 40 GB (what the notebooks are sized for). `sinfo -s` lists all.
export DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"
# --------------------------------------------------------------------------------------------

# Notebook 08 uses 1 GPU and is billed for 1 GPU. Set to 1 to keep other jobs off its node (billed for all 4).
# The 4-GPU jobs (07, 09, 10) always take the whole node, so they are exclusive at no extra cost.
export NB08_EXCLUSIVE="${NB08_EXCLUSIVE:-0}"

# Run settings passed to the notebooks (each notebook keeps its own default when unset).
export NB07_N_GSM="${NB07_N_GSM:-250}"
export NB07_RUN_CONFIGS="${NB07_RUN_CONFIGS:-stock (patched);certified, sc full;certified, sc topk renorm;naive topk renorm}"
export NB09_N_QUESTIONS="${NB09_N_QUESTIONS:-100}"
export NB09_RUN_AR="${NB09_RUN_AR:-1}"

# ---- paths (normally no need to change) ----------------------------------------------------
export REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Large files (environments, model cache ~110 GB, run outputs) go on work/project storage, not $HOME.
_acct_short="${DELTA_ACCOUNT%%-*}"
if [ -z "${BIG_DIR:-}" ]; then
    if [ -d "/work/hdd/${_acct_short}" ]; then BIG_DIR="/work/hdd/${_acct_short}/${USER}/anlp"
    elif [ -d "/projects/${_acct_short}" ]; then BIG_DIR="/projects/${_acct_short}/${USER}/anlp"
    else BIG_DIR="${HOME}/anlp"; fi
fi
export BIG_DIR
export VENV_KS="${VENV_KS:-${BIG_DIR}/venv-ksampling}"   # notebooks 07, 08   (requirements.txt)
export VENV_VLLM="${VENV_VLLM:-${BIG_DIR}/venv-vllm}"    # notebooks 09a, 09, 10 (vLLM 0.31.0 + dg_topk_plugin)
export HF_HOME="${HF_HOME:-${BIG_DIR}/hf_cache}"         # same cache on login and compute nodes (token lives here too)
export UV_CACHE_DIR="${UV_CACHE_DIR:-${BIG_DIR}/uv_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${BIG_DIR}/triton_cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${BIG_DIR}/vllm_cache}"

# PyTorch wheels for the ksampling env. cu130 matches requirements.txt. If the GPU check reports that the
# driver is too old for CUDA 13, set TORCH_INDEX=https://download.pytorch.org/whl/cu128 and rerun setup_env.sh.
export TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

export MODEL_ID="google/diffusiongemma-26B-A4B-it"
export AR_ID="google/gemma-4-26B-A4B-it"
export AR_DRAFT_ID="google/gemma-4-26B-A4B-it-assistant"
