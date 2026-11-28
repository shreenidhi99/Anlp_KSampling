# Sourced at the start of every job script.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source delta/config.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # everything was prefetched
export PYTHONUNBUFFERED=1
echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-none} | start $(date)"
echo "commit $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
trap 'echo "end $(date)"' EXIT
