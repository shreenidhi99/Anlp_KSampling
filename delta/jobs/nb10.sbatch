#!/bin/bash
# Notebook 10: profiling. Part A (Nsight Compute; skipped if counters are not permitted), Part B (noise
# ablation), Part C (vLLM profiler, tensor parallel 4, same as notebook 09). Whole node.
#SBATCH --job-name=nb10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=delta/logs/%x-%j.out
source delta/common.sh
module load cuda >/dev/null 2>&1 || true          # provides ncu for Part A if available
export NB_RUN_PART_C=1 NB_N_GPUS=4
"$VENV_VLLM/bin/python" delta/run_notebook.py 10_profiling.ipynb anlp-vllm
du -sh runs/nb10/
