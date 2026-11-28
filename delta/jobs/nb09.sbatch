#!/bin/bash
# Notebook 09a (plugin smoke test, ~5 min) and, only if it passes, notebook 09 (vLLM end to end:
# stock vs top-k self-conditioning, plus the AR references). Whole node: tensor parallel 4 over 4x A100.
# Results append to runs/nb09/e2e.csv, so a job that hits the time limit resumes when resubmitted.
#SBATCH --job-name=nb09
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=delta/logs/%x-%j.out
source delta/common.sh
"$VENV_VLLM/bin/python" delta/run_notebook.py 09a_plugin_smoke_test.ipynb anlp-vllm

export NB_N_GPUS=4 NB_N_QUESTIONS="$NB09_N_QUESTIONS" NB_RUN_AR="$NB09_RUN_AR"
echo "tensor parallel $NB_N_GPUS | questions $NB_N_QUESTIONS | AR references: $NB_RUN_AR"
"$VENV_VLLM/bin/python" delta/run_notebook.py 09_vllm_end_to_end.ipynb anlp-vllm
ls -la runs/nb09/
