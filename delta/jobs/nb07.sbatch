#!/bin/bash
# Notebook 07: certified sampler in the real diffusers pipeline: pad check, Test A, Test B, GSM8K.
# Whole node: the ~52 GB bf16 model is split over 4x A100 40 GB. GSM8K results append to a CSV,
# so a job that hits the time limit can simply be resubmitted and resumes.
#SBATCH --job-name=nb07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=delta/logs/%x-%j.out
source delta/common.sh
export NB_N_GPUS=4 NB_N_GSM="$NB07_N_GSM" NB_RUN_CONFIGS="$NB07_RUN_CONFIGS"
echo "N_GSM=$NB_N_GSM | configs: $NB_RUN_CONFIGS"
"$VENV_KS/bin/python" delta/run_notebook.py 07_certified_full_pipeline.ipynb anlp-ksampling
ls -la runs/nb07/
