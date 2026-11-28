#!/bin/bash
# Notebook 08: sampler microbenchmark. 1 GPU, no model (embedding table + saved logits). ~1 h.
#SBATCH --job-name=nb08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=delta/logs/%x-%j.out
source delta/common.sh
"$VENV_KS/bin/python" delta/run_notebook.py 08_sampler_microbenchmark.ipynb anlp-ksampling
ls -la runs/nb08/
