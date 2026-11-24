# Runbook: notebooks 08 and 07

Step-by-step instructions for running the current experiments on the GPU machine. Written so that a person or a coding agent (e.g. Claude Code in the terminal or VS Code) can follow them top to bottom and report back.

## 0. Setup (once)

```bash
cd /data/users/shreenis/private/Anlp_KSampling          # the repo folder; run everything from here
git pull                                                 # need commit 3e66675 or later
source .venv-2/bin/activate                              # the environment in requirements.txt
pip install nbconvert                                    # headless notebook execution (not in requirements.txt)
python -m ipykernel install --user --name anlp-ksampling # register this env as a Jupyter kernel
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
```

Checks before running anything:

```bash
# the updated module must be the one next to the notebooks (an older copy once ran from another folder)
python -c "import dg_certified, inspect; s = inspect.getsource(dg_certified); print('float64 fix:', 'PADS' in s)"
ls runs/nb04/step_*.pt                                   # logits saved by notebook 04; notebook 08 and 07's pad check need them
```

If `runs/nb04/step_*.pt` is missing, run notebook 04 section 5 first (it saves 6 steps).

Running a notebook headless (outputs are saved into the notebook):

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=anlp-ksampling --ExecutePreprocessor.timeout=-1 NOTEBOOK.ipynb
```

For long runs, use `tmux` or `nohup ... > NAME.log 2>&1 &` so the run survives a closed terminal.

## 1. Notebook 08: sampler microbenchmark (1 GPU, no model, ~30 min)

Needs one A100 with **~30 GB free** for batch sizes up to 64 (larger batches are skipped automatically if memory is short). It downloads only the embedding table from the Hugging Face cache.

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=anlp-ksampling --ExecutePreprocessor.timeout=-1 08_sampler_microbenchmark.ipynb
```

Outputs: `runs/nb08/sampler_bench.csv`, `runs/nb08/sampler_bench.png`, tables in the notebook.

Success criteria:
- Section 2: `entropy max |kernel - float64|` around 1e-5 or smaller; top-k hint error well under 1% at k = 64 (`renorm` should be lowest).
- Section 4: a `Summary` table with `speedup vs vllm` per batch size, plus memory per path. `OOM` entries for the HF path at large batch are expected.

Report back: the correctness table, the `Summary` table, the memory table, and the plot.

## 2. Notebook 07: exactness checks and GSM8K (2 GPUs, bf16 model)

Needs ~70 GB of free GPU memory across the selected GPUs (model ~54 GB + headroom). In the first cell set `N_GPUS = 2` and, if needed, `MANUAL_GPUS = '0,1'` (the two free GPUs). `REFERENCE = 'stock'` is the default.

Run order, and what to look for:

1. **Pad check** (cell "Pad check on saved logits"): `violations` must be 0 for every row; note `refined_pct` for each pad and reference.
2. **Test A**: all three columns `True` for every prompt (the `DETERMINISTIC` switch is on). If `original_matches_itself` is `False`, the GPUs are still non-deterministic and the other two columns cannot be judged.
3. **Test B (shadow)**: `sh_commit_mismatch_rows` should equal `sh_commit_mismatch_ties` (any mismatch is only an exact tie), `sh_stop_mismatch` = 0, `sh_argmax_mismatch` = 0, `sh_sc_bound_violations` = 0. Also read `sh_commit_mismatch_exact` / `sh_stop_mismatch_exact` (against float64 entropies).
4. **GSM8K**: set `N_GSM = 250` (then 1319 later). Time: about 5 s per question per config, about 20 s for `z=0`, so 250 questions x 7 configs is roughly 4 hours. To save time, drop configs from `RUN_CONFIGS` (keep `stock (patched)`, `certified, sc full`, `certified, sc topk renorm`, `naive topk renorm`). Results append to `runs/nb07/gsm8k_<N>_k64.csv`, so an interrupted run resumes.

Report back: the pad-check table, Test A and Test B tables, and the GSM8K summary + paired comparison.

## 3. After a run

```bash
git add 08_sampler_microbenchmark.ipynb 07_certified_full_pipeline.ipynb
git commit -m "Run outputs: notebook 08 sampler benchmark, notebook 07 checks (date)"
git push
```

`runs/` is git-ignored, so copy any CSV or PNG you want to keep into the notebook outputs or attach them separately.
