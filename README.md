The notebooks at a glance (01-07)
#	Notebook	Needs the model?	GPU	Answers
01	topk sampling harness	No	1 GPU (T4 or better)	Is sampling expensive, and does top-k fix it?
02	diffusers smoke test	Yes (4-bit)	1 A100	Does the model run, and does my scheduler reproduce the stock decisions?
03	certified topk sampler	No	1 GPU	Does the certified method give identical decisions?
04	bf16 two gpu split	Yes (bf16)	Both A100s	Does certification hold on real, unquantized logits, and is accuracy unchanged?
05	single gpu speed tests	Yes (4-bit/int8)	1 A100	How much faster and smaller is the sampler on real data?
06	vllm tensor parallel sweep	Yes (bf16)	Both A100s	Where does AR overtake diffusion in real serving?
01: topk sampling harness

Input: synthetic logits with DiffusionGemma’s shape (batch × 256 × 262,144).
Does: implements the full sampler, an optimized full sampler (version B), and several top-k variants. It then:

times each one vs. batch size, with memory
checks whether top-k variants make the same commit and stop decisions
runs the full sampler loop (Algorithm 1) with a toy model, showing that plain top-k stops too early
Output: time and memory table vs. batch size (H1); decision-agreement table (H2, synthetic).
02: diffusers smoke test

Input: the real model, 4-bit, through the diffusers pipeline.
Does:

loads the model and generates one answer
records per-step logits and the exact temperature schedule
swaps in your scheduler and checks that its commits match the stock scheduler’s exactly (gate 1)
runs a 20-question GSM8K comparison
prints where the self-conditioning and stopping code live, which is what you patch next
Output: pass/fail on gate 1, saved logits, first accuracy numbers.
03: certified topk sampler

Input: synthetic logits, plus a replay cell for real ones.
Does: implements your method (entropy ranges, certified commits and stopping, fallback for loose slots) and:

asserts zero decision mismatches against the full sampler across many cases
reports how many slots needed the full-vocabulary fallback (the refinement rate)
checks the self-conditioning error bound
tests the optional exact-sampling branch
Output: proof that certification works (H4, synthetic), refinement rates.
04: bf16 two gpu split

Input: the real model in bf16, split across both GPUs. Slower than one GPU, so it’s not used for speed.
Does: during real generation on math, code, and chat prompts, checks the certified sampler on every denoising step for several k values, then runs GSM8K with stock, certified, and plain top-k.
Output: mismatch counts (must be 0), real refinement rates by step and task (H4), the unquantized accuracy baseline (H2).

05: single gpu speed tests

Input: the real model, quantized, pinned to one GPU.
Does:

Part A: takes real logits from a generation, scales them to batch 1–32, and times the scheduler and the self-conditioning mix separately: full vs. top-k.
Part B: end-to-end generation vs. batch size, stock vs. certified scheduler.
Output: speed and memory savings on real data (H1). Part B is the baseline to rerun after the self-conditioning patch.
06: vllm tensor parallel sweep

Input: stock DiffusionGemma and Gemma 4 AR in vLLM, each split across both GPUs.
Does: measures throughput at batch sizes 1–64 for both models, and optionally records a profiler trace of DiffusionGemma.
Output: the crossover plot (H3 baseline); the share of each step spent on sampling inside vLLM.

dg_sampling

The shared sampler code (certified decisions, top-k mix, the drop-in scheduler, helpers). Notebooks 04 and 05 write it to disk automatically.

How they fit together
01, 03  → method works (synthetic)
02      → model runs, scheduler is correct (gate 1)
04      → method works on real logits; accuracy unchanged
05      → method is faster on real data
06      → where it matters in real serving
Issues log

Problems found while running the notebooks, newest last. Each entry: what happened, the cause, the fix, and what still needs doing.

1. Notebook 02/04: model does not fit on the GPUs as written (2026-11-15)
   - Symptom: Notebook 04 hard-codes two GPUs with 37 GiB each, but on the shared 8 x A100 40GB box no GPU had more than ~14 GB free; ~52 GB of bf16 weights need 4-5 GPUs there.
   - Fix: both notebooks now start with a GPU-selection cell that sets CUDA_VISIBLE_DEVICES before torch is imported (auto-picks the N_GPUS GPUs with the most free memory, or MANUAL_GPUS), and size max_memory from each GPU's real free memory minus 4 GB headroom.
   - Note: setting CUDA_VISIBLE_DEVICES after CUDA has started in a kernel does nothing; restart the kernel first. Free memory changes as other users' jobs grow; raise N_GPUS if loading fails with an offload error.

2. Notebook 02: Colab-only code (2026-11-15)
   - Symptom: the logits-saving cell mounted Google Drive (google.colab), which crashes outside Colab.
   - Fix: USE_DRIVE = False by default; logits go to ./logits.

3. Notebook 02: GSM8K too small to compare variants (2026-11-15)
   - Change: N_GSM = 250 (was 20). Results are appended per problem to ./runs/nb02/gsm8k_250.csv and the cell resumes from it after an interruption.
   - Caveat: scoring takes the last number in the answer and compares strings, so "18.0" vs "18" or a trailing number counts as wrong. Same for all variants, so comparisons are fair; absolute accuracy may read low.

4. Notebook 04/05: certified sampler made wrong commit decisions (2026-11-15)
   - Symptom: Notebook 04 run reported 8 of 400 step checks with commit_mismatch = 1 (k=64: 4/100, k=256: 2/100, k=1024: 2/100, k=16: 0/100; stop mismatches 0/400). Mismatches must be 0.
   - Cause: topk_stats in dg_sampling.py computed the bounds in float32 with (a) the normalizer lse unshifted (large values lose ~1e-6) and (b) tail mass m = 1 - sum(p) (catastrophic cancellation when m is ~1e-7). The resulting [lo, hi] did not contain the true entropy: on the 6 saved logits steps, 77-256 of 256 slots violated the bounds at every k. A mismatch appears only when such a slot sits at the commit threshold, hence the low rate. Notebook 03 had already been fixed for this (commit 6e21297) but the fix was never copied into the %%writefile dg_sampling.py cell of Notebooks 04 and 05.
   - Fix: ported Notebook 03's topk_stats (normalizer shifted by the row max, m = -expm1(logsumexp(lp)), [lo, hi] widened by NUM_EPS) into both %%writefile cells.
   - Margin choice: on the saved logits, NUM_EPS = 2e-4 (Notebook 03's value) gives 0 violations but refines ~47% of slots, because the commit rule sums 256 per-slot margins against a 0.1 budget. 2e-5 gives 0 violations and ~10% refined; 5e-6 and smaller already violate. Set to 2e-5. This is validated on only 6 steps of one prompt; the Notebook 04 rerun checks it on every step of 9 prompts.
   - Old results invalid: runs/nb04/certified_real_bf16.csv, refinement_by_step.png and the "certified k64" rows of gsm8k_bf16.csv were produced with the broken bounds. Rerun Notebook 04 from section 5.
   - Open: even after the shift, float32 leaves ~5e-6 of error (and exact_entropy, the reference, is itself float32). Computing the top-k entropy terms and the reference in float64 may allow a much smaller margin and bring refinement back toward ~1%.
   - Also affected: Notebook 02 has its own inline copy of topk_stats (section 5) with the same float32 bug. Its GSM8K-250 run (runs/nb02/gsm8k_250.csv, finished 2026-11-15) gave stock 96.8%, certified k64 96.4%, topk64 renorm 94.4%; the certified k64 row used the broken bounds. stock and topk64 renorm rows are unaffected. Notebook 02's copy is not yet fixed.

5. Notebook 04: load failed with ValueError in accelerate_dispatch (2026-11-15)
   - Symptom: with N_GPUS = 6 and HEADROOM_GB = 4, max_memory summed to 61 GiB, weights loaded, then from_pretrained raised ValueError while dispatching.
   - Cause: not enough GPU budget. The loaded model takes ~54 GB (Notebook 02's run used ~54 GB), and device_map='auto' places whole decoder layers (~1.5 GB each with experts), so each GPU can waste up to a layer at its budget edge; the overflow went to CPU/disk offload, which the dispatch step rejects.
   - Fix: restart the kernel (the failed load keeps holding GPU memory), then N_GPUS = 7 and HEADROOM_GB = 3. Aim for a max_memory total of ~70 GiB or more.
   - Rerun result (2026-11-15, NUM_EPS = 2e-5, 7 GPUs): 0/400 commit and 0/400 stop mismatches at every k. Refinement ~10% of slots at every k (k=16: 10.8%, k=64/256/1024: 10.0%), flat in k, so it is set by the eps margin rather than the tail mass; reducing it needs the float64 route above. GSM8K-30: stock 96.7%, certified k64 96.7%, topk64 renorm 93.3%.

6. dg_certified.py (notebook 07): same float32 bounds bug (2026-11-15)
   - Found after pulling commit c275708. Its topk_stats still uses the unshifted lse and m = 1 - sum(p), with a pad of 1e-7 + 1e-6 * hi.
   - On the 6 saved logits steps (runs/nb04/step_*.pt), 515-802 of 1536 slots fall outside [lo, hi] (k=16: 515, k=64: 583, k=256: 671, k=1024: 802). The fixed dg_sampling.topk_stats gives 0. Expect non-zero decision mismatches from notebook 07 until it is fixed.
   - Not yet fixed. Port the shifted normalizer and expm1 tail mass; the pad then needs to be about 2e-5 absolute (5e-6 still violated in issue 4's test).
   - Fixed (2026-11-16): topk_stats now computes lse, the top-k terms and the tail mass in float64 (m = -expm1(logsumexp(top-k log-probs))). The pad depends on the reference (dc.PADS): 'stock' = 2e-5 (matches the float32 stock decisions; the stock entropy itself carries ~1e-5 of float32 rounding because logits / T reach ~75), 'exact' = 1e-9 (matches float64 entropies). Notebook 07 has a pad-check cell (dc.pad_sweep on runs/nb04/step_*.pt) to confirm 0 violations before Test B.
   - The 2026-11-16 notebook 07 run (commit 2e1fc06) used the unfixed file: its GSM8K accuracy/steps are usable, its Test B mismatch counts are not. Rerun the pad check and Test B.

7. Notebook 07 checks: ties, argmax and self-conditioning bound (2026-11-16)
   - The LM head runs in bf16, so exactly equal logits and entropies are common. torch.topk orders ties arbitrarily while torch.argmax takes the lowest index; the certified stop now breaks ties like argmax (this caused the sh_argmax_mismatch counts at k=16). The stock commit sort is not stable either: Test B now reports sh_commit_mismatch_ties (mismatches where the differing slots have identical stock entropy, i.e. both choices follow the rule).
   - Test B also reports mismatches against float64 entropies (sh_commit_mismatch_exact, sh_stop_mismatch_exact).
   - sh_sc_bound_violations counted thousands of slots because the stock self-conditioning reference is itself bf16-rounded (~0.2% relative error even when the tail is empty); the check now allows 1% of the reference norm.
   - Test A: on the code and chat prompts the original pipeline does not reproduce itself with the same seed (GPU non-determinism across 4 GPUs), so only the math prompt is a valid check. Open: deterministic mode or a step-level replay test.

8. Notebook 02: certified GSM8K-250 row still from buggy bounds (open)
   - Its inline topk_stats was never fixed (issue 4). Use notebook 07 for the certified GSM8K numbers instead.
