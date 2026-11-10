The six notebooks at a glance
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