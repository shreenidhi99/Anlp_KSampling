"""Copy of vLLM's DiffusionGemma row-statistics kernel (vllm/model_executor/models/diffusion_gemma_sampler.py,
Apache-2.0, Copyright contributors to the vLLM project), plus one extra output: the row's log-normalizer (lse).
Same random-number scheme as vLLM's kernel, so argmax, samples and entropies match it for the same seed."""
import torch
import triton
import triton.language as tl


# ============================================================================ vLLM kernel (copied, + lse output)
@triton.jit(do_not_specialize=["seed", "CL"])
def _row_stats_kernel(
    logits_ptr, logits_stride, temps_ptr, argmax_ptr, sample_ptr, entropy_ptr, lse_ptr,
    probs_ptr, probs_stride, seed, V, CL,
    WRITE_PROBS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    temp = tl.load(temps_ptr + row // CL).to(tl.float32)
    inv = 1.0 / tl.maximum(temp, 1e-10)
    noise = (temp > 0).to(tl.float32)
    base = logits_ptr + row.to(tl.int64) * logits_stride
    rng_base = row.to(tl.int64) * V

    m = float("-inf")
    z = 0.0
    s = 0.0
    best = float("-inf")
    best_idx = 0
    best_noisy = float("-inf")
    best_noisy_idx = 0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)
        block_max = tl.max(x, 0)
        new_m = tl.maximum(m, block_max)
        safe_m = tl.where(new_m > float("-inf"), new_m, 0.0)
        alpha = tl.where(m > float("-inf"), tl.exp((m - safe_m) * inv), 0.0)
        shift = tl.where(z > 0, (m - safe_m) * inv * z, 0.0)
        d = (x - safe_m) * inv
        e = tl.exp(d)
        z = z * alpha + tl.sum(e, 0)
        s = (s + shift) * alpha + tl.sum(tl.where(e > 0, d * e, 0.0), 0)
        m = new_m

        block_idx = tl.argmax(x, 0)
        take = block_max > best
        best_idx = tl.where(take, start + block_idx, best_idx)
        best = tl.where(take, block_max, best)

        u = tl.rand(seed, rng_base + offs)
        u = tl.maximum(u, 1e-20)
        gumbel = -tl.log(-tl.log(u))
        noisy = tl.where(mask, x * inv + gumbel * noise, float("-inf"))
        noisy_max = tl.max(noisy, 0)
        noisy_idx = tl.argmax(noisy, 0)
        take_n = noisy_max > best_noisy
        best_noisy_idx = tl.where(take_n, start + noisy_idx, best_noisy_idx)
        best_noisy = tl.where(take_n, noisy_max, best_noisy)

    tl.store(argmax_ptr + row, best_idx)
    tl.store(sample_ptr + row, best_noisy_idx)
    tl.store(entropy_ptr + row, tl.log(z) - s / z)
    tl.store(lse_ptr + row, m * inv + tl.log(z))          # added: log sum_j exp(x_j / T)

    if WRITE_PROBS:
        out = probs_ptr + row.to(tl.int64) * probs_stride
        for start in range(0, V, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < V
            x = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)
            p = tl.exp((x - m) * inv) / z
            tl.store(out + offs, p.to(probs_ptr.dtype.element_ty), mask=mask)


def row_stats(logits, temps, canvas_len, seed=0, probs_dtype=None):
    """logits [rows, V] fp32; temps [rows // canvas_len]. Returns argmax, sample, entropy, lse, probs (or None)."""
    rows, V = logits.shape
    dev = logits.device
    argmax = torch.empty(rows, dtype=torch.int64, device=dev)
    sample = torch.empty(rows, dtype=torch.int64, device=dev)
    entropy = torch.empty(rows, dtype=torch.float32, device=dev)
    lse = torch.empty(rows, dtype=torch.float32, device=dev)
    probs = torch.empty(rows, V, dtype=probs_dtype, device=dev) if probs_dtype is not None else None
    _row_stats_kernel[(rows,)](
        logits, logits.stride(0), temps, argmax, sample, entropy, lse,
        probs if probs is not None else entropy, probs.stride(0) if probs is not None else 0,
        seed, V, canvas_len, WRITE_PROBS=probs is not None, BLOCK=4096, num_warps=8,
    )
    return argmax, sample, entropy, lse, probs


