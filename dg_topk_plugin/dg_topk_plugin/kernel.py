"""Kernels for the dg_topk plugin, kept in sync with hint_bench.py (notebook 08).

`_row_stats_kernel` is copied from vLLM (vllm/model_executor/models/diffusion_gemma_sampler.py, Apache-2.0,
Copyright contributors to the vLLM project) with an extra lse output and a DO_SAMPLE switch. `_row_stats_topk_kernel`
is the same pass plus a running top-k. Same random-number scheme as vLLM's kernel, so argmax, samples and entropies
match it for the same seed when DO_SAMPLE is on."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================================ vLLM kernel (copied, + lse output)
@triton.jit(do_not_specialize=["seed", "CL"])
def _row_stats_kernel(
    logits_ptr, logits_stride, temps_ptr, argmax_ptr, sample_ptr, entropy_ptr, lse_ptr,
    probs_ptr, probs_stride, seed, V, CL,
    WRITE_PROBS: tl.constexpr, BLOCK: tl.constexpr, DO_SAMPLE: tl.constexpr = True,
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

        if DO_SAMPLE:                                       # added switch; True = vLLM's kernel exactly
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
    if DO_SAMPLE:
        tl.store(sample_ptr + row, best_noisy_idx)
    else:
        tl.store(sample_ptr + row, best_idx)
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


def row_stats(logits, temps, canvas_len, seed=0, probs_dtype=None, do_sample=True):
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
        seed, V, canvas_len, WRITE_PROBS=probs is not None, BLOCK=4096, DO_SAMPLE=do_sample, num_warps=8,
    )
    return argmax, sample, entropy, lse, probs


# ============================================================================ fused: vLLM's statistics + a running top-k
# Order-preserving int keys: a float's bits mapped so that integer order = float order, then (key << 32) | index.
_NEG_INF_KEY32 = (-8388608) ^ 0x7FFFFFFF          # key of -inf (bits 0xFF800000 as int32 = -8388608)
NEG_KEY = _NEG_INF_KEY32 << 32                    # key of (-inf, index 0): below every real logit's key


@triton.jit(do_not_specialize=["seed", "CL"])
def _row_stats_topk_kernel(
    logits_ptr, logits_stride, temps_ptr, argmax_ptr, sample_ptr, entropy_ptr, topi_ptr, topp_ptr,
    seed, V, CL, NEG_KEY,
    K: tl.constexpr, BLOCK: tl.constexpr, DO_SAMPLE: tl.constexpr,
):
    """One pass per row: vLLM's argmax / Gumbel sample / entropy (same code, same random numbers) plus the row's
    top-K logits, kept as a running list. A block is searched for candidates only if its max beats the current
    K-th best, which after the first few blocks is rare (about K * (1 + ln(V / BLOCK)) insertions per row)."""
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

    kidx = tl.arange(0, K)
    top = tl.zeros([K], dtype=tl.int64) + NEG_KEY       # running top-K keys (unsorted)
    min_key = tl.min(top, 0)                            # the K-th best key so far
    min_val = float("-inf")                             # ... and its logit

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

        if DO_SAMPLE:
            u = tl.rand(seed, rng_base + offs)
            u = tl.maximum(u, 1e-20)
            gumbel = -tl.log(-tl.log(u))
            noisy = tl.where(mask, x * inv + gumbel * noise, float("-inf"))
            noisy_max = tl.max(noisy, 0)
            noisy_idx = tl.argmax(noisy, 0)
            take_n = noisy_max > best_noisy
            best_noisy_idx = tl.where(take_n, start + noisy_idx, best_noisy_idx)
            best_noisy = tl.where(take_n, noisy_max, best_noisy)

        # running top-K: only blocks whose max can enter the list
        if block_max >= min_val:
            bits = x.to(tl.int32, bitcast=True)
            k32 = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
            key = (k32.to(tl.int64) << 32) | offs.to(tl.int64)
            cand = tl.where(mask & (key > min_key), key, NEG_KEY)
            bkey = tl.max(cand, 0)
            while bkey > min_key:
                pos = tl.argmin(top, 0)
                top = tl.where(kidx == pos, bkey, top)       # replace the current K-th best
                min_key = tl.min(top, 0)
                cand = tl.where(cand == bkey, NEG_KEY, cand)
                bkey = tl.max(cand, 0)
            mk32 = (min_key >> 32).to(tl.int32)
            mbits = tl.where(mk32 >= 0, mk32, mk32 ^ 0x7FFFFFFF)
            min_val = mbits.to(tl.float32, bitcast=True)

    tl.store(argmax_ptr + row, best_idx)
    if DO_SAMPLE:
        tl.store(sample_ptr + row, best_noisy_idx)
    else:
        tl.store(sample_ptr + row, best_idx)
    tl.store(entropy_ptr + row, tl.log(z) - s / z)

    ti = (top & 0xFFFFFFFF).to(tl.int32)
    tk32 = (top >> 32).to(tl.int32)
    tbits = tl.where(tk32 >= 0, tk32, tk32 ^ 0x7FFFFFFF)
    tval = tbits.to(tl.float32, bitcast=True)
    p = tl.where(top > NEG_KEY, tl.exp((tval - m) * inv) / z, 0.0)  # exact top-K probabilities
    out = row.to(tl.int64) * K + kidx
    tl.store(topi_ptr + out, ti)
    tl.store(topp_ptr + out, p)


def row_stats_topk(logits, temps, canvas_len, k=64, seed=0, do_sample=True, block=4096, num_warps=8):
    """Fused statistics + top-k. Returns argmax, sample, entropy, top-k ids [rows, k] (unsorted), probs [rows, k]."""
    assert k & (k - 1) == 0, 'k must be a power of two'
    rows, V = logits.shape
    dev = logits.device
    argmax = torch.empty(rows, dtype=torch.int64, device=dev)
    sample = torch.empty(rows, dtype=torch.int64, device=dev)
    entropy = torch.empty(rows, dtype=torch.float32, device=dev)
    topi = torch.empty(rows, k, dtype=torch.int32, device=dev)
    topp = torch.empty(rows, k, dtype=torch.float32, device=dev)
    _row_stats_topk_kernel[(rows,)](
        logits, logits.stride(0), temps, argmax, sample, entropy, topi, topp,
        seed, V, canvas_len, NEG_KEY, K=k, BLOCK=block, DO_SAMPLE=do_sample, num_warps=num_warps,
    )
    return argmax, sample, entropy, topi, topp


def hint_from_topk(topi, topp, E, scale, tail='renorm'):
    w = topp / topp.sum(-1, keepdim=True) if tail == 'renorm' else topp
    return F.embedding_bag(topi.long(), E, per_sample_weights=w.to(E.dtype), mode='sum') * scale


def path_topk_fused(L, temps, CL, E, scale, k=64, tail='renorm'):
    """Ours, fused: one pass for statistics + top-k (identical decisions to vLLM), then the k-row hint."""
    argmax, sample, ent, topi, topp = row_stats_topk(L, temps, CL, k)
    return dict(argmax=argmax, entropy=ent, hint=hint_from_topk(topi, topp, E, scale, tail))


def path_topk_fused_nogumbel(L, temps, CL, E, scale, k=64, generator=None):
    """Timing estimate only: no per-element random numbers; the token is drawn from the top-k probabilities.
    (An exact sampler would also draw from the tail with probability = tail mass; that branch is rare and omitted.)"""
    argmax, _, ent, topi, topp = row_stats_topk(L, temps, CL, k, do_sample=False)
    w = topp / topp.sum(-1, keepdim=True)
    sample = topi.gather(1, torch.multinomial(w, 1, generator=generator)).squeeze(1)
    return dict(argmax=argmax, sample=sample, entropy=ent, hint=hint_from_topk(topi, topp, E, scale))



# ============================================================================ exact sampling from the top-k (no per-logit RNG)
def tail_mass_from_topk(topp):
    """Probability outside the top-k, accurate even when it is tiny: -expm1(log(sum top-k probs))."""
    return (-torch.expm1(torch.log(topp.clamp_min(1e-38)).logsumexp(-1))).clamp_min(0.0)


def sample_topk_exact(L, temps, CL, topi, topp, generator=None):
    """Draw one token per row from the FULL softmax(L / T), using the top-k:
    with probability (1 - tail mass) pick among the top-k (proportional to their probabilities), otherwise draw from
    the tail (the full row with the top-k masked out). The tail branch is rare and touches only those rows.
    Returns (tokens [rows] int64, number of tail draws)."""
    rows = L.shape[0]
    tm = tail_mass_from_topk(topp)
    u = torch.rand(rows, device=L.device, generator=generator)
    pick = torch.multinomial(topp / topp.sum(-1, keepdim=True), 1, generator=generator).squeeze(1)
    tok = topi.gather(1, pick[:, None]).squeeze(1).long()
    tail = u < tm
    n_tail = int(tail.sum())
    if n_tail:
        r = tail.nonzero().squeeze(1)
        inv = (1.0 / temps.float().clamp_min(1e-10)).repeat_interleave(CL)[r]
        lt = L[r].float() * inv[:, None]
        lt.scatter_(1, topi[r].long(), float('-inf'))
        tok[r] = torch.multinomial(torch.softmax(lt, -1), 1, generator=generator).squeeze(1)
    return tok, n_tail


def path_topk_exact(L, temps, CL, E, scale, k=64, generator=None):
    """Ours, fastest: fused statistics + top-k WITHOUT per-logit random numbers, exact sampling from the top-k with a
    tail branch, and the k-row hint. Argmax and entropy are identical to vLLM's; the sampled tokens follow the same
    distribution but use different random numbers."""
    argmax, _, ent, topi, topp = row_stats_topk(L, temps, CL, k, do_sample=False)
    sample, n_tail = sample_topk_exact(L, temps, CL, topi, topp, generator)
    return dict(argmax=argmax, sample=sample, entropy=ent, hint=hint_from_topk(topi, topp, E, scale), n_tail=n_tail)


