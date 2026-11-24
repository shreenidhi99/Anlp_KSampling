"""Sampler microbenchmark for DiffusionGemma (notebook 08).

Compares, on real logits and the real embedding table, with no model loaded:

  vllm   : vLLM's fused per-row statistics kernel (one pass: argmax, Gumbel sample, entropy) that also writes the
           softmax in bf16, followed by the full self-conditioning matmul  probs @ E * sqrt(d)   (vLLM's current path)
  topk   : the same statistics kernel WITHOUT writing probs, plus torch.topk and a k-row self-conditioning sum
  hf     : the Hugging Face / diffusers stock path (several full-vocabulary fp32 passes), for reference

`_row_stats_kernel` below is copied from vLLM (vllm/model_executor/models/diffusion_gemma_sampler.py, Apache-2.0,
Copyright contributors to the vLLM project), unchanged except that it also stores the row's log-normalizer (lse),
which the top-k path needs to turn top-k logits into probabilities. The commit/re-noise/stop kernel (vLLM's
`sample_step`) is identical in both vLLM and top-k paths and is not timed.
"""
import glob
import json
import math
import os
import time

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


# ============================================================================ the three paths
def path_vllm(L, temps, CL, E, scale):
    """vLLM today: stats + bf16 probs in the kernel, then the full self-conditioning matmul."""
    argmax, sample, ent, lse, probs = row_stats(L, temps, CL, probs_dtype=E.dtype)
    hint = torch.matmul(probs, E) * scale
    return dict(argmax=argmax, entropy=ent, hint=hint)


def topk_hint(L, temps, CL, lse, E, scale, k, tail='renorm', E_sum=None):
    inv = (1.0 / temps.clamp_min(1e-10)).repeat_interleave(CL)            # [rows]
    vals, idx = torch.topk(L, k, dim=-1)                                   # extra read of the logits (unfused)
    p = torch.exp(vals * inv[:, None] - lse[:, None])                      # exact top-k probabilities
    if tail == 'renorm':
        w = p / p.sum(-1, keepdim=True)
    else:
        w = p
    try:
        hint = F.embedding_bag(idx, E, per_sample_weights=w.to(E.dtype), mode='sum')
    except RuntimeError:
        hint = torch.cat([torch.einsum('rk,rkd->rd', w[s:s + 2048].to(E.dtype), E[idx[s:s + 2048]])
                          for s in range(0, idx.shape[0], 2048)])
    if tail == 'mean':
        V = E.shape[0]
        m = (1.0 - p.sum(-1)).clamp_min(0)
        top_sum = F.embedding_bag(idx, E, mode='sum').float()
        hint = hint.float() + m[:, None] * (E_sum[None] - top_sum) / (V - k)
        hint = hint.to(E.dtype)
    return hint * scale


def path_topk(L, temps, CL, E, scale, k, tail='renorm', E_sum=None):
    """Ours: same stats kernel without the probs write, then a k-row self-conditioning sum."""
    argmax, sample, ent, lse, _ = row_stats(L, temps, CL, probs_dtype=None)
    hint = topk_hint(L, temps, CL, lse, E, scale, k, tail, E_sum)
    return dict(argmax=argmax, entropy=ent, hint=hint)


def path_hf(L, temps, CL, E, scale, b=0.1):
    """Hugging Face / diffusers stock: scheduler step + pipeline stop check + decoder self-conditioning."""
    T = temps.repeat_interleave(CL)[:, None]
    mo = L / T                                                             # scheduler: model_output / temperature
    probs = torch.softmax(mo, dim=-1)                                      # _sample_from_logits: probs
    scaled = torch.softmax(mo / 1.0, dim=-1)                               #   and scaled_probs (temperature=1.0)
    tok = torch.multinomial(scaled, 1)
    sampled_probs = probs.gather(-1, tok)
    del probs, scaled
    ent = torch.distributions.Categorical(logits=mo).entropy()             # commit entropy
    s, idx = torch.sort(ent.view(-1, CL), dim=-1)
    keep = (s.cumsum(-1) - s) <= b
    ent2 = torch.distributions.Categorical(logits=mo.float()).entropy()    # pipeline stop check (recomputed)
    argmax = mo.argmax(-1)
    hint = torch.matmul(mo.softmax(-1, dtype=torch.float32).to(E.dtype), E) * scale   # decoder self-conditioning
    return dict(argmax=argmax, entropy=ent, hint=hint)


# ============================================================================ timing helpers
def time_cuda(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


def peak_extra_gb(fn):
    """Peak memory the call adds on top of what is already allocated (inputs, embedding table)."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return (peak - base) / 2**30


# ============================================================================ inputs
def load_embedding_table(model_id, device, cache_dir=None):
    """Read only the tied embedding table from the checkpoint (no model). Returns (E [V, d] bf16, key)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    idx_path = hf_hub_download(model_id, 'model.safetensors.index.json', cache_dir=cache_dir)
    weight_map = json.load(open(idx_path))['weight_map']
    cands = [k for k in weight_map if k.endswith('embed_tokens.weight') and 'vision' not in k and 'audio' not in k]
    if not cands:
        raise KeyError(f'no embed_tokens.weight in index; keys like: {list(weight_map)[:10]}')
    key = sorted(cands, key=len)[0]
    shard = hf_hub_download(model_id, weight_map[key], cache_dir=cache_dir)
    with safe_open(shard, framework='pt', device='cpu') as f:
        E = f.get_tensor(key)
    return E.to(device=device, dtype=torch.bfloat16).contiguous(), key


def load_saved_logits(pattern):
    """Saved notebook 04 steps: list of ([rows, V] fp32 on CPU, step index)."""
    out = []
    for f in sorted(glob.glob(pattern)):
        L = torch.load(f, map_location='cpu').float()
        step = int(os.path.basename(f).split('_')[-1].split('.')[0])
        out.append((L.reshape(-1, L.shape[-1]), step))
    return out


def make_batch(saved, B, CL, device, n_steps=48, t_min=0.4, t_max=0.8):
    """B sequences x CL rows, cycling through the saved steps; temperature from each step's schedule."""
    rows, temps = [], []
    for i in range(B):
        L, step = saved[i % len(saved)]
        rows.append(L[:CL])
        temps.append(t_min + (t_max - t_min) * (n_steps - step) / n_steps)
    L = torch.cat(rows).to(device).contiguous()
    return L, torch.tensor(temps, dtype=torch.float32, device=device)

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


# ============================================================================ the real vLLM kernel (optional)
def load_vllm_kernel():
    """vLLM's own `sample_row_stats`, if vLLM (with DiffusionGemma support) is installed in this environment.
    Returns (function, version string) or (None, reason)."""
    try:
        import vllm
        from vllm.model_executor.models.diffusion_gemma_sampler import sample_row_stats
        return sample_row_stats, getattr(vllm, '__version__', '?')
    except Exception as e:                      # not installed, or a version without DiffusionGemma
        return None, f'{type(e).__name__}: {e}'


def path_vllm_pkg(L, temps, CL, E, scale, vllm_fn, seed=0):
    """vLLM's current path using the installed vLLM kernel: stats + bf16 probs, then the full hint matmul."""
    argmax, sample, ent, probs = vllm_fn(L, temps, CL, seed, E.dtype)
    hint = torch.matmul(probs, E) * scale
    return dict(argmax=argmax, entropy=ent, hint=hint)
