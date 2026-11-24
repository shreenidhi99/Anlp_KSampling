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
