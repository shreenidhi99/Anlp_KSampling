"""Kernel-level profiling target for notebook 10.

Runs each sampler statistics kernel once on real logits, between cudaProfilerStart/Stop, so that Nsight Compute
(`ncu --profile-from-start off ...`) profiles exactly these launches and nothing else (warm-up and Triton compilation
happen before the profiled region). Launch order is printed and saved, so the notebook can label ncu's results.

  python profile_kernels.py --batch 8 --out runs/nb10/launch_order.json

Also usable without ncu: --ablation times a noise-ablation kernel (no random numbers / random numbers only / random
numbers + the two logarithms of Gumbel-max) to show which part of the per-logit sampling costs the time.
"""
import argparse
import json
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hint_bench as hb  # noqa: E402


@triton.jit
def _noise_ablation_kernel(logits_ptr, logits_stride, temps_ptr, out_ptr, seed, V, CL,
                           NOISE_MODE: tl.constexpr, BLOCK: tl.constexpr):
    """Same streaming structure as vLLM's statistics pass (online max/sum/entropy + argmax), with the sampling noise
    chosen by NOISE_MODE: 0 = none, 1 = tl.rand only (no logs), 2 = tl.rand + -log(-log(u)) (vLLM's Gumbel-max),
    3 = two logs of a cheap hash instead of tl.rand (isolates the cost of the random-number generator)."""
    row = tl.program_id(0)
    temp = tl.load(temps_ptr + row // CL).to(tl.float32)
    inv = 1.0 / tl.maximum(temp, 1e-10)
    base = logits_ptr + row.to(tl.int64) * logits_stride
    rng_base = row.to(tl.int64) * V
    m = float("-inf")
    z = 0.0
    s = 0.0
    best_noisy = float("-inf")
    best_noisy_idx = 0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)
        new_m = tl.maximum(m, tl.max(x, 0))
        alpha = tl.where(m > float("-inf"), tl.exp((m - new_m) * inv), 0.0)
        shift = tl.where(z > 0, (m - new_m) * inv * z, 0.0)
        d = (x - new_m) * inv
        e = tl.exp(d)
        z = z * alpha + tl.sum(e, 0)
        s = (s + shift) * alpha + tl.sum(tl.where(e > 0, d * e, 0.0), 0)
        m = new_m
        if NOISE_MODE == 0:
            noisy = x * inv
        elif NOISE_MODE == 1:
            noisy = x * inv + tl.rand(seed, rng_base + offs)
        elif NOISE_MODE == 2:
            u = tl.maximum(tl.rand(seed, rng_base + offs), 1e-20)
            noisy = x * inv - tl.log(-tl.log(u))
        else:
            h = ((rng_base + offs) * 2654435761) & 0xFFFFFF
            u = tl.maximum(h.to(tl.float32) * (1.0 / 16777216.0), 1e-20)
            noisy = x * inv - tl.log(-tl.log(u))
        noisy = tl.where(mask, noisy, float("-inf"))
        nm = tl.max(noisy, 0)
        ni = tl.argmax(noisy, 0)
        take = nm > best_noisy
        best_noisy_idx = tl.where(take, start + ni, best_noisy_idx)
        best_noisy = tl.where(take, nm, best_noisy)
    tl.store(out_ptr + row, (tl.log(z) - s / z) + best_noisy_idx.to(tl.float32) * 0.0)


def noise_ablation(L, temps, CL, mode):
    out = torch.empty(L.shape[0], dtype=torch.float32, device=L.device)
    _noise_ablation_kernel[(L.shape[0],)](L, L.stride(0), temps, out, 0, L.shape[1], CL,
                                          NOISE_MODE=mode, BLOCK=4096, num_warps=8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--logits', default='./runs/nb04/step_*.pt')
    ap.add_argument('--out', default='./runs/nb10/launch_order.json')
    ap.add_argument('--ablation', action='store_true', help='time the noise-ablation kernel instead of profiling')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    CL = 256
    saved = hb.load_saved_logits(args.logits)
    assert saved, f'no logits at {args.logits}'
    L, temps = hb.make_batch(saved, args.batch, CL, 'cuda')
    bytes_read = L.numel() * 4

    if args.ablation:
        names = {0: 'no noise', 1: 'random numbers only', 2: 'random numbers + 2 logs (vLLM Gumbel-max)',
                 3: '2 logs of a cheap hash (no RNG)'}
        rows = []
        for mode, name in names.items():
            ms = hb.time_cuda(lambda: noise_ablation(L, temps, CL, mode))
            rows.append(dict(mode=mode, variant=name, ms=ms, GBps=bytes_read / 1e9 / (ms / 1e3)))
            print(f'{name:45s} {ms:8.3f} ms  {rows[-1]["GBps"]:7.0f} GB/s')
        json.dump(dict(batch=args.batch, rows=rows), open(args.out.replace('.json', '_ablation.json'), 'w'), indent=1)
        return

    vllm_fn, vllm_info = hb.load_vllm_kernel()
    variants = [
        ('vllm_copy_stats',          lambda: hb.row_stats(L, temps, CL)),
        ('vllm_copy_stats_no_rng',   lambda: hb.row_stats(L, temps, CL, do_sample=False)),
        ('vllm_copy_stats+probs',    lambda: hb.row_stats(L, temps, CL, probs_dtype=torch.bfloat16)),
        ('fused_topk64',             lambda: hb.row_stats_topk(L, temps, CL, 64)),
        ('fused_topk64_no_rng',      lambda: hb.row_stats_topk(L, temps, CL, 64, do_sample=False)),
    ]
    if vllm_fn is not None:
        variants.append(('vllm_installed_stats+probs', lambda: vllm_fn(L, temps, CL, 0, torch.bfloat16)))

    for _, fn in variants:          # compile + warm up outside the profiled region
        fn(); fn()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _, fn in variants:
        fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    order = [name for name, _ in variants]
    json.dump(dict(batch=args.batch, bytes_read=bytes_read, order=order, vllm=vllm_info), open(args.out, 'w'), indent=1)
    print('profiled launches, in order:', order)


if __name__ == '__main__':
    main()
