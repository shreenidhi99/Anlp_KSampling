"""vLLM general plugin: top-k self-conditioning for DiffusionGemma.

Off unless the environment variable DG_TOPK_SC_K is set to a positive k (e.g. 64). Then, in every vLLM process:

  * `sample_row_stats` runs the same fused statistics pass as vLLM (argmax, sample, entropy: identical for the
    same seed, so commit / re-noise / stop decisions are unchanged) but does NOT write the full softmax. Instead it
    returns the top-k token ids and their probabilities.
  * `_store_self_conditioning` builds the self-conditioning soft embedding from those k rows of the embedding
    table (an embedding_bag) instead of the full [rows, vocab] @ [vocab, hidden] matmul. Under tensor parallelism
    each rank sums the top-k ids in its vocabulary shard and the partial sums are all-reduced, as in vLLM.

DG_TOPK_SC_TAIL = renorm (default: weights rescaled to sum to 1) | drop (raw top-k probabilities).
DG_TOPK_SAMPLE  = vllm (default: vLLM's per-logit Gumbel sample, identical tokens to stock vLLM for the same seed)
                | topk_exact (no per-logit random numbers: exact sampling from the top-k with a rare tail branch;
                  same token distribution, different random numbers, ~4x faster statistics pass)
k must be a power of two (the fused kernel's running top-k).
"""
import logging
import os

logger = logging.getLogger("dg_topk_plugin")


class TopKProbs:
    """Stands in for vLLM's [rows, vocab] probs tensor: top-k ids and weights per row."""

    def __init__(self, idx, w):
        self.idx, self.w = idx, w

    def view(self, *shape):                       # vLLM calls probs_rows.view(n, W, -1)
        n, W = shape[0], shape[1]
        return TopKProbs(self.idx.reshape(n, W, -1), self.w.reshape(n, W, -1))


def register():
    k = int(os.environ.get("DG_TOPK_SC_K", "0") or 0)
    if k <= 0:
        return
    import torch
    import torch.nn.functional as F
    import vllm.model_executor.models.diffusion_gemma as dg
    if getattr(dg, "_dg_topk_patched", False):
        return
    from .kernel import row_stats, row_stats_topk, sample_topk_exact

    tail = os.environ.get("DG_TOPK_SC_TAIL", "renorm")
    mode = os.environ.get("DG_TOPK_SAMPLE", "vllm")
    if k & (k - 1):
        raise ValueError(f"DG_TOPK_SC_K must be a power of two, got {k}")
    if mode not in ("vllm", "topk_exact"):
        raise ValueError(f"DG_TOPK_SAMPLE must be vllm or topk_exact, got {mode}")
    orig_store = dg._store_self_conditioning
    orig_stats = dg.sample_row_stats

    def sample_row_stats(logits, temps, canvas_len, seed, probs_dtype):
        if probs_dtype is None and mode == "vllm":          # no self-conditioning needed and vLLM's sampling
            argmax, sample, entropy, _, _ = row_stats(logits, temps, canvas_len, seed, None)
            return argmax, sample, entropy, None
        # one pass: vLLM's statistics + running top-k (+ vLLM's Gumbel sample when mode == vllm)
        argmax, sample, entropy, topi, topp = row_stats_topk(
            logits, temps, canvas_len, min(k, 1 << (logits.shape[-1].bit_length() - 1)), seed,
            do_sample=(mode == "vllm"))
        if mode == "topk_exact":
            g = torch.Generator(device=logits.device)
            g.manual_seed(int(seed))
            sample, _ = sample_topk_exact(logits, temps, canvas_len, topi, topp, g)
        if probs_dtype is None:
            return argmax, sample, entropy, None
        p = topp / topp.sum(-1, keepdim=True) if tail == "renorm" else topp
        return argmax, sample, entropy, TopKProbs(topi.long(), p)

    def _store_self_conditioning(sc_embeds, decode_slots, keep, probs, pin_mask, embed_weight, normalizer,
                                 sc_vocab_start, sc_vocab_end, tp_size, tp_group_name):
        if not isinstance(probs, TopKProbs):
            return orig_store(sc_embeds, decode_slots, keep, probs, pin_mask, embed_weight, normalizer,
                              sc_vocab_start, sc_vocab_end, tp_size, tp_group_name)
        n, W, kk = probs.idx.shape
        idx = probs.idx.reshape(-1, kk)
        w = probs.w.reshape(-1, kk)
        local = (idx >= sc_vocab_start) & (idx < sc_vocab_end)       # ids in this rank's vocabulary shard
        lidx = torch.where(local, idx - sc_vocab_start, torch.zeros_like(idx))
        lw = torch.where(local, w, torch.zeros_like(w)).to(embed_weight.dtype)
        table = embed_weight[: sc_vocab_end - sc_vocab_start]
        soft = F.embedding_bag(lidx, table, per_sample_weights=lw, mode="sum").view(n, W, -1)
        if tp_size > 1:
            soft = torch.ops.vllm.all_reduce(soft, group_name=tp_group_name)
        keep2 = keep[:, None, None] & ~pin_mask[decode_slots, :, None]
        sc_embeds[decode_slots] = (soft * normalizer * keep2).to(sc_embeds.dtype)

    # the model calls sample_row_stats.register_warmup() at init; keep vLLM's warmup of its own kernel
    if hasattr(orig_stats, "register_warmup"):
        sample_row_stats.register_warmup = orig_stats.register_warmup
    dg.sample_row_stats = sample_row_stats
    dg._store_self_conditioning = _store_self_conditioning
    dg._dg_topk_patched = True
    logger.warning("[dg_topk] top-k self-conditioning enabled: k=%d, tail=%s, sample=%s (pid %d)", k, tail, mode, os.getpid())
