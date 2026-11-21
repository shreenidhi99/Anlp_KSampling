"""Certified top-k sampling for the Hugging Face DiffusionGemma sampler (notebook 07).

Built against the real source:
  diffusers   pipelines/diffusion_gemma/pipeline_diffusion_gemma.py   (stop check, self-cond hand-off)
  diffusers   schedulers/scheduling_entropy_bound.py                  (commit rule, temperature)
  transformers models/diffusion_gemma/modeling_diffusion_gemma.py     (self-conditioning matmul)

What the stock code does on every denoising step (per sequence, V = 262,144):
  1. scheduler:  model_output / T                      -> full fp32 [256, V]
  2. scheduler:  softmax + multinomial                 -> full fp32 [256, V]
  3. scheduler:  Categorical(...).entropy()            -> full pass, commit rule on the entropies
  4. pipeline:   Categorical(pred_logits).entropy()    -> SAME entropies recomputed for the stop check
  5. decoder:    softmax(pred_logits) @ E * sqrt(d)    -> full [256, V] x [V, 2816] matmul (self-conditioning)

What this module changes (nothing in the model weights or the commit/stop rules):
  * commit:  certified from top-k entropy intervals [lo, hi]; ambiguous slots refined exactly
  * sample:  top-k with an exact tail branch (distribution-exact, the tail is rarely touched)
  * stop:    same rule (argmax unchanged AND mean entropy < 0.005), certified from the same [lo, hi]
  * self-conditioning: 'full' | 'zero' | 'topk' with tail 'drop' | 'renorm' | 'mean'

How it is wired in, without editing library files:
  * a scheduler subclass (CertifiedEBScheduler)
  * a pipeline subclass whose __call__ is the INSTALLED stock __call__ with four small, asserted text edits
  * a wrapper on decoder.self_conditioning.forward that injects our soft embeddings
"""
import inspect
import math
import re
import textwrap
import time

import torch
import torch.nn.functional as F

E_STOP = 0.005


# ============================================================================ basic pieces
def stock_entropy(rows, tau):
    """Entropy computed exactly the way the stock scheduler and pipeline compute it."""
    return torch.distributions.Categorical(logits=rows.float() / tau).entropy()


def n_commit(e, b):
    s, _ = torch.sort(e)
    return int(((s.cumsum(0) - s) <= b).sum())


def commit_exact(e, b):
    """Stock commit rule: sort ascending, accept while (cumsum - current) <= b."""
    order = torch.argsort(e, stable=True)
    keep = torch.zeros_like(e, dtype=torch.bool)
    keep[order[:n_commit(e, b)]] = True
    return keep


def chunked_logsumexp_shifted(x, tau, shift, chunk=32_768):
    """log-sum-exp of (x - shift) / tau, one vocabulary chunk at a time. shift is the row max, so the
    result is near 0, where float32 is precise (an unshifted value near 45 at tau=0.4 loses ~4e-6)."""
    out = None
    for s in range(0, x.shape[-1], chunk):
        part = torch.logsumexp((x[..., s:s + chunk].float() - shift) / tau, dim=-1)
        out = part if out is None else torch.logaddexp(out, part)
    return out


# Margin on [lo, hi] for float32 rounding. 2e-5 is the smallest tested value with 0 bound violations on the
# saved real logits (5e-6 already violates); 2e-4 is also valid but refines ~47% of slots. See README, issue 4.
NUM_EPS = 2e-5


def topk_stats(logits, tau, k, eps=NUM_EPS):
    """Exact top-k probabilities, tail mass m, and an entropy interval [lo, hi] that contains the true entropy.
    lo: tail mass on as few tokens as possible.  hi: tail mass spread uniformly over the other V-k tokens.
    float32-safe (same as notebooks 03-05): normalizer relative to the row max, tail mass via expm1
    instead of 1 - sum(p), and [lo, hi] widened by eps for rounding error."""
    V = logits.shape[-1]
    vals, idx = torch.topk(logits, k, dim=-1)                     # sorted, largest first
    top1 = vals[..., :1].float()                                  # row max
    lse_s = chunked_logsumexp_shifted(logits, tau, top1)          # normalizer relative to the max (near 0)
    lp = (vals.float() - top1) / tau - lse_s.unsqueeze(-1)
    p = lp.exp()
    h = -(p * lp).sum(-1)
    m = (-torch.expm1(torch.logsumexp(lp, dim=-1))).clamp(0.0, 1.0)   # 1 - sum(p), computed stably
    hi = h - m * torch.log(m.clamp_min(1e-30)) + (m * math.log(V - k) if V > k else 0.0)
    lo = h - m * torch.log(torch.minimum(m, p[..., -1]).clamp_min(1e-30))
    lse = top1.squeeze(-1) / tau + lse_s                          # unshifted normalizer, same meaning as before
    return dict(idx=idx, p=p, m=m, lse=lse, h=h, lo=(lo - eps).clamp_min(0.0), hi=hi + eps)


# ============================================================================ certified decisions
def certified_commit_row(lo, hi, refine, b, chunk=8):
    """Commit mask identical to commit_exact(true entropies). Returns (mask, number of slots refined)."""
    lo, hi = lo.clone(), hi.clone()
    exact = torch.zeros_like(lo, dtype=torch.bool)
    refined = 0
    while True:
        if bool(exact.all()):
            return commit_exact(hi, b), refined
        n_lo, n_hi = n_commit(lo, b), n_commit(hi, b)
        if n_lo == n_hi:
            order = torch.argsort(hi, stable=True)
            S, R = order[:n_lo], order[n_lo:]
            if R.numel() == 0 or hi[S].max() < lo[R].min():
                keep = torch.zeros_like(exact)
                keep[S] = True
                return keep, refined
            amb = torch.cat([S[hi[S] >= lo[R].min()], R[lo[R] <= hi[S].max()]])
        else:
            amb = torch.argsort(lo, stable=True)[:n_lo + 1]
        amb = amb[~exact[amb]]
        if amb.numel() == 0:
            amb = (~exact).nonzero().squeeze(-1)
        if amb.numel() > chunk:
            amb = amb[torch.argsort(hi[amb] - lo[amb], descending=True)[:chunk]]
        e = refine(amb)
        lo[amb], hi[amb] = e, e
        exact[amb] = True
        refined += amb.numel()


def certified_mean_below(lo, hi, refine, thr, chunk=32):
    """Is mean(true entropy) < thr ?  (strict '<', as in the stock pipeline). Returns (answer, slots refined)."""
    lo, hi = lo.clone(), hi.clone()
    exact = torch.zeros_like(lo, dtype=torch.bool)
    refined = 0
    while True:
        if hi.mean() < thr:
            return True, refined
        if lo.mean() >= thr:
            return False, refined
        cand = (~exact).nonzero().squeeze(-1)
        if cand.numel() == 0:
            return bool(hi.mean() < thr), refined
        pick = cand[torch.argsort(hi[cand] - lo[cand], descending=True)[:chunk]]
        e = refine(pick)
        lo[pick], hi[pick] = e, e
        exact[pick] = True
        refined += pick.numel()


def renorm_entropy(st):
    q = st['p'] / st['p'].sum(-1, keepdim=True)
    return -(q * torch.log(q.clamp_min(1e-30))).sum(-1)


# ============================================================================ self-conditioning
def soft_full(logits, tau, W, scale):
    """Exactly the stock decoder: softmax(logits/T) in fp32, cast to the weight dtype, matmul, times sqrt(d)."""
    probs = (logits.float() / tau).softmax(dim=-1).to(W.dtype)
    return torch.matmul(probs, W) * scale.to(W.dtype)


def soft_topk(st, W, scale, tail='drop', W_sum=None, row_chunk=128):
    """sum_k w_k * E[idx_k] * sqrt(d) over the top-k only.
    tail='drop'   : w = p (the missing tail mass is simply left out)
    tail='renorm' : w = p / sum(p)
    tail='mean'   : w = p, plus m * (mean embedding of the tail tokens)
    Error of 'drop' vs full is at most m * max_row_norm(E) * sqrt(d)."""
    B, C, k = st['p'].shape
    V = W.shape[0]
    idx = st['idx'].reshape(-1, k).to(W.device)
    w = st['p'].reshape(-1, k).to(W.device)
    if tail == 'renorm':
        w = w / w.sum(-1, keepdim=True)
    out = torch.empty(idx.shape[0], W.shape[1], device=W.device, dtype=torch.float32)
    for s in range(0, idx.shape[0], row_chunk):
        rows = W[idx[s:s + row_chunk]].float()                     # [r, k, d]
        out[s:s + row_chunk] = torch.einsum('rk,rkd->rd', w[s:s + row_chunk], rows)
        if tail == 'mean':
            tail_sum = W_sum.unsqueeze(0) - rows.sum(1)            # sum of the V-k tail embeddings
            m = st['m'].reshape(-1)[s:s + row_chunk].to(W.device).unsqueeze(-1)
            out[s:s + row_chunk] += m * tail_sum / (V - k)
    return (out * scale.float()).to(W.dtype).view(B, C, -1)


# ============================================================================ scheduler
def make_scheduler_class():
    from diffusers import EntropyBoundScheduler
    try:
        from diffusers.schedulers.scheduling_entropy_bound import EntropyBoundSchedulerOutput
    except ImportError:
        from diffusers import EntropyBoundSchedulerOutput

    class CertifiedEBScheduler(EntropyBoundScheduler):
        """mode='stock'     : the parent step, unchanged (optionally shadowed by certified decisions for checking)
           mode='certified' : certified commit, top-k sampling with an exact tail branch, no full pred_logits
           mode='naive'     : renormalized top-k entropy for the commit (ablation: not exact)"""

        def configure(self, rt, mode='certified', k=64, tail_sampling=True):
            self.rt, self.mode, self.k, self.tail_sampling = rt, mode, k, tail_sampling
            return self

        def temperature(self, timestep):
            # identical to the parent: anneal from t_max (step 0) to t_min
            frac = (self.num_inference_steps - int(timestep)) / self.num_inference_steps
            return self.config.t_min + (self.config.t_max - self.config.t_min) * frac

        def step(self, model_output, timestep, sample, *, entropy_bound=None, generator=None, return_dict=True):
            rt = self.rt
            b = float(self.config.entropy_bound) if entropy_bound is None else entropy_bound
            T = self.temperature(timestep)
            rt.last = dict(logits=model_output, tau=T, st=None)
            if self.mode == 'stock':
                out = super().step(model_output, timestep, sample, entropy_bound=entropy_bound,
                                   generator=generator, return_dict=True)
                if rt.shadow:
                    rt.shadow_commit(model_output, T, b, out.accepted_index)
                return out if return_dict else out.to_tuple()

            t0 = rt.tic()
            logits = model_output
            B, C, V = logits.shape
            g = generator if (generator is not None and generator.device == logits.device) else None
            st = topk_stats(logits, T, self.k)
            rt.last['st'] = st
            refined = 0
            if self.mode == 'certified':
                rows = []
                for i in range(B):
                    refine = lambda pos, i=i: stock_entropy(logits[i, pos], T)
                    kp, r = certified_commit_row(st['lo'][i], st['hi'][i], refine, b)
                    rows.append(kp)
                    refined += r
                keep = torch.stack(rows)
            elif self.mode == 'naive':
                ent = renorm_entropy(st)
                keep = torch.stack([commit_exact(ent[i], b) for i in range(B)])
            else:
                raise ValueError(self.mode)

            # top-k sample; with probability m draw from the tail instead -> exactly the full distribution
            pick = torch.multinomial(st['p'].reshape(-1, self.k), 1, generator=g).view(B, C)
            tok = st['idx'].gather(-1, pick.unsqueeze(-1)).squeeze(-1)
            n_tail = 0
            if self.tail_sampling and self.mode == 'certified':
                u = torch.rand(B, C, device=logits.device, generator=g)
                tail = u < st['m']
                n_tail = int(tail.sum())
                if n_tail:
                    bi, ci = tail.nonzero(as_tuple=True)
                    lt = logits[bi, ci].float() / T
                    lt.scatter_(-1, st['idx'][bi, ci], float('-inf'))
                    tok[bi, ci] = torch.multinomial(lt.softmax(-1), 1, generator=g).squeeze(-1)

            noise = torch.randint(0, V, (B, C), device=logits.device, generator=g)
            prev = torch.where(keep, tok, noise)
            sampled_probs = torch.exp(logits.gather(-1, tok.unsqueeze(-1)).squeeze(-1).float() / T - st['lse'])
            rt.add(refined_commit=refined, tail_samples=n_tail, slots=B * C, t_sampler=rt.toc(t0))
            out = EntropyBoundSchedulerOutput(prev_sample=prev, accepted_index=keep, sampled_tokens=tok,
                                              sampled_probs=sampled_probs, pred_logits=None)
            return out if return_dict else out.to_tuple()

    return EntropyBoundScheduler, CertifiedEBScheduler


_EMB_CACHE = {}


def _embedding_stats(W, chunk=16_384):
    """Sum of all embedding rows and the largest row norm, computed once per table, in chunks (no 3 GB fp32 copy)."""
    key = (W.data_ptr(), tuple(W.shape))
    if key not in _EMB_CACHE:
        s = torch.zeros(W.shape[1], device=W.device, dtype=torch.float32)
        mx = 0.0
        for i in range(0, W.shape[0], chunk):
            blk = W[i:i + chunk].float()
            s += blk.sum(0)
            mx = max(mx, float(blk.norm(dim=-1).max()))
        _EMB_CACHE[key] = (s, mx)
    return _EMB_CACHE[key]


# ============================================================================ runtime: self-conditioning + stop hooks
class Runtime:
    """Holds the settings and the counters, and implements the hooks the patched pipeline calls.
    sc   : 'stock' (pass pred_logits through, stock scheduler only) | 'full' | 'zero' | 'topk'
    stop : 'stock' (stock scheduler only) | 'certified' | 'naive'
    shadow=True (with the stock scheduler): also compute the certified decisions and the top-k self-cond,
          and record any disagreement with the stock ones. Nothing the pipeline uses is changed."""

    def __init__(self, decoder, sc='full', sc_tail='drop', stop='certified', k=64, shadow=False, timing=False):
        self.sc, self.sc_tail, self.stop, self.k, self.shadow, self.timing = sc, sc_tail, stop, k, shadow, timing
        self.W = decoder.embed_tokens.weight
        self.scale = decoder.embed_tokens.embed_scale
        self.W_sum, self.max_row_norm = _embedding_stats(self.W)
        self.pending = None
        self.last = {}
        self.reset_counters()
        mod = decoder.self_conditioning
        if not hasattr(mod, '_orig_forward'):
            mod._orig_forward = mod.forward
        rt = self

        def fwd(inputs_embeds, self_conditioning_signal):
            if rt.pending is not None:
                self_conditioning_signal = rt.pending.to(self_conditioning_signal.device,
                                                         self_conditioning_signal.dtype)
                rt.pending = None
            return mod._orig_forward(inputs_embeds, self_conditioning_signal)
        mod.forward = fwd

    # ---- counters
    def reset_counters(self):
        self.c = dict(steps=0, blocks=0, slots=0, refined_commit=0, refined_stop=0, tail_samples=0,
                      t_sampler=0.0, t_sc=0.0, t_stop=0.0,
                      sh_steps=0, sh_commit_mismatch_rows=0, sh_stop_mismatch=0, sh_stop_checks=0,
                      sh_argmax_mismatch=0, sh_refined_commit=0, sh_refined_stop=0, sh_slots=0,
                      sh_tail_mass=0.0, sh_sc_err_drop=0.0, sh_sc_err_renorm=0.0, sh_sc_err_mean=0.0,
                      sh_sc_bound_violations=0)
        self.block_steps = []

    def add(self, **kw):
        for k_, v in kw.items():
            self.c[k_] += v

    def tic(self):
        if self.timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def toc(self, t0):
        if self.timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    # ---- hooks called from the patched pipeline
    def reset_block(self):
        self.pending = None
        self.block_steps.append(0)
        self.c['blocks'] += 1

    def prepare_self_conditioning(self, scheduler_output, pred_logits):
        """Returns what the pipeline passes as self_conditioning_logits next step (None = we inject instead)."""
        self.block_steps[-1] += 1
        self.c['steps'] += 1
        t0 = self.tic()
        L, T, st = self.last['logits'], self.last['tau'], self.last.get('st')
        if self.shadow and pred_logits is not None:
            self.shadow_sc(L, T)
        if self.sc == 'stock':
            assert pred_logits is not None, "sc='stock' needs the stock scheduler"
            ret = pred_logits
        elif self.sc == 'zero':
            ret = None
        elif self.sc == 'full':
            self.pending = soft_full(L.to(self.W.device), T, self.W, self.scale)
            ret = None
        elif self.sc == 'topk':
            if st is None:
                st = topk_stats(L, T, self.k)
                self.last['st'] = st
            self.pending = soft_topk(st, self.W, self.scale, self.sc_tail, self.W_sum)
            ret = None
        else:
            raise ValueError(self.sc)
        self.c['t_sc'] += self.toc(t0)
        return ret

    def stop_argmax(self, scheduler_output, pred_logits):
        st = self.last.get('st')
        if self.stop == 'stock' or st is None:
            assert pred_logits is not None, "stop='stock' needs the stock scheduler"
            a = pred_logits.argmax(dim=-1)
            if self.shadow and st is not None:
                self.c['sh_argmax_mismatch'] += int((a != st['idx'][..., 0]).sum())
            return a
        return st['idx'][..., 0]                        # top-1 of the top-k is the argmax

    def stop_confident(self, scheduler_output, pred_logits, threshold, stable):
        t0 = self.tic()
        L, T, st = self.last['logits'], self.last['tau'], self.last.get('st')
        if self.stop == 'stock':
            conf = torch.distributions.Categorical(logits=pred_logits.float()).entropy().mean(-1) < threshold
            if self.shadow and st is not None:
                cert = self._certified_confident(L, T, st, threshold, torch.ones_like(conf), shadow=True)
                self.c['sh_stop_mismatch'] += int((cert != conf).sum())
                self.c['sh_stop_checks'] += conf.numel()
        elif self.stop == 'certified':
            conf = self._certified_confident(L, T, st, threshold, stable)
        elif self.stop == 'naive':
            conf = renorm_entropy(st).mean(-1) < threshold
        else:
            raise ValueError(self.stop)
        self.c['t_stop'] += self.toc(t0)
        return conf

    def _certified_confident(self, L, T, st, threshold, stable, shadow=False):
        out = torch.zeros(L.shape[0], dtype=torch.bool, device=L.device)
        for i in range(L.shape[0]):
            if not bool(stable[i]):                     # the pipeline ANDs with `stable`, so skip the work
                continue
            refine = lambda pos, i=i: stock_entropy(L[i, pos], T)
            ans, r = certified_mean_below(st['lo'][i], st['hi'][i], refine, threshold)
            out[i] = ans
            self.c['sh_refined_stop' if shadow else 'refined_stop'] += r
        return out

    # ---- shadow checks (stock scheduler drives; we only compare)
    def shadow_commit(self, logits, T, b, stock_keep):
        st = topk_stats(logits, T, self.k)
        self.last['st'] = st
        for i in range(logits.shape[0]):
            refine = lambda pos, i=i: stock_entropy(logits[i, pos], T)
            kp, r = certified_commit_row(st['lo'][i], st['hi'][i], refine, b)
            self.c['sh_commit_mismatch_rows'] += int((kp != stock_keep[i]).any())
            self.c['sh_refined_commit'] += r
        self.c['sh_steps'] += 1
        self.c['sh_slots'] += logits.shape[0] * logits.shape[1]
        self.c['sh_tail_mass'] += float(st['m'].mean())

    def shadow_sc(self, L, T):
        st = self.last.get('st')
        if st is None:
            return
        full = soft_full(L.to(self.W.device), T, self.W, self.scale).float()
        nf = full.norm(dim=-1).clamp_min(1e-6)
        for tail in ('drop', 'renorm', 'mean'):
            approx = soft_topk(st, self.W, self.scale, tail, self.W_sum).float()
            err = (approx - full).norm(dim=-1)
            self.c[f'sh_sc_err_{tail}'] += float((err / nf).mean())
            if tail == 'drop':
                bound = st['m'].to(err.device) * self.max_row_norm * float(self.scale) * 1.01 + 0.05
                self.c['sh_sc_bound_violations'] += int((err > bound).sum())

    def summary(self):
        c = self.c
        s = dict(steps=c['steps'], blocks=c['blocks'],
                 steps_per_block=c['steps'] / max(c['blocks'], 1))
        if c['slots']:
            s.update(refined_commit_pct=100 * c['refined_commit'] / c['slots'],
                     refined_stop_pct=100 * c['refined_stop'] / c['slots'],
                     tail_sample_pct=100 * c['tail_samples'] / c['slots'])
        if c['sh_steps']:
            n = c['sh_steps']
            s.update(sh_commit_mismatch_rows=c['sh_commit_mismatch_rows'],
                     sh_stop_mismatch=c['sh_stop_mismatch'], sh_stop_checks=c['sh_stop_checks'],
                     sh_argmax_mismatch=c['sh_argmax_mismatch'],
                     sh_refined_commit_pct=100 * c['sh_refined_commit'] / c['sh_slots'],
                     sh_refined_stop_pct=100 * c['sh_refined_stop'] / c['sh_slots'],
                     sh_tail_mass_mean=c['sh_tail_mass'] / n,
                     sh_sc_relerr_drop=c['sh_sc_err_drop'] / n,
                     sh_sc_relerr_renorm=c['sh_sc_err_renorm'] / n,
                     sh_sc_relerr_mean=c['sh_sc_err_mean'] / n,
                     sh_sc_bound_violations=c['sh_sc_bound_violations'])
        if self.timing:
            s.update(t_sampler_s=c['t_sampler'], t_sc_s=c['t_sc'], t_stop_s=c['t_stop'])
        return s


# ============================================================================ patched pipeline
# Four edits to the installed stock __call__. Each anchor must match exactly once, or we stop with an error.
EDITS = [
    ("block start",
     r"(?P<i>[ \t]*)self_conditioning_logits = None\n",
     r"\g<i>self_conditioning_logits = None\n\g<i>self._rt.reset_block()\n"),
    ("self-conditioning hand-off",
     r"(?P<i>[ \t]*)pred_logits = scheduler_output\.pred_logits\n[ \t]*self_conditioning_logits = pred_logits\n",
     r"\g<i>pred_logits = scheduler_output.pred_logits\n"
     r"\g<i>self_conditioning_logits = self._rt.prepare_self_conditioning(scheduler_output, pred_logits)\n"),
    ("stop: argmax",
     r"next_argmax_canvas = pred_logits\.argmax\(dim=-1\)",
     r"next_argmax_canvas = self._rt.stop_argmax(scheduler_output, pred_logits)"),
    ("stop: mean entropy",
     r"confident = torch\.distributions\.Categorical\(logits=pred_logits\.float\(\)\)\.entropy\(\)\.mean\(-1\) < \(\s*confidence_threshold\s*\)",
     r"confident = self._rt.stop_confident(scheduler_output, pred_logits, confidence_threshold, stable)"),
]


def patch_call_source(src):
    """Apply EDITS to the source text of DiffusionGemmaPipeline.__call__. Returns the patched source."""
    src = textwrap.dedent(src)
    lines = src.split('\n')
    while lines and lines[0].lstrip().startswith('@'):      # drop @torch.no_grad / @replace_example_docstring
        lines.pop(0)
    src = '\n'.join(lines)
    for name, pat, rep in EDITS:
        src, n = re.subn(pat, rep, src)
        if n != 1:
            raise RuntimeError(f"patch '{name}': anchor matched {n} times (expected 1). "
                               f"Your diffusers version differs; print inspect.getsource(DiffusionGemmaPipeline.__call__).")
    return src


def make_patched_pipeline_class(save_source_to=None):
    import diffusers.pipelines.diffusion_gemma.pipeline_diffusion_gemma as mod
    from diffusers import DiffusionGemmaPipeline
    src = patch_call_source(inspect.getsource(DiffusionGemmaPipeline.__call__))
    if save_source_to:
        open(save_source_to, 'w').write(src)
    ns = dict(mod.__dict__)
    exec(compile(src, '<patched DiffusionGemmaPipeline.__call__>', 'exec'), ns)
    patched_call = torch.no_grad()(ns['__call__'])

    class CertifiedDiffusionGemmaPipeline(DiffusionGemmaPipeline):
        __call__ = patched_call

        def attach(self, rt):
            self._rt = rt
            return self

    return CertifiedDiffusionGemmaPipeline


# ============================================================================ eval helpers
def last_number(s):
    nums = re.findall(r'-?\d[\d,]*\.?\d*', s or '')
    return nums[-1].replace(',', '').rstrip('.') if nums else None


def gsm8k_gold(ans):
    return ans.split('####')[-1].strip().replace(',', '')


def same_number(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False
