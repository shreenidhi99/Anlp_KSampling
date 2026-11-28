#!/bin/bash
# 15-minute GPU check of both environments on a real Delta GPU. Must pass before delta/submit_all.sh.
#SBATCH --job-name=check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=delta/logs/%x-%j.out
source delta/common.sh
rm -f delta/logs/CHECK_PASSED

echo "== ksampling env: CUDA, a Triton kernel, the embedding table, the saved logits"
"$VENV_KS/bin/python" - <<'EOF'
import glob, os, torch, triton, triton.language as tl
print('torch', torch.__version__, 'CUDA', torch.version.cuda, '| triton', triton.__version__)
assert torch.cuda.is_available(), 'torch cannot see the GPU: driver too old for this build? (TORCH_INDEX in delta/config.sh)'
print(torch.cuda.get_device_name(0), '| free GB', round(torch.cuda.mem_get_info()[0] / 2**30, 1))

@triton.jit
def add(x, y, o, n, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B); m = i < n
    tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)
x = torch.randn(10_000, device='cuda'); y = torch.randn_like(x); z = torch.empty_like(x)
add[(triton.cdiv(x.numel(), 1024),)](x, y, z, x.numel(), B=1024)
assert torch.allclose(z, x + y); print('triton kernel ok')

import hint_bench as hb
E, key = hb.load_embedding_table(os.environ['MODEL_ID'], 'cuda')
print('embedding table', key, tuple(E.shape))
saved = hb.load_saved_logits('./runs/nb04/step_*.pt')
assert len(saved) == 6 and saved[0][0].shape[-1] == E.shape[0], 'saved logits missing or wrong vocabulary'
print('saved logits', [tuple(L.shape) for L, _ in saved][:2], '...')
EOF

echo "== vLLM env: CUDA, vLLM sampler kernels, the plugin"
"$VENV_VLLM/bin/python" - <<'EOF'
import os, torch
os.environ['DG_TOPK_SC_K'] = '64'
print('torch', torch.__version__, 'CUDA', torch.version.cuda)
assert torch.cuda.is_available(), 'vLLM env torch cannot see the GPU'
import vllm, vllm.model_executor.models.diffusion_gemma as dg
from vllm.plugins import load_general_plugins
load_general_plugins()
assert getattr(dg, '_dg_topk_patched', False), 'dg_topk plugin did not patch vLLM'
print('vllm', vllm.__version__, '| plugin patched vLLM')
x = torch.randn(4, 1024, device='cuda').softmax(-1); print('cuda op ok', float(x.sum()))
EOF

echo "== ncu (notebook 10 Part A; optional)"
(module load cuda >/dev/null 2>&1 || true; command -v ncu && ncu --version | tail -1) || echo "ncu not found: Part A will be skipped, Part B still runs"

git rev-parse HEAD > delta/logs/CHECK_PASSED
echo "GPU CHECK PASSED. Next: bash delta/submit_all.sh"
