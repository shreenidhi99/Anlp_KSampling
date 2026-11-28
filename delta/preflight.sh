#!/bin/bash
# Login-node checks: everything the jobs need, without using a GPU. Must pass before delta/submit_all.sh.
#   bash delta/preflight.sh
set -uo pipefail
source "$(dirname "$0")/config.sh"
cd "$REPO_DIR"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
fails=0
ok()   { printf '  [ OK ] %s\n' "$1"; }
bad()  { printf '  [FAIL] %s\n' "$1"; fails=$((fails + 1)); }
warn() { printf '  [WARN] %s\n' "$1"; }
check() { local msg="$1"; shift; if out=$("$@" 2>&1); then ok "$msg"; else bad "$msg"; echo "$out" | tail -5 | sed 's/^/         /'; fi; }

echo "== Slurm"
if [[ "$DELTA_ACCOUNT" == FILL_IN* ]]; then bad "DELTA_ACCOUNT not set in delta/config.sh (run: accounts)"
else ok "account $DELTA_ACCOUNT"; fi
check "partition $DELTA_PARTITION exists" bash -c "sinfo -h -p '$DELTA_PARTITION' | grep -q ."

echo "== files"
n=$(ls runs/nb04/step_*.pt 2>/dev/null | wc -l)
[ "$n" -eq 6 ] && ok "saved logits: 6 files in runs/nb04" || bad "saved logits: $n of 6 (run delta/upload_logits.sh on your laptop)"
[ -f "$HF_HOME/token" ] && ok "Hugging Face token present" || bad "no Hugging Face token (run: $VENV_KS/bin/hf auth login)"
check "kernels anlp-ksampling and anlp-vllm registered" bash -c \
    "'$VENV_KS/bin/jupyter' kernelspec list | grep -q anlp-ksampling && '$VENV_KS/bin/jupyter' kernelspec list | grep -q anlp-vllm"

echo "== ksampling env (07, 08)"
check "torch, triton, transformers, diffusers, datasets import" "$VENV_KS/bin/python" -c "
import torch, triton, transformers, diffusers, datasets, nbclient, pandas, matplotlib
from transformers import DiffusionGemmaForBlockDiffusion
from diffusers import DiffusionGemmaPipeline
import hint_bench, dg_certified
print(torch.__version__, triton.__version__, transformers.__version__, diffusers.__version__)"

echo "== vLLM env (09a, 09, 10)"
# No `import vllm` here: on a GPU-less login node it hangs probing for devices. The GPU check job imports it.
check "vllm 0.31.0 with the DiffusionGemma sampler (installed files)" "$VENV_VLLM/bin/python" -c "
import importlib.metadata as md, importlib.util, os
v = md.version('vllm'); assert v.startswith('0.31.0'), v
root = os.path.dirname(importlib.util.find_spec('vllm').origin)
src = open(os.path.join(root, 'model_executor', 'models', 'diffusion_gemma_sampler.py')).read()
assert 'def sample_row_stats' in src, 'no sample_row_stats in the DiffusionGemma sampler'"
check "dg_topk plugin registered with vLLM" "$VENV_VLLM/bin/python" -c "
import importlib.metadata as md
assert any(e.name == 'dg_topk' for e in md.entry_points(group='vllm.general_plugins'))"
check "openai, datasets, nbclient import" "$VENV_VLLM/bin/python" -c "import openai, datasets, nbclient, pandas, matplotlib"

echo "== downloads (offline)"
models="$MODEL_ID"; [ "$NB09_RUN_AR" = 1 ] && models="$models $AR_ID $AR_DRAFT_ID"
for m in $models; do
    check "model cached: $m" "$VENV_KS/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('$m', allow_patterns=['*.json', '*.safetensors'], local_files_only=True)"
done
for py in "$VENV_KS/bin/python" "$VENV_VLLM/bin/python"; do
    check "GSM8K cached for $(basename "$(dirname "$(dirname "$py")")")" "$py" -c \
        "from datasets import load_dataset; assert len(load_dataset('openai/gsm8k', 'main', split='test')) == 1319"
done

echo "== storage"
df -h "$BIG_DIR" | tail -1 | awk '{print "  free on large storage: " $4}'
command -v quota >/dev/null && quota 2>/dev/null | tail -n +1 | sed 's/^/  /' | head -15 || true

echo
if [ "$fails" -eq 0 ]; then
    date > delta/logs/PREFLIGHT_PASSED
    echo "PREFLIGHT PASSED. Next: bash delta/submit.sh check   (15-minute GPU check)"
else
    rm -f delta/logs/PREFLIGHT_PASSED
    echo "PREFLIGHT FAILED: $fails problem(s) above. Nothing was submitted."; exit 1
fi
