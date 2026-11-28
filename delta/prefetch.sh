#!/bin/bash
# Download everything the jobs read from the internet, on the LOGIN node, so the GPU jobs run fully offline:
#   DiffusionGemma (07, 08, 09, 10), Gemma 4 AR + its MTP draft model (09), GSM8K (07, 09, 10).
#   nohup bash delta/prefetch.sh > delta/logs/prefetch.log 2>&1 &
# Safe to rerun: finished files are skipped.
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$REPO_DIR"

"$VENV_KS/bin/python" - <<'EOF'
import os, time
from huggingface_hub import snapshot_download
models = [os.environ['MODEL_ID']]
if os.environ.get('NB09_RUN_AR', '1') == '1':
    models += [os.environ['AR_ID'], os.environ['AR_DRAFT_ID']]
for m in models:
    t = time.time()
    p = snapshot_download(m, allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', 'tokenizer*'])
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs) / 2**30
    print(f'{m}: {size:.1f} GB in {time.time() - t:.0f}s -> {p}', flush=True)
EOF

# GSM8K into the datasets cache of each environment (their datasets versions may differ).
for py in "$VENV_KS/bin/python" "$VENV_VLLM/bin/python"; do
    "$py" -c "from datasets import load_dataset; d = load_dataset('openai/gsm8k', 'main', split='test'); print('gsm8k test:', len(d), 'rows ($py)')"
done
echo "PREFETCH DONE $(date)"
