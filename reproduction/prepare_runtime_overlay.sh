#!/usr/bin/env bash
set -euo pipefail

# The public MapFlow checkout vendors modified Python vLLM files but does not
# vendor compiled vLLM extensions.  This script overlays only the MapFlow
# integration files into the separately installed vLLM runtime.  The source
# checkout remains the auditable artifact; the copied files are ephemeral
# environment state and are never committed.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MAPFLOW_PYTHON:-/home/wby/.venvs/mapflow-repro312/bin/python}"
SITE="$($PYTHON_BIN -c 'import site; print(site.getsitepackages()[0])')"
test -d "$SITE/vllm" || { echo "vLLM is not installed in $SITE" >&2; exit 1; }

for rel in \
  vllm/envs.py \
  vllm/transformers_utils/configs/mistral.py \
  vllm/v1/attention/selector.py \
  vllm/v1/attention/backends/registry.py \
  vllm/v1/attention/backends/bsr_attn.py \
  vllm/v1/attention/backends/flash_attn.py \
  vllm/v1/attention/backends/flex_prefill_kernel.py \
  vllm/v1/core/kv_cache_manager.py \
  vllm/v1/core/sched/scheduler.py \
  vllm/model_executor/layers/attention/attention.py \
  vllm/v1/worker/gpu_model_runner.py; do
  # The upstream checkout is a source distribution whose Python package is
  # rooted at vllm/vllm, while the installed wheel is rooted at site/vllm.
  src="$ROOT/vllm/$rel"
  dst="$SITE/$rel"
  test -f "$src" || { echo "missing source overlay: $src" >&2; exit 1; }
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  echo "OVERLAY $rel"
done

echo "RUNTIME_SITE $SITE"
PYTHONPATH="$ROOT" "$PYTHON_BIN" - <<'PY'
import torch, vllm
print("VLLM_RUNTIME", vllm.__version__, vllm.__file__)
print("TORCH_RUNTIME", torch.__version__, torch.version.cuda)
from vllm.platforms import current_platform
print("PLATFORM", type(current_platform).__name__, current_platform.is_cuda())
from vllm.v1.attention.backends.bsr_attn import BSRAttnBackend
print("BSR_BACKEND_CLASS", BSRAttnBackend.get_name())
try:
    import triton
    print("TRITON_RUNTIME", triton.__version__)
except Exception as exc:
    print("TRITON_RUNTIME_ERROR", type(exc).__name__, str(exc))
PY
