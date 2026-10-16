#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${MAPFLOW_PYTHON:-/home/wby/.venvs/mapflow-repro312/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "DATE $(date -u +%FT%TZ)"
echo "ROOT $ROOT"
echo "PYTHON $($PYTHON_BIN --version)"
echo "DRIVER"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv,noheader
echo "NVCC"
nvcc --version 2>&1 || true
PYTHONPATH="$ROOT" "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import torch
print("TORCH", torch.__version__, "CUDA", torch.version.cuda, "GPU", torch.cuda.is_available(), torch.cuda.device_count())
for name in ("transformers", "vllm", "triton", "pyzmq", "sacrebleu", "rouge-score", "nvtx"):
    try: print("PACKAGE", name, md.version(name))
    except md.PackageNotFoundError: print("PACKAGE", name, "MISSING")
PY
