#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT/examples/models"
mkdir -p "$MODEL_DIR"

link_model() {
  local name="$1" path="$2"
  test -d "$path" || { echo "missing model directory: $path" >&2; return 1; }
  ln -sfn "$path" "$MODEL_DIR/$name"
  echo "MODEL_LINK $name -> $(readlink -f "$MODEL_DIR/$name")"
}

link_model Qwen3-8B /home/wby/models/Qwen3-8B
link_model Qwen3-14B /home/wby/models/Qwen3-14B
link_model Ministral3-3B /home/wby/models/Ministral-3-3B-Reasoning-2512
link_model Ministral3-8B /home/wby/models/Ministral-3-8B-Reasoning-2512
link_model Ministral3-14B /home/wby/models/Ministral-3-14B-Reasoning-2512

if [ ! -d "$MODEL_DIR/Qwen3-1.7B" ]; then
  PYTHON_BIN="${MAPFLOW_PYTHON:-/home/wby/.venvs/mapflow-repro312/bin/python}"
  "$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="Qwen/Qwen3-1.7B",
    local_dir="/home/wby/models/Qwen3-1.7B",
)
print("DOWNLOADED_QWEN3_1.7B", path)
PY
fi
link_model Qwen3-1.7B /home/wby/models/Qwen3-1.7B

for model in "$MODEL_DIR"/*; do
  [ -e "$model" ] || continue
  printf 'MODEL_CONFIG %s ' "$(basename "$model")"
  sha256sum "$(readlink -f "$model")/config.json" | cut -d' ' -f1
done
