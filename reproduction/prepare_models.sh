#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT/examples/models"
mkdir -p "$MODEL_DIR"

link_model() {
  local name="$1" path="$2"
  test -d "$path" || { echo "missing model directory: $path" >&2; return 1; }
  # A partially interrupted Hugging Face download can leave a directory and
  # config.json behind while one or more safetensors shards are absent.  Do
  # not create a seemingly valid symlink in that state: vLLM would fail much
  # later during worker startup and the reproduction record would be unclear.
  MODEL_TO_VALIDATE="$path" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["MODEL_TO_VALIDATE"])
index = root / "model.safetensors.index.json"
if not (root / "config.json").is_file():
    raise SystemExit(f"missing config.json: {root}")
if index.is_file():
    data = json.loads(index.read_text())
    shards = sorted(set(data.get("weight_map", {}).values()))
    missing = [name for name in shards if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"incomplete safetensors checkout {root}: {missing}")
    partial = sorted(root.glob("*.safetensors.part"))
    if partial:
        raise SystemExit(f"partial safetensors files remain in {root}: {partial}")
PY
  ln -sfn "$path" "$MODEL_DIR/$name"
  echo "MODEL_LINK $name -> $(readlink -f "$MODEL_DIR/$name")"
}

PYTHON_BIN="${MAPFLOW_PYTHON:-/home/wby/.venvs/mapflow-repro312/bin/python}"

link_model Qwen3-8B /home/wby/models/Qwen3-8B
link_model Qwen3-14B /home/wby/models/Qwen3-14B
link_model Ministral3-3B /home/wby/models/Ministral-3-3B-Reasoning-2512
link_model Ministral3-8B /home/wby/models/Ministral-3-8B-Reasoning-2512
link_model Ministral3-14B /home/wby/models/Ministral-3-14B-Reasoning-2512

if ! MODEL_TO_VALIDATE="/home/wby/models/Qwen3-1.7B" "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import json
from pathlib import Path

root = Path(__import__("os").environ["MODEL_TO_VALIDATE"])
index = root / "model.safetensors.index.json"
if not root.is_dir() or not (root / "config.json").is_file():
    raise SystemExit(1)
if index.is_file():
    files = set(json.loads(index.read_text()).get("weight_map", {}).values())
    if any(not (root / name).is_file() for name in files):
        raise SystemExit(1)
    if any(root.glob("*.safetensors.part")):
        raise SystemExit(1)
PY
then
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
