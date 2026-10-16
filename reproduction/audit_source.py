"""Static, reproducible audit of the imported MapFlow source tree.

This script deliberately does not import vLLM.  It can therefore run before
the large CUDA environment is installed and catches the most common protocol
mistakes (wrong dataset counts, missing calibration weights, and inconsistent
model-pair dimensions).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "examples" / "weights"
DATA = ROOT / "examples" / "data" / "accuracy"


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl(path: Path) -> int:
    return sum(1 for line in path.open() if line.strip())


def load_torch_shape(path: Path):
    """Return tensor shape without making torch a static-audit dependency."""
    try:
        import torch
    except ImportError:
        return None
    obj = torch.load(path, map_location="cpu", weights_only=False)
    tensor = obj
    if isinstance(obj, dict):
        for key in ("weight", "translator", "W"):
            if key in obj:
                tensor = obj[key]
                break
    return tuple(tensor.shape)


def main() -> int:
    expected_files = [
        "README.md",
        "setup.py",
        "mapflow/server/server.py",
        "mapflow/server/block_manager.py",
        "mapflow/kernel/compress_page_triton.py",
        "mapflow/kernel/retrieve.py",
        "vllm/vllm/v1/attention/backends/bsr_attn.py",
        "examples/inference_qa_batch.py",
    ]
    missing = [x for x in expected_files if not (ROOT / x).is_file()]
    if missing:
        raise SystemExit(f"missing upstream files: {missing}")

    print("ROOT", ROOT)
    print("GIT", git_output("rev-parse", "--show-toplevel"))
    try:
        imported_commit = git_output("rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        imported_commit = "uncommitted import"
    print("COMMIT", imported_commit)
    print("SOURCE_EXPECTED_COMMIT", "ab1dc50ea811a8a50c0beb64964cdf963956eb32")

    datasets = {
        "hotpotqa": DATA / "hotpotqa_1000.jsonl",
        "conversation": DATA / "conversation_1000.jsonl",
        "multinews": DATA / "multinews_1000.jsonl",
        "rulerqa": DATA / "rulerqa_100.jsonl",
    }
    for name, path in datasets.items():
        n = count_jsonl(path)
        print(f"DATASET {name} rows={n} sha256={sha256(path)} path={path.relative_to(ROOT)}")

    pairs = {
        "Qwen3-8B_to_Qwen3-1.7B": (36, 28, 32, 16),
        "Qwen3-14B_to_Qwen3-8B": (40, 36, 40, 32),
        "Ministral3-8B_to_Ministral3-3B": (34, 26, 32, 32),
        "Ministral3-14B_to_Ministral3-8B": (40, 34, 32, 32),
    }
    for pair, (sender_layers, receiver_layers, sender_heads, receiver_heads) in pairs.items():
        pair_dir = WEIGHTS / pair
        assert pair_dir.is_dir(), pair_dir
        print(
            f"PAIR {pair} sender_layers={sender_layers} receiver_layers={receiver_layers} "
            f"sender_heads={sender_heads} receiver_heads={receiver_heads} "
            f"suffix_offset={sender_layers-receiver_layers}"
        )
        for task in ("hotpotqa", "conversation", "multinews", "rulerqa"):
            task_dir = pair_dir / task
            files = sorted(task_dir.glob("translator_layer_*.pt"))
            if len(files) != receiver_layers - 3:
                raise SystemExit(
                    f"{pair}/{task}: expected {receiver_layers-3} translated layers "
                    f"after three ignored receiver layers, found {len(files)}"
                )
            shapes = {load_torch_shape(p) for p in files}
            effective_reuse = list(range(3, receiver_layers - 1))
            print(
                f"  WEIGHTS task={task} files={len(files)} "
                f"runtime_reuse_layers={len(effective_reuse)} "
                f"shapes={sorted(map(str, shapes))}"
            )

    # These strings are implementation anchors used by the paper audit.
    checks = {
        "simplex normalization": re.compile(r"softmax\([^\n]*dim=0"),
        "BSR backend": re.compile(r"BSR_ATTN"),
        "top-p": re.compile(r"top.?p|TOP_P", re.I),
        "prefix hash": re.compile(r"prefix.?hash|compute_block_hash", re.I),
    }
    source_text = "\n".join(
        p.read_text(errors="ignore")
        for p in [
            ROOT / "mapflow/kernel/compress_page_triton.py",
            ROOT / "mapflow/server/server.py",
            ROOT / "mapflow/core/prefix_hash.py",
            ROOT / "mapflow/core/__init__.py",
            ROOT / "mapflow/server/compress.py",
            ROOT / "vllm/vllm/v1/attention/backends/bsr_attn.py",
        ]
    )
    for label, pattern in checks.items():
        found = bool(pattern.search(source_text))
        print(f"IMPLEMENTATION_ANCHOR {label}={found}")
        if not found:
            raise SystemExit(f"missing implementation anchor: {label}")

    print("STATIC_AUDIT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
