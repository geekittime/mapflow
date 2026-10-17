# MapFlow reproduction harness

Current H20 accuracy work: [`h20_accuracy/PROCESS.md`](h20_accuracy/PROCESS.md).
The commands and RTX 4090 limitations below describe the earlier host, not the current H20 runs.

This directory contains the audit and launcher files used for the MapFlow
reproduction.  The implementation under `../mapflow/`, the modified vLLM
under `../vllm/`, the paper-provided calibration weights under
`../examples/weights/`, and the paper-format evaluation data under
`../examples/data/` are retained from the public MapFlow source commit.

The main experiment log is [`MAPFLOW_REPRODUCTION_LOG.md`](../MAPFLOW_REPRODUCTION_LOG.md).

## Quick checks

```bash
cd /home/wby/mapflow
python reproduction/audit_source.py
python reproduction/verify_sparse_math.py
python reproduction/hardware_audit.py
/home/wby/.venvs/mapflow-repro312/bin/python reproduction/audit_models.py
```

The checks above do not claim that a GPU run succeeded.  They only verify the
source provenance, dataset/weight inventory, layer/head configuration, and the
mathematical invariants of block top-p compression.  GPU accuracy and latency
commands are recorded in the main log and must be run in the separate
`mapflow-repro312` environment.

`audit_models.py` is the stricter model-artifact gate: it checks the nested
text configuration, every safetensors shard referenced by the index, and
opens each shard before a model is linked into `examples/models/`.

The public implementation uses CUDA IPC for sender/cache/receiver buffers.  A
multi-GPU run therefore requires peer access between the participating GPUs.
`hardware_audit.py` records this prerequisite explicitly.  On the current
RTX 4090 host the peer matrix is all false, so an end-to-end MapFlow result
cannot be called a faithful run on this host; use an NVLink/P2P-capable host
(the paper used A800s), or implement and separately label a host-staging
communication backend.

For a 24-GB development GPU, the copied environment accepts
`MAPFLOW_IPC_BUFFER_SIZE` to lower the sender's default 3,096,000-block IPC
buffer.  The default is unchanged; lowering it is only a memory-footprint
workaround and does not remove the CUDA-IPC/P2P prerequisite.

## Model paths

The upstream shell scripts intentionally use `examples/models/`, which is
ignored by git.  Create links to local model snapshots (never copy or commit
model weights):

```bash
mkdir -p examples/models
ln -sfn /home/wby/models/Qwen3-8B examples/models/Qwen3-8B
ln -sfn /home/wby/models/Qwen3-14B examples/models/Qwen3-14B
ln -sfn /home/wby/models/Ministral-3-3B-Reasoning-2512 examples/models/Ministral3-3B
ln -sfn /home/wby/models/Ministral-3-8B-Reasoning-2512 examples/models/Ministral3-8B
ln -sfn /home/wby/models/Ministral-3-14B-Reasoning-2512 examples/models/Ministral3-14B
```

Qwen3-1.7B is required for the Qwen3-8B → Qwen3-1.7B paper pair.  Run
`reproduction/prepare_models.sh` with the copied environment's Python; it
validates every shard referenced by `model.safetensors.index.json` and refuses
to link a directory containing missing or `.part` files.  Its acquisition and
checksums are recorded in the main log before that pair is run.
