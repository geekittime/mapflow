#!/usr/bin/env python3
"""Print hardware/runtime facts that determine whether MapFlow can run.

MapFlow's public implementation passes CUDA IPC tensors between sender,
cache-server and receiver processes.  CUDA IPC across distinct GPUs requires
peer access on this topology; a False matrix is therefore a hard execution
limitation, not an accuracy result.
"""

from __future__ import annotations

import os


def main() -> int:
    import torch

    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available(), "count", torch.cuda.device_count())
    if not torch.cuda.is_available():
        return 0
    for i in range(torch.cuda.device_count()):
        print("gpu", i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
    print("peer_access_matrix")
    for i in range(torch.cuda.device_count()):
        row = [int(torch.cuda.can_device_access_peer(i, j)) for j in range(torch.cuda.device_count())]
        print(i, row)
    print("visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
