#!/usr/bin/env bash
export MAPFLOW_ROOT=/home/tiger/mapflow
export PATH="$MAPFLOW_ROOT/.venv/bin:$PATH"
export TMPDIR="$MAPFLOW_ROOT/.tmp"
export XDG_CACHE_HOME="$MAPFLOW_ROOT/.cache"
export HF_HOME="$MAPFLOW_ROOT/.cache/huggingface"
export UV_CACHE_DIR=/home/tiger/.cache/uv
export TRITON_CACHE_DIR="$MAPFLOW_ROOT/.cache/triton"
export VLLM_CACHE_ROOT="$MAPFLOW_ROOT/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$MAPFLOW_ROOT/.cache/inductor"
export CUDA_MPS_PIPE_DIRECTORY="$MAPFLOW_ROOT/.tmp/mps-pipe"
export CUDA_MPS_LOG_DIRECTORY="$MAPFLOW_ROOT/.tmp/mps-log"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONPATH="$MAPFLOW_ROOT"
mkdir -p "$TMPDIR" "$HF_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
