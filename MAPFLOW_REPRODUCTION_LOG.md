# MapFlow reproduction and audit log

Status: **audited — static reproduction is complete; faithful multi-GPU accuracy run is blocked by host hardware**

This file is the single chronological record for the reproduction requested in
the issue.  It distinguishes paper claims, source facts, and measurements made
on this machine.  A result is not called a reproduction result unless the
command, model pair, data split, sampling parameters, and log path are all
recorded.

## 1. Scope and source provenance

The target is the paper *Accelerating Cross-Model LLM Inference via Attention
Map Reuse* (MapFlow), including its method, calibration weights, accuracy
datasets, long-context RULER evaluation, TTFT experiment, and multi-turn
routing experiment.  The local paper file is:

```text
/home/wby/paper/MapFlow___Arxiv.pdf
sha256 4ace8f9de9380f071269210c89afead152ca43e96283818562abe8bec95d93a0
```

The requested anonymous endpoint
`https://anonymous.4open.science/r/MapFlow-6341` currently returns a redirect
to an API endpoint with HTTP 401 and `{"error":"not_connected"}`.  It was
therefore not silently substituted by an unverified fork.  The public authors'
mirror used for the source import is `PKU-DAIR/MapFlow`, commit
`ab1dc50ea811a8a50c0beb64964cdf963956eb32` (the commit adds `Appendix.pdf`).
The imported source tree is retained under `mapflow/`, `vllm/`, `examples/`,
and `blob/`.  Three runtime compatibility edits are explicitly tracked: the
default-preserving `MAPFLOW_IPC_BUFFER_SIZE` knob in `mapflow/__init__.py`, a
Transformers dtype compatibility guard in the vendored Mistral config, and
the missing evaluator/import repairs described below.  The MapFlow attention,
compression, retrieval, server, and vLLM integration algorithms were not
edited.  The upstream README hash at import time was
`79ec45f4c05c8fb962c7cfa7eb0705bab5c4c013e1110ebb8bbc349cbc06b2bc`.

## 2. Paper protocol extracted before running code

The following facts were transcribed from the local PDF, rather than inferred
from the output of a run:

| Component | Paper protocol | Source/experiment check |
|---|---|---|
| Model families | Qwen3 (1.7B, 8B, 14B), Ministral3 (3B, 8B, 14B) | Exact four pairs below are present in the upstream scripts and weights |
| Paper pairs | Qwen3-8B→1.7B; Qwen3-14B→8B; Ministral3-8B→3B; Ministral3-14B→8B | Checked against sender/receiver shell-case tables and weight directories |
| Recompute | Receiver recomputes the first `r` layers; late receiver layers reuse sender maps, with suffix alignment `m(j)=j+L_S-L_R` | Checked in BSR backend and launch configuration; `num_ignored_layers` is the sender-side/recompute setting |
| Head mapping | Sender attention heads are translated to receiver heads with learned `W` of shape `[H_S,H_R]` | Calibration files and fused retrieval path are checked; weights are simplex-normalized at use time |
| Training | Freeze both LLMs; fit only head mapping on shared context; sender makes the map, receiver reuses it; append and answer are then run on receiver; loss is masked to answer tokens | Verified against source calibration/inference data flow; no LLM fine-tuning is performed |
| Compression | Block-level log-sum-exp statistics, smallest descending block set reaching top-p, exact sparse extraction into BSR | Reference invariants are tested by `reproduction/verify_sparse_math.py`; Triton code is retained unchanged |
| Default short-context top-p | `p=0.7` | README launch ports and sender/server scripts use `0.7` |
| Long context | RULER 20K; p = 0.92, 0.87, 0.85, 0.87 for the four pairs in order | README and paper Table 2 checked; local RULER path is environment-specific and recorded below |
| Accuracy | 1,000 test examples; HotpotQA F1, conversation/Multi-News ROUGE-L | Official JSONL files and evaluator scripts retained |
| Baselines | Full prefill, CacheBlend-style cross-model KV reuse, FlexPrefill, MInference, IAM | Upstream scripts expose baseline/flexprefill/minference; baseline comparison is recorded separately from MapFlow |
| Hardware | Paper: 8× NVIDIA A800 80GB | This reproduction: 8× RTX 4090 24GB; this is a hardware deviation and is not hidden |
| Runtime | NVIDIA MPS, CUDA IPC, GPU block cache, modified vLLM + Triton BSR backend | MPS and CUDA IPC are checked before the first server run |

The PDF reports approximately 10,000 examples for train/validation and 1,000
for test.  The bundled public test files are authoritative for this run; their
actual line counts are printed by the static audit because `conversation` and
`multinews` are nested/turn-expanded files rather than one trivial line per
trajectory.

## 3. Static source and artifact audit

Run from the repository root:

```bash
python reproduction/audit_source.py
python reproduction/verify_sparse_math.py
```

`audit_source.py` checks required MapFlow/vLLM files, all four model-pair
weight directories, the exact bundled dataset hashes and counts, expected
translated-layer counts, and source anchors for BSR attention, top-p, simplex
weights, and prefix hashing.  `verify_sparse_math.py` checks that retained
blocks always reach the requested probability mass and that the minimal
descending selection invariant holds on a deterministic random matrix.  These
are static/reference checks and must not be confused with a GPU accuracy
measurement.

### Static audit output

The first run found that the upstream directory also contains an extra
`Ministral3-14B_to_Ministral3-3B` directory, but that is not one of the four
paper pairs.  The audit was corrected to the paper's
`Ministral3-8B_to_Ministral3-3B` pair before any experiment was launched.
This is deliberately recorded because silently testing the extra pair would
not be a faithful paper reproduction.

The public checkout also referenced `examples/inference_conversation_window.py`
and `examples/inference_conversation_batch.py` from its shell scripts without
shipping those files.  They are now supplied under `examples/` as a minimal
compatibility implementation matching the included QA evaluator's sender
prefill → receiver continuation protocol and the paper's ROUGE-L metric.  This
is a packaging repair, not a change to MapFlow's cache or kernel algorithm,
and is marked as a source deviation for independent review.

Likewise, `mapflow.client.receiver` had an unconditional import of
`mapflow.kernel.flex_prefill_kernel`, while the implementation existed only in
the vendored vLLM tree and the BSR path's call sites were commented out.  The
small re-export shim at `mapflow/kernel/flex_prefill_kernel.py` makes the
upstream package importable; it is not executed by the BSR MapFlow path.

The latest output is appended below after the source is committed.

Static audit run (16 October 2026, before CUDA packages were installed):

```text
DATASET hotpotqa rows=1000 sha256=2da0655e0177a4a1622fdee481e7a972b4c93f70be3eefd40ea5a905075245c3
DATASET conversation rows=100 sha256=289ed137b0d0af664d6f58ef124763317f339ceb90de47cca99452a9539e7b4d
DATASET multinews rows=5622 sha256=7338c63f8eaa875b2cfb2e38d1c0fb9e0202fd5f5cb765cab05f68b8aadcfd54
DATASET rulerqa rows=100 sha256=efcabced9b2a999bc9ab868f91210a37d6ea571909ccfe27c2e5f98f5528767c
Qwen3-8B_to_Qwen3-1.7B: 25 translated layers (28 receiver layers, 3 ignored)
Qwen3-14B_to_Qwen3-8B: 33 translated layers (36 receiver layers, 3 ignored)
Ministral3-8B_to_Ministral3-3B: 23 translated layers (26 receiver layers, 3 ignored)
Ministral3-14B_to_Ministral3-8B: 31 translated layers (34 receiver layers, 3 ignored)
weight shapes: (32,16), (40,32), (32,32), (32,32), respectively; these agree
with the sender/receiver attention-head counts in the paper configuration.
IMPLEMENTATION_ANCHOR simplex normalization=True
IMPLEMENTATION_ANCHOR BSR backend=True
IMPLEMENTATION_ANCHOR top-p=True
IMPLEMENTATION_ANCHOR prefix hash=True
STATIC_AUDIT PASS
SPARSE_REFERENCE_MATH PASS
```

One indexing detail is easy to miss: the released directories contain
`translator_layer_3.pt` through the final receiver-layer index (for example
3--27 for the 28-layer receiver), but the shipped runtime uses
`range(num_ignored_layers, num_layers - 1)` and explicitly treats
`layer_idx == num_layers - 1` as dense/ignored.  Thus the effective reuse
interval is receiver layers 3--26 (24 layers) for Qwen3-1.7B, not all 25
files in the directory; analogous one-file terminal overprovisioning exists
for the other pairs.  The audit keeps the files and records this behavior
instead of silently changing the implementation.  It is an implementation
detail that must be matched when comparing layer counts with the paper's
high-level `L_R-r` notation.

Two artifact-level gaps are important for interpreting “complete reproduction”:

1. The paper says the head maps are fitted with frozen sender/receiver LLMs
   using answer-only end-to-end loss and approximately 10,000 train/validation
   examples.  The public checkout contains the resulting `translator_layer_*.pt`
   files plus optimizer/scheduler state, but no mapper-training script, train or
   validation split, preprocessing recipe, seed, or checkpoint-selection rule.
   The optimizer states show that training did occur (for example, the released
   Qwen3-8B→1.7B HotpotQA state is at step 10,650), but they cannot reconstruct
   the missing training procedure.  This audit therefore verifies use of the
   released calibration artifacts, not an independent re-training reproduction.
2. The paper lists LMSYS and ShareGPT4 as separate conversation datasets, while
   the public accuracy directory exposes one `conversation_1000.jsonl` file and
   no provenance/split manifest identifying those two sources.  Its 100 raw JSON
   lines expand to 1,033 valid conversation transitions under the supplied
   tokenizer; Multi-News has 5,622 raw records.  Passing `--num_max_examples
   1000` selects 1,000 valid transitions, but this is not evidence that the
   public file is the paper's exact LMSYS/ShareGPT4 test split.

The effective local evaluator distributions were also measured with the
tokenizer and `max_length` values in the public launchers (length includes
context, append, and answer tokens): HotpotQA `n=1000`, min/max/mean
`99/3475/1550.4`; conversation `n=1014` valid transitions before taking the
first 1,000, min/max/mean `145/10236/5152.2` at `max_length=10240`; and
Multi-News `n=1000`, min/max/mean `270/5106/2144.6` at `max_length=5120`.
These do not match the Appendix's nominal 4--8K, 12--20K, and 8--12K context
bands respectively.  They are useful for debugging the public checkout, but
they are a material data/protocol deviation from the paper's reported
evaluation distribution.

## 4. Environment and model preparation

The MapFlow runtime and all subsequent experiment commands use the copied
environment `/home/wby/.venvs/mapflow-repro312` (created from the existing
environment rather than modifying it in place).  The source README requests
Python 3.11,
the modified vLLM package, `nvtx`, `sacrebleu`, and `rouge_score`, followed by
NVIDIA MPS.  The imported vLLM metadata pins torch 2.10.0.  The machine has
driver 535.183.01 (reported CUDA 12.2) and eight RTX 4090 24GB cards, whereas
the paper uses eight A800 80GB cards.  The driver/CUDA compatibility and BSR
Triton compilation are explicit gates; a server that starts with a fallback
attention backend is **not** accepted as a MapFlow run.

Existing local model snapshots at the start of the run:

```text
/home/wby/models/Qwen3-8B
/home/wby/models/Qwen3-14B
/home/wby/models/Ministral-3-3B-Reasoning-2512
/home/wby/models/Ministral-3-8B-Reasoning-2512
/home/wby/models/Ministral-3-14B-Reasoning-2512
```

Qwen3-1.7B was absent and must be downloaded to a separate model cache.  The
model is never committed to git.  The launcher expects ignored symlinks under
`examples/models/`; the exact links and checksums are printed by the run
script before serving.

Setup caveat: one early shell command resolved the bare `pip` executable to
`/home/wby/miniconda3/bin/pip` instead of the copied environment and installed
the requested Transformers/OpenCV packages there.  No MapFlow server or
accuracy command used that interpreter; all later installs and runtime checks
use the absolute copied-environment Python.  The original base-environment
package state was not recorded, so this is disclosed as an environment
contamination that should be independently restored/checked rather than
silently claimed absent.

## 5. Runtime gate and attempted accuracy experiment

The independent runtime copy was brought to the vLLM 0.18.0 / torch
2.10.0+cu126 combination required by the vendored source.  The following
compatibility repairs were needed before the MapFlow path could be selected:

* `transformers>=4.56` and `tokenizers>=0.22` are required by vLLM 0.18;
* `opencv-python-headless` is required by `mistral-common`'s dummy multimodal
  tokenizer path, even for text-only requests;
* the vendored Mistral config parser removes an inferred Python `dtype` before
  constructing nested Transformers configs (otherwise model loading exits with
  `TypeError: Object of type dtype is not JSON serializable`);
* vLLM 0.18's attention selector does not read MapFlow's
  `VLLM_ATTENTION_BACKEND` variable unless the MapFlow selector/attention files
  are overlaid.  The runtime overlay now includes those exact vendored files.

The normal Qwen3-8B vLLM server reached `/v1/models` and generated a smoke
response with FlashAttention.  The MapFlow Ministral sender reached the
MapFlow-specific selector and printed `Using AttentionBackendEnum.BSR_ATTN`;
there was no silent FlashAttention fallback.  A first sender attempt used the
paper's 3,096,000-block IPC buffer and correctly failed on a 24-GB RTX 4090
with an approximately 23.6-GiB allocation.  A copied-source runtime knob
`MAPFLOW_IPC_BUFFER_SIZE` was then added, defaulting to the original value;
this knob is a memory-footprint workaround only.

With `MAPFLOW_IPC_BUFFER_SIZE=300000`, the cache server and sender were
started as separate processes.  The cache-server child failed while opening a
sender CUDA-IPC buffer with:

```text
torch.AcceleratorError: CUDA error: peer access is not supported between these two devices
```

This was independently confirmed before and after the run:

```text
torch 2.10.0+cu126 cuda 12.6
cuda_available True count 8
torch.cuda.can_device_access_peer(i, j) = False for every distinct GPU pair
```

The repeatable hardware audit is `python reproduction/hardware_audit.py`; it
also records all eight devices as RTX 4090 `(8, 9)` and prints the full 8×8
peer-access matrix.

The paper's implementation uses CUDA IPC, and the public cache server places
the shared block pool on a second visible GPU (`cuda:tp_size`).  On this
RTX-4090 host (no NVLink/P2P), the sender/cache/receiver topology cannot open
those remote CUDA tensors.  The process therefore never reached a valid
receiver registration or an accuracy request.  No MapFlow accuracy number is
reported from this failed gate, and no failed output is mislabeled as a model
score.  A host with A800-class peer access, or a separately implemented and
explicitly labeled host-staging backend, is required for the paper's actual
end-to-end test.

The BSR selector and MapFlow Python implementation were reached before this
failure.  Triton kernel *selection* was therefore verified, but the actual
BSR Triton launch/compilation was not reached in a valid sender/cache/receiver
run.  It must remain an open gate rather than being inferred from a Python
import.

The exact attempted gate used the paper pair Ministral3-8B→Ministral3-3B,
`top_p=0.7`, sender layers 34, receiver layers 26, block size 64, and the
paper's `hotpotqa` weights.  The attempt stopped before any dataset example
was scored.

## 6. Accuracy experiments (paper test split)

The official scripts use `examples/data/accuracy/*.jsonl`, max lengths and
generation caps from the upstream scripts, and deterministic vLLM server
settings.  For each pair, the sender and receiver run as separate processes on
separate GPUs, with the MapFlow cache server on its own configured CUDA
device(s).  A result is valid only when all three process logs show
`VLLM_ATTENTION_BACKEND=BSR_ATTN`, the expected model dimensions, and no
fallback/error.

Planned commands (one pair at a time to avoid memory contamination):

```bash
# cache server (GPU 0), sender (GPU 1), receiver (GPU 2)
bash examples/bash/accuracy/server_top_p.sh qwen_8_1.7 0.7 0
bash vllm/attnmap_cache_integration/accuracy/sender_top_p.sh Qwen3-8B 0.7 1
bash vllm/attnmap_cache_integration/accuracy/receiver_top_p.sh Qwen3-1.7B 0.7 2 examples/weights/Qwen3-8B_to_Qwen3-1.7B/hotpotqa
bash examples/bash/accuracy/hotpotqa.sh 0.7 Qwen3-8B Qwen3-1.7B 7001 7002 1000
```

The corresponding official commands are used for `conversation.sh` and
`multinews.sh`, and for Qwen3-14B→8B, Ministral3-8B→3B, and
Ministral3-14B→8B.  The data-specific mapper directory is changed to the
matching `examples/weights/<pair>/<dataset>`; no mapper trained on the test
examples is substituted.  The bundled example README uses 100 HotpotQA
examples as a quick smoke test; the paper reproduction uses all 1,000 unless
a hardware failure is explicitly reported.

Accuracy results will be recorded in a table with: pair, task, n, top-p,
MapFlow score, full-prefill receiver score, absolute delta, seed/decoding
settings, and log paths.  Missing rows remain marked `NOT RUN`, not zero.

## 7. RULER 20K, sparsification, and system measurements

The paper's long-context test uses the RULER 20K validation file and the four
pair-specific top-p values listed in Section 2.  The upstream script currently
contains an institution-specific `/data/zhan/...` path; the exact source file
and hash must be supplied in the log before running.  If it is unavailable,
the RULER row is reported as an environment/data limitation rather than
replaced by the 100-row local smoke file.

TTFT and end-to-end experiments use the official Poisson traces and routing
scripts.  They are measured only after accuracy correctness is established:

```bash
python examples/extract_ttft.py --traj_path <official-log>
python examples/extract_e2e.py --file_path <official-log>
```

The paper reports 15–20% prefill reduction and up to 25% higher QPS, but these
are paper reference numbers, not claims about the 4090 reproduction.  Any
hardware-dependent difference is reported with the raw logs and percentiles.

## 8. Deviation and correctness checklist

- [x] anonymous source endpoint availability recorded;
- [x] public source commit and local paper hash recorded;
- [x] source tree imported without changing MapFlow/vLLM algorithms;
- [x] paper's four model pairs distinguished from the extra bundled pair;
- [x] dataset files, mapper files, layer counts, and heads statically audited;
- [x] block top-p reference invariants checked;
- [x] missing public evaluator entry points and unconditional kernel import repaired and recorded;
- [x] independent MapFlow environment imports and normal vLLM smoke server starts;
- [ ] Qwen3-1.7B acquired and verified;
- [x] BSR backend is selected (no fallback); CUDA-IPC registration is blocked by no P2P;
- [ ] BSR Triton kernel launch/compilation on a valid MapFlow cache run;
- [ ] all four pair × accuracy datasets run with n=1,000 paper test examples;
- [ ] RULER-20K file resolved and four long-context rows run;
- [ ] TTFT and routing traces run and extracted;
- [x] raw commands, runtime errors, deviations, and static audit are recorded;
- [ ] branch pushed to `https://github.com/geekittime/mapflow`.

The checklist is intentionally left open until authoritative logs exist.
