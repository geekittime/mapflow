# H20 accuracy reproduction — MapFlow

## Scope and completion gate

User requested two important paper model pairs, paper methods and accuracy benchmarks, stage-by-stage correctness checks, and pushing source/logs to `geekittime/mapflow`. TTFT is excluded. Chosen pairs: **Qwen3-8B → Qwen3-1.7B**, **Qwen3-14B → Qwen3-8B**. Original scope remains open until completed and verified.

For each pair: run the paper's HotpotQA, Conversation, Multi-News quality evaluation and RULER-20K; compare MapFlow to native receiver full prefill on identical examples. Retain upstream task-specific translator checkpoints and top-p settings. Record sampling parameters, actual loader-selected examples, per-item predictions, scores, failures, server/backend evidence, file hashes, and explicit deviations. Additional paper quality ablations/baselines will be inventoried against available code before the completion audit. No missing metric is to be called zero or passed.

## 2026-10-17 — initial checks

- Fresh clone of requested repository: `2c6e49af82f75f8e04cebe802401df145df624cc`, branch `reproduction/h20-accuracy-two-pairs`. Previous logs concern a different host (`/home/wby`, RTX 4090); they are retained as history, not current-host evidence.
- Anonymous endpoint returned HTTP 401 (`not_connected`). Fresh author mirror `/home/tiger/mapflow-upstream`, commit `ab1dc50ea811a8a50c0beb64964cdf963956eb32`, matches the source recorded in earlier work. The mirror is separate and not substituted silently.
- Current hardware: 4 × H20 96 GB. All cross-GPU peer access checks passed. Existing `/home/tiger/exp.py` was inspected: it is an infinite GPU memory/matmul stress test. Stopped parent PID 2919023 with SIGINT; workers exited and all four GPUs became free. No model jobs were running.
- All three local Qwen3 snapshots passed indexed safetensors shard-opening and layer/head configuration checks. Created model symlinks under `examples/models`.
- Initial source inventory and reference sparse-block math passed; these are static checks, not evidence of a successful MapFlow inference run.
- Raw bundled data: HotpotQA 1,000 records; Conversation **100 conversations**, despite filename; Multi-News **5,622 records**, despite filename; RULER 100 records whose context length must still be verified. Actual eligible window counts must be measured using the author loader.
- Isolated `.venv` installation started for vLLM 0.18.0 + Transformers 4.57.6 and metric dependencies. No edits to global Python environment. Installation log: `logs/install.log`.
- The author release includes task-specific translation weights; no calibration trainer was found in the public source. Using released weights is allowed by the request; any newly trained substitute must be explicitly labelled and audited, not described as the original checkpoint.

Status: **in progress**. No H20 accuracy results yet. GPU IPC execution and BSR retrieval still require runtime validation.

## 2026-10-17 — detailed stage audit

- Full details: [AUDIT_20261017.md](AUDIT_20261017.md). Fixed state_dict loading in the input audit; all 232 released translator files now match the author mirror and pass tensor checks. Exact loader-selected examples and token hashes recorded, including 1,014 conversation windows and bundled RULER QA lengths 17,249–20,445.
- Fixed the Multi-News evaluator CLI incompatibility. Installed isolated torch 2.10.0+cu128 / vLLM 0.18.0 / Transformers 4.57.6 and applied audited runtime overlay; BSR imports pass.
- Added actual causal GQA GPU compression tests (nine cases) instead of relying on standalone reference arithmetic. Passed in the final runtime.
- Cross-process IPC test initially reproduced a cold-context SIGSEGV. Identified private storage import before CUDA initialization; moved public empty-tensor allocation before storage reconstruction. Four cross-GPU cold-worker read/write cases now pass. Original failure logs retained.
- Also re-audited prior KV-Lingo canonical results separately, recorded in that repository; no cross-paper result mixing.
- Status remains **in progress**: no H20 end-to-end MapFlow model inference or complete accuracy results yet. End-to-end reuse evidence, structured evaluator, baseline comparability and paper/data deviations remain explicit gates.
- Audit changes committed locally on `reproduction/h20-accuracy-two-pairs`. Authorized push was attempted but failed because Git has no available HTTPS username/credentials (`could not read Username for https://github.com` with prompts disabled); `gh` is not installed. No remote publication is claimed. After configuring credentials, push with `git push -u origin reproduction/h20-accuracy-two-pairs`.
