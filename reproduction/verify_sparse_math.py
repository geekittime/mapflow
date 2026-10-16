"""Reference-only check for MapFlow's block top-p invariants.

The paper's CUDA/Triton kernels use the same two-stage operation: aggregate
block log-sum-exp values, choose the smallest descending set whose cumulative
probability reaches p, and extract exact probabilities for retained blocks.
This NumPy implementation is intentionally tiny and is not used for serving.
"""

from __future__ import annotations

import math
import random


def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def block_top_p(logits, block_size, p):
    n_rows = len(logits)
    n_cols = len(logits[0])
    out = []
    for r0 in range(0, n_rows, block_size):
        for c0 in range(0, n_cols, block_size):
            block = [
                logits[r][c]
                for r in range(r0, min(r0 + block_size, n_rows))
                for c in range(c0, min(c0 + block_size, n_cols))
            ]
            out.append((r0 // block_size, c0 // block_size, logsumexp(block)))
    total = logsumexp([x[2] for x in out])
    probs = [(r, c, math.exp(score - total)) for r, c, score in out]
    probs.sort(key=lambda x: x[2], reverse=True)
    kept, cumulative = [], 0.0
    for item in probs:
        kept.append(item)
        cumulative += item[2]
        if cumulative >= p:
            break
    return kept, probs


def main():
    random.seed(7)
    rows, cols, block = 16, 20, 4
    logits = [[random.uniform(-9.0, 3.0) for _ in range(cols)] for _ in range(rows)]
    for p in (0.7, 0.8, 0.9, 0.92):
        kept, all_probs = block_top_p(logits, block, p)
        kept_mass = sum(x[2] for x in kept)
        assert kept_mass >= p - 1e-12
        if len(kept) > 1:
            previous = sum(x[2] for x in all_probs[: len(kept) - 1])
            assert previous < p + 1e-12
        assert len(kept) <= len(all_probs)
        print(f"p={p:.2f} kept_blocks={len(kept)}/{len(all_probs)} retained_mass={kept_mass:.8f}")
    print("SPARSE_REFERENCE_MATH PASS")


if __name__ == "__main__":
    main()
