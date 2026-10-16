"""Compatibility shim for the public MapFlow source tree.

The upstream ``mapflow.client.receiver`` imports this symbol from
``mapflow.kernel.flex_prefill_kernel`` although the implementation is kept in
the bundled modified vLLM tree.  The two call sites in the receiver are
disabled comments for the BSR path, but the import is unconditional and makes
the unmodified checkout fail before a server can start.  Re-exporting the
upstream vLLM implementation fixes that packaging omission without changing
the MapFlow BSR algorithm or its runtime path.
"""

from vllm.v1.attention.backends.flex_prefill_kernel import flex_prefill_varlen_func

__all__ = ["flex_prefill_varlen_func"]
