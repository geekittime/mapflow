"""Compatibility entry point for the public MapFlow conversation launcher.

The upstream shell script passes the same arguments as the batch evaluator and
adds ``--window_size``.  The included dataset builder already constructs each
valid context/append/answer window, so the batch implementation is the faithful
execution path; this file intentionally delegates to it rather than silently
changing the cache protocol.
"""

from inference_conversation_batch import main_async
import asyncio


if __name__ == "__main__":
    asyncio.run(main_async())
