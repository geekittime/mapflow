"""Accuracy evaluator for conversation and Multi-News MapFlow runs.

The public source distribution referenced this file from its shell launcher but
did not include it.  This implementation follows the included QA evaluator's
protocol exactly: prefill each sender prefix with one token, then issue the
receiver request over the shared context plus append, and score only the
receiver answer with ROUGE-L (the paper metric for conversation/summarization).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from typing import Any

from openai import AsyncOpenAI
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm

from data import get_dataset
from metrics import get_bleu_rouge


SERVERS: dict[str, AsyncOpenAI] = {}


def _extra(model: str) -> dict[str, Any]:
    # Qwen3 defaults to thinking mode.  The upstream README's accuracy path is
    # a short-answer task, so match the QA evaluator and disable it explicitly.
    return {} if "Ministral" in model else {"chat_template_kwargs": {"enable_thinking": False}}


async def process_batch(batch: list[dict[str, Any]], args):
    sender = SERVERS["sender"]
    receiver = SERVERS["receiver"]

    await asyncio.gather(*[
        sender.chat.completions.create(
            model=args.send_model,
            messages=x["sender_messages"],
            max_tokens=1,
            extra_body=_extra(args.send_model),
        )
        for x in batch
    ])
    await asyncio.sleep(0.5)

    responses = await asyncio.gather(*[
        receiver.chat.completions.create(
            model=args.receive_model,
            messages=x["receiver_messages"],
            max_tokens=args.max_new_tokens,
            extra_body=_extra(args.receive_model),
        )
        for x in batch
    ])
    out = []
    for x, response in zip(batch, responses):
        prediction = response.choices[0].message.content or ""
        scores = get_bleu_rouge(prediction, x["ans"])
        out.append((prediction, x["ans"], scores))
    return out


def write(path: str, text: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


async def main_async():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["conversation", "multinews"], default=None,
                   help="Dataset type; omitted only for legacy filename inference")
    p.add_argument("--eval_data", required=True)
    p.add_argument("--num_max_examples", type=int, default=-1)
    p.add_argument("--max_length", type=int, default=10240)
    p.add_argument("--send_model", required=True)
    p.add_argument("--receive_model", required=True)
    p.add_argument("--send_model_port", type=int, default=8000)
    p.add_argument("--recv_model_port", type=int, default=8001)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--window_size", type=int, default=1, help="retained for upstream CLI compatibility")
    p.add_argument("--log_file", required=True)
    p.add_argument("--result_file", required=True)
    args = p.parse_args()

    for path in (args.log_file, args.result_file):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    SERVERS.update(
        sender=AsyncOpenAI(base_url=f"http://localhost:{args.send_model_port}/v1", api_key="EMPTY"),
        receiver=AsyncOpenAI(base_url=f"http://localhost:{args.recv_model_port}/v1", api_key="EMPTY"),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.send_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset_name = args.dataset or ("multinews" if "multinews" in args.eval_data.lower() else "conversation")
    dataset = get_dataset(
        dataset_name, args.eval_data, tokenizer,
        max_length=args.max_length, num_max_examples=args.num_max_examples,
    )
    write(args.log_file, f"dataset={dataset_name} examples={len(dataset)} window_size={args.window_size}")

    rouge_l, bleu, rouge1, rouge2 = [], [], [], []
    for start in tqdm(range(0, len(dataset), args.batch_size), desc="Processing Batches"):
        batch = [dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))]
        for j, (prediction, gold, scores) in enumerate(await process_batch(batch, args), start=1):
            idx = start + j
            rouge_l.append(scores["rougeL"])
            bleu.append(scores["bleu"])
            rouge1.append(scores["rouge1"])
            rouge2.append(scores["rouge2"])
            write(args.log_file, f"[{idx:4d}/{len(dataset)}] ROUGE-L={scores['rougeL']:.6f} | Pred: {prediction} | Gold: {gold}")

    n = len(rouge_l)
    if not n:
        raise RuntimeError("dataset produced zero valid transition examples")
    result = (
        f"Send Model: {args.send_model}\nReceive Model: {args.receive_model}\n"
        f"Samples: {n}\nROUGE-L: {sum(rouge_l)/n:.6f}\n"
        f"ROUGE-1: {sum(rouge1)/n:.6f}\nROUGE-2: {sum(rouge2)/n:.6f}\n"
        f"BLEU: {sum(bleu)/n:.6f}\n"
    )
    write(args.result_file, result)
    print(result)


if __name__ == "__main__":
    asyncio.run(main_async())
