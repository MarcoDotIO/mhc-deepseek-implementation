#!/usr/bin/env python
from __future__ import annotations

import argparse
import math

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, default_data_collator

from mhc.transformers.convert_gpt2 import convert_gpt2_lm_head_model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate perplexity for an mHC GPT-2 model.")
    p.add_argument("--base", type=str, default="gpt2", help="Base model name/path (if converting)")
    p.add_argument("--mhc-checkpoint", type=str, default=None, help="Path to saved mHC model (save_pretrained dir)")
    p.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help=(
            "Tokenizer name/path. Default: gpt2. "
            "Note: model-only checkpoints (like outputs/mhc-gpt2-init) may not include tokenizer files."
        ),
    )
    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--mhc-n", type=int, default=4)
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Default is 'cuda' (fail fast if unavailable). Use 'auto' to fall back to CPU.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build or pass --device cpu/auto."
            )
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer: the mHC checkpoint directory may not include tokenizer files.
    # Default to a known GPT-2 tokenizer unless the user overrides.
    tokenizer_ref = args.tokenizer or args.base
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_ref)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(args.dataset, args.dataset_config, split=args.split)

    def tokenize_fn(examples):
        return tokenizer(examples["text"])

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=raw.column_names)

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // args.block_size) * args.block_size
        result = {
            k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_ds = tokenized.map(group_texts, batched=True)
    dl = DataLoader(lm_ds, batch_size=args.batch_size, collate_fn=default_data_collator)

    if args.mhc_checkpoint is not None:
        from mhc.transformers.gpt2_mhc import MhcGPT2LMHeadModel

        model = MhcGPT2LMHeadModel.from_pretrained(args.mhc_checkpoint)
    else:
        base = GPT2LMHeadModel.from_pretrained(args.base)
        base.config.use_cache = False
        model = convert_gpt2_lm_head_model(base, mhc_n=args.mhc_n, equivalence_init=False)

    model.config.use_cache = False
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, use_cache=False)
            loss = out.loss
            # loss is mean over tokens in batch; scale by token count
            n_tokens = batch["labels"].numel()
            total_loss += float(loss) * n_tokens
            total_tokens += n_tokens

    ppl = math.exp(total_loss / max(total_tokens, 1))
    print(f"perplexity={ppl:.4f}")


if __name__ == "__main__":
    main()


