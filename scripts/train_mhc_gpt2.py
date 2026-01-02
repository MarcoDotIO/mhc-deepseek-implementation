#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

from mhc.transformers.convert_gpt2 import convert_gpt2_lm_head_model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finetune mHC parameters on a GPT-2 model (single-device).")
    p.add_argument("--base", type=str, default="gpt2", help="HF model name or local path")
    p.add_argument("--output-dir", type=str, default="./outputs/mhc-gpt2", help="Where to save checkpoints")
    p.add_argument("--dataset", type=str, default="wikitext", help="HF dataset name (default: wikitext)")
    p.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--eval-split", type=str, default="validation")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--mhc-n", type=int, default=4)
    p.add_argument("--mhc-tmax", type=int, default=20)
    p.add_argument("--mhc-alpha-init", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--freeze-base", action="store_true", help="Train only mHC params + readout")
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Default is 'cuda' (fail fast if unavailable). Use 'auto' to fall back to CPU.",
    )
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--compile", action="store_true", help="Use torch.compile (if supported)")
    p.add_argument("--profile", action="store_true", help="Print rough wall-time breakdown for train/eval steps")
    return p.parse_args()


def _freeze_except_mhc(model) -> None:
    allow = ("mhc_attn.", "mhc_mlp.", "mhc_readout_logits")
    for name, p in model.named_parameters():
        p.requires_grad = any(k in name for k in allow)


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

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

    print(f"[info] torch={torch.__version__} device={device} cuda_available={torch.cuda.is_available()}")

    tokenizer = GPT2TokenizerFast.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(args.dataset, args.dataset_config)

    def tokenize_fn(examples):
        return tokenizer(examples["text"])

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=raw[args.train_split].column_names)

    def group_texts(examples):
        # Concatenate and split into blocks.
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
    train_ds = lm_ds[args.train_split]
    eval_ds = lm_ds[args.eval_split]

    base = GPT2LMHeadModel.from_pretrained(args.base)
    base.config.use_cache = False
    base.eval()

    model = convert_gpt2_lm_head_model(
        base,
        mhc_n=args.mhc_n,
        mhc_tmax=args.mhc_tmax,
        mhc_alpha_init=args.mhc_alpha_init,
        equivalence_init=True,
        offdiag_bias=-50.0,
    )
    model.config.use_cache = False

    if args.freeze_base:
        _freeze_except_mhc(model)

    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()

    model.to(device)

    if args.compile:
        model = torch.compile(model)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        use_cpu=(device.type == "cpu"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        report_to=None,
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if args.profile:
        from mhc.profiling import WallTimer, timed

        train_timer = WallTimer("trainer.train")
        with timed(train_timer):
            trainer.train()
        print(f"[profile] {train_timer.name}: {train_timer.seconds:.2f}s")
    else:
        trainer.train()
    trainer.save_model(args.output_dir)

    # Generation smoke test
    model.eval()
    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=False)
    print("[gen]", tokenizer.decode(gen[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()


