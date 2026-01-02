#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

from datasets import load_dataset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download/cache a HuggingFace dataset for offline training.")
    p.add_argument("--dataset", type=str, default="wikitext", help="HF dataset name")
    p.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1", help="HF dataset config")
    p.add_argument("--splits", type=str, default="train,validation,test", help="Comma-separated splits to cache")
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional HF datasets cache dir (otherwise uses HF defaults).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise SystemExit("No splits specified")

    if args.cache_dir is not None:
        os.makedirs(args.cache_dir, exist_ok=True)
        print(f"[info] cache_dir={args.cache_dir}")

    for split in splits:
        print(f"[download] {args.dataset}/{args.dataset_config} split={split}")
        ds = load_dataset(args.dataset, args.dataset_config, split=split, cache_dir=args.cache_dir)
        print(f"[done] rows={len(ds)}")


if __name__ == "__main__":
    main()


