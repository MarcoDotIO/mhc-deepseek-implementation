#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Optional

import requests
import torch
from transformers import GPT2TokenizerFast

from mhc.transformers.gpt2_mhc import MhcGPT2LMHeadModel


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optional qualitative comparison against an Ollama model.")
    p.add_argument("--ollama-host", type=str, default="http://localhost:11434", help="Ollama server base URL")
    p.add_argument(
        "--ollama-model",
        "--model",
        dest="ollama_model",
        type=str,
        default="auto",
        help="Ollama model name. Use 'auto' to pick the first installed model from /api/tags.",
    )
    p.add_argument(
        "--hf-model",
        type=str,
        default="./outputs/mhc-gpt2-init",
        help="Path to a saved mHC HF model directory (e.g. outputs/mhc-gpt2-init).",
    )
    p.add_argument(
        "--hf-tokenizer",
        type=str,
        default="gpt2",
        help="HF tokenizer name/path (default: gpt2).",
    )
    p.add_argument(
        "--hf-device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Default is 'cuda'. Use 'auto' to fall back to CPU.",
    )
    p.add_argument("--prompts", type=str, default=None, help="Path to a JSONL file with {prompt: str}")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def ollama_health(host: str, timeout_s: float = 2.0) -> bool:
    try:
        r = requests.get(f"{host}/api/tags", timeout=timeout_s)
        return r.status_code == 200
    except requests.RequestException:
        return False


def ollama_tags(host: str, timeout_s: float = 5.0) -> dict[str, Any]:
    r = requests.get(f"{host}/api/tags", timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def pick_ollama_model(host: str, requested: str) -> str:
    tags = ollama_tags(host)
    models = [m.get("name") for m in tags.get("models", []) if isinstance(m, dict)]
    models = [m for m in models if isinstance(m, str) and m]
    if not models:
        raise RuntimeError(f"No models found in {host}/api/tags. Pull a model first (e.g. `ollama pull ...`).")

    if requested == "auto":
        return models[0]

    if requested not in models:
        raise RuntimeError(
            f"Ollama model '{requested}' not found. Available models: {', '.join(models)}"
        )
    return requested


def ollama_generate(
    host: str,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float = 60.0,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    r = requests.post(f"{host}/api/generate", json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


def pick_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA requested but torch.cuda.is_available() is False. Use --hf-device cpu/auto."
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def hf_generate(
    *,
    model: MhcGPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    do_sample = temperature is not None and temperature > 0
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    completion_ids = gen[0, input_ids.shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def main() -> None:
    args = _parse_args()

    ok = ollama_health(args.ollama_host)
    if not ok:
        raise SystemExit(
            f"Ollama not reachable at {args.ollama_host}. Start it (or change --ollama-host) and retry."
        )

    ollama_model = pick_ollama_model(args.ollama_host, args.ollama_model)
    hf_device = pick_device(args.hf_device)
    hf_dtype = torch.float16 if hf_device.type == "cuda" else torch.float32

    tokenizer = GPT2TokenizerFast.from_pretrained(args.hf_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # `torch_dtype` is deprecated in this transformers version; prefer `dtype` if supported.
    try:
        model = MhcGPT2LMHeadModel.from_pretrained(args.hf_model, dtype=hf_dtype)
    except TypeError:
        model = MhcGPT2LMHeadModel.from_pretrained(args.hf_model, torch_dtype=hf_dtype)
    model.eval().to(hf_device)

    print(
        f"[info] hf_model={args.hf_model} hf_tokenizer={args.hf_tokenizer} hf_device={hf_device} hf_dtype={hf_dtype} "
        f"ollama_model={ollama_model} ollama_host={args.ollama_host}"
    )

    prompt_list: list[str] = [
        "Explain mHC (manifold-constrained hyper-connections) in two sentences.",
        "Write a short haiku about residual connections.",
        "Complete: The quick brown fox",
    ]

    if args.prompts is not None:
        prompt_list = []
        with open(args.prompts, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                prompt_list.append(obj["prompt"])

    for i, prompt in enumerate(prompt_list):
        t0 = time.time()
        hf_resp = hf_generate(
            model=model,
            tokenizer=tokenizer,
            device=hf_device,
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        t_hf = time.time() - t0

        t1 = time.time()
        ollama_resp = ollama_generate(
            args.ollama_host,
            model=ollama_model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        t_ollama = time.time() - t1

        print(f"\n=== prompt[{i}] ===\n{prompt}")
        print(f"\n--- hf ({t_hf:.2f}s) ---\n{hf_resp}")
        print(f"\n--- ollama:{ollama_model} ({t_ollama:.2f}s) ---\n{ollama_resp}")


if __name__ == "__main__":
    main()


