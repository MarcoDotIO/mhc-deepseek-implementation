from __future__ import annotations

import argparse
from typing import Optional

import torch
from transformers import GPT2LMHeadModel

from .gpt2_mhc import MhcGPT2Config, MhcGPT2LMHeadModel


@torch.no_grad()
def convert_gpt2_lm_head_model(
    base: GPT2LMHeadModel,
    *,
    mhc_n: int = 4,
    mhc_tmax: int = 20,
    mhc_alpha_init: float = 0.01,
    mhc_rmsnorm_eps: float = 1e-6,
    equivalence_init: bool = True,
    offdiag_bias: float = -50.0,
) -> MhcGPT2LMHeadModel:
    """
    Convert a GPT-2-style HF `GPT2LMHeadModel` into an `MhcGPT2LMHeadModel`.

    The conversion copies all standard GPT-2 weights (embeddings, attention, MLP, LNs).

    If `equivalence_init=True`, we additionally initialize mHC coefficients to behave like a
    vanilla residual connection when the stream is initialized as **identical copies**
    (see the mHC paper Table 1 ablation defaults):
      - H_pre = uniform 1/n
      - H_post = ones
      - H_res ≈ I (via Sinkhorn on strongly diagonal-biased logits)
      - stream init = "copy"
      - readout init = "mean"

    This makes the converted model's logits match the base model (up to tiny numerical noise).
    """
    base_cfg = base.config.to_dict()
    base_cfg.pop("model_type", None)

    mhc_stream_init = "copy" if equivalence_init else "paper"
    mhc_readout_init = "mean" if equivalence_init else "first"

    cfg = MhcGPT2Config(
        **base_cfg,
        mhc_n=mhc_n,
        mhc_tmax=mhc_tmax,
        mhc_alpha_init=mhc_alpha_init,
        mhc_rmsnorm_eps=mhc_rmsnorm_eps,
        mhc_stream_init=mhc_stream_init,
        mhc_readout_init=mhc_readout_init,
    )

    mhc = MhcGPT2LMHeadModel(cfg)

    # Copy embeddings + final LN
    mhc.transformer.wte.load_state_dict(base.transformer.wte.state_dict())
    mhc.transformer.wpe.load_state_dict(base.transformer.wpe.state_dict())
    mhc.transformer.ln_f.load_state_dict(base.transformer.ln_f.state_dict())

    # Copy blocks
    if len(mhc.transformer.h) != len(base.transformer.h):
        raise ValueError("n_layer mismatch between base and mhc configs")

    for i, (mhc_block, base_block) in enumerate(zip(mhc.transformer.h, base.transformer.h)):
        mhc_block.ln_1.load_state_dict(base_block.ln_1.state_dict())
        mhc_block.attn.load_state_dict(base_block.attn.state_dict())
        mhc_block.ln_2.load_state_dict(base_block.ln_2.state_dict())
        mhc_block.mlp.load_state_dict(base_block.mlp.state_dict())

        if equivalence_init:
            mhc_block.mhc_attn.init_gpt2_equivalence(offdiag_bias=offdiag_bias, alpha=0.0)
            mhc_block.mhc_mlp.init_gpt2_equivalence(offdiag_bias=offdiag_bias, alpha=0.0)

    mhc.tie_weights()
    return mhc


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a GPT-2 model to an mHC GPT-2 model.")
    p.add_argument("--base", type=str, default="gpt2", help="HF model name or local path")
    p.add_argument("--out", type=str, required=True, help="Output directory (save_pretrained)")
    p.add_argument("--mhc-n", type=int, default=4)
    p.add_argument("--mhc-tmax", type=int, default=20)
    p.add_argument("--mhc-alpha-init", type=float, default=0.01)
    p.add_argument("--mhc-rmsnorm-eps", type=float, default=1e-6)
    p.add_argument("--no-equivalence-init", action="store_true", help="Disable GPT-2 equivalence init")
    p.add_argument("--offdiag-bias", type=float, default=-50.0)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    base = GPT2LMHeadModel.from_pretrained(args.base)
    mhc = convert_gpt2_lm_head_model(
        base,
        mhc_n=args.mhc_n,
        mhc_tmax=args.mhc_tmax,
        mhc_alpha_init=args.mhc_alpha_init,
        mhc_rmsnorm_eps=args.mhc_rmsnorm_eps,
        equivalence_init=not args.no_equivalence_init,
        offdiag_bias=args.offdiag_bias,
    )
    mhc.save_pretrained(args.out)


if __name__ == "__main__":
    main()


