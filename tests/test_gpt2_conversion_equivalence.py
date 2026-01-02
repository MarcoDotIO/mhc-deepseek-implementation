import torch
from transformers import GPT2Config, GPT2LMHeadModel

from mhc.transformers.convert_gpt2 import convert_gpt2_lm_head_model


def test_gpt2_to_mhc_conversion_logits_match_at_init():
    torch.manual_seed(0)
    cfg = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    base = GPT2LMHeadModel(cfg)
    base.eval()

    mhc = convert_gpt2_lm_head_model(base, mhc_n=4, equivalence_init=True, offdiag_bias=-50.0)
    mhc.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (2, 13))
    with torch.no_grad():
        base_logits = base(input_ids=input_ids).logits
        mhc_logits = mhc(input_ids=input_ids, use_cache=False).logits

    assert torch.allclose(base_logits, mhc_logits, atol=1e-4, rtol=1e-4)


