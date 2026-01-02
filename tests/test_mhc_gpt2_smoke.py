import torch

from mhc.transformers.gpt2_mhc import MhcGPT2Config, MhcGPT2LMHeadModel


def _tiny_cfg():
    # Keep IDs in-vocab to avoid HF warnings about eos_token_id/pad_token_id.
    vocab_size = 64
    return MhcGPT2Config(
        vocab_size=vocab_size,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=True,
        eos_token_id=vocab_size - 1,
        pad_token_id=vocab_size - 1,
        mhc_n=4,
        mhc_stream_init="paper",
        mhc_readout_init="first",
    )


def test_mhc_gpt2_forward_shapes_and_loss_finite():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = MhcGPT2LMHeadModel(cfg)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (2, 11))
    out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    assert out.logits.shape == (2, 11, cfg.vocab_size)
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_mhc_gpt2_generate_smoke_no_cache_and_cache():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = MhcGPT2LMHeadModel(cfg)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 7))
    attention_mask = torch.ones_like(input_ids)

    out1 = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=5,
        do_sample=False,
        use_cache=False,
    )
    assert out1.shape == (1, 12)

    out2 = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=5,
        do_sample=False,
        use_cache=True,
    )
    assert out2.shape == (1, 12)


def test_mhc_gpt2_cache_logits_match_full_forward():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = MhcGPT2LMHeadModel(cfg)
    model.eval()

    # Sequence length >= 2 so we can split into prefix + last token.
    input_ids = torch.randint(0, cfg.vocab_size, (1, 9))

    with torch.no_grad():
        full = model(input_ids=input_ids, use_cache=False).logits  # (1,9,V)
        full_last = full[:, -1, :]  # logits at last position

        prefix = input_ids[:, :-1]
        last = input_ids[:, -1:]

        out1 = model(input_ids=prefix, use_cache=True)
        past = out1.past_key_values
        out2 = model(input_ids=last, past_key_values=past, use_cache=True)

        cached_last = out2.logits[:, -1, :]

    assert torch.allclose(full_last, cached_last, atol=1e-4, rtol=1e-4)


