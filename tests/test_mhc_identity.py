import torch

from mhc.mhc import MhcProjector
from mhc.stream_ops import mhc_update, stream_weighted_sum


def test_mhc_projector_gpt2_equivalence_constants():
    torch.manual_seed(0)
    n, c = 4, 8
    proj = MhcProjector(n_streams=n, hidden_dim=c, tmax=20, alpha_init=0.01)
    proj.init_gpt2_equivalence(offdiag_bias=-50.0, alpha=0.0)

    x = torch.randn(2, 3, c)
    x_stream = x.unsqueeze(-2).expand(2, 3, n, c).contiguous()

    maps = proj(x_stream)
    assert torch.allclose(maps.h_pre, torch.full_like(maps.h_pre, 1.0 / n), atol=1e-4, rtol=0.0)
    assert torch.allclose(maps.h_post, torch.ones_like(maps.h_post), atol=1e-6, rtol=0.0)

    eye = torch.eye(n).reshape(1, 1, n, n).expand_as(maps.h_res)
    assert torch.allclose(maps.h_res, eye, atol=1e-3, rtol=0.0)


def test_mhc_update_matches_residual_when_streams_identical():
    torch.manual_seed(0)
    n, c = 4, 8
    proj = MhcProjector(n_streams=n, hidden_dim=c, tmax=20, alpha_init=0.01)
    proj.init_gpt2_equivalence(offdiag_bias=-50.0, alpha=0.0)

    x = torch.randn(2, 3, c)
    x_stream = x.unsqueeze(-2).expand(2, 3, n, c).contiguous()

    maps = proj(x_stream)
    x_in = stream_weighted_sum(x_stream, maps.h_pre)
    assert torch.allclose(x_in, x, atol=1e-5, rtol=0.0)

    # Dummy residual function: y = 0.5 * x_in
    y = 0.5 * x_in

    x_out = mhc_update(x_stream, h_post=maps.h_post, h_res=maps.h_res, y=y)
    expected = (x + y).unsqueeze(-2).expand_as(x_out)
    assert torch.allclose(x_out, expected, atol=1e-3, rtol=0.0)


