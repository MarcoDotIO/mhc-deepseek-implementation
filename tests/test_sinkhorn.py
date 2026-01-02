import torch

from mhc.sinkhorn import sinkhorn_knopp


def test_sinkhorn_output_is_doubly_stochastic_approx():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4, 4)  # (B,T,n,n)
    m = sinkhorn_knopp(logits, tmax=30)

    row_sums = m.sum(dim=-1)  # (B,T,n)
    col_sums = m.sum(dim=-2)  # (B,T,n)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-3, rtol=1e-3)
    assert (m >= 0).all()


def test_sinkhorn_identity_like_init_is_close_to_identity():
    n = 4
    logits = torch.full((1, 1, n, n), -20.0)
    logits[..., torch.arange(n), torch.arange(n)] = 0.0
    m = sinkhorn_knopp(logits, tmax=20, clamp_min=0.0)

    eye = torch.eye(n).reshape(1, 1, n, n)
    assert torch.allclose(m, eye, atol=5e-3, rtol=0.0)


def test_sinkhorn_backward_has_gradients():
    logits = torch.randn(2, 4, 4, requires_grad=True)
    m = sinkhorn_knopp(logits, tmax=10)
    loss = m.sum()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


