from __future__ import annotations

import torch


def stream_weighted_sum(x_stream: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Weighted sum over the stream dimension.

    Args:
        x_stream: (B, T, n, C)
        weights:  (B, T, n)

    Returns:
        (B, T, C)
    """
    if x_stream.ndim != 4:
        raise ValueError(f"x_stream must be (B,T,n,C), got {tuple(x_stream.shape)}")
    if weights.ndim != 3:
        raise ValueError(f"weights must be (B,T,n), got {tuple(weights.shape)}")
    if x_stream.shape[:3] != weights.shape:
        raise ValueError(f"shape mismatch: {tuple(x_stream.shape)} vs {tuple(weights.shape)}")

    # Ensure dtype match for einsum (common when coefficients are fp32 but activations are fp16/bf16).
    if weights.dtype != x_stream.dtype:
        weights = weights.to(dtype=x_stream.dtype)

    # (B,T,n,C) + (B,T,n) -> (B,T,C)
    return torch.einsum("btn,btnc->btc", weights, x_stream)


def stream_mix(x_stream: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
    """
    Mix streams with residual mapping H_res.

    Paper Eq. (3): x_{l+1} = H_res x_l + ...

    Args:
        x_stream: (B, T, n, C)
        h_res:    (B, T, n, n)

    Returns:
        (B, T, n, C)
    """
    if x_stream.ndim != 4:
        raise ValueError(f"x_stream must be (B,T,n,C), got {tuple(x_stream.shape)}")
    if h_res.ndim != 4:
        raise ValueError(f"h_res must be (B,T,n,n), got {tuple(h_res.shape)}")
    if x_stream.shape[0] != h_res.shape[0] or x_stream.shape[1] != h_res.shape[1]:
        raise ValueError(f"batch/time mismatch: {tuple(x_stream.shape)} vs {tuple(h_res.shape)}")
    if x_stream.shape[2] != h_res.shape[2] or x_stream.shape[2] != h_res.shape[3]:
        raise ValueError(f"stream mismatch: {tuple(x_stream.shape)} vs {tuple(h_res.shape)}")

    # Ensure dtype match for einsum (Sinkhorn is computed in fp32 for stability by default).
    if h_res.dtype != x_stream.dtype:
        h_res = h_res.to(dtype=x_stream.dtype)

    return torch.einsum("btij,btjc->btic", h_res, x_stream)


def stream_write(y: torch.Tensor, h_post: torch.Tensor) -> torch.Tensor:
    """
    Write a C-dim layer output back into the n-stream residual.

    Paper Eq. (3): ... + H_post^T F(...)

    Args:
        y:      (B, T, C)
        h_post: (B, T, n)

    Returns:
        (B, T, n, C)
    """
    if y.ndim != 3:
        raise ValueError(f"y must be (B,T,C), got {tuple(y.shape)}")
    if h_post.ndim != 3:
        raise ValueError(f"h_post must be (B,T,n), got {tuple(h_post.shape)}")
    if y.shape[0] != h_post.shape[0] or y.shape[1] != h_post.shape[1]:
        raise ValueError(f"batch/time mismatch: {tuple(y.shape)} vs {tuple(h_post.shape)}")

    # Ensure dtype match for broadcast multiply.
    if h_post.dtype != y.dtype:
        h_post = h_post.to(dtype=y.dtype)

    return h_post.unsqueeze(-1) * y.unsqueeze(-2)


def mhc_update(x_stream: torch.Tensor, *, h_post: torch.Tensor, h_res: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Apply the residual mix + write-in update of Eq. (3):

      x_{l+1} = H_res x_l + H_post^T y

    where y = F(H_pre x_l, W_l).
    """
    return stream_mix(x_stream, h_res) + stream_write(y, h_post)


