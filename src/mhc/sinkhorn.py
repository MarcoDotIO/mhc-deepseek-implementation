from __future__ import annotations

import torch


def sinkhorn_knopp(
    logits: torch.Tensor,
    *,
    tmax: int = 20,
    eps: float = 1e-8,
    clamp_min: float = 0.0,
) -> torch.Tensor:
    """
    Entropic projection to the **Birkhoff polytope** (doubly-stochastic matrices) via
    the **Sinkhorn-Knopp** algorithm.

    This matches the paper's description:
      - Start from a positive matrix M^(0) = exp( H~_res )        (Eq. 8/9)
      - Iterate: M^(t) = T_r( T_c( M^(t-1) ) )                    (Eq. 9)
        where T_r / T_c normalize rows / columns to sum to 1.
      - Use tmax = 20 in practice (paper default).

    In the paper, the constrained residual mapping H_res is:
      H_res = Sinkhorn-Knopp( H~_res )                            (Eq. 8)

    Args:
        logits: Tensor with shape [..., n, n]. Interpreted as H~_res in the paper.
        tmax: Number of Sinkhorn iterations (paper uses 20).
        eps: Small constant to avoid division by zero.
        clamp_min: Optional lower bound applied after exp() to avoid exact zeros during normalization.
            Use 0.0 to allow exact sparsity / permutation-matrix behavior.

    Returns:
        Tensor with shape [..., n, n] approximately doubly-stochastic (rows and columns ~ 1).
    """
    if logits.ndim < 2 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError(f"logits must have shape [..., n, n], got {tuple(logits.shape)}")

    # Do the projection in float32 for stability, regardless of input dtype.
    log_m = logits.float()

    # Normalization is scale-invariant: subtract a per-matrix max to avoid overflow.
    log_m = log_m - log_m.amax(dim=(-2, -1), keepdim=True)

    # Log-space Sinkhorn: dividing by sums becomes subtracting logsumexp.
    # This is numerically more stable than alternating exp-space divisions.
    for _ in range(tmax):
        log_m = log_m - torch.logsumexp(log_m, dim=-1, keepdim=True)  # rows
        log_m = log_m - torch.logsumexp(log_m, dim=-2, keepdim=True)  # cols

    m = torch.exp(log_m)

    # Optional clamp to avoid exact zeros (helps some downstream ops), then renormalize once.
    if clamp_min is not None and clamp_min > 0:
        m = m.clamp_min(clamp_min)
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)

    return m


