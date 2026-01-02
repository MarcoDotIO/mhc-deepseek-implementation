from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .rmsnorm import RMSNorm
from .sinkhorn import sinkhorn_knopp


@dataclass(frozen=True)
class MhcMappings:
    """
    Per-token mHC mappings.

    Shapes assume transformer-style batching:
      - H_pre:  (B, T, n)
      - H_post: (B, T, n)
      - H_res:  (B, T, n, n)
    """

    h_pre: torch.Tensor
    h_post: torch.Tensor
    h_res: torch.Tensor


class MhcProjector(nn.Module):
    """
    Compute mHC mappings (H_pre, H_post, H_res) from the current n-stream residual state.

    Key equations from the paper (`2512.24880v1.pdf` / arXiv:2512.24880v1):

    HC forward (Eq. 3):
      x_{l+1} = H_res x_l + H_post^T * F(H_pre x_l, W_l)

    mHC manifold constraint on H_res (Eq. 6):
      H_res is projected onto the Birkhoff polytope (doubly-stochastic matrices).

    Parameterization + projection (Eq. 7–9, formatting adapted):
      x'_l = RMSNorm(vec(x_l))
      H~_pre  = α_pre  * (x'_l Φ_pre)  + b_pre
      H~_post = α_post * (x'_l Φ_post) + b_post
      H~_res  = α_res  * mat(x'_l Φ_res) + b_res

      H_pre  = sigmoid(H~_pre)
      H_post = 2 * sigmoid(H~_post)
      H_res  = Sinkhorn-Knopp(H~_res)   # exp + alternating row/col normalization, tmax=20

    Notes:
      - The paper enforces **non-negativity** for H_pre/H_post via sigmoid (Section 4.1/4.2),
        and enforces **double stochasticity** for H_res via Sinkhorn-Knopp.
      - α_* are learnable gating scalars initialized to small values (paper Table 5 uses 0.01).
    """

    def __init__(
        self,
        *,
        n_streams: int,
        hidden_dim: int,
        tmax: int = 20,
        alpha_init: float = 0.01,
        rmsnorm_eps: float = 1e-6,
    ):
        super().__init__()
        if n_streams < 1:
            raise ValueError("n_streams must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")

        self.n = int(n_streams)
        self.c = int(hidden_dim)
        self.tmax = int(tmax)

        flat_dim = self.n * self.c
        self.rmsnorm = RMSNorm(flat_dim, eps=rmsnorm_eps, elementwise_affine=True)

        # Dynamic projection weights (Φ in the paper)
        self.phi_pre = nn.Parameter(torch.empty(flat_dim, self.n))
        self.phi_post = nn.Parameter(torch.empty(flat_dim, self.n))
        self.phi_res = nn.Parameter(torch.empty(flat_dim, self.n * self.n))

        # Static biases (b in the paper)
        self.b_pre = nn.Parameter(torch.zeros(self.n))
        self.b_post = nn.Parameter(torch.zeros(self.n))
        self.b_res = nn.Parameter(torch.zeros(self.n, self.n))

        # Learnable gating scalars (α in the paper)
        self.alpha_pre = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_post = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_res = nn.Parameter(torch.tensor(float(alpha_init)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Small init similar to transformer init scales.
        std = 0.02
        nn.init.normal_(self.phi_pre, mean=0.0, std=std)
        nn.init.normal_(self.phi_post, mean=0.0, std=std)
        nn.init.normal_(self.phi_res, mean=0.0, std=std)
        nn.init.zeros_(self.b_pre)
        nn.init.zeros_(self.b_post)
        nn.init.zeros_(self.b_res)

    @torch.no_grad()
    def init_gpt2_equivalence(
        self,
        *,
        offdiag_bias: float = -20.0,
        alpha: float = 0.0,
    ) -> None:
        """
        Initialize mappings to behave like a standard residual connection *when the stream is
        initialized as identical copies*.

        This mirrors the paper's ablation fallback defaults (Table 1):
          - H_pre: uniform 1/n
          - H_post: all ones
          - H_res: identity

        We achieve constant mappings by zeroing dynamic weights (Φ) and setting α=0 so only
        biases remain.
        """
        self.phi_pre.zero_()
        self.phi_post.zero_()
        self.phi_res.zero_()

        self.alpha_pre.fill_(alpha)
        self.alpha_post.fill_(alpha)
        self.alpha_res.fill_(alpha)

        # H_pre = sigmoid(b_pre) = 1/n  -> b_pre = logit(1/n)
        p = 1.0 / float(self.n)
        logit_p = math.log(p / (1.0 - p)) if p not in (0.0, 1.0) else 0.0
        self.b_pre.fill_(logit_p)

        # H_post = 2*sigmoid(b_post) = 1 -> b_post = 0
        self.b_post.zero_()

        # H_res = Sinkhorn(exp(b_res)) ~ I by making off-diagonal very small.
        self.b_res.fill_(offdiag_bias)
        self.b_res.diagonal().fill_(0.0)

    def forward(self, x_stream: torch.Tensor) -> MhcMappings:
        """
        Args:
            x_stream: (B, T, n, C)
        Returns:
            MhcMappings with shapes:
              - h_pre:  (B, T, n)
              - h_post: (B, T, n)
              - h_res:  (B, T, n, n)
        """
        if x_stream.ndim != 4:
            raise ValueError(f"x_stream must be (B,T,n,C), got {tuple(x_stream.shape)}")
        b, t, n, c = x_stream.shape
        if n != self.n or c != self.c:
            raise ValueError(
                f"Expected (B,T,n={self.n},C={self.c}), got (B,T,n={n},C={c})"
            )

        # Flatten stream into a vector, per token: vec(x_l) in Eq. (7).
        x_flat = x_stream.reshape(b * t, n * c)
        x_flat = self.rmsnorm(x_flat)

        # Dynamic + static mappings (Eq. 7)
        h_pre_tilde = self.alpha_pre * (x_flat @ self.phi_pre) + self.b_pre
        h_post_tilde = self.alpha_post * (x_flat @ self.phi_post) + self.b_post

        h_res_dyn = x_flat @ self.phi_res  # (BT, n*n)
        h_res_tilde = self.alpha_res * h_res_dyn.reshape(b * t, n, n) + self.b_res

        # Constrained mappings (Eq. 8)
        h_pre = torch.sigmoid(h_pre_tilde).reshape(b, t, n)
        h_post = (2.0 * torch.sigmoid(h_post_tilde)).reshape(b, t, n)
        h_res = sinkhorn_knopp(h_res_tilde.reshape(b, t, n, n), tmax=self.tmax)

        return MhcMappings(h_pre=h_pre, h_post=h_post, h_res=h_res)


