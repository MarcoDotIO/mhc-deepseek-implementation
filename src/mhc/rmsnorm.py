from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    RMSNorm (Zhang & Sennrich, 2019).

    The mHC paper uses RMSNorm for coefficient generation (Eq. 5 / Eq. 7), applied to the last
    dimension.
    """

    def __init__(self, dim: int, *, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        self.weight = nn.Parameter(torch.ones(self.dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {x.shape[-1]}")
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(rms + self.eps)
        if self.weight is not None:
            x_norm = x_norm * self.weight
        return x_norm.to(dtype=x.dtype)


