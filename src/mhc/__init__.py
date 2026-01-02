"""
PyTorch implementation of **mHC (Manifold-Constrained Hyper-Connections)**.

Paper: "mHC: Manifold-Constrained Hyper-Connections" (arXiv:2512.24880v1)
  - https://arxiv.org/abs/2512.24880v1
  - https://doi.org/10.48550/arXiv.2512.24880
"""

from .mhc import MhcProjector
from .sinkhorn import sinkhorn_knopp

__all__ = ["MhcProjector", "sinkhorn_knopp"]


