from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass


@dataclass
class WallTimer:
    name: str
    seconds: float = 0.0


@contextlib.contextmanager
def timed(timer: WallTimer):
    """
    Very small helper for profiling without external deps.

    Intended for quick bottleneck checks on Thor/Tegra (where installing profilers can be annoying).
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timer.seconds += time.perf_counter() - t0


