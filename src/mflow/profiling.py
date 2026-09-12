"""Latency and peak memory measurement.

The edge-deployment claim in the paper rests on inference cost, so every forecaster call
is measured rather than estimated. Peak memory is taken from :mod:`tracemalloc` for the
Python heap and, when a CUDA device is in use, from ``torch.cuda.max_memory_allocated``,
because a foundation model's real footprint lives in the accelerator and not in the heap.
"""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator


@dataclass
class ResourceUsage:
    """Wall-clock latency and peak memory of a measured block."""

    latency_ms: float = 0.0
    peak_memory_mb: float = 0.0
    device: str = "cpu"

    def as_dict(self) -> dict[str, float | str]:
        """Flat mapping for the metrics table."""
        return {
            "latency_ms": round(self.latency_ms, 4),
            "peak_memory_mb": round(self.peak_memory_mb, 4),
            "device": self.device,
        }


@contextmanager
def measure(device: str = "cpu") -> Iterator[ResourceUsage]:
    """Measure wall-clock time and peak memory of the enclosed block.

    Args:
        device: ``cuda`` switches peak memory to the CUDA allocator's high-water mark.

    Yields:
        A :class:`ResourceUsage` populated when the block exits.

    Example:
        >>> with measure() as usage:
        ...     total = sum(range(1000))
        >>> usage.latency_ms >= 0.0
        True
    """
    usage = ResourceUsage(device=device)
    torch_cuda = None
    if device == "cuda":
        import torch

        if torch.cuda.is_available():
            torch_cuda = torch.cuda
            torch_cuda.synchronize()
            torch_cuda.reset_peak_memory_stats()

    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[1]

    start = time.perf_counter()
    try:
        yield usage
    finally:
        if torch_cuda is not None:
            torch_cuda.synchronize()
        usage.latency_ms = (time.perf_counter() - start) * 1000.0
        heap_peak = tracemalloc.get_traced_memory()[1]
        if started_tracing:
            tracemalloc.stop()
        if torch_cuda is not None:
            usage.peak_memory_mb = torch_cuda.max_memory_allocated() / 1024**2
        else:
            usage.peak_memory_mb = max(0.0, heap_peak - baseline) / 1024**2
