"""Measure Python wrapper latency and native batch throughput.

Run against a release-mode installed wheel:
    python python/benchmarks/wrapper_overhead.py
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import numpy as np

import pykep_rust as pk


def median_seconds(operation: Callable[[], object], repeats: int) -> float:
    """Return the median elapsed time without I/O in the measured region."""
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - started) * 1e-9)
    return statistics.median(samples)


def main() -> None:
    """Run cheap-call and propagation throughput comparisons."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.size <= 0 or args.repeats <= 0:
        parser.error("--size and --repeats must be positive")

    values = [1e-12] * args.size
    states = np.tile(
        np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
        (args.size, 1),
    )
    times = np.full(args.size, 0.1, dtype=np.float64)

    scalar_stumpff = median_seconds(
        lambda: [pk.stumpff_c(value) for value in values], args.repeats
    )
    batch_stumpff = median_seconds(
        lambda: pk.stumpff_c_batch(values), args.repeats
    )
    scalar_propagation = median_seconds(
        lambda: [
            pk.propagate_lagrangian(state, duration, 1.0)
            for state, duration in zip(states, times, strict=True)
        ],
        args.repeats,
    )
    batch_propagation = median_seconds(
        lambda: pk.propagate_lagrangian_batch(states, times, 1.0),
        args.repeats,
    )

    print(f"items: {args.size:,}; median of {args.repeats}")
    print(
        "stumpff_c scalar: "
        f"{1e9 * scalar_stumpff / args.size:.1f} ns/item; "
        f"batch: {1e9 * batch_stumpff / args.size:.1f} ns/item; "
        f"speedup {scalar_stumpff / batch_stumpff:.2f}x"
    )
    print(
        "Lagrange scalar: "
        f"{1e6 * scalar_propagation / args.size:.2f} us/item; "
        f"batch: {1e6 * batch_propagation / args.size:.2f} us/item; "
        f"speedup {scalar_propagation / batch_propagation:.2f}x"
    )


if __name__ == "__main__":
    main()
