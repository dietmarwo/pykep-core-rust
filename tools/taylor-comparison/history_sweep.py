#!/usr/bin/env python3
"""Measure complete Python ZOH history calls against independent endpoint solves."""

import csv
import statistics
import sys
import time

import numpy as np
import pykep_rust as pk


def main():
    initial = [0.8, -0.2, 0.1, 0.03, 1.0, 0.02, 1.1]
    final = [0.4, 0.9, 0.08, -0.8, 0.3, -0.04, 1.0]
    controls = [[0.02, 0.3, -0.4, 0.5], [0.01, 0.0, 1.0, 0.0]]
    leg = pk.ZohLeg(pk.ZohModel.Kepler, initial, controls, final, [0.0, 0.1, 0.25], [0.01])
    writer = csv.writer(sys.stdout)
    writer.writerow(("model", "method", "tolerance", "min_us", "median_us", "max_us",
                     "scaled_endpoint_error", "samples_us"))
    reference_options = dict(relative_tolerance=1e-14, absolute_tolerance=1e-14, method="dop853")
    for points in (2, 10, 101, 1001):
        history = leg.state_history(points, method="taylor")
        expected = [[initial], [final]]
        for sample in range(1, points):
            fraction = sample / (points - 1)
            expected[0].append(pk.propagate_zoh_kepler(
                initial, [0.0, 0.1 * fraction], [controls[0]], 0.01, **reference_options,
            ))
            expected[1].append(pk.propagate_zoh_kepler(
                final, [0.25 - 0.15 * fraction, 0.25], [controls[1]], 0.01,
                backward=True, **reference_options,
            ))
        reference = np.asarray(expected)
        actual = np.asarray([history[0][0], history[1][0]])
        error = float(np.max(np.abs(actual - reference) / np.maximum(1.0, np.abs(reference))))
        if not np.isfinite(error) or error > 2e-11:
            raise RuntimeError(f"history {points}: scaled sample error {error}")
        samples = []
        for _ in range(7):
            started = time.perf_counter_ns()
            for _ in range(64):
                leg.state_history(points, method="taylor")
            samples.append((time.perf_counter_ns() - started) / 64000.0)
        writer.writerow((f"zoh_history_{points}", "taylor", 1e-12, min(samples), statistics.median(samples),
                         max(samples), error, "|".join(map(str, samples))))


if __name__ == "__main__":
    main()
