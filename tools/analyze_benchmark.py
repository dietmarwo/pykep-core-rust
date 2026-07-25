"""Summarize release benchmark CSV files with deterministic bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile from sorted values."""
    ordered = sorted(values)
    location = fraction * (len(ordered) - 1)
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def median_interval(values: list[float]) -> tuple[float, float]:
    """Return a fixed-seed percentile-bootstrap 95% median interval."""
    generator = random.Random(0x5EED)
    medians = [
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(20_000)
    ]
    return percentile(medians, 0.025), percentile(medians, 0.975)


def read_samples(path: Path) -> dict[str, list[float]]:
    """Load benchmark samples grouped by workload."""
    groups: dict[str, list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            groups[row["workload"]].append(float(row["ns_per_call"]))
    return groups


def main() -> None:
    """Print a Markdown distribution table for labelled CSV inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        nargs="+",
        metavar="LABEL=CSV",
        help="benchmark label and CSV path",
    )
    args = parser.parse_args()
    print("| Implementation | Workload | N | Mean | Median | P05–P95 | Median 95% CI |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for item in args.input:
        label, separator, filename = item.partition("=")
        if not separator:
            parser.error(f"expected LABEL=CSV, got {item!r}")
        for workload, values in sorted(read_samples(Path(filename)).items()):
            low, high = median_interval(values)
            print(
                f"| {label} | `{workload}` | {len(values)} | "
                f"{statistics.mean(values):.2f} ns | "
                f"{statistics.median(values):.2f} ns | "
                f"{percentile(values, 0.05):.2f}–"
                f"{percentile(values, 0.95):.2f} ns | "
                f"{low:.2f}–{high:.2f} ns |"
            )


if __name__ == "__main__":
    main()
