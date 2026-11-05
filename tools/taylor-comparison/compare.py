#!/usr/bin/env python3
"""Compare identical Taylor benchmark cases in independent, bounded ABBA rounds."""

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile

from run import digest, run


def rows(result):
    if result["status"] != "success":
        raise RuntimeError(f"{result['command']}: {result['status']}\n{result['stderr_tail']}")
    records = {}
    for row in csv.DictReader(io.StringIO(result["stdout_tail"])):
        key = (row["model"], row["method"], float(row["tolerance"]))
        elapsed = float(row["median_us"])
        if key in records or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"invalid or duplicated timing: {row}")
        records[key] = row
    if not records:
        raise ValueError("benchmark produced no timing rows")
    return records


def summarize(rounds):
    keys = rounds[0][0].keys()
    if any(records.keys() != keys for group in rounds for records in group):
        raise ValueError("benchmark fixture sets differ")
    rng = random.Random(42)
    summary = []
    for key in sorted(keys):
        baseline = []
        candidate = []
        for first, second, third, fourth in rounds:
            baseline.append(statistics.median(float(run[key]["median_us"]) for run in (first, fourth)))
            candidate.append(statistics.median(float(run[key]["median_us"]) for run in (second, third)))
        pairs = [a / b for a, b in zip(baseline, candidate)]
        interval = None
        if len(pairs) >= 3:
            bootstrap = sorted(statistics.median(rng.choices(pairs, k=len(pairs))) for _ in range(2000))
            interval = [bootstrap[50], bootstrap[1949]]
        summary.append({
            "model": key[0], "method": key[1], "tolerance": key[2],
            "baseline_us": statistics.median(baseline),
            "candidate_us": statistics.median(candidate),
            "speedup": statistics.median(pairs), "round_ratios": pairs,
            "round_bootstrap_95": interval,
            "baseline_max_error": max(float(group[i][key]["scaled_endpoint_error"])
                                      for group in rounds for i in (0, 3)),
            "candidate_max_error": max(float(group[i][key]["scaled_endpoint_error"])
                                       for group in rounds for i in (1, 2)),
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpus", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("benchmark_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 20 or not 0 < args.timeout < math.inf:
        parser.error("expected 1–20 rounds and a positive finite per-pass timeout")
    tail = args.benchmark_arguments
    if tail[:1] == ["--"]:
        tail = tail[1:]
    binaries = [args.baseline.resolve(), args.candidate.resolve()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = {"schema_version": 1, "binary_sha256": [digest(p) for p in binaries],
                "affinity": args.cpus, "arguments": tail, "rounds": [], "status": "failed"}
    parsed = []
    memory = 4 * 1024**3
    limit = Path("/sys/fs/cgroup/memory.max")
    if limit.exists() and (value := limit.read_text().strip()).isdecimal():
        memory = min(memory, int(value) // 2)
    try:
        for _ in range(args.rounds):
            group = []
            evidence["rounds"].append(group)
            for build in (0, 1, 1, 0):
                result = run(["taskset", "-c", args.cpus, str(binaries[build]), *tail],
                             args.timeout, memory, args.output)
                group.append(result)
                if result["status"] != "success":
                    rows(result)  # Raise after saving this failure to the evidence.
            parsed.append([rows(result) for result in group])
        evidence["summary"] = summarize(parsed)
        evidence["status"] = "success"
    finally:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as out:
            json.dump(evidence, out, indent=2, allow_nan=False)
            out.write("\n")
        os.replace(out.name, args.output)
    print(f"{args.output}: {len(evidence['summary'])} cases, {args.rounds} ABBA rounds")


if __name__ == "__main__":
    main()
