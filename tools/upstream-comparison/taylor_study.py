#!/usr/bin/env python3
"""Compare the eleven native Taylor fixtures against one tight endpoint reference."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import median

from matched_study import invoke, paired_interval


def fixtures(binary, output, selected):
    command = [binary, "--reference-fixtures"]
    if selected:
        command.append("--only=" + selected)
    receipt = invoke(command, output / "reference.json")
    records = [json.loads(line) for line in receipt["stdout_tail"].splitlines()]
    path = output / "fixtures.txt"
    with path.open("w") as stream:
        for case in records:
            state, parameters = case["state"], case["parameters"]
            print(case["model"], 0., case["final_time"], len(state), len(parameters),
                  *state, *parameters, *case["reference"], file=stream)
    return path, [case["model"] for case in records]


def rows(receipt):
    return {row["model"]: row for row in csv.DictReader(io.StringIO(receipt["stdout_tail"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("rust", "native", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--only", help="Comma-separated model prefixes")
    parser.add_argument("--cpu")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--tolerance", choices=("1e-9", "1e-12", "1e-14"), default="1e-12")
    parser.add_argument("--error-gate", type=float, default=1e-11)
    parser.add_argument("--repeat-scale", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 20 or not 1 <= args.repeat_scale <= 32:
        parser.error("invalid round or repeat count")
    args.output.mkdir(parents=True, exist_ok=True)
    fixture, models = fixtures(args.rust, args.output, args.only)
    if not models:
        parser.error("no models selected")
    repeat = f"--repeat-scale={args.repeat_scale}"
    rust = [args.rust, "--scalar-suite", "--taylor-only", f"--run-tolerance={args.tolerance}", repeat]
    if args.only:
        rust.append("--only=" + args.only)
    arms = {"rust": rust, "upstream": [args.native, fixture, f"--tolerance={args.tolerance}", repeat]}
    report = {"schema_version": 1, "tolerance": float(args.tolerance), "error_gate": args.error_gate,
              "cpu": args.cpu, "fixtures_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
              "raw": [], "summary": []}
    for k in range(args.rounds):
        order = list(arms) if k % 2 == 0 else list(reversed(arms))
        pair = {name: rows(invoke(arms[name], args.output / f"{k:02d}-{name}.json",
                                  200, [fixture], args.cpu)) for name in order}
        if any(set(data) != set(models) for data in pair.values()):
            raise RuntimeError("model mismatch between comparison arms")
        report["raw"].append(pair)
        report["summary"] = []
        for name in models:
            rust_rows = [p["rust"][name] for p in report["raw"]]
            upstream_rows = [p["upstream"][name] for p in report["raw"]]
            rt, ut = [[float(row["median_us"]) for row in group] for group in (rust_rows, upstream_rows)]
            err = [max(float(row["scaled_endpoint_error"]) for row in group)
                   for group in (rust_rows, upstream_rows)]
            ratios = [r / u for r, u in zip(rt, ut)]
            report["summary"].append({
                "model": name, "rust_median_us": median(rt), "upstream_median_us": median(ut),
                "rust_over_upstream": median(ratios), "paired_interval_95": paired_interval(ratios),
                "rust_endpoint_error": err[0], "upstream_endpoint_error": err[1],
                "status": "passed" if max(err) <= args.error_gate else "error_gate",
            })
        temp = args.output / "report.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(args.output / "report.json")
        print(f"Taylor comparison pair {k+1}/{args.rounds}", flush=True)


if __name__ == "__main__":
    main()
