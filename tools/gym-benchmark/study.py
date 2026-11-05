#!/usr/bin/env python3
"""Run paired native TOPS optimizers at equal logical work and equal wall budgets."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "upstream-comparison"))
from matched_study import invoke

CASES = (
    ("tops.twobody.P4", 16, "uniform"),
    ("tops.mee.P8", 24, "softmax"),
    ("tops.cr3bp.P0", 30, "uniform"),
    ("tops.sail.P4", 24, "softmax"),
)
PILOT = (11, 17, 42)
HELD_OUT = tuple(range(21001, 21021))


def arm(args, case, seed, backend, budget):
    id, segments, encoding = case
    key = f"{id}-{seed}-{backend}-{budget}"
    folder = args.output / key
    run_file = folder / "run.json"
    command = [args.binary, "optimize", "--problem", id, "--nseg", segments,
               "--time-encoding", encoding, "--optimizer", "de-cma", "--backend", backend,
               "--seed", seed, "--workers", 1, "--output", folder]
    command += ["--evaluations", 128] if budget == "evaluations" else [
        "--evaluations", 100000, "--seconds", .08]
    try:
        invoke(command, args.output / f"{key}.json", 30, [Path(__file__)], args.cpu)
        run = json.loads(run_file.read_text())
        spec = (folder / "spec.json").read_bytes()
    except (OSError, ValueError, RuntimeError) as error:
        return {"status": "failed", "reason": str(error)}
    best = run["best"]
    row = {**{key: run[key] for key in (
        "status", "requested_evaluations", "actual_evaluations", "elapsed_seconds",
        "feasible_results", "failed_evaluations",
    )}, "spec_sha256": hashlib.sha256(spec).hexdigest(), "pykep": run["pykep"], "best": best}
    if run["feasible_results"]:
        try:
            receipt = invoke([args.binary, "validate", "--run", run_file],
                             args.output / f"{key}-validation.json", 30, [run_file], args.cpu)
            checked = json.loads(receipt["stdout_tail"])
            row["forward_residual"] = checked["forward_residual"]
            row["independently_feasible"] = (
                max(map(abs, checked["forward_residual"])) <= 1e-6
                and max(map(abs, checked["evaluation"]["equalities"]), default=0.) <= 1e-6
                and max(checked["evaluation"]["inequalities"], default=0.) <= 1e-8
            )
        except (RuntimeError, KeyError, ValueError) as error:
            row["validation_error"] = str(error)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "cpu": args.cpu, "pilot": args.pilot,
              "seeds": PILOT if args.pilot else HELD_OUT, "cases": CASES,
              "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(), "runs": []}
    for case in CASES:
        for seed in report["seeds"]:
            backends = ("taylor", "dop853") if seed % 2 else ("dop853", "taylor")
            for budget in ("evaluations", "wall"):
                for backend in backends:
                    row = arm(args, case, seed, backend, budget)
                    report["runs"].append({"case": case[0], "seed": seed, "backend": backend,
                                           "budget": budget, **row})
            temp = args.output / "report.tmp"
            temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temp.replace(args.output / "report.json")
            print(case[0], seed, flush=True)


if __name__ == "__main__":
    main()
