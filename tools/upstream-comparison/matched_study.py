#!/usr/bin/env python3
"""Export, validate, and time a matched workload with bounded processes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import median
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parent / "taylor-comparison" / "run.py"


def invoke(command, receipt, timeout=300, inputs=(), cpus=None):
    flags = [sys.executable, str(RUNNER), "--output", str(receipt), "--resume",
             "--timeout", str(timeout), "--memory-mib", "4096"]
    for path in inputs:
        flags += ["--input", str(path)]
    if cpus:
        command = ["taskset", "-c", cpus, *command]
    result = subprocess.run([*flags, "--", *map(str, command)], check=False)
    record = json.loads(receipt.read_text())
    if result.returncode != 0 or record["status"] != "success":
        raise RuntimeError(f"{receipt.name} {record['status']}: {record['stderr_tail'][-1500:]}")
    return record


def parse(text):
    rows, points = {}, {}
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if "ns_per_call" in row:
            rows[row["operation"]] = row
        if "values" in row:
            points[(row["operation"], row["case"])] = row["values"]
    if not rows:
        raise RuntimeError("benchmark produced no timing records")
    return rows, points


def tolerance(operation):
    if operation == "m2e":
        return 5e-8
    if operation == "udp-numerical":
        return 4e-5
    if operation in ("jacobian", "udp-analytic"):
        return 1e-6
    return 1e-8


def max_error(a, b):
    if len(a) != len(b) or any(v is None for v in a+b):
        return None
    return max((abs(x-y)/max(1.,abs(y)) for x,y in zip(a,b)), default=0.)


def paired_interval(ratios):
    if len(ratios) < 2:
        return None
    rng = random.Random(9503)
    samples = sorted(median(rng.choices(ratios, k=len(ratios))) for _ in range(4000))
    return [samples[100], samples[3899]]


def benchmark(args, suite, count):
    directory = args.output / suite
    directory.mkdir(parents=True, exist_ok=True)
    fixture = directory / "inputs.bin"
    invoke([args.exporter, suite, count, fixture], directory/"export.json", args.timeout)
    if not fixture.is_file():
        # The runner's receipt can outlive a deleted output.
        (directory/"export.json").unlink()
        invoke([args.exporter, suite, count, fixture], directory/"export.json", args.timeout)
    values = 1 if suite.startswith("sf") else min(8,count)
    repeats = 2048 if suite=="sf-simple" else max(1, min(1024, 1024//count))
    if suite.startswith(("sf","zoh")) and suite!="sf-simple":
        repeats = 1
    options = [f"--passes={args.passes}", f"--repetitions={repeats}",
               f"--values={values}", f"--zoh-derivative-tol={args.derivative_tol}"]
    observed = []
    for k in range(args.rounds):
        arms = []
        for index, (label,binary) in enumerate((
                ("rust",args.rust),("upstream",args.native),("upstream",args.native),("rust",args.rust))):
            receipt = directory / f"{k:02d}-{index}-{label}.json"
            result = invoke([binary,fixture,*options],receipt,args.timeout,[fixture],args.cpu)
            arms.append(parse(result["stdout_tail"]))
        observed.append(arms)
    first_rust, first_upstream = observed[0][:2]
    rows = []
    for name, r in first_rust[0].items():
        u = first_upstream[0][name]
        keys = [key for key in first_rust[1] if key[0]==name]
        errors = [max_error(first_rust[1][key],first_upstream[1][key])
                  for key in keys if key in first_upstream[1]]
        error = max(errors,default=None) if all(x is not None for x in errors) else None
        ratios, rust_ns, upstream_ns = [], [], []
        for arms in observed:
            rust = median([s for i in (0,3) for s in arms[i][0][name]["ns_per_call"]])
            upstream = median([s for i in (1,2) for s in arms[i][0][name]["ns_per_call"]])
            ratios.append(upstream/rust)
            rust_ns.append(rust)
            upstream_ns.append(upstream)
        ok = (r["failed"]==u["failed"]==0 and r["output_scalars"]==u["output_scalars"]
              and error is not None and len(errors)==len(keys) and error<=tolerance(name))
        rows.append({"operation":name,"cases":r["cases"],"rust_failed":r["failed"],
            "upstream_failed":u["failed"],"output_scalars":[r["output_scalars"],u["output_scalars"]],
            "sampled_cases":len(errors),"scaled_difference":error,"gate":tolerance(name),
            "status":"passed" if ok else "discrepancy",
            "rust_round_median_ns":rust_ns,"upstream_round_median_ns":upstream_ns,
            "upstream_over_rust":ratios, "paired_interval_95":paired_interval(ratios)})
    return {"suite":suite,"fixture_sha256":hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "status":"passed" if all(r["status"]=="passed" for r in rows) else "discrepancy",
            "rows":rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("exporter","rust","native","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--profile",choices=("smoke","full"),default="smoke")
    parser.add_argument("--only",action="append")
    parser.add_argument("--rounds",type=int,default=1)
    parser.add_argument("--passes",type=int,default=5)
    parser.add_argument("--timeout",type=float,default=300)
    parser.add_argument("--cpu")
    parser.add_argument("--derivative-tol",type=float,default=1e-8)
    args=parser.parse_args()
    if not 1<=args.rounds<=20:
        parser.error("rounds must lie in 1..=20")
    spec=json.loads((HERE/"specs.json").read_text())
    suites=args.only or [s for s in spec["full"] if not s.startswith("sf-conv")]
    args.output.mkdir(parents=True,exist_ok=True)
    report={"schema_version":1,"profile":args.profile,"rounds":args.rounds,"cpu":args.cpu,
            "upstream_revision":spec["upstream"],"derivative_tolerance":args.derivative_tol,
            "binaries":{name:hashlib.sha256(getattr(args,name).read_bytes()).hexdigest()
                         for name in ("exporter","rust","native")},"suites":[]}
    for suite in suites:
        count=spec["full"][suite]
        if args.profile=="smoke":
            count=min(count,spec["sampling"]["smoke_max_cases"])
        try:
            result=benchmark(args,suite,count)
        except (OSError,RuntimeError,ValueError,KeyError) as exc:
            result={"suite":suite,"status":"failed","reason":str(exc)}
        report["suites"].append(result)
        temp=args.output/"report.tmp"
        temp.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        os.replace(temp,args.output/"report.json")
        print(suite,result["status"],flush=True)
    return int(any(s["status"]=="failed" for s in report["suites"]))


if __name__=="__main__":
    raise SystemExit(main())
