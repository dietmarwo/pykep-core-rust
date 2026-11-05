#!/usr/bin/env python3
"""Evaluate a mission or replay a published TOPS decision vector."""

import argparse
import json
from pathlib import Path

import numpy as np

import pykep_rust as pk
from pykep_rust.trajopt import gym


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="gym.eve_mga1dsm")
    parser.add_argument("--nseg", type=int, default=None)
    parser.add_argument("--solution", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    options = {"nseg": args.nseg}
    x = None
    if args.solution:
        solution = json.loads(args.solution.read_text())
        args.problem = solution["id"]
        options = solution["options"]
        x = solution["x"]
    problem = gym.Problem(args.problem, **options)
    if x is None:
        x = np.mean(problem.get_bounds(),axis=0).tolist()
    values = problem.evaluate(x)
    decoded = problem.decode(x, args.samples)
    result = {"problem":problem.metadata(),"x":x,"evaluation":values,"trajectory":decoded}
    if args.problem.startswith("tops."):
        result["forward_residual"] = problem.forward_residual(x)
        if args.solution:
            assert max(map(abs,result["forward_residual"])) <= 1e-6
            assert max(map(abs,values["equalities"]),default=0.) <= 1e-6
            assert max(values["inequalities"],default=0.) <= 1e-8
    else:
        errors = []
        for arc in decoded["trajectory"]["arcs"]:
            end = pk.propagate_lagrangian(arc["start"], arc["duration_seconds"], decoded["central_mu_si"])
            ref = np.array(arc["end"])
            errors.append(float(np.max(np.abs(end-ref)/np.maximum(1.,np.abs(ref)))))
        result["maximum_scaled_arc_closure"] = max(errors,default=0.)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({key:result[key] for key in ("evaluation","forward_residual","maximum_scaled_arc_closure")
                      if key in result},indent=2))


if __name__ == "__main__":
    main()
