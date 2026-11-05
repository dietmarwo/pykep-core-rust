#!/usr/bin/env python3
"""Compare official PyKEP and Rust Python calls on identical fixed fixtures."""

import argparse
import csv
import importlib.metadata
import json
import os
import statistics
import sys
import time
from pathlib import Path

import heyoka
import numpy as np
import pykep

FACTORIES = {
    "kepler": "kep", "cr3bp": "cr3bp", "bcp": "bcp", "zoh_kepler": "zoh_kep",
    "zoh_cr3bp": "zoh_cr3bp", "zoh_equinoctial": "zoh_eq", "solar_sail": "zoh_ss",
    "cartesian_mass": "pc", "cartesian_time": "pc",
    "equinoctial_mass": "peq", "equinoctial_time": "peq",
}


def upstream(fixture, tol):
    model = fixture["model"]
    factory = getattr(pykep.ta, "get_" + FACTORIES[model])
    started = time.perf_counter()
    if model.startswith(("cartesian_", "equinoctial_")):
        optimality = getattr(pykep.optimality_type, model.split("_")[1].upper())
        integrator = factory(tol, optimality)
    else:
        integrator = factory(tol)
    setup = time.perf_counter() - started
    initial = np.array(fixture["state"], dtype=float)
    parameters = fixture["parameters"]
    if integrator.pars.size != len(parameters):
        raise ValueError(f"{model}: expected {integrator.pars.size} upstream parameters, got {len(parameters)}")
    integrator.pars[:] = parameters
    final_time = fixture["final_time"]

    def run():
        integrator.time = 0.
        integrator.state[:] = initial
        outcome, *_ = integrator.propagate_until(final_time)
        if outcome != heyoka.taylor_outcome.time_limit:
            raise RuntimeError(f"{model}: upstream termination {outcome}")
        return integrator.state

    return run, setup


def rust_call(fixture, tol, method):
    import pykep_rust as rust

    model = fixture["model"]
    state, pars, target = (fixture[key] for key in ("state", "parameters", "final_time"))
    options = dict(relative_tolerance=tol, absolute_tolerance=tol, method=method)
    if model in {"kepler", "cr3bp", "bcp"}:
        fn = getattr(rust, "propagate_" + ("kepler_dynamics" if model == "kepler" else model))
        return lambda: fn(state, target, *pars, **options)
    if model in {"zoh_kepler", "zoh_equinoctial", "zoh_cr3bp"}:
        fn = getattr(rust, "propagate_" + model)
        thrust, x, y, z, *constants = pars
        return lambda: fn(state, [0., target], [[thrust, x, y, z]], *constants, **options)
    if model == "solar_sail":
        return lambda: rust.propagate_zoh_solar_sail(state, [0., target], [pars[:2]], pars[2], **options)
    family, objective = model.split("_")
    opt = rust.Optimality.Mass if objective == "mass" else rust.Optimality.Time
    fn = getattr(rust, "propagate_pontryagin_" + family)
    return lambda: fn(state, target, opt, pars, **options)


def samples(run, repeats):
    run()
    times = []
    for _ in range(7):
        start = time.perf_counter_ns()
        for _ in range(repeats):
            run()
        times.append((time.perf_counter_ns() - start) / (1000 * repeats))
    return min(times), statistics.median(times), max(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("--repeats", type=int, default=256)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--isolated-exit", action="store_true", help=(
        "retain upstream integrators until process exit to bypass a wheel teardown crash"
    ))
    args = parser.parse_args()
    if args.reference_only and args.references is None:
        parser.error("--reference-only requires --references")
    if args.repeats <= 0:
        parser.error("repeats must be positive")
    print(f"pykep={importlib.metadata.version('pykep')}, heyoka={heyoka.__version__}", file=sys.stderr)
    out = csv.writer(sys.stdout)
    out.writerow(("model", "backend", "tolerance", "min_us", "median_us", "max_us",
                  "scaled_endpoint_error", "setup_seconds", "status"))
    retained = []
    references = []
    for line in args.fixtures.read_text().splitlines():
        fixture = json.loads(line)
        model = fixture["model"]
        try:
            reference, _ = upstream(fixture, 2.22e-16)
            if args.isolated_exit:
                retained.append(reference)
            reference = reference().copy()
            references.append({**fixture, "reference": reference.tolist()})
            if args.reference_only:
                continue
        except (ValueError, RuntimeError, AttributeError) as error:
            out.writerow((model, "upstream", "", "", "", "", "", "", "failed"))
            print(error, file=sys.stderr)
            continue
        for tolerance in (1e-9, 1e-12, 1e-14):
            for backend in ("upstream", "taylor", "dop853"):
                try:
                    run, setup = upstream(fixture, tolerance) if backend == "upstream" else (
                        rust_call(fixture, tolerance, backend), 0)
                    if args.isolated_exit and backend == "upstream":
                        retained.append(run)
                    state = np.array(run())
                    error = float(np.max(np.abs(state - reference) / np.maximum(1., np.abs(reference))))
                    out.writerow((model, backend, tolerance, *samples(run, args.repeats), error, setup, "success"))
                except (ValueError, RuntimeError, TypeError, AttributeError) as error:
                    out.writerow((model, backend, tolerance, "", "", "", "", "", "failed"))
                    print(error, file=sys.stderr)
            sys.stdout.flush()
    if args.references:
        args.references.write_text(json.dumps({
            "schema_version": 1,
            "pykep_version": importlib.metadata.version("pykep"),
            "heyoka_version": heyoka.__version__,
            "tolerance": 2.22e-16,
            "cases": references,
        }, indent=2) + "\n")
    if args.isolated_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
