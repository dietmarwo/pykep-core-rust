#!/usr/bin/env python3
"""Check complete Rust gym vectors against the pinned PyKEP source."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pykep_rust.trajopt import gym

PIN = "c01e19debad5ea22b7079189f786f473046a79d5"
FAMILIES = {"twobody": "twobody", "mee": "mee", "cr3bp": "cr3bp", "sail": "ss"}


def upstream_problem(pk, id, nseg, encoding, cut=0.5, direction_inequalities=None):
    if id.startswith("gym."):
        return getattr(pk.trajopt.gym, id[4:])
    _, family, case, *moving = id.split(".")
    constructor = "tops_" + FAMILIES[family] + ("_mb" if moving else "")
    options = {"nseg": nseg, "time_encoding": encoding, "cut": cut}
    if direction_inequalities is not None and (family != "sail" or moving):
        options["inequalities_for_tc"] = direction_inequalities
    return getattr(pk.trajopt.gym, constructor)(case, **options)


def reference_derivative_tolerance(pk, ref, family, tolerance):
    """Retain the source's nominal integrator and parameters in a tighter oracle."""
    model = {"twobody": "kep", "mee": "eq", "cr3bp": "cr3bp", "sail": "ss"}[family]
    old = ref.udp.leg
    var = getattr(pk.ta, f"get_zoh_{model}_var")(tolerance)
    var.pars[:] = old.ta_var.pars
    ref.udp.leg = pk.leg.zoh(old.state0, old.controls, old.state1, old.tgrid, old.cut,
                            (old.ta, var), old.max_steps, old.dim_dynamics, old.dim_controls)


def regularize_mee_endpoints(pk, ref, case):
    """Use the source's symbolic MEE conversion without the classical-element detour."""
    raw = pk.trajopt.gym.tops_mee_json[case]
    udp = ref.udp
    mu = udp.L * udp.V**2
    for attribute, state, days in (
        ("pls", raw["state_s"], 0.),
        ("plf", raw["state_f"], raw["tof_bounds"][0] * raw["TIME"] / pk.DAY2SEC),
    ):
        mee = np.array(state)
        mee[0] *= raw["L"] / udp.L
        posvel = np.asarray(udp.state2cart(mee)).reshape(2, 3) * np.array([[udp.L], [udp.V]])
        body = pk.udpla.keplerian(pk.epoch(days), posvel, mu)
        setattr(udp, attribute, pk.planet(body))


def finite_difference_columns(problem, x):
    """Richardson extrapolation of the source fitness, including fixed columns."""
    columns = []
    for i, value in enumerate(x):
        perturbation = np.zeros_like(x)
        perturbation[i] = 1e-5 * max(1., min(4., abs(value)))
        upper = np.asarray(problem.fitness(x + perturbation))
        lower = np.asarray(problem.fitness(x - perturbation))
        fine_upper = np.asarray(problem.fitness(x + .5 * perturbation))
        fine_lower = np.asarray(problem.fitness(x - .5 * perturbation))
        columns.append((8 * (fine_upper - fine_lower) - (upper - lower)) / (6 * perturbation[i]))
    return np.asarray(columns).T.ravel()


def points(spec, seed, count):
    rng = np.random.default_rng(seed)
    bounds = np.asarray(spec["bounds"])
    low, high = bounds.T
    result = []
    for index in range(count):
        x = (low + high) * 0.5 if index == 0 else low + rng.uniform(.18, .82, len(low)) * (high-low)
        for i, name in enumerate(spec["decisions"]):
            if name.startswith("throttle_"):
                x[i] = low[i] + (.2 + .1 * rng.random()) * (high[i] - low[i])
            if name.startswith("direction_") and name.endswith("_x"):
                d = rng.normal(size=3)
                x[i:i+3] = d / np.linalg.norm(d) * .8
            if name in ("departure_direction_x", "arrival_direction_x"):
                x[i:i+3] = [.8, .3, -.25]
            if name.startswith("alpha_"):
                x[i] = .18 + .04 * rng.normal()
            if name.startswith("beta_"):
                x[i] = 1.3 + .1 * rng.normal()
        result.append(x)
    return result


def error(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"invalid comparison {a.shape} vs {b.shape}")
    return float(np.max(np.abs(a-b)/np.maximum(1., np.abs(b)), initial=0.))


def difference(problem, x, jacobian, seed):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=len(x))
    d *= np.minimum(np.maximum(np.abs(x), .05), 2.)
    d /= np.linalg.norm(d)
    analytic = jacobian.reshape(-1, len(x)) @ d
    errors = []
    for h in (1e-4, 1e-5, 1e-6):
        measured = (np.asarray(problem.fitness(x+h*d)) - problem.fitness(x-h*d))/(2*h)
        errors.append(error(analytic, measured))
    return errors


def timed(fn, x, calls, passes=7):
    for _ in range(3):
        fn(x)
    times = []
    for _ in range(passes):
        start = perf_counter()
        for _ in range(calls):
            fn(x)
        times.append((perf_counter()-start)*1e6/calls)
    return times


def row(pk, id, encoding, args):
    nseg = args.nseg
    start = perf_counter()
    problem = gym.Problem(id, nseg=nseg, time_encoding=encoding, cut=args.cut,
                          inequalities_for_tc=args.direction_inequalities, backend=args.backend,
                          tolerance=args.rust_tolerance, derivative_tolerance=args.rust_derivative_tol)
    rust_setup = perf_counter()-start
    spec = problem.metadata()
    result = {"id": id, "encoding": encoding, "nseg": nseg,
              "rust_setup_seconds": rust_setup, "spec": spec,
              "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
              "formulation": spec["formulation"]}
    if id.startswith("tops.cr3bp.") and id.endswith(".moving") and encoding == "softmax":
        result["comparison"] = "rust extension, upstream constructor ignores softmax"
        sample = points(spec, 30, 1)[0]
        j = np.asarray(problem.gradient(sample))
        result["independent_gradient_errors"] = difference(problem, sample, j, 117)
        result["rust_fitness"] = problem.fitness(sample)
        result["status"] = "extension"
        return result
    start = perf_counter()
    ref = upstream_problem(pk, id, nseg, encoding, args.cut, args.direction_inequalities)
    regularized = args.regularized_mee and id.startswith("tops.mee.") and id.endswith(".moving")
    if regularized:
        regularize_mee_endpoints(pk, ref, id.split(".")[2])
        result["comparison"] = "upstream symbolic endpoint adapter and full finite differences"
    if id.startswith("tops.") and args.reference_derivative_tol is not None:
        reference_derivative_tolerance(pk, ref, id.split(".")[1], args.reference_derivative_tol)
    result["reference_setup_seconds"] = perf_counter()-start
    if problem.has_gradient():
        result["reference_tolerances"] = [ref.udp.leg.ta.tol, ref.udp.leg.ta_var.tol]
    actual_bounds = np.array(problem.get_bounds())
    ref_bounds = np.array(ref.get_bounds())
    result["bounds_error"] = error(actual_bounds, ref_bounds)
    result["dimensions_match"] = (problem.get_nobj(), problem.get_nec(), problem.get_nic()) == (
        getattr(ref, "get_nobj", lambda: 1)(), getattr(ref, "get_nec", lambda: 0)(), getattr(ref, "get_nic", lambda: 0)())
    vectors = []
    gradient_max = 0.
    frozen_winding = regularized and id.split(".")[2] in ("P5", "P8") and not args.off_anchor
    free = actual_bounds[0] < actual_bounds[1]
    if frozen_winding:
        result["gradient_scope"] = "free coordinates at fixed-time longitude winding transition"
    for i, x in enumerate(points(spec, 90210, args.samples)):
        if args.off_anchor and regularized:
            x[spec["decisions"].index("time_of_flight")] += 0.01
        d = {"x": x.tolist()}
        try:
            rust = problem.fitness(x)
            d["rust"] = rust
            reference = list(ref.fitness(x))
            d["reference"] = [float(v) if np.isfinite(v) else None for v in reference]
            d["fitness_error"] = error(rust, reference)
            if problem.has_gradient():
                jac = np.asarray(problem.gradient(x))
                jref = finite_difference_columns(ref, x) if regularized else np.asarray(ref.gradient(x)).flatten()
                d["gradient_error"] = error(jac, jref)
                if frozen_winding:
                    d["free_gradient_error"] = error(jac.reshape(-1, len(x))[:, free],
                                                     jref.reshape(-1, len(x))[:, free])
                gradient_max = max(gradient_max, d.get("free_gradient_error", d["gradient_error"]))
                d["independent_gradient_errors"] = difference(problem, x, jac, 117+i)
                d["gradient_shape"] = [len(rust), len(x)]
                if id.split(".")[2] == "P0":
                    d["reference_gradient"] = jref.tolist()
        except Exception as exc:
            d["error"] = str(exc)
        vectors.append(d)
    result["vectors"] = vectors
    fitness_tol = 1e-8 if problem.has_gradient() else 1e-10
    gradient_tol = 5e-5 if regularized else 1e-7
    result["gradient_gate"] = gradient_tol
    result["status"] = "failed" if any("error" in v for v in vectors) else "passed" if (
        result["bounds_error"] <= 2e-13 and result["dimensions_match"]
        and all(v["fitness_error"] <= fitness_tol for v in vectors)
        and gradient_max <= gradient_tol
    ) else "discrepancy"
    if args.timing and result["status"] == "passed" and not regularized:
        x = np.array(vectors[0]["x"])
        result["timings_us"] = {"fitness": {
            "rust": timed(problem.fitness, x, 64), "upstream": timed(ref.fitness, x, 64)
        }}
        if problem.has_gradient():
            result["timings_us"]["gradient"] = {
                "rust": timed(problem.gradient, x, 16), "upstream": timed(ref.gradient, x, 16)
            }
            result["timings_us"]["rust_fitness_and_gradient"] = timed(
                lambda d: (problem.fitness(d), problem.gradient(d)), x, 16)
            result["timings_us"]["rust_history_export_3_samples"] = timed(
                lambda d: problem.decode(d, 3), x, 8)
        population = np.asarray([v["x"] for v in vectors])
        batch = problem.fitness_batch(population, workers=1)
        np.testing.assert_array_equal(batch, [v["rust"] for v in vectors])
        result["timings_us"]["rust_batch_one_worker"] = timed(
            lambda pop: problem.fitness_batch(pop, workers=1), population, 16)
        result["work"] = problem.evaluate(x)["work"]
    return result


def launcher_rows(pk):
    launchers = pk.trajopt.launchers
    output = []
    for name in ("ariane5", "ariane64", "atlas501", "atlas551", "soyuzf", "falcon9"):
        call = getattr(launchers, name)
        for speed, dec in ((0., -90.), (2.5, 0.), (3.2, -13.), (4.17, 37.), (-1., 130.), (9., -100.)):
            ref = call(speed) if name in ("atlas551", "falcon9") else call(speed, dec)
            ref = float(np.asarray(ref).reshape(-1)[0])
            ours = gym.launcher_mass(name, speed, dec)
            output.append({"launcher":name,"speed":speed,"declination":dec,"rust":ours,
                           "upstream":ref,"scaled_error":error([ours],[ref])})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", action="append", help="Problem ID; repeat to select several.")
    parser.add_argument("--nseg", type=int, default=4)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--cut", type=float, default=0.5)
    parser.add_argument("--direction-inequalities", action=argparse.BooleanOptionalAction)
    parser.add_argument("--backend", choices=("taylor", "dop853"), default="taylor")
    parser.add_argument("--rust-tolerance", type=float, default=1e-13)
    parser.add_argument("--rust-derivative-tol", type=float, default=1e-13)
    parser.add_argument("--reference-derivative-tol", type=float)
    parser.add_argument("--regularized-mee", action="store_true",
                        help="Separate reference arm for moving MEE with regularized endpoints and numerical gradients")
    parser.add_argument("--off-anchor", action="store_true",
                        help="Probe an off-box arrival time on a smooth MEE winding branch")
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--allow-historical-wheel", action="store_true")
    parser.add_argument("--isolated-exit", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.nseg < 1:
        parser.error("sample and segment counts must be positive")
    import os
    import sys
    import pykep as pk
    import heyoka
    if not pk.__version__.startswith("3.1.") and not args.allow_historical_wheel:
        parser.error("the current comparison requires the pinned PyKEP 3.1 source build")
    ids = args.only or [i for i in gym.ids() if not args.classical_only or i.startswith("gym.")]
    out = {"schema_version": 1, "pinned_definition": PIN,
           "pykep_version":pk.__version__, "heyoka_py_version":heyoka.__version__,
           "pykep_module_sha256":hashlib.sha256(Path(pk.__file__).read_bytes()).hexdigest(),
           "pykep_core_sha256":hashlib.sha256(Path(pk.core.__file__).read_bytes()).hexdigest(),
           "reference_derivative_tolerance":args.reference_derivative_tol or 1e-8,
           "problems": [], "launchers": launcher_rows(pk)}
    if args.isolated_exit:
        # Retain integrator owners until process exit on historical wheels.
        keep = []
        original = pk.trajopt.gym
        for name in [s for s in dir(original) if s.startswith("tops_") and not s.endswith("_json")]:
            ctor = getattr(original, name)
            def retain(*a, _ctor=ctor, **kw):
                p = _ctor(*a, **kw)
                keep.append(p)
                return p
            setattr(original, name, retain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for id in ids:
        for encoding in (("uniform",) if id.startswith("gym.") else ("uniform","softmax")):
            try:
                result = row(pk, id, encoding, args)
            except Exception as exc:
                result = {"id":id,"encoding":encoding,"status":"failed","error":str(exc)}
            out["problems"].append(result)
            temp = args.output.with_suffix(".tmp")
            temp.write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")
            temp.replace(args.output)
            print(id, encoding, result["status"], flush=True)
    if args.isolated_exit:
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
