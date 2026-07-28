#!/usr/bin/env python3
"""Generate independent heyoka reference states for the Taylor backend.

This is development tooling, not a build or runtime dependency. Install
heyoka.py explicitly, run this script from the repository root, and review the
fixture diff before committing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import heyoka as hy

OUTPUT = Path("crates/pykep-core/tests/data/taylor-heyoka-v1.json")
TOLERANCE = 1e-14


def propagate(
    name: str,
    system: list[tuple[hy.expression, hy.expression]],
    initial: list[float],
    parameters: list[float],
    final_time: float,
    *,
    stm: bool = False,
) -> dict[str, object]:
    equations: list[tuple[hy.expression, hy.expression]] | hy.var_ode_sys = system
    if stm:
        equations = hy.var_ode_sys(system, hy.var_args.vars, order=1)
    integrator = hy.taylor_adaptive(
        equations,
        initial,
        pars=parameters,
        tol=TOLERANCE,
        high_accuracy=True,
        compact_mode=True,
    )
    integrator.propagate_until(final_time)
    dimension = len(initial)
    result: dict[str, object] = {
        "name": name,
        "initial_time": 0.0,
        "final_time": final_time,
        "initial_state": initial,
        "parameters": parameters,
        "state": [float(value) for value in integrator.state[:dimension]],
    }
    if stm:
        variational = integrator.state[integrator.get_vslice(order=1)]
        result["stm"] = [
            [float(value) for value in variational[row * dimension : (row + 1) * dimension]]
            for row in range(dimension)
        ]
    return result


def kepler() -> dict[str, object]:
    x, y, z, vx, vy, vz = hy.make_vars("kx", "ky", "kz", "kvx", "kvy", "kvz")
    radius_squared = x * x + y * y + z * z
    gravity = -hy.par[0] * radius_squared**-1.5
    system = [
        (x, vx),
        (y, vy),
        (z, vz),
        (vx, gravity * x),
        (vy, gravity * y),
        (vz, gravity * z),
    ]
    return propagate(
        "kepler",
        system,
        [0.8, -0.2, 0.1, 0.03, 1.0, 0.02],
        [1.0],
        0.5,
        stm=True,
    )


def cr3bp() -> dict[str, object]:
    x, y, z, vx, vy, vz = hy.make_vars("cx", "cy", "cz", "cvx", "cvy", "cvz")
    mu = hy.par[0]
    dx1 = x + mu
    dx2 = x + mu - 1.0
    r1 = (dx1 * dx1 + y * y + z * z) ** -1.5
    r2 = (dx2 * dx2 + y * y + z * z) ** -1.5
    system = [
        (x, vx),
        (y, vy),
        (z, vz),
        (vx, 2.0 * vy + x - (1.0 - mu) * dx1 * r1 - mu * dx2 * r2),
        (vy, -2.0 * vx + y - ((1.0 - mu) * r1 + mu * r2) * y),
        (vz, -((1.0 - mu) * r1 + mu * r2) * z),
    ]
    return propagate(
        "cr3bp",
        system,
        [0.8, -0.2, 0.1, 0.03, -0.04, 0.02],
        [0.01215058560962404],
        0.75,
    )


def bcp() -> dict[str, object]:
    x, y, z, vx, vy, vz = hy.make_vars("bx", "by", "bz", "bvx", "bvy", "bvz")
    mu, mu_sun, rho_sun, omega_sun = (hy.par[index] for index in range(4))
    sine = hy.sin(omega_sun * hy.time)
    cosine = hy.cos(omega_sun * hy.time)
    dx1 = x + mu
    dx2 = x + mu - 1.0
    sun_x = x - rho_sun * cosine
    sun_y = y - rho_sun * sine
    r1 = (dx1 * dx1 + y * y + z * z) ** -1.5
    r2 = (dx2 * dx2 + y * y + z * z) ** -1.5
    rs = (sun_x * sun_x + sun_y * sun_y + z * z) ** -1.5
    indirect = -mu_sun / (rho_sun * rho_sun)
    system = [
        (x, vx),
        (y, vy),
        (z, vz),
        (
            vx,
            2.0 * vy
            + x
            - (1.0 - mu) * dx1 * r1
            - mu * dx2 * r2
            - mu_sun * sun_x * rs
            + indirect * cosine,
        ),
        (
            vy,
            -2.0 * vx
            + y
            - (1.0 - mu) * y * r1
            - mu * y * r2
            - mu_sun * sun_y * rs
            + indirect * sine,
        ),
        (vz, -(1.0 - mu) * z * r1 - mu * z * r2 - mu_sun * z * rs),
    ]
    return propagate(
        "bcp",
        system,
        [0.8, -0.2, 0.1, 0.03, -0.04, 0.02],
        [0.01215058560962404, 328900.56, 389.172, -0.925195985520347],
        0.25,
    )


def main() -> None:
    document = {
        "schema_version": 1,
        "generator": "tools/heyoka-cross-validation/generate.py",
        "heyoka_version": hy.__version__,
        "tolerance": TOLERANCE,
        "cases": [kepler(), cr3bp(), bcp()],
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
