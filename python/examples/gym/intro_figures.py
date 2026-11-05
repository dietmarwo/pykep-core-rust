#!/usr/bin/env python3
"""Render the figures of docs/astrodynamics-introduction.md.

Requires matplotlib. Writes an Earth-Mars porkchop plot and the Cassini1
trajectory found by the chapter's optimizer examples.

    python python/examples/gym/intro_figures.py docs/images
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pykep_rust as pk
from pykep_rust.trajopt import gym

AU = pk.ASTRONOMICAL_UNIT
DAY = pk.DAY_TO_SECONDS
CASSINI1 = [
    -772.4793517902046, 136.48521715578022, 449.40073064324264,
    57.37453338756138, 948.5084493971979, 4434.572372085129,
]


def porkchop(output: Path) -> None:
    earth = pk.Planet.jpl_low_precision("earth")
    mars = pk.Planet.jpl_low_precision("mars")
    start = pk.Epoch.from_iso("2033-01-01").mjd2000
    departures = start + np.arange(0.0, 361.0, 2.0)
    flight_days = np.arange(100.0, 401.0, 4.0)
    t0, tof = np.meshgrid(departures, flight_days)
    r0 = earth.states(t0.ravel())
    r1 = mars.states((t0 + tof).ravel())
    problems = pk.lambert_problem_batch(
        np.ascontiguousarray(r0[:, :3]), np.ascontiguousarray(r1[:, :3]),
        tof.ravel() * DAY, pk.MU_SUN, maximum_revolutions=0,
    )
    vinf = np.array([
        np.linalg.norm(np.subtract(p.solutions[0].departure_velocity, a[3:]))
        + np.linalg.norm(np.subtract(p.solutions[0].arrival_velocity, b[3:]))
        for p, a, b in zip(problems, r0, r1)
    ]).reshape(t0.shape) / 1e3
    figure, axes = plt.subplots(figsize=(7, 4.6), layout="constrained")
    levels = np.arange(6.0, 16.1, 1.0)
    contour = axes.contourf(t0 - start, tof, np.clip(vinf, 0, 16), levels=levels, cmap="viridis_r")
    axes.contour(t0 - start, tof, vinf, levels=levels, colors="white", linewidths=0.4)
    best = np.unravel_index(np.argmin(vinf), vinf.shape)
    axes.plot(t0[best] - start, tof[best], "r*", ms=12, label=f"minimum {vinf[best]:.2f} km/s")
    figure.colorbar(contour, label="departure + arrival $v_\\infty$ (km/s)")
    axes.set(xlabel="departure (days after 2033-01-01)", ylabel="time of flight (days)",
             title="Earth to Mars, zero-revolution Lambert arcs")
    axes.legend(loc="upper right")
    figure.savefig(output, dpi=110)


def cassini(output: Path) -> None:
    problem = gym.Problem("gym.cassini1")
    trace = problem.decode(CASSINI1)
    figure, axes = plt.subplots(figsize=(6.4, 6.4), layout="constrained")
    epoch = CASSINI1[0]
    for name in ["venus", "earth", "jupiter", "saturn"]:
        planet = pk.Planet.jpl_low_precision(name)
        period = planet.period(epoch) / DAY
        orbit = planet.states(epoch + np.linspace(0.0, period, 300))
        axes.plot(*(orbit[:, :2] / AU).T, color="0.75", lw=0.7)
    labels = ["Earth-Venus", "Venus-Venus", "Venus-Earth", "Earth-Jupiter", "Jupiter-Saturn"]
    for arc, label in zip(trace["trajectory"]["arcs"], labels):
        times = np.linspace(0.0, arc["duration_seconds"], 200)
        states = np.asarray(pk.propagate_lagrangian_grid(arc["start"], times, trace["central_mu_si"]))
        axes.plot(*(states[:, :2] / AU).T, lw=1.4, label=label)
    encounters = np.asarray(trace["trajectory"]["body_states"])[:, :2] / AU
    axes.scatter(*encounters.T, color="black", s=14, zorder=3, label="encounters")
    axes.scatter([0], [0], color="orange", s=40, zorder=3)
    axes.set(xlabel="x (AU)", ylabel="y (AU)", aspect="equal",
             title=f"gym.cassini1: {problem.fitness(CASSINI1)[0]:.0f} m/s")
    axes.legend(loc="lower left", fontsize=8)
    axes.grid(alpha=0.25)
    figure.savefig(output, dpi=110)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    porkchop(args.directory / "astro-porkchop.png")
    cassini(args.directory / "astro-cassini1.png")


if __name__ == "__main__":
    main()
