#!/usr/bin/env python3
"""Render a saved gym replay in its physical trajectory plane projection."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pykep_rust as pk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    replay = json.loads(args.replay.read_text())
    spec, trace = replay["problem"], replay["trajectory"]
    figure, axes = plt.subplots(figsize=(7, 6), layout="constrained")
    if spec["id"].startswith("gym."):
        mu = trace["central_mu_si"]
        for arc in trace["trajectory"]["arcs"]:
            times = np.linspace(0., arc["duration_seconds"], 80)
            states = np.asarray(pk.propagate_lagrangian_grid(arc["start"], times, mu))
            axes.plot(*(states[:, :2] / pk.ASTRONOMICAL_UNIT).T, lw=1.2)
        positions = np.asarray(trace["trajectory"]["body_states"])[:, :2] / pk.ASTRONOMICAL_UNIT
        axes.scatter(*positions.T, color="black", s=16, label="Encounters")
        units = "AU"
    else:
        frame = spec["settings"]["frame"]
        if "forward" not in trace:
            parser.error("replay TOPS with a positive --samples value")
        for key, color in (("forward", "tab:blue"), ("backward", "tab:orange")):
            for i, path in enumerate(trace[key]):
                if frame.startswith("prograde MEE"):
                    path = [pk.modified_equinoctial_to_cartesian(state[:6], 1., False) for state in path]
                axes.plot(*np.asarray(path)[:, :2].T, color=color, lw=1.2,
                          label=key.capitalize() if i == 0 else None)
        units = "normalized distance"
    axes.set(title=spec["id"], xlabel=f"x ({units})", ylabel=f"y ({units})", aspect="equal")
    axes.legend()
    axes.grid(alpha=.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
