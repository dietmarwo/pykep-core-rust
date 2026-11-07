#!/usr/bin/env python3
"""Draw docs/images/taylor-results.svg from the one-revolution benchmark rows.

Reads docs/data/taylor-kepler-benchmark.csv (written by
`cargo run --release -p pykep-taylor-benchmark`) and writes the figure used by
docs/taylor-integration.md: wall-time speed-up bars on a linear axis and the
maximum state error of both integrators on a log axis.

usage: python tools/taylor_results_svg.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data" / "taylor-kepler-benchmark.csv"
OUTPUT = ROOT / "docs" / "images" / "taylor-results.svg"

LABELS = {1e-9: "1e−9", 1e-12: "1e−12", 1e-14: "1e−14"}
CENTERS = (180, 390, 600, 810)


def label(tolerance: float) -> str:
    for value, text in LABELS.items():
        if math.isclose(tolerance, value, rel_tol=1e-9):
            return text
    if math.isclose(tolerance, 2.220446049250313e-16, rel_tol=1e-9):
        return "epsilon"
    raise ValueError(f"unexpected tolerance {tolerance}")


def sci(value: float) -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    return (
        f"{mantissa}e−{abs(int(exponent))}"
        if int(exponent) < 0
        else f"{mantissa}e{int(exponent)}"
    )


def main() -> None:
    rows = [r for r in csv.DictReader(DATA.open()) if r["revolutions"] == "1"]
    by_tolerance: dict[float, dict[str, dict]] = {}
    for row in rows:
        by_tolerance.setdefault(float(row["tolerance"]), {})[row["method"]] = row
    tolerances = sorted(by_tolerance, reverse=True)
    if len(tolerances) != len(CENTERS):
        raise SystemExit(
            f"expected {len(CENTERS)} one-revolution tolerances, got {len(tolerances)}"
        )

    speedups, errors = [], []
    for tolerance in tolerances:
        pair = by_tolerance[tolerance]
        dop, taylor = pair["dop853"], pair["taylor"]
        speedups.append(float(dop["median_seconds"]) / float(taylor["median_seconds"]))
        errors.append((float(dop["max_state_error"]), float(taylor["max_state_error"])))

    out: list[str] = []
    add = out.append
    add(
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="470" viewBox="0 0 960 470" role="img" aria-labelledby="title desc">'
    )
    add('  <title id="title">Taylor speed-up and accuracy by tolerance</title>')
    faster = ", ".join(
        f"{label(t)} {s:.2f} times" for t, s in zip(tolerances, speedups)
    )
    add(
        '  <desc id="desc">Wall-time speed-up of Taylor over DOP853 for one revolution: '
        f"{faster}. Maximum state error on a log scale: Taylor is smaller than DOP853 at every tolerance.</desc>"
    )
    add("""  <style>
    .axis { stroke:#304b5f; stroke-width:2 }
    .grid { stroke:#d9e2e8; stroke-width:1 }
    .bar { fill:#26886b }
    .slow { fill:#c87822 }
    .dop { fill:#557a95; stroke:#557a95 }
    .taylor { fill:#26886b; stroke:#26886b }
    .series { fill:none; stroke-width:2 }
    .text { font-family:sans-serif; font-size:14px; fill:#263b49 }
    .small { font-family:sans-serif; font-size:12px; fill:#263b49 }
    .halo { paint-order:stroke; stroke:white; stroke-width:4px; stroke-linejoin:round }
    .heading { font-family:sans-serif; font-size:18px; font-weight:600; fill:#172b3a }
    .value { font-family:sans-serif; font-size:14px; font-weight:600; fill:#172b3a; text-anchor:middle }
  </style>""")
    add('  <rect width="960" height="470" fill="white"/>')

    # Speed-up panel: linear axis, 75 px per 1x, baseline at y=195.
    add(
        '  <text class="heading" x="30" y="32">One revolution: wall-time speed-up (DOP853 time / Taylor time)</text>'
    )
    add('  <line class="axis" x1="80" y1="195" x2="900" y2="195"/>')
    for y in (120, 45):
        add(f'  <line class="grid" x1="80" y1="{y}" x2="900" y2="{y}"/>')
    for multiple, y in ((0, 200), (1, 125), (2, 50)):
        add(f'  <text class="text" x="46" y="{y}">{multiple}×</text>')
    for center, speedup in zip(CENTERS, speedups):
        height = round(75 * speedup)
        css = "slow" if speedup < 1 else "bar"
        add(
            f'  <rect class="{css}" x="{center - 55}" y="{195 - height}" width="110" height="{height}"/>'
        )
        add(
            f'  <text class="value" x="{center}" y="{195 - height - 8}">{speedup:.2f}×</text>'
        )
    for center, tolerance in zip(CENTERS, tolerances):
        add(
            f'  <text class="text" x="{center}" y="219" text-anchor="middle">{label(tolerance)}</text>'
        )

    # Error panel: log10 axis from 1e-7 (top) to 1e-16 (bottom).
    top, bottom, high, low = 290, 425, -7, -16

    def y_of(value: float) -> float:
        return top + (high - math.log10(value)) / (high - low) * (bottom - top)

    add(
        '  <text class="heading" x="30" y="262">Maximum state error (log scale; lower is better)</text>'
    )
    add('  <circle class="dop" cx="676" cy="256" r="6"/>')
    add('  <text class="text" x="688" y="261">DOP853</text>')
    add('  <rect class="taylor" x="770" y="250" width="12" height="12"/>')
    add('  <text class="text" x="788" y="261">Taylor</text>')
    for exponent in range(-8, -17, -2):
        y = y_of(10.0**exponent)
        add(f'  <line class="grid" x1="80" y1="{y:.1f}" x2="900" y2="{y:.1f}"/>')
        add(
            f'  <text class="small" x="74" y="{y + 4:.1f}" text-anchor="end">1e−{-exponent}</text>'
        )
    add(f'  <line class="axis" x1="80" y1="{bottom}" x2="900" y2="{bottom}"/>')
    for index, css in ((0, "dop"), (1, "taylor")):
        points = " ".join(f"{c},{y_of(e[index]):.1f}" for c, e in zip(CENTERS, errors))
        add(f'  <polyline class="{css} series" points="{points}" style="fill:none"/>')
    for center, (dop, taylor) in zip(CENTERS, errors):
        yd, yt = y_of(dop), y_of(taylor)
        add(f'  <circle class="dop" cx="{center}" cy="{yd:.1f}" r="6"/>')
        add(
            f'  <text class="small halo" x="{center - 12}" y="{yd + 4:.1f}" text-anchor="end">{sci(dop)}</text>'
        )
        add(
            f'  <rect class="taylor" x="{center - 6}" y="{yt - 6:.1f}" width="12" height="12"/>'
        )
        add(
            f'  <text class="small halo" x="{center + 12}" y="{yt + 4:.1f}">{sci(taylor)}</text>'
        )
    for center, tolerance in zip(CENTERS, tolerances):
        add(
            f'  <text class="text" x="{center}" y="449" text-anchor="middle">{label(tolerance)}</text>'
        )
    add(
        '  <text class="small" x="490" y="466" text-anchor="middle">requested tolerance</text>'
    )
    add("</svg>")
    OUTPUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
