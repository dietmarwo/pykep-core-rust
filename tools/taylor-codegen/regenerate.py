#!/usr/bin/env python3
"""Export the built-in model graphs and emit deterministic Rust kernels."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "crates/pykep-core/src/integration/taylor/generated.rs"
ARITY = {"constant": 1, "time": 0, "state": 1, "parameter": 1,
         "add": 2, "subtract": 2, "multiply": 2, "negate": 1, "power": 2,
         "exp": 1, "sin": 1, "cos": 1, "identity": 1}
LEAVES = {"constant", "time", "state", "parameter"}


def dependencies(op):
    name, *args = op
    if name in LEAVES:
        return []
    return args[:1] if name == "power" else args


def float_bits(bits):
    return f"f64::from_bits(0x{bits:016x})"


def analyze(graph, optimized=True):
    """Find graph reachability, time invariants, and recurrence histories."""
    ops = graph["operations"]
    companions = graph["companions"]
    live = set()
    pending = list(graph["outputs"])
    while pending:
        index = pending.pop()
        if index not in live:
            live.add(index)
            pending.extend(dependencies(ops[index]))
            if companions[index] is not None:
                pending.append(companions[index])
    if not optimized:
        live = set(range(len(ops)))
    invariant = []
    for op in ops:
        invariant.append(optimized and op[0] not in {"state", "time"}
                         and all(invariant[j] for j in dependencies(op)))
    history = set()
    for i in sorted(live):
        kind, *args = ops[i]
        if invariant[i]:
            continue
        if kind == "multiply" and not invariant[args[0]] and not invariant[args[1]]:
            history.update(args)
        elif kind in {"power", "exp"}:
            history.update([i, args[0]])
        elif kind in {"sin", "cos"}:
            history.update([args[0], companions[i]])
    history = {i for i in history if not invariant[i] and ops[i][0] not in {"state", "time"}}
    return sorted(live), invariant, sorted(history)


def generate(name, graph, optimized, compact=False):
    ops = graph["operations"]
    companions = graph["companions"]
    outputs = graph["outputs"]
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or len(companions) != len(ops):
        raise ValueError("invalid graph identity")
    for i, op in enumerate(ops):
        kind, *args = op
        if len(args) != ARITY[kind] or any(not isinstance(x, int) for x in args):
            raise ValueError(f"{name}: malformed operation {i}")
        if any(not 0 <= d < i for d in dependencies(op)):
            raise ValueError(f"{name}: graph is not topologically sorted at {i}")
        if kind in {"sin", "cos"}:
            partner = companions[i]
            other = "cos" if kind == "sin" else "sin"
            if partner is None or not 0 <= partner < len(ops) or ops[partner] != [other, args[0]]:
                raise ValueError(f"{name}: invalid trigonometric companion")
        elif kind == "parameter" and not 0 <= args[0] < graph["parameters"]:
            raise ValueError(f"{name}: invalid parameter")
        elif kind == "state" and not 0 <= args[0] < len(outputs):
            raise ValueError(f"{name}: invalid state")
    if any(not 0 <= i < len(ops) for i in outputs):
        raise ValueError(f"{name}: invalid output")
    live, invariant, history = analyze(graph, optimized)
    if compact:
        return generate_compact(name, graph, live, invariant, history)
    indices = {old: new for new, old in enumerate(sorted(live))}
    fingerprint = hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()
    lines = [
        f"// Graph SHA-256: {fingerprint}",
        "#[inline(never)]",
        f"pub(super) fn {name}<T: Coefficient>(",
        f"    _time: T, state: &[T; {len(outputs)}], parameters: &[T; {graph['parameters']}],",
        f"    order: usize, jet: &mut [[T; WIDTH]; {len(outputs)}],",
        ") {",
        f"    let mut c = [[T::ZERO; WIDTH]; {len(live)}];",
        "    for (row, &value) in jet.iter_mut().zip(state) { row[0] = value; }",
        "    for n in 0..order {",
    ]
    for i in sorted(live):
        kind, *args = ops[i]
        dest = f"c[{indices[i]}][n]"
        refs = [f"c[{indices[j]}]" for j in dependencies(ops[i])]
        if kind == "constant":
            expr = f"if n == 0 {{ T::splat({float_bits(args[0])}) }} else {{ T::ZERO }}"
        elif kind == "parameter":
            expr = f"if n == 0 {{ parameters[{args[0]}] }} else {{ T::ZERO }}"
        elif kind == "time":
            expr = "match n { 0 => _time, 1 => T::splat(1.0), _ => T::ZERO }"
        elif kind == "state":
            expr = f"jet[{args[0]}][n]"
        elif kind in {"add", "subtract"}:
            symbol = "+" if kind == "add" else "-"
            expr = f"{refs[0]}[n] {symbol} {refs[1]}[n]"
        elif kind in {"negate", "identity"}:
            expr = ("-" if kind == "negate" else "") + refs[0] + "[n]"
        elif kind == "multiply":
            if invariant[args[0]]:
                expr = f"{refs[0]}[0] * {refs[1]}[n]"
            elif invariant[args[1]]:
                expr = f"{refs[0]}[n] * {refs[1]}[0]"
            else:
                expr = f"product(&{refs[0]}, &{refs[1]}, n)"
        elif kind == "power":
            expr = f"power(&{refs[0]}, &c[{indices[i]}], {float_bits(args[1])}, n)"
        elif kind == "exp":
            expr = f"exponential(&{refs[0]}, &c[{indices[i]}], n)"
        else:
            partner = indices[companions[i]]
            expr = f"trigonometric(&{refs[0]}, &c[{partner}], {str(kind == 'sin').lower()}, n)"
        if invariant[i]:
            lines.append(f"        if n == 0 {{ {dest} = {expr}; }}")
        else:
            lines.append(f"        {dest} = {expr};")
    lines.append("        let divisor = T::splat((n + 1) as f64);")
    for component, index in enumerate(outputs):
        lines.append(f"        jet[{component}][n + 1] = c[{indices[index]}][n] / divisor;")
    lines.extend(["    }", "}", ""])
    return "\n".join(lines)


def generate_compact(name, graph, live, invariant, history):
    """Keep current-degree values in locals and retain only recurrence inputs."""
    ops = graph["operations"]
    slots = {node: slot for slot, node in enumerate(history)}

    def series(node):
        kind, *args = ops[node]
        if kind == "state":
            return f"jet[{args[0]}]"
        if kind == "time":
            return "time_series"
        return f"c[{slots[node]}]"

    def value(node):
        kind, *args = ops[node]
        if kind == "state":
            return f"jet[{args[0]}][n]"
        if kind == "time":
            return "time_series[n]"
        return f"v{node}"

    fingerprint = hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()
    lines = [f"// Graph SHA-256: {fingerprint}",
             f"// Reachable nodes: {len(live)}. Intermediate history rows: {len(history)}.",
             "#[allow(clippy::needless_range_loop)]",
             "#[inline(never)]", f"pub(super) fn {name}<T: Coefficient>(",
             f"    _time: T, state: &[T; {len(graph['outputs'])}], parameters: &[T; {graph['parameters']}],",
             f"    order: usize, jet: &mut [[T; WIDTH]; {len(graph['outputs'])}],", ") {",
             f"    let mut c = [[T::ZERO; WIDTH]; {len(history)}];",
             "    for (row, &value) in jet.iter_mut().zip(state) { row[0] = value; }"]
    if any(ops[i][0] == "time" for i in live):
        lines.extend(["    let mut time_series = [T::ZERO; WIDTH];",
                      "    time_series[0] = _time;", "    time_series[1] = T::splat(1.0);"])
    for i in live:
        if not invariant[i]:
            continue
        kind, *args = ops[i]
        refs = [f"v{j}" for j in dependencies(ops[i])]
        if kind == "constant":
            expr = f"T::splat({float_bits(args[0])})"
        elif kind == "parameter":
            expr = f"parameters[{args[0]}]"
        elif kind in {"add", "subtract", "multiply"}:
            symbol = {"add": "+", "subtract": "-", "multiply": "*"}[kind]
            expr = f"{refs[0]} {symbol} {refs[1]}"
        elif kind in {"negate", "identity"}:
            expr = ("-" if kind == "negate" else "") + refs[0]
        elif kind == "power":
            expr = f"{refs[0]}.powf({float_bits(args[1])})"
        else:
            expr = f"{refs[0]}.{kind}()"
        lines.append(f"    let v{i} = {expr};")
    lines.append("    for n in 0..order {")
    for i in live:
        kind, *args = ops[i]
        if invariant[i] or kind in {"state", "time"}:
            continue
        refs = [value(j) for j in dependencies(ops[i])]
        if kind in {"add", "subtract"}:
            # Invariant series contribute only at degree zero, even with tangent seeds.
            refs = [f"if n == 0 {{ {ref} }} else {{ T::ZERO }}" if invariant[j] else ref
                    for j, ref in zip(dependencies(ops[i]), refs)]
            symbol = "+" if kind == "add" else "-"
            expr = f"({refs[0]}) {symbol} ({refs[1]})"
        elif kind in {"negate", "identity"}:
            expr = ("-" if kind == "negate" else "") + refs[0]
        elif kind == "multiply":
            if invariant[args[0]] or invariant[args[1]]:
                expr = f"{refs[0]} * {refs[1]}"
            elif args[0] == args[1] and name in {"bcp", "cr3bp"}:
                expr = f"square(&{series(args[0])}, n)"
            else:
                expr = f"product(&{series(args[0])}, &{series(args[1])}, n)"
        elif kind == "power":
            expr = f"power(&{series(args[0])}, &{series(i)}, {float_bits(args[1])}, n)"
        elif kind == "exp":
            expr = f"exponential(&{series(args[0])}, &{series(i)}, n)"
        else:
            partner = graph["companions"][i]
            expr = f"trigonometric(&{series(args[0])}, &{series(partner)}, {str(kind == 'sin').lower()}, n)"
        lines.append(f"        let v{i} = {expr};")
        if i in slots:
            lines.append(f"        c[{slots[i]}][n] = v{i};")
    lines.append("        let divisor = T::splat((n + 1) as f64);")
    for component, node in enumerate(graph["outputs"]):
        expr = value(node)
        if invariant[node]:
            expr = f"if n == 0 {{ {expr} }} else {{ T::ZERO }}"
        lines.append(f"        jet[{component}][n + 1] = ({expr}) / divisor;")
    lines.extend(["    }", "}", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--unoptimized", action="store_true")
    parser.add_argument("--legacy-layout", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        manifest = args.manifest or (Path(tmp) / "graph.json")
        if args.manifest is None:
            subprocess.run(
                ["cargo", "test", "--locked", "-p", "pykep-core", "--no-default-features",
                 "--lib", "export_taylor_graphs", "--", "--ignored"],
                cwd=ROOT, env={**os.environ, "PYKEP_TAYLOR_GRAPH_OUTPUT": str(manifest)}, check=True,
                timeout=600,
            )
        graph = json.loads(manifest.read_text())
        if graph["schema"] != 1:
            raise ValueError("unsupported Taylor manifest schema")
        output = Path(tmp) / "generated.rs"
        compact = not (args.unoptimized or args.legacy_layout)
        kernels = "Coefficient, WIDTH, exponential, power, product, "
        if compact:
            kernels += "square, "
        kernels += "trigonometric"
        source = [
            "// SPDX-License-Identifier: MPL-2.0",
            "// Generated by tools/taylor-codegen/regenerate.py.",
            f"use super::kernel::{{{kernels}}};",
            "",
        ]
        source.extend(generate(name, model, not args.unoptimized, compact)
                      for name, model in sorted(graph["models"].items()))
        if args.report:
            report = {}
            for name, model in sorted(graph["models"].items()):
                live, invariant, history = analyze(model, not args.unoptimized)
                report[name] = {"reachable_nodes": len(live), "history_rows": len(history),
                                "invariant_nodes": sum(invariant[i] for i in live)}
            args.report.write_text(json.dumps({"schema": 1, "models": report}, indent=2) + "\n")
        output.write_text("\n".join(source))
        subprocess.run(["rustfmt", "--edition", "2024", str(output)], check=True, timeout=30)
        if args.check:
            if not args.output.is_file() or args.output.read_bytes() != output.read_bytes():
                raise SystemExit("generated Taylor kernels differ")
        else:
            args.output.write_bytes(output.read_bytes())


if __name__ == "__main__":
    main()
