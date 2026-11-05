#!/usr/bin/env python3
"""Export pinned numeric source tables without importing PyKEP."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

SOURCE = "c01e19debad5ea22b7079189f786f473046a79d5"


def literal(node):
    if isinstance(node, ast.ListComp):
        return [math.sqrt(x) for x in ast.literal_eval(node.generators[0].iter)]
    if isinstance(node, ast.Call):
        return ast.literal_eval(node.args[0])
    return ast.literal_eval(node)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--output", type=Path, default=Path("crates/pykep-trajopt/data"))
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != SOURCE:
        parser.error(f"expected upstream {SOURCE}, got {revision}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tops").mkdir(exist_ok=True)
    inventory = {"source": SOURCE, "tops": {}, "sha256": {}}
    tops_dir = args.upstream / "pykep/trajopt/gym/tops"
    for path in sorted(tops_dir.glob("_tops_*.json")):
        dest = args.output / "tops" / path.name
        dest.write_bytes(path.read_bytes())
        inventory["tops"][path.stem] = list(json.loads(dest.read_text()))
        inventory["sha256"][f"tops/{dest.name}"] = hashlib.sha256(dest.read_bytes()).hexdigest()
    text = (args.upstream / "pykep/trajopt/_launchers.py").read_text()
    definitions = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key.startswith(("_data_", "_vinfs_", "_decls_")):
                definitions[key] = literal(node.value)
    tables = {}
    for key, prefix in [("ariane5", "Ariane5"), ("ariane64", "A64"), ("atlas501", "A5"),
                        ("atlas551", "A551"), ("soyuzf", "SF"), ("falcon9", "F9")]:
        mass = definitions["_data_" + prefix]
        dec = definitions.get("_decls_" + prefix, [])
        tables[key] = {"speed": definitions["_vinfs_" + prefix], "declination": dec,
                       "mass": mass if dec else [mass]}
    dest = args.output / "launchers.json"
    dest.write_text(json.dumps(tables, allow_nan=False, separators=(",", ":")) + "\n",
                    newline="\n")
    inventory["sha256"]["launchers.json"] = hashlib.sha256(dest.read_bytes()).hexdigest()
    (args.output / "provenance.json").write_text(json.dumps(inventory, indent=2) + "\n",
                                                 newline="\n")


if __name__ == "__main__":
    main()
