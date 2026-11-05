#!/usr/bin/env python3
"""Convert the pinned all-model reference JSON into the native benchmark's text input."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("references", type=Path)
    args = parser.parse_args()
    document = json.loads(args.references.read_text())
    if document["schema_version"] != 1:
        parser.error("unsupported reference schema")
    for case in document["cases"]:
        initial = case["state"]
        parameters = case["parameters"]
        reference = case["reference"]
        times = [case.get("initial_time", 0.0), case["final_time"]]
        if len(initial) != len(reference) or not all(
            math.isfinite(x) for x in [*times, *initial, *parameters, *reference]
        ):
            parser.error(f"invalid reference for {case['model']}")
        print(case["model"], *times, len(initial), len(parameters), *initial, *parameters, *reference)


if __name__ == "__main__":
    main()
