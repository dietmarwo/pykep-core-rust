#!/usr/bin/env python3
"""Run a dependency-ordered manifest through the existing bounded job runner."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

RUNNER = Path(__file__).resolve().parents[1] / "taylor-comparison" / "run.py"


def write(path, data):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(f.name, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("version") != 1:
        parser.error("expected manifest version 1")
    args.output.mkdir(parents=True, exist_ok=True)
    statuses = {}
    for job in manifest["jobs"]:
        key = job["id"]
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", key) or key in statuses:
            parser.error(f"invalid or duplicate job id {key}")
        deps = job.get("depends_on", [])
        if not all(dep in statuses for dep in deps):
            parser.error(f"dependencies of {key} must precede it")
        if any(statuses[dep] != "success" for dep in deps):
            statuses[key] = "blocked"
        else:
            dest = args.output / f"{key}.json"
            command = [sys.executable, str(RUNNER), "--output", str(dest), "--resume",
                       "--timeout", str(job.get("timeout", 120)),
                       "--memory-mib", str(min(4096, job.get("memory_mib", 4096))),
                       "--input", str(args.manifest)]
            for path in job.get("inputs", []):
                command += ["--input", path]
            run = subprocess.run(command + ["--", *job["command"]], check=False)
            statuses[key] = "success" if run.returncode == 0 else "failed"
        write(args.output / "progress.json", {"version": 1, "jobs": statuses})
    return int(any(status != "success" for status in statuses.values()))


if __name__ == "__main__":
    raise SystemExit(main())
