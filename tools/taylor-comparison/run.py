#!/usr/bin/env python3
"""Run one reproducible, bounded Taylor comparison job on Linux."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import tempfile
import time


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def experiment_identity(command, inputs, timeout, memory):
    """Fingerprint inputs to a bounded run, including a rebuilt binary at the same path."""
    files = {Path(__file__).resolve(), *(path.resolve() for path in inputs)}
    for argument in command:
        path = Path(argument)
        try:
            if path.is_file():
                files.add(path.resolve())
            elif executable := shutil.which(argument):
                files.add(Path(executable).resolve())
        except OSError:
            # Arguments such as inline Python programs need not be valid paths.
            # Their full text is still part of the command identity.
            continue
    source = None
    try:
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], stderr=subprocess.DEVNULL, timeout=10
        )
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], timeout=10).decode().strip()
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"], timeout=10
        ).split(b"\0")
        for name in untracked:
            if name:
                path = Path(os.fsdecode(name))
                if path.is_file():
                    files.add(path.resolve())
        source = {"head": head, "diff_sha256": hashlib.sha256(diff).hexdigest()}
    except (OSError, subprocess.SubprocessError):
        pass
    compiler = None
    if rustc := shutil.which("rustc"):
        try:
            compiler = subprocess.check_output([rustc, "-Vv"], text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "directory": os.getcwd(),
        "command": command,
        "source": source,
        "inputs": {str(path): digest(path) for path in sorted(files)},
        "rustc": compiler,
        "environment": {name: os.environ.get(name) for name in (
            "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "CARGO_BUILD_JOBS",
            "CARGO_TARGET_DIR", "RAYON_NUM_THREADS", "PYKEP_TAYLOR_SAMPLES",
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "PYTHONPATH", "LD_LIBRARY_PATH",
        )},
        "affinity": sorted(os.sched_getaffinity(0)),
        "timeout": timeout,
        "memory_limit_bytes": memory,
    }


def group_rss(group):
    total = 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            if int(stat[2]) == group:
                total += int(stat[21]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            continue
    return total


def stop_group(group):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            return
        if sig == signal.SIGTERM:
            time.sleep(0.25)


def tail(path, size=16384):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - size))
        return stream.read().decode("utf-8", errors="replace")


def run(command, timeout, memory, output):
    started = time.monotonic()
    peak = 0
    next_progress = started + 30
    status = "failed"
    prlimit = shutil.which("prlimit")
    if prlimit is None:
        raise RuntimeError("util-linux prlimit is required")
    # The address-space limit is inherited, including by children that create
    # a new session or are hidden by a process namespace.
    limited_command = [prlimit, f"--as={memory}", "--", *command]
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stdout = Path(temporary) / "stdout"
        stderr = Path(temporary) / "stderr"
        code = None
        with stdout.open("wb") as out, stderr.open("wb") as err:
            try:
                child = subprocess.Popen(limited_command, start_new_session=True, stdout=out, stderr=err)
            except OSError as error:
                err.write(str(error).encode())
                status = "unavailable"
            else:
                try:
                    while child.poll() is None:
                        peak = max(peak, group_rss(child.pid))
                        if peak > memory:
                            status = "memory_limit"
                            break
                        if time.monotonic() - started > timeout:
                            status = "timeout"
                            break
                        if stdout.stat().st_size + stderr.stat().st_size > 16 * 1024**2:
                            status = "output_limit"
                            break
                        now = time.monotonic()
                        if now >= next_progress:
                            print(f"{output.name}: running for {now - started:.0f}s, "
                                  f"sampled peak RSS {peak / 1024**2:.0f} MiB", flush=True)
                            next_progress = now + 30
                        time.sleep(0.1)
                    else:
                        status = "success" if child.returncode == 0 else "failed"
                finally:
                    stop_group(child.pid)
                    try:
                        code = child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        status = "cleanup_timeout"
                if stdout.stat().st_size + stderr.stat().st_size > 16 * 1024**2:
                    status = "output_limit"
        return {
            "schema_version": 2,
            "command": command,
            "status": status,
            "returncode": code,
            "seconds": time.monotonic() - started,
            "sampled_peak_rss_bytes": peak or None,
            "stdout_tail": tail(stdout, 1024**2),
            "stderr_tail": tail(stderr),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--memory-mib", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--input", action="append", type=Path, default=[],
                        help="Fingerprint an additional fixture, configuration, or source file.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command or not 0 < args.timeout < float("inf") or args.memory_mib <= 0:
        parser.error("a command and positive finite limits are required")
    if not Path("/proc/self/stat").is_file():
        parser.error("this runner requires Linux procfs")
    if any(not path.is_file() for path in args.input):
        parser.error("every --input must name an existing file")
    memory = args.memory_mib * 1024**2
    limit_path = Path("/sys/fs/cgroup/memory.max")
    if limit_path.is_file():
        limit = limit_path.read_text().strip()
        if limit.isdecimal():
            memory = min(memory, int(limit) // 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    identity = experiment_identity(command, args.input, args.timeout, memory)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if args.resume and args.output.exists():
        try:
            previous = json.loads(args.output.read_text())
        except (OSError, ValueError):
            previous = {}
        if previous.get("job_key") == key and previous.get("status") == "success":
            print(f"{args.output}: {previous['status']} (existing)")
            return 0
    result = run(command, args.timeout, memory, args.output)
    result["job_key"] = key
    result["identity"] = identity
    result["memory_limit_bytes"] = memory
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    os.replace(stream.name, args.output)
    print(f"{args.output}: {result['status']} ({result['seconds']:.2f}s)")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
