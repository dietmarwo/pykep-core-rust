#!/usr/bin/env python3
"""Fail when a relative Markdown link points to a missing repository file."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
IGNORED_DIRECTORIES = {".git", ".mypy_cache", ".pytest_cache", ".venv", "target"}
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:")


def markdown_files(root: Path) -> list[Path]:
    """Return tracked-source Markdown candidates below *root*."""
    return sorted(
        path
        for path in root.rglob("*.md")
        if not IGNORED_DIRECTORIES.intersection(path.relative_to(root).parts)
    )


def local_target(raw_target: str) -> str | None:
    """Extract a local path from a simple inline Markdown link target."""
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if not target or target.startswith("#") or target.startswith(EXTERNAL_PREFIXES):
        return None
    return unquote(target.split("#", 1)[0])


def main() -> int:
    """Check links below the requested root and print every missing target."""
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    missing: list[str] = []
    for document in markdown_files(root):
        text = document.read_text(encoding="utf-8")
        for match in LINK.finditer(text):
            target = local_target(match.group(1))
            if target is None:
                continue
            destination = (document.parent / target).resolve()
            if not destination.exists():
                missing.append(
                    f"{document.relative_to(root)}: missing link target {match.group(1)!r}"
                )
    if missing:
        print("\n".join(missing), file=sys.stderr)
        return 1
    print(f"checked relative links in {len(markdown_files(root))} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
