"""Fail when JOSE contains external research-repository runtime references."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


RUNTIME_SUFFIXES = {".py", ".sh", ".bat", ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini"}
AUDIT_ONLY = {"check_solo_amp_parity.py", "check_humanoid_amp_parity.py"}
FORBIDDEN_RUNTIME_PATTERNS = (
    re.compile(r"(?:from|import)\s+isaaclab_tasks\.direct\.SOLO(?:\b|\.)"),
    re.compile(r"(?:PYTHONPATH|sys\.path|subprocess)[^\n]*[/\\]SOLO(?:[/\\]|\b)", re.IGNORECASE),
)


def violations(root: Path) -> list[str]:
    failures = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".git" in relative.parts or any(part in {"logs", "outputs", "__pycache__"} for part in relative.parts):
            continue
        if path.is_symlink():
            failures.append(f"symlink: {relative} -> {path.readlink()}")
        if (
            path.is_file()
            and path.suffix.lower() in RUNTIME_SUFFIXES
            and path.name != Path(__file__).name
            and path.name not in AUDIT_ONLY
        ):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(pattern.search(text) for pattern in FORBIDDEN_RUNTIME_PATTERNS):
                failures.append(f"forbidden SOLO runtime dependency: {relative}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Check standalone JOSE runtime independence")
    parser.add_argument("--root", default=str(Path(__file__).parent))
    args = parser.parse_args()
    failures = violations(Path(args.root).resolve())
    if failures:
        raise SystemExit("JOSE independence check failed:\n" + "\n".join(failures))
    print("JOSE independence check passed: no symlinks or forbidden runtime references")


if __name__ == "__main__":
    main()
