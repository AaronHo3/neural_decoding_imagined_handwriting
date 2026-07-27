"""
Guard against em dashes re-entering the repository.

Repository style uses ASCII punctuation (commas, colons, semicolons,
parentheses, hyphens) rather than em dashes. This test fails with the exact
file and line so the offending character is easy to find and replace.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Built from its codepoint so this file does not itself contain the character
# it forbids, which would make the test fail on itself.
EM_DASH = chr(0x2014)

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv",
             "handwritingBCIData", "node_modules", ".ipynb_checkpoints"}

# Text formats we author. Binary and data files are not scanned.
SCAN_SUFFIXES = {".py", ".md", ".ipynb", ".txt", ".sh", ".toml",
                 ".yaml", ".yml", ".cfg", ".json"}


def iter_text_files():
    for path in REPO.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in SCAN_SUFFIXES:
            continue
        # Committed result artifacts contain decoded model output, not prose.
        if path.suffix == ".json" and "results" in path.parts:
            continue
        yield path


def test_no_em_dashes_anywhere():
    offenders = []
    for path in iter_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if EM_DASH not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if EM_DASH in line:
                rel = path.relative_to(REPO).as_posix()
                offenders.append(f"{rel}:{lineno}: {line.strip()[:100]}")

    assert not offenders, (
        f"Found {len(offenders)} em dash(es). Replace with a comma, colon, "
        "semicolon, parentheses, or hyphen:\n  " + "\n  ".join(offenders)
    )


def test_scanner_actually_scans_something():
    """A guard on the guard: an empty file list would make the test vacuous."""
    files = list(iter_text_files())
    assert len(files) > 10, f"expected to scan the repo, only found {len(files)} files"
    assert any(f.suffix == ".py" for f in files)
    assert any(f.suffix == ".md" for f in files)


if __name__ == "__main__":
    test_scanner_actually_scans_something()
    try:
        test_no_em_dashes_anywhere()
    except AssertionError as e:
        print(e)
        sys.exit(1)
    print("clean: no em dashes found")
