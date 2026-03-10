#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Parse every .mojo file in a directory tree using mojo-lsp-simple-client.

For each file the command is printed before it runs. Exits 1 if any invocation
crashes or reports a server-side failure. Server output is suppressed on success
and shown on failure so the log stays readable across large directories.

The directory defaults to the stdlib root derived from the
MODULAR_MOJO_MAX_IMPORT_PATH environment variable, which is set automatically
when the enclosing lit_tests target has mojo_deps = ["@mojo//:std"].

Files are processed sequentially; parallelism is provided by Bazel running
multiple per-subfolder test targets concurrently.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def check_subdir_coverage(scan_dir: Path, known_subdirs: set[str]) -> int:
    """Check that every stdlib subfolder is listed in the BUILD.bazel parse-stdlib lists."""
    actual = sorted(d.name for d in scan_dir.iterdir() if d.is_dir())
    missing = [s for s in actual if s not in known_subdirs]
    if missing:
        for s in missing:
            print(
                f"error: stdlib subfolder '{s}' is not listed in BUILD.bazel.\n"
                f"  Fix: add '{s}' to _ENABLED_SUBDIRS or _DISABLED_SUBDIRS in\n"
                f"  KGEN/test/mojo-lsp-server/BUILD.bazel",
                file=sys.stderr,
            )
        return 1
    print(
        f"ok: all {len(actual)} stdlib subfolder(s) are listed in BUILD.bazel."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dir",
        nargs="?",
        type=Path,
        help=(
            "Directory to scan for .mojo files. "
            "Defaults to the stdlib root from MODULAR_MOJO_MAX_IMPORT_PATH."
        ),
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan subdirectories recursively (default: True).",
    )
    parser.add_argument(
        "--check-subdir-coverage",
        metavar="SUBDIR,...",
        help=(
            "Comma-separated list of all stdlib subdirs known to BUILD.bazel "
            "(_ENABLED_SUBDIRS + _DISABLED_SUBDIRS). "
            "Exits 1 with an actionable message if any subfolder of DIR is missing."
        ),
    )
    parser.add_argument(
        "--file-regex",
        metavar="REGEX",
        help=(
            "Only process files whose stem (filename without extension) matches "
            "this regular expression. Anchored at the start (re.match)."
        ),
    )
    args = parser.parse_args()

    if args.dir:
        scan_dir = args.dir
    else:
        import_path_env = os.environ.get("MODULAR_MOJO_MAX_IMPORT_PATH", "")
        if not import_path_env:
            print(
                "error: no directory given and MODULAR_MOJO_MAX_IMPORT_PATH is not set",
                file=sys.stderr,
            )
            return 1
        # MODULAR_MOJO_MAX_IMPORT_PATH is the stdlib root directory itself
        # (the directory that contains io/, math/, etc.).
        scan_dir = Path(import_path_env.split(",")[0])

    if args.check_subdir_coverage:
        known = set(args.check_subdir_coverage.split(","))
        return check_subdir_coverage(scan_dir, known)

    if args.recursive:
        mojo_files = sorted(scan_dir.rglob("*.mojo"))
    else:
        mojo_files = sorted(scan_dir.glob("*.mojo"))

    if args.file_regex:
        pat = re.compile(args.file_regex)
        mojo_files = [f for f in mojo_files if pat.match(f.stem)]

    if not mojo_files:
        print(f"error: no .mojo files found in {scan_dir}", file=sys.stderr)
        return 1

    failed: list[Path] = []
    t_start = time.monotonic()

    for i, f in enumerate(mojo_files, 1):
        print(f"[{i}/{len(mojo_files)}] mojo-lsp-simple-client {f}", flush=True)
        t0 = time.monotonic()
        result = subprocess.run(
            ["mojo-lsp-simple-client", f], capture_output=True, text=True
        )
        elapsed = time.monotonic() - t0
        print(f"  -> {elapsed:.1f}s", flush=True)
        if result.returncode != 0:
            print(
                f"FAILED: {f}\n"
                f"  Reproduce: bazel run //KGEN/tools/mojo-lsp-simple-client -- {f}",
                file=sys.stderr,
            )
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            failed.append(f)

    total = time.monotonic() - t_start
    print(
        f"TIMING: total={total:.1f}s files={len(mojo_files)} dir={scan_dir}",
        flush=True,
    )

    if failed:
        print(f"\n{len(failed)} file(s) failed:", file=sys.stderr)
        for f in failed:
            print(f"  {f}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
