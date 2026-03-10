#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Profile LSP parse time per stdlib subdir.

Runs parse-mojo-files-with-lsp sequentially for each subdir, collects the
TIMING: summary line, and prints a table sorted by total time descending.

Usage:
    bazel run //KGEN/test/mojo-lsp-server:profile-parse-stdlib
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subdirs",
        required=True,
        help="Comma-separated list of stdlib subdir names to profile.",
    )
    parser.add_argument(
        "--stdlib-root",
        help=(
            "Path to the stdlib root directory. "
            "Defaults to the first entry of MODULAR_MOJO_MAX_IMPORT_PATH."
        ),
    )
    args = parser.parse_args()

    subdirs = [s for s in args.subdirs.split(",") if s]

    if args.stdlib_root:
        stdlib_root = Path(args.stdlib_root)
    else:
        import_path_env = os.environ.get("MODULAR_MOJO_MAX_IMPORT_PATH", "")
        if not import_path_env:
            print(
                "error: --stdlib-root not given and MODULAR_MOJO_MAX_IMPORT_PATH is not set",
                file=sys.stderr,
            )
            return 1
        stdlib_root = Path(import_path_env.split(",")[0])

    results: list[tuple[str, int, float]] = []  # (subdir, files, total_s)

    timing_re = re.compile(
        r"TIMING:\s+total=([0-9.]+)s\s+files=(\d+)\s+dir=\S+"
    )

    # Locate tools via the runfiles tree.  When run with `bazel run`, __file__
    # is inside the .runfiles directory:
    #   .../profile-parse-stdlib.runfiles/_main/KGEN/test/mojo-lsp-server/profile-parse-stdlib.py
    # Walking up 5 levels gives the runfiles root.
    # Do NOT resolve() — that follows symlinks and escapes the runfiles tree.
    this_file = Path(__file__)
    runfiles_root = this_file.parents[4]  # …/profile-parse-stdlib.runfiles
    workspace = runfiles_root / "_main"

    run_env = dict(os.environ)
    if (
        workspace / "KGEN/test/mojo-lsp-server/parse-mojo-files-with-lsp"
    ).exists():
        tool_dirs = [
            str(workspace / "KGEN/test/mojo-lsp-server"),
            str(workspace / "KGEN/tools/mojo-lsp-simple-client"),
            str(workspace / "KGEN/tools/mojo-lsp-server"),
        ]
        run_env["PATH"] = ":".join(tool_dirs) + ":" + run_env.get("PATH", "")
        run_env["RUNFILES_DIR"] = str(runfiles_root)

    parse_tool = "parse-mojo-files-with-lsp"

    for subdir in subdirs:
        scan_dir = stdlib_root / subdir
        print(f"profiling {subdir} ...", flush=True)
        try:
            proc = subprocess.run(
                [parse_tool, str(scan_dir)],
                capture_output=True,
                text=True,
                env=run_env,
            )
            output = proc.stdout + proc.stderr
            m = timing_re.search(output)
            if m:
                total_s = float(m.group(1))
                files = int(m.group(2))
                results.append((subdir, files, total_s))
            else:
                print(
                    f"  WARNING: no TIMING line found for {subdir}",
                    file=sys.stderr,
                )
                results.append((subdir, 0, float("nan")))
        except Exception as exc:
            print(f"  FAIL: {subdir}: {exc}", file=sys.stderr)
            results.append((subdir, 0, float("nan")))

    # Sort by total time descending (NaN last).
    results.sort(key=lambda r: r[2] if r[2] == r[2] else -1, reverse=True)

    col_w = max(len(r[0]) for r in results) + 2
    print()
    print(f"{'subdir':<{col_w}}  {'files':>5}  {'total':>8}  {'avg/file':>8}")
    print(f"{'-' * col_w}  {'-' * 5}  {'-' * 8}  {'-' * 8}")
    for subdir, files, total_s in results:
        if total_s != total_s:  # NaN
            print(f"{subdir:<{col_w}}  {'?':>5}  {'FAIL':>8}  {'?':>8}")
        else:
            avg = total_s / files if files else 0.0
            print(
                f"{subdir:<{col_w}}  {files:>5}  {total_s:>7.1f}s  {avg:>7.1f}s"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
