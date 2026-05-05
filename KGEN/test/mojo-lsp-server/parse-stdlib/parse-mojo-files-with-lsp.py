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

Files are processed sequentially; parallelism is provided by Bazel running
multiple shard instances concurrently.

When run as a Bazel test with shard_count > 1, TEST_SHARD_INDEX and
TEST_TOTAL_SHARDS partition the file list so each shard processes exactly one
file. JUnit XML is written to XML_OUTPUT_FILE when set.
"""

import argparse
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def _find_lsp_client() -> str:
    """Find mojo-lsp-simple-client binary via runfiles or PATH.

    Returns:
        Path to the binary or the bare name ``mojo-lsp-simple-client`` if not
        found via runfiles (falls back to PATH lookup by subprocess).
    """
    test_srcdir = os.environ.get("TEST_SRCDIR", "")
    if test_srcdir:
        p = (
            Path(test_srcdir)
            / "_main"  # Bzlmod workspace name for the main repo
            / "KGEN"
            / "tools"
            / "mojo-lsp-simple-client"
            / "mojo-lsp-simple-client"
        )
        if p.exists():
            return str(p)
    return "mojo-lsp-simple-client"


def _write_junit_xml(
    xml_file: str,
    suite_name: str,
    results: list[tuple[str, float, str | None]],
) -> None:
    """Write JUnit XML for the given test results.

    Args:
        xml_file: Path to write the XML to.
        suite_name: ``testsuite`` name attribute.
        results: List of ``(test_name, elapsed_seconds, error_or_None)`` tuples.
    """
    failures = sum(1 for _, _, err in results if err is not None)
    suite = ET.Element(
        "testsuite",
        name=suite_name,
        tests=str(len(results)),
        failures=str(failures),
        errors="0",
    )
    for name, elapsed, error in results:
        tc = ET.SubElement(
            suite,
            "testcase",
            name=name,
            classname=suite_name,
            time=f"{elapsed:.2f}",
        )
        if error is not None:
            failure = ET.SubElement(
                tc, "failure", message=error.splitlines()[0]
            )
            failure.text = error
    root = ET.Element("testsuites")
    root.append(suite)
    ET.ElementTree(root).write(
        xml_file, encoding="unicode", xml_declaration=True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-root-file",
        metavar="FILE",
        action="append",
        dest="scan_root_files",
        default=[],
        help=(
            "Treat the parent directory of FILE as a scan root and collect "
            "all .mojo files under it. May be repeated for multiple roots. "
            "A sentinel file such as __init__.mojo works well; the Bazel "
            "$(rootpath) expansion makes it easy to pin a root to a specific "
            "package without hard-coding paths."
        ),
    )
    parser.add_argument(
        "--skip-file",
        metavar="REL_PATH",
        action="append",
        default=[],
        help=(
            "Skip a file by its path relative to its scan root "
            "(e.g. 'collections/string/string.mojo'). May be repeated. "
            "Used to blocklist files with known crashes (MOCO-3399)."
        ),
    )
    parser.add_argument(
        "--no-docstring-checks-file",
        metavar="REL_PATH",
        action="append",
        default=[],
        help=(
            "Run this file without docstring code-block validation. Passes "
            "--no-docstring-checks to the LSP client so the file is checked "
            "for structural correctness but docstring examples are not "
            "type-checked. May be repeated."
        ),
    )
    args = parser.parse_args()

    if not args.scan_root_files:
        parser.error("at least one --scan-root-file is required")

    # Resolve to absolute paths so mojo-lsp-simple-client receives an
    # absolute file path.  That produces a well-formed file:///abs/path URI
    # inside simple-client.cpp ("file://" + path).  A relative path yields
    # file://relative/path where the first component is mis-parsed as the
    # URI authority/host, causing the LSP server to treat the document as
    # having an unknown location and skip import resolution entirely.
    scan_roots = [Path(f).resolve().parent for f in args.scan_root_files]

    # Collect all .mojo files from all scan roots, paired with their root so
    # relative paths (for --skip-file and --no-docstring-checks-file matching)
    # can be computed correctly regardless of which root the file came from.
    all_files: list[tuple[Path, Path]] = []
    for root in scan_roots:
        for f in root.rglob("*.mojo"):
            all_files.append((f, root))
    all_files.sort()

    if args.skip_file:
        skip = {Path(p) for p in args.skip_file}
        all_files = [
            (f, r) for f, r in all_files if f.relative_to(r) not in skip
        ]

    no_docstring_checks = {Path(p) for p in args.no_docstring_checks_file}

    # Shard-aware file selection: shard i processes files where index % total == i.
    shard_index = int(os.environ.get("TEST_SHARD_INDEX", "0"))
    total_shards = int(os.environ.get("TEST_TOTAL_SHARDS", "1"))
    if total_shards > 1:
        all_files = [
            (f, r)
            for i, (f, r) in enumerate(all_files)
            if i % total_shards == shard_index
        ]

    # Signal to Bazel that this test runner is shard-aware.
    shard_status_file = os.environ.get("TEST_SHARD_STATUS_FILE")
    if shard_status_file:
        Path(shard_status_file).touch()

    suite_name = "lsp-parse/" + "+".join(r.name for r in scan_roots)

    if not all_files:
        if total_shards > 1:
            print(
                f"Shard {shard_index}/{total_shards}: no files assigned, skipping.",
                flush=True,
            )
        else:
            roots_str = ", ".join(str(r) for r in scan_roots)
            print(
                f"error: no .mojo files found in {roots_str}", file=sys.stderr
            )
            return 1
        xml_file = os.environ.get("XML_OUTPUT_FILE")
        if xml_file:
            _write_junit_xml(xml_file, suite_name, [])
        return 0

    client = _find_lsp_client()
    results: list[tuple[str, float, str | None]] = []
    failed: list[Path] = []
    t_start = time.monotonic()

    for i, (f, root) in enumerate(all_files, 1):
        rel = f.relative_to(root)
        skip_docstrings = rel in no_docstring_checks
        print(
            f"[{i}/{len(all_files)}] mojo-lsp-simple-client"
            f"{' --no-docstring-checks' if skip_docstrings else ''} {f}",
            flush=True,
        )
        t0 = time.monotonic()
        cmd = [client, "--fail-on-diagnostics"]
        if skip_docstrings:
            cmd.append("--no-docstring-checks")
        cmd.append(str(f))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        elapsed = time.monotonic() - t0
        print(f"  -> {elapsed:.1f}s", flush=True)
        error: str | None = None
        if result.returncode != 0:
            error = result.stderr or f"exit code {result.returncode}"
            reproduce_flags = " ".join(cmd[1:])
            print(
                f"FAILED: {f}\n"
                f"  Reproduce: bazel run //KGEN/tools/mojo-lsp-simple-client"
                f" -- {reproduce_flags}",
                file=sys.stderr,
            )
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            failed.append(f)
        results.append((str(rel), elapsed, error))

    total = time.monotonic() - t_start
    roots_str = ", ".join(str(r) for r in scan_roots)
    print(
        f"TIMING: total={total:.1f}s files={len(all_files)} roots={roots_str}",
        flush=True,
    )

    xml_file = os.environ.get("XML_OUTPUT_FILE")
    if xml_file:
        _write_junit_xml(xml_file, suite_name, results)

    if failed:
        print(f"\n{len(failed)} file(s) failed:", file=sys.stderr)
        for f in failed:
            print(f"  {f}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
