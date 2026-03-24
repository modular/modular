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
        "dir",
        type=Path,
        help="Directory to scan for .mojo files.",
    )
    parser.add_argument(
        "--skip-file",
        metavar="REL_PATH",
        action="append",
        default=[],
        help=(
            "Skip a file by its path relative to the scan directory "
            "(e.g. 'collections/string/string.mojo'). May be repeated. "
            "Used to blocklist files with known crashes (MOCO-3399)."
        ),
    )
    args = parser.parse_args()

    # Resolve to an absolute path so mojo-lsp-simple-client receives an
    # absolute file path.  That produces a well-formed file:///abs/path URI
    # inside simple-client.cpp ("file://" + path).  A relative path yields
    # file://relative/path where the first component is mis-parsed as the
    # URI authority/host, causing the LSP server to treat the document as
    # having an unknown location and skip import resolution entirely.
    scan_dir = args.dir.resolve()

    mojo_files = sorted(scan_dir.rglob("*.mojo"))

    if args.skip_file:
        skip = {Path(p) for p in args.skip_file}
        mojo_files = [
            f for f in mojo_files if f.relative_to(scan_dir) not in skip
        ]

    # Shard-aware file selection: shard i processes files where index % total == i.
    shard_index = int(os.environ.get("TEST_SHARD_INDEX", "0"))
    total_shards = int(os.environ.get("TEST_TOTAL_SHARDS", "1"))
    if total_shards > 1:
        mojo_files = [
            f
            for i, f in enumerate(mojo_files)
            if i % total_shards == shard_index
        ]

    # Signal to Bazel that this test runner is shard-aware.
    shard_status_file = os.environ.get("TEST_SHARD_STATUS_FILE")
    if shard_status_file:
        Path(shard_status_file).touch()

    suite_name = f"lsp-parse/{scan_dir.name}"

    if not mojo_files:
        if total_shards > 1:
            print(
                f"Shard {shard_index}/{total_shards}: no files assigned, skipping.",
                flush=True,
            )
        else:
            print(f"error: no .mojo files found in {scan_dir}", file=sys.stderr)
            return 1
        xml_file = os.environ.get("XML_OUTPUT_FILE")
        if xml_file:
            _write_junit_xml(xml_file, suite_name, [])
        return 0

    client = _find_lsp_client()
    results: list[tuple[str, float, str | None]] = []
    failed: list[Path] = []
    t_start = time.monotonic()

    for i, f in enumerate(mojo_files, 1):
        print(f"[{i}/{len(mojo_files)}] mojo-lsp-simple-client {f}", flush=True)
        t0 = time.monotonic()
        result = subprocess.run(
            [client, "--fail-on-diagnostics", str(f)],
            capture_output=True,
            text=True,
        )
        elapsed = time.monotonic() - t0
        print(f"  -> {elapsed:.1f}s", flush=True)
        error: str | None = None
        if result.returncode != 0:
            error = result.stderr or f"exit code {result.returncode}"
            print(
                f"FAILED: {f}\n"
                f"  Reproduce: bazel run //KGEN/tools/mojo-lsp-simple-client"
                f" -- --fail-on-diagnostics {f.resolve()}",
                file=sys.stderr,
            )
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            failed.append(f)
        results.append((str(f.relative_to(scan_dir)), elapsed, error))

    total = time.monotonic() - t_start
    print(
        f"TIMING: total={total:.1f}s files={len(mojo_files)} dir={scan_dir}",
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
