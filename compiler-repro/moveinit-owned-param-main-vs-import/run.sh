#!/bin/sh
# Reproduces: `def __moveinit__(out self, owned existing: Self):` parses
# fine when the struct lives in a file that gets IMPORTED as a module,
# but fails to parse the byte-for-byte identical source when that file is
# run directly as the `mojo run` entry point.
set -e
cd "$(dirname "$0")"

echo "--- via_import.mojo (imports lib.mojo): expected PASS, prints -1 ---"
mojo run -I . via_import.mojo

echo
echo "--- main_vs_import.mojo (identical struct, defined inline, run directly): expected FAIL to parse ---"
mojo run main_vs_import.mojo
