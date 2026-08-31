# M1.3 (#126) compiler defect reproducer, PASS side: see
# m1-3-owned-param-main-vs-import.mojo for the full write-up. This file
# imports the BYTE-FOR-BYTE IDENTICAL struct (m1_3_owned_param_lib.mojo,
# same directory) instead of defining it inline, and compiles/runs
# cleanly — the same `owned existing: Self` __moveinit__ parameter that
# fails to PARSE as a main-file entry point parses fine as an imported
# module member.
#
# Toolchain: Mojo 1.0.0b2 (2cf4d08a), macOS arm64 (Darwin 25.6.0).
# Run: mojo run -I <this-directory> m1-3-owned-param-via-import.mojo
# Expected: PASSES, prints "-1".

from lib import IoEventLike


def main():
    var e = IoEventLike()
    print(e.fd)
