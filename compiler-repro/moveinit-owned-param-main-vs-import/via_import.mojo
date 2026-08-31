# Defect half of this reproducer: imports the byte-for-byte identical
# struct from lib.mojo (same directory) instead of defining it inline,
# and compiles and runs cleanly, even though the legacy `__moveinit__`
# spelling it contains was removed in 1.0.0b1 and the same source is
# rejected when compiled as the entry file. See main_vs_import.mojo for
# the full write-up and the per-spelling error matrix.
#
# Verified on Mojo 1.0.0b2 (2cf4d08a) and 1.0.0 (ed45d567), macOS arm64.
# Run: mojo run -I <this-directory> via_import.mojo
# Today: PASSES and prints "-1" (that acceptance is the bug).

from lib import IoEventLike


def main():
    var e = IoEventLike()
    print(e.fd)
