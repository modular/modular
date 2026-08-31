# M1.3 (#126) compiler defect reproducer: `owned` as a parameter-convention
# keyword in a __moveinit__ signature PARSES when the struct's file is
# IMPORTED as a module, but FAILS TO PARSE the identical source when that
# same file is run directly as the `mojo run` entry file.
#
# This is not a hypothetical: mojito_sys/io/poller.mojo:78 and
# mojito_sys/io/handle.mojo:65 both already contain
# `def __moveinit__(out self, owned existing: Self):` (or `fn` in
# handle.mojo's case) in the committed repository, and every existing test
# that imports those modules (e.g. tests/s6/io/poller/conformance.mojo)
# compiles them fine. But running EITHER FILE DIRECTLY as a `mojo run`
# entry point fails to parse:
#
#     mojo run -I . mojito_sys/io/poller.mojo
#     # error: expected ')' in argument list
#     #     def __moveinit__(out self, owned existing: Self):
#     #                                ^
#
# Minimized here to a plain struct with no mojito_sys dependency, in two
# files:
#   - THIS file (m1-3-owned-param-main-vs-import.mojo) reproduces the FAIL
#     path: `mojo run m1-3-owned-param-main-vs-import.mojo` fails to parse.
#   - m1-3-owned-param-via-import.mojo (same directory) reproduces the
#     PASS path: importing the identical struct from a SEPARATE main file
#     compiles and runs cleanly.
#
# This repo's OWN dominant convention for __moveinit__ (14 of 16 call
# sites, grep-counted) uses `mut self, mut existing: Self` instead, which
# parses identically in BOTH contexts (also verified, not shown here) —
# that is almost certainly why this asymmetry has not caused a visible
# failure anywhere in this repo's test suite to date: nothing happens to
# `mojo run` poller.mojo or handle.mojo directly, only import them.
#
# Toolchain: Mojo 1.0.0b2 (2cf4d08a), macOS arm64 (Darwin 25.6.0).
# Run: mojo run m1-3-owned-param-main-vs-import.mojo
# Expected (defect): fails to parse — "expected ')' in argument list" at
# the `owned existing: Self` parameter.
# Contrast: mojo run m1-3-owned-param-via-import.mojo — PASSES, importing
# the byte-for-byte identical struct definition from its own module file.

struct IoEventLike(ImplicitlyCopyable):
    var token: UInt64
    var fd: Int32
    var events: UInt32

    def __init__(out self):
        self.token = 0
        self.fd = -1
        self.events = 0

    def __moveinit__(out self, owned existing: Self):
        self.token = existing.token
        self.fd = existing.fd
        self.events = existing.events


def main():
    var e = IoEventLike()
    print(e.fd)
