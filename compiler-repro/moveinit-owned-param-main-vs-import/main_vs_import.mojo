# Reproducer for: the removal of the legacy `__moveinit__`/`__copyinit__`
# spellings (deprecated in 0.26.2, removed in 1.0.0b1 per its release
# notes: "no longer auto-rewritten... now fail to compile") is enforced
# ONLY when the file is the compilation entry point. The byte-for-byte
# identical struct, imported as a module, still accepts the legacy
# spelling silently.
#
# THIS file defines the struct inline and is meant to be run directly:
#   mojo run main_vs_import.mojo
# It fails, which is the CORRECT half (the spelling is removed), though
# with a misleading diagnostic: `owned existing` gets
# "expected ')' in argument list" (owned itself was removed in 0.26.2),
# and the current-convention spellings get something worse. The same
# struct with `deinit existing: Self` or `var existing: Self`, or a
# legacy `__copyinit__(out self, existing: Self)`, fails as an entry
# file with "'None' has no attributes" pointing at the first use of the
# parameter. None of these is the advertised "no matching function in
# initialization" error, and none carries a fix-it toward the unified
# `__init__` rename.
#
# via_import.mojo (same directory) is the DEFECT half: it imports the
# identical struct from lib.mojo and compiles and runs cleanly, so a
# library full of legacy spellings keeps working for everyone who
# imports it and breaks only for whoever compiles a file directly.
#
# This is NOT a general legacy-leniency mode on the import path: the
# removed `inout` keyword is rejected in both contexts. The acceptance
# is specific to the legacy dunder handling.
#
# The unified spelling, def __init__(out self, *, deinit take: Self),
# compiles in BOTH contexts (verified).
#
# Verified identical on Mojo 1.0.0b2 (2cf4d08a) and 1.0.0 (ed45d567),
# macOS arm64 (Darwin 25.6.0).
#
# Run: mojo run main_vs_import.mojo
# Expected today: fails, "expected ')' in argument list" at the
# `owned existing: Self` parameter.
# Contrast: mojo run -I . via_import.mojo passes and prints -1.

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
