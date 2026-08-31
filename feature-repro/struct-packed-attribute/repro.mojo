# M1.2 (#124) feature-gap reproducer (not a crash — an expressiveness
# gap issue #124 explicitly asks to be written up "with what it would
# need"): Mojo 1.0.0b2 has no attribute or mechanism to declare a struct
# with alignment LESS than its members' natural alignment — i.e. no
# equivalent of C's `#pragma pack(N)` / `__attribute__((packed))`.
#
# This matters for real: Apple's <sys/event.h> wraps `struct kevent` in
# `#pragma pack(4)`, which forces the struct's own alignment to 4 despite
# every member being naturally 8-byte aligned (ident/data/udata are
# uintptr_t/intptr_t/void*) — confirmed on this host via
# spike/abi/oracle.c (`_Alignof(struct kevent)` reports 4). A plain Mojo
# struct with the identical field types (spike/abi/types.mojo's `Kevent`)
# computes alignment 8, matching neither `@packed` (doesn't exist, tried
# below) nor any other override.
#
# Consequence for THIS specific struct: harmless, because kevent's last
# field ends at byte offset 32, already a multiple of 8, so the extra
# alignment Mojo computes doesn't change the struct's own size or its
# stride inside an array (verified: spike/abi/struct_layout_test.mojo's
# kevent size/offset checks all pass). It WOULD matter for a
# hypothetical struct whose tail field ended on a 4-but-not-8 byte
# boundary, where Mojo's over-alignment would insert real trailing
# padding a C caller does not expect.
#
# What full fidelity would need: a struct-level attribute (e.g.
# `@packed(4)`) or an alternate declaration form that lets a struct's
# reported/enforced alignment be set below its natural maximum-member
# alignment — either a compiler feature or, short of that, a documented
# policy that any struct needing sub-natural alignment is described as a
# byte blob with manual offset arithmetic instead of a native Mojo struct
# (the AGGREGATE RULE convention mojito_sys/io/externs.mojo already uses
# for cases it treats as opaque).
#
# Toolchain: Mojo 1.0.0b2 (2cf4d08a), macOS arm64.
# Run: mojo run m1-2-no-packed-struct-attribute.mojo
# Expected (gap): fails to parse — "use of unknown declaration 'packed'".

@packed
struct Kevent:
    var ident: UInt64
    var filter: Int16
    var flags: UInt16
    var fflags: UInt32
    var data: Int64
    var udata: UInt64

    def __init__(out self, ident: UInt64, filter: Int16, flags: UInt16, fflags: UInt32, data: Int64, udata: UInt64):
        self.ident = ident
        self.filter = filter
        self.flags = flags
        self.fflags = fflags
        self.data = data
        self.udata = udata


def main():
    pass
