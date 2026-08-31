# TDD red test for a feature gap: Mojo has no way to force a struct's
# alignment BELOW its members' natural alignment, i.e. no equivalent of
# C's `#pragma pack(N)` / `__attribute__((packed))`.
#
# What exists today is `@align(N)`, and it is explicitly a MINIMUM: the
# decorator reference says "You can't reduce alignment below the natural
# alignment of the struct." Verified on both 1.0.0b2 and 1.0.0:
# `@align(4)` on a struct of 8-aligned members is accepted SILENTLY and
# clamped to 8 (align_of still reports 8). That silent clamp is its own
# little trap for anyone mirroring a `#pragma pack(4)` C struct, since
# it reads like success; the linked issue asks for a warning there even
# if the packing feature itself is declined.
#
# Why it matters: Apple's <sys/event.h> wraps `struct kevent` in
# `#pragma pack(4)`, forcing alignment 4 despite every member being
# naturally 8-byte aligned. Confirmed with a C oracle on macOS arm64:
# _Alignof(struct kevent) == 4, sizeof == 32. A Mojo struct with the
# identical field types computes alignment 8. For kevent itself that is
# harmless (the last field ends on an 8-byte boundary), but any packed
# struct whose tail ends on a 4-but-not-8 boundary would silently gain
# trailing padding a C caller does not have.
#
# Toolchains checked: Mojo 1.0.0b2 (2cf4d08a) and 1.0.0 (ed45d567),
# macOS arm64.
# Run: mojo run repro.mojo
# Expected (gap, today): fails to parse,
#   "use of unknown declaration 'packed'".
# Goes green once some below-natural alignment override ships (whether
# that is @packed, @packed(N), or @align(N) learning to reduce).

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
